# %%
# FastAPI side
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import psycopg2
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg2 import errors
from psycopg2.pool import SimpleConnectionPool

from ragpon import (
    BaseDocument,
    BM25Repository,
    ChromaDBEmbeddingAdapter,
    ChromaDBRepository,
    Config,
    Document,
    RuriLargeEmbedder,
)
from ragpon._utils.logging_helper import get_library_logger
from ragpon.api.client_init import create_openai_client
from ragpon.domain.chat import (
    DeleteRoundPayload,
    Message,
    PatchFeedbackPayload,
    SessionCreate,
    SessionUpdate,
)
from ragpon.tokenizer import SudachiTokenizer

app = FastAPI()

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host="postgres",
    dbname="postgres",
    user="postgres",
    password="postgres123",
)


def get_connection():
    """Fetch a connection from the pool."""
    return db_pool.getconn()


def put_connection(conn):
    """Return a connection to the pool."""
    db_pool.putconn(conn)


base_path = Path(__file__).parent
config = Config(config_file=base_path / "config" / "sample_config.yml")

# Load feature flags from config
raw_use_bm25 = config.get("DATABASES.USE_BM25", False)
if isinstance(raw_use_bm25, str):
    use_bm25: bool = raw_use_bm25.lower() in ("true", "1", "yes")
else:
    use_bm25 = bool(raw_use_bm25)
raw_use_chromadb = config.get("DATABASES.USE_CHROMADB", False)
if isinstance(raw_use_chromadb, str):
    use_chromadb: bool = raw_use_chromadb.lower() in ("true", "1", "yes")
else:
    use_chromadb = bool(raw_use_chromadb)

embedder = ChromaDBEmbeddingAdapter(RuriLargeEmbedder(config=config))

# Initialize logger
logger = get_library_logger(__name__)

# logging settings for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Instantiate repositories only if enabled in config
chroma_repo = (
    ChromaDBRepository(
        collection_name="pdf_collection",
        embed_func=embedder,
        metadata_class=BaseDocument,
        result_class=Document,
        similarity="cosine",
        connection_mode="http",
        folder_path=None,
        http_url="chromadb",
        port=8007,
    )
    if use_chromadb
    else None
)

bm25_repo = (
    BM25Repository(
        db_path=config.get("DATABASES.BM25_PATH"),
        schema=BaseDocument,
        result_class=Document,
        tokenizer=SudachiTokenizer(),
    )
    if use_bm25
    else None
)

client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE = create_openai_client()


def insert_three_records(
    user_id: str,
    session_id: str,
    app_name: str,
    round_id: int,
    user_msg_id: str,
    system_msg_id: str,
    assistant_msg_id: str,
    user_query: str,
    retrieved_contexts_str: str,
    total_stream_content: str,
    llm_model: str,
    rerank_model: str | None,
    rag_mode: str,
    optimized_queries: list[str] | None = None,
    use_reranker: bool = False,
    connection: psycopg2.extensions.connection | None = None,
) -> None:
    """
    Inserts three message records (user, system, assistant) into the messages table
    using explicit UUIDs and a single atomic transaction.

    Args:
        user_id: user id.
        session_id: UUID of the session.
        app_name: Application name.
        round_id (int): The round number.
        user_msg_id: UUID for the user message.
        system_msg_id: UUID for the system message.
        assistant_msg_id: UUID for the assistant message.
        user_query: User input text.
        retrieved_contexts_str: Context text retrieved from a knowledge base.
        total_stream_content: Assistant's response content.
        llm_model: Name of the LLM used (optional).
        rerank_model: Name of the reranker used (optional).
        rag_mode: Retrieval-Augmented Generation mode (optional).
        use_reranker: Whether reranking was used.
        connection: Active psycopg2 PostgreSQL connection.
    """
    created_at = datetime.now(timezone.utc)

    system_content = retrieved_contexts_str
    if optimized_queries:
        system_content += "\n\n--- Optimized Queries ---\n" + "\n".join(
            optimized_queries
        )

    records = [
        {
            "id": user_msg_id,
            "message_type": "user",
            "content": user_query,
            "created_by": user_id,
        },
        {
            "id": system_msg_id,
            "message_type": "system",
            "content": system_content,
            "created_by": "system",
        },
        {
            "id": assistant_msg_id,
            "message_type": "assistant",
            "content": total_stream_content,
            "created_by": "assistant",
        },
    ]

    with connection:
        with connection.cursor() as cursor:
            for record in records:
                cursor.execute(
                    """
                    INSERT INTO messages (
                        id, round_id, user_id, session_id, app_name,
                        content, message_type,
                        created_at, created_by,
                        is_deleted,
                        llm_model, use_reranker, rerank_model,
                        rag_mode
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s,
                        %s, %s, %s,
                        %s
                    )
                    """,
                    (
                        record["id"],
                        round_id,
                        user_id,
                        session_id,
                        app_name,
                        record["content"],
                        record["message_type"],
                        created_at,
                        record["created_by"],
                        False,
                        llm_model,
                        use_reranker,
                        rerank_model,
                        rag_mode,
                    ),
                )


def generate_queries_from_history(
    messages_list: list[dict[str, str]], system_instructions: str
) -> list[str]:
    """
    Generate multiple search queries from a given conversation history.

    This function takes the existing conversation (messages_list) and appends
    final instructions (system_instructions) to request a JSON array of objects
    like: [{"query": "..."}]. The model's response is parsed into Python objects,
    and a list of query strings is returned.

    Args:
        messages_list (list[dict[str, str]]):
            The conversation so far, e.g.:
            [
              {"role": "user", "content": "User's question 1"},
              {"role": "assistant", "content": "Assistant's answer 1"},
              ...
            ]
        system_instructions (str):
            The final system message that instructs the model to produce
            multiple queries in JSON format, for example:
            "You are a helpful assistant. Return an array of objects with
             a 'query' key in each."

    Returns:
        list[str]: A list of query strings extracted from the model's JSON output.

    Raises:
        ValueError: If the JSON returned by the model is malformed or missing
            the expected structure.
    """
    # 1) Build the messages for the ChatCompletion request
    #    We'll append a final system message at the end
    final_messages = messages_list + [
        {"role": "system", "content": system_instructions}
    ]

    # 2) Prepare ChatCompletion arguments
    kwargs = {
        "model": MODEL_NAME,
        "messages": final_messages,
        "temperature": 0.0,  # Lower temperature => more deterministic JSON
    }

    if OPENAI_TYPE == "azure" and DEPLOYMENT_ID is not None:
        kwargs["model"] = DEPLOYMENT_ID

    # 3) Call the OpenAI ChatCompletion API
    response = client.chat.completions.create(**kwargs)
    raw_text = response.choices[0].message.content.strip()

    # 4) Parse the JSON output (expected: [{"query": "..."}])
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from model: {raw_text}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got: {type(parsed)}")

    # 5) Collect queries
    queries = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict at index {i}, got: {type(item)}")
        if "query" not in item:
            raise ValueError(f"Missing 'query' key at index {i} in: {item}")
        queries.append(item["query"])

    return queries


def stream_chat_completion(
    user_id: str,
    session_id: str,
    app_name: str,
    round_id: int,
    user_msg_id: str,
    system_msg_id: str,
    assistant_msg_id: str,
    messages: list[dict],
    retrieved_contexts_str: str,
    rag_mode: str,
    optimized_queries: list[str] | None = None,
    use_reranker: bool = False,
    conn: psycopg2.extensions.connection | None = None,
) -> Generator[str, None, None]:
    """
    Generates a streaming response for the user's query by calling the LLM, then
    inserts three records into a mock database once the streaming completes.

    This function yields chunks of text to the client via Server-Sent Events (SSE)
    and accumulates the entire assistant response locally. After the final chunk
    is yielded, it performs a mock database insertion for user/system/assistant
    messages, including relevant metadata.

    Args:
        user_id (str): The ID of the user initiating the query.
        session_id (str): The ID of the session associated with this conversation.
        app_name (str): The name of the application or service.
        round_id (int): The conversation round number (used to group messages).
        user_msg_id (str): Unique identifier for the user's message.
        system_msg_id (str): Unique identifier for the system message.
        assistant_msg_id (str): Unique identifier for the assistant's message.
        messages (list[dict]): A list of messages in the format:
            [ { "role": "user"/"assistant"/"system", "content": "..." }, ... ]
        retrieved_contexts_str (str): A string containing retrieved context data
            (e.g., from RAG search) to be provided to the LLM.
        rag_mode (str): The mode to use for generating queries from rag results.
        optimized_queries (list[str] | None, optional): A list of optimized queries.
        use_reranker (bool, optional): Whether to use a reranker model. Defaults to False.
        conn (psycopg2.extensions.connection, optional): A live PostgreSQL connection used to insert records.

    Yields:
        str: SSE data chunks in the format "data: ...\\n\\n" for each partial response
        from the LLM. Also yields a
        final "[DONE]" marker if streaming completes
        successfully.

    Raises:
        Exception: Propagates any exceptions encountered during the LLM streaming
        process, yielding an error message to the SSE client.
    """
    accumulated_content = ""  # Will hold the entire assistant response
    rerank_model = None  # If you want to pass something else, do so

    try:
        # 1) Build the OpenAI messages array
        if rag_mode == "No RAG":
            openai_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please answer in Japanese.",
                },
            ]
            openai_messages.extend(messages)
        else:
            #    We can prepend system instructions + RAG context, then append the conversation
            openai_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. In your responses, please "
                        "include the file names and page numbers that serve as "
                        "the basis for your statements.  Please answer in Japanese."
                    ),
                },
                {
                    "role": "system",
                    "content": f"Context: {retrieved_contexts_str}",
                },
            ]
            # Now extend with each item in 'messages'
            # (We assume each item in messages is already {"role": ..., "content": ...})
            openai_messages.extend(messages)

        kwargs = {
            "model": MODEL_NAME,
            "messages": openai_messages,
            "temperature": 0.7,
            "stream": True,
        }
        if OPENAI_TYPE == "azure":
            kwargs["model"] = DEPLOYMENT_ID

        response = client.chat.completions.create(**kwargs)

        # (A) Yield chunks to the client and accumulate them
        for chunk in response:
            try:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    accumulated_content += content
                    yield f"data: {json.dumps({'data': content}, ensure_ascii=False)}\n\n"
            except (IndexError, AttributeError) as e:
                logger.warning(f"Error processing chunk: {e}")
                logger.debug(f"Problematic chunk: {chunk}")

        yield f"data: {json.dumps({'data': '[DONE]'}, ensure_ascii=False)}\n\n"

        user_query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        # (B) Once streaming is finished, do the mock DB insert

        conn = get_connection()
        try:
            insert_three_records(
                user_id=user_id,
                session_id=session_id,
                app_name=app_name,
                round_id=round_id,
                user_msg_id=user_msg_id,
                system_msg_id=system_msg_id,
                assistant_msg_id=assistant_msg_id,
                user_query=user_query,
                retrieved_contexts_str=retrieved_contexts_str,
                total_stream_content=accumulated_content,
                llm_model=MODEL_NAME,
                rerank_model=rerank_model,
                rag_mode=rag_mode,
                optimized_queries=optimized_queries,
                use_reranker=use_reranker,
                connection=conn,
            )
        finally:
            put_connection(conn)

    except Exception as e:
        yield f"data: Error during streaming: {str(e)}\n\n"


def build_context_string(
    rag_results: dict[str, list[Document]],
    use_chromadb: bool,
    use_bm25: bool,
) -> str:
    """
    Convert the dictionary of RAG results into a text block
    that includes relevant metadata and passages.
    Args:
        rag_results (dict[str, list[Document]]):
            Mapping from backend name ("chroma" or "bm25") to list of docs.
        use_chromadb (bool): Whether chromadb results should be included.
        use_bm25 (bool): Whether bm25 results should be included.
    Returns:
        str: Multiline string with file, page, and text from each doc.
    """
    # Determine which backends to include based on feature flags
    db_types: list[str] = []
    if use_chromadb:
        db_types.append("chroma")
    if use_bm25:
        db_types.append("bm25")

    lines: list[str] = []
    for db_type in db_types:
        for doc in rag_results.get(db_type, []):
            # Append metadata and enhanced text
            lines.append(f"File: {doc.base_document.metadata['file_path']}")
            lines.append(f"Page: {doc.base_document.metadata['page_number']}")
            lines.append(f"Text: {doc.enhanced_text}")
            lines.append("---")

    return "\n".join(lines)


@app.get("/users/{user_id}/apps/{app_name}/sessions")
async def list_sessions(user_id: str, app_name: str) -> list[dict]:
    """
    Returns a list of sessions for the specified user and application.

    This fetches session_id, session_name, and is_private_session
    from the sessions table where is_deleted = FALSE.

    Args:
        user_id (str): The ID of the user.
        app_name (str): The name of the application.

    Returns:
        list[dict]: A list of session objects.
    """
    logger.info(f"Fetching sessions for user_id={user_id}, app_name={app_name}")

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.session_id,
                    s.session_name,
                    s.is_private_session,
                    COALESCE(m.last_user_query_at, s.created_at) AS last_touched_at
                FROM sessions AS s
                LEFT JOIN (
                    SELECT
                        session_id,
                        MAX(created_at) AS last_user_query_at
                    FROM messages
                    WHERE message_type = 'user'
                    AND is_deleted = FALSE
                    GROUP BY session_id
                ) AS m ON s.session_id = m.session_id
                WHERE s.user_id = %s
                AND s.app_name = %s
                AND s.is_deleted = FALSE
                ORDER BY last_touched_at ASC
                """,
                (user_id, app_name),
            )
            rows = cursor.fetchall()

        sessions = [
            {
                "session_id": str(row[0]),
                "session_name": row[1],
                "is_private_session": row[2],
                "last_touched_at": row[3].isoformat(),
            }
            for row in rows
        ]
        return sessions

    except Exception as e:
        logger.exception("Failed to fetch sessions.")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions.")
    finally:
        put_connection(conn)


@app.get(
    "/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries",
    response_model=list[Message],
)
async def list_session_queries(
    user_id: str, app_name: str, session_id: str
) -> list[Message]:
    """
    Retrieve user and assistant messages for a specific session,
    ensuring user_id and app_name match, ordered by round_id.
    """
    conn = get_connection()

    logger.info(
        f"Querying messages: user_id={user_id}, app_name={app_name}, session_id={session_id}"
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, round_id, message_type, content, is_deleted
                FROM messages
                WHERE user_id = %s
                  AND app_name = %s
                  AND session_id = %s
                  AND message_type IN ('user', 'assistant')
                ORDER BY round_id ASC
                """,
                (user_id, app_name, session_id),
            )
            rows = cursor.fetchall()

        messages = [
            Message(
                id=str(row[0]),
                round_id=row[1],
                role=row[2],
                content=row[3],
                is_deleted=row[4],
            )
            for row in rows
        ]

        return messages

    except Exception as e:
        logger.exception("Failed to list session queries.")
        return []
    finally:
        put_connection(conn)


@app.put("/users/{user_id}/apps/{app_name}/sessions/{session_id}")
async def create_session(
    user_id: str,
    app_name: str,
    session_id: str,
    session_data: SessionCreate = Body(...),
):
    """
    Mock endpoint to create a new session. Prints out the received fields instead of actually creating a new session.

    Args:
        user_id (str): The user ID associated with the session.
        app_name (str): The name of the application context.
        session_id (str): The ID of the new session.
        session_data (SessionUpdate): The body payload containing the new session name, privacy flag, and deletion flag.

    Returns:
        dict: A simple confirmation response indicating success.
    """
    logger.info(
        f"Creating session: user_id={user_id}, app_name={app_name}, session_id={session_id}, "
        f"session_name='{session_data.session_name}', "
        f"is_private_session={session_data.is_private_session}, "
        f"is_deleted={session_data.is_deleted}"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO sessions (
                    user_id, session_id, app_name, session_name,
                    is_private_session, created_at, created_by,
                    updated_at, updated_by, is_deleted, deleted_by
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s
                )
                """,
                (
                    user_id,
                    session_id,
                    app_name,
                    session_data.session_name,
                    session_data.is_private_session,
                    datetime.now(timezone.utc),
                    user_id,
                    datetime.now(timezone.utc),
                    user_id,
                    session_data.is_deleted,
                    user_id,
                ),
            )
            conn.commit()
        return {"status": "ok", "detail": "Session created successfully."}

    except errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=409, detail="Session already exists.")
    except Exception as e:
        conn.rollback()
        logger.exception("Failed to create session.")
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        put_connection(conn)


@app.patch("/users/{user_id}/sessions/{session_id}")
async def patch_session_info(
    user_id: str,
    session_id: str,
    update_data: SessionUpdate = Body(...),
):
    """
    Updates session information in the sessions table.

    Args:
        user_id (str): The user ID associated with the session.
        session_id (str): The ID of the session to be updated.
        update_data (SessionUpdate): New session name, flags, etc.

    Returns:
        dict: Status response.
    """
    logger.info(
        f"PATCH session info: user_id={user_id}, session_id={session_id}, "
        f"session_name='{update_data.session_name}', "
        f"is_private_session={update_data.is_private_session}, "
        f"is_deleted={update_data.is_deleted}"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE sessions
                SET session_name = %s,
                    is_private_session = %s,
                    is_deleted = %s,
                    updated_at = %s,
                    updated_by = %s
                WHERE user_id = %s
                  AND session_id = %s
                """,
                (
                    update_data.session_name,
                    update_data.is_private_session,
                    update_data.is_deleted,
                    datetime.now(timezone.utc),
                    user_id,
                    user_id,
                    session_id,
                ),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Session not found")

            conn.commit()

        return JSONResponse({"status": "ok", "detail": "Session updated successfully."})

    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.exception("Failed to update session.")
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        put_connection(conn)


@app.delete("/sessions/{session_id}/rounds/{round_id}")
async def delete_round(session_id: str, round_id: int, payload: DeleteRoundPayload):
    """
    Logically deletes all messages in the given round for the specified session.

    Args:
        session_id (str): The UUID of the session.
        round_id (int): The round number (e.g., 0, 1, 2).
        payload (DeleteRoundPayload): Includes who performed the deletion.

    Returns:
        JSONResponse: Confirmation message.
    """
    logger.info(
        f"Deleting round {round_id} from session {session_id} by {payload.deleted_by}"
    )
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE messages
                SET is_deleted = TRUE,
                    deleted_by = %s,
                    updated_at = %s,
                    updated_by = %s
                WHERE session_id = %s
                  AND round_id = %s
                  AND is_deleted = FALSE
                """,
                (
                    payload.deleted_by,
                    datetime.now(timezone.utc),
                    payload.deleted_by,
                    session_id,
                    round_id,
                ),
            )
            conn.commit()

        return JSONResponse(
            {
                "status": "ok",
                "msg": f"Round {round_id} deleted in session {session_id}.",
            }
        )

    except Exception:
        conn.rollback()
        logger.exception("Failed to delete round.")
        raise HTTPException(status_code=500, detail="Failed to delete round.")
    finally:
        put_connection(conn)


@app.patch("/llm_outputs/{llm_output_id}")
async def patch_feedback(llm_output_id: str, payload: PatchFeedbackPayload):
    """
    Patch feedback for a specific LLM output message in the messages table.

    Args:
        llm_output_id (str): The UUID of the assistant message.
        payload (PatchFeedbackPayload): Contains feedback type and reason.

    Returns:
        JSONResponse: Confirmation message.
    """
    logger.info(
        f"Patching feedback for id={llm_output_id}: feedback={payload.feedback}, reason={payload.reason}"
    )

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE messages
                SET feedback = %s,
                    feedback_reason = %s,
                    feedback_at = %s,
                    updated_at = %s
                WHERE id = %s
                  AND message_type = 'assistant'
                """,
                (
                    payload.feedback,
                    payload.reason,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    llm_output_id,
                ),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="LLM output not found")

            conn.commit()

        return JSONResponse({"status": "ok", "msg": "Feedback patched successfully."})

    except HTTPException:
        raise
    except Exception:
        conn.rollback()
        logger.exception("Failed to patch feedback.")
        raise HTTPException(status_code=500, detail="Failed to patch feedback.")
    finally:
        put_connection(conn)


@app.post("/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries")
async def handle_query(
    user_id: str, app_name: str, session_id: str, request: Request
) -> StreamingResponse:
    try:
        data = await request.json()
        user_msg_id = data.get("user_msg_id", "")
        system_msg_id = data.get("system_msg_id", "")
        assistant_msg_id = data.get("assistant_msg_id", "")
        round_id = data.get("round_id", 0)
        messages_list = data.get("messages", [])  # an array of {role, content}
        rag_mode = data.get("rag_mode", "RAG (Optimized Query)")
        use_reranker = data.get("use_reranker", False)
    except Exception as e:
        return StreamingResponse(
            iter([f"data: Error parsing request: {str(e)}\n\n"]),
            media_type="text/event-stream",
        )

    user_query = ""
    for msg in reversed(messages_list):
        if msg["role"] == "user":
            user_query = msg["content"]
            break

    if rag_mode == "No RAG":
        retrieved_contexts_str = ""
        try:
            conn = get_connection()
            return StreamingResponse(
                stream_chat_completion(
                    user_id=user_id,
                    session_id=session_id,
                    app_name=app_name,
                    round_id=round_id,
                    user_msg_id=user_msg_id,
                    system_msg_id=system_msg_id,
                    assistant_msg_id=assistant_msg_id,
                    messages=messages_list,
                    retrieved_contexts_str=retrieved_contexts_str,
                    rag_mode=rag_mode,
                    optimized_queries=None,
                    use_reranker=use_reranker,
                    conn=conn,
                ),
                media_type="text/event-stream",
            )
        finally:
            put_connection(conn)

    if rag_mode == "RAG (Optimized Query)":
        # Example usage of generate_queries_from_history, if needed:
        instructions = (
            "Your task is to reconstruct the user’s request from the conversation above so it can "
            "effectively retrieve relevant documents using both vector search and BM25+ search in **Japanese**. "
            "If needed, split the request into multiple queries. You must respond **only** with "
            "strictly valid JSON, containing an array of objects, each with a single 'query' key. "
            "No extra text or explanation is allowed. For example:\n"
            "[\n"
            '  {"query": "Example query 1"},\n'
            '  {"query": "Example query 2"}\n'
            "]\n"
            "If only one query is sufficient, you may include just one object. Make sure you "
            "accurately capture the user’s intent, adding any relevant context if necessary to "
            "produce clear, natural-sounding queries."
        )

        try:
            optimized_queries = generate_queries_from_history(
                messages_list, instructions
            )
            logger.info(f"Optimized queries from LLM: {optimized_queries}")
        except ValueError as exc:
            logger.warning(
                f"Failed to generate optimized queries. Using fallback. Error: {exc}"
            )
            optimized_queries = [user_query]
    else:
        optimized_queries = [user_query]

    search_results = {
        "chroma": [],
        "bm25": [],
    }

    for query in optimized_queries:
        top_k_for_chroma = 12 // (len(optimized_queries))
        top_k_for_bm25 = 4 // (len(optimized_queries))
        enhance_num_brefore = 2
        enhance_num_after = 3

        if use_chromadb and chroma_repo:
            embedded_query = embedder(query)
            logger.info(f"Searching in ChromaDB for query: {query}")
            partial_chroma = chroma_repo.search(
                query, top_k=top_k_for_chroma, query_embeddings=embedded_query
            )
            search_results["chroma"].extend(partial_chroma)

        if use_bm25 and bm25_repo:
            logger.info(f"Searching in BM25 for query: {query}")
            partial_bm25 = bm25_repo.search(query, top_k=top_k_for_bm25)
            search_results["bm25"].extend(partial_bm25)

    enhanced_results = {}
    if use_chromadb and chroma_repo:
        enhanced_results["chroma"] = chroma_repo.enhance(
            search_results["chroma"],
            num_before=enhance_num_brefore,
            num_after=enhance_num_after,
        )
    if use_bm25 and chroma_repo:
        enhanced_results["bm25"] = chroma_repo.enhance(
            search_results["bm25"],
            num_before=enhance_num_brefore,
            num_after=enhance_num_after,
        )

    # please write rerank process
    if use_reranker:
        pass

    retrieved_contexts_str = build_context_string(
        enhanced_results, use_chromadb=use_chromadb, use_bm25=use_bm25
    )
    conn = get_connection()
    try:
        return StreamingResponse(
            stream_chat_completion(
                user_id=user_id,
                session_id=session_id,
                app_name=app_name,
                round_id=round_id,
                user_msg_id=user_msg_id,
                system_msg_id=system_msg_id,
                assistant_msg_id=assistant_msg_id,
                messages=messages_list,
                retrieved_contexts_str=retrieved_contexts_str,
                rag_mode=rag_mode,
                optimized_queries=optimized_queries,
                use_reranker=use_reranker,
                conn=conn,
            ),
            media_type="text/event-stream",
        )
    finally:
        put_connection(conn)
