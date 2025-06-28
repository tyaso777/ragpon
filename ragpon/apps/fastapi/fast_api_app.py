# %%
# FastAPI side
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Literal, NoReturn, cast

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion
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
from ragpon.apps.fastapi.db.db_session import get_database_client
from ragpon.apps.fastapi.openai.client_init import (
    call_llm_async_with_handling,
    call_llm_sync_with_handling,
    create_async_openai_client,
    create_openai_client,
)
from ragpon.apps.chat_domain import (
    DeleteRoundPayload,
    Message,
    PatchFeedbackPayload,
    SessionCreate,
    SessionUpdate,
)
from ragpon.tokenizer import SudachiTokenizer

# Initialize logger
logger = get_library_logger(__name__)

# logging settings for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI()

MAX_CHUNK_LOG_LEN = 300
DB_TYPE = "postgres"

try:
    client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE = create_openai_client()
    logger.info(
        f"OpenAI client initialized: model={MODEL_NAME}, deployment={DEPLOYMENT_ID}, type={OPENAI_TYPE}"
    )
except Exception as e:
    logger.exception("Failed to initialize OpenAI client during startup")
    raise

try:
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=32,
        host="postgres",
        dbname="postgres",
        user="postgres",
        password="postgres123",
    )
    logger.info("PostgreSQL connection pool initialized")
except Exception:
    logger.exception("Failed to initialize PostgreSQL connection pool")
    raise

base_path = Path(__file__).parent
try:
    config = Config(config_file=base_path / "config" / "sample_config.yml")
    logger.info("Config file loaded successfully")
except Exception:
    logger.exception("Failed to load configuration file")
    raise


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

try:
    embedder = ChromaDBEmbeddingAdapter(RuriLargeEmbedder(config=config))
    logger.info("Embedder initialized successfully")
except Exception:
    logger.exception("Failed to initialize embedder")
    raise

# Instantiate repositories only if enabled in config
if use_chromadb:
    try:
        chroma_repo = ChromaDBRepository(
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
        logger.info("ChromaDB repository initialized")
    except Exception:
        logger.exception("Failed to initialize ChromaDB repository")
        raise
else:
    chroma_repo = None
    logger.warning("ChromaDB is disabled, skipping vector retrieval.")

if use_bm25:
    try:
        bm25_repo = BM25Repository(
            db_path=config.get("DATABASES.BM25_PATH"),
            schema=BaseDocument,
            result_class=Document,
            tokenizer=SudachiTokenizer(),
        )
        logger.info("BM25 repository initialized")
    except Exception:
        logger.exception("Failed to initialize BM25 repository")
        raise
else:
    bm25_repo = None
    logger.warning("BM25_DB is disabled, skipping bm25 retrieval.")


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
) -> None:
    """
    Inserts three message records (user, system, assistant) into the messages table
    using explicit UUIDs and a single atomic transaction.

    This function stores a single conversational round composed of:
    - The user's input (`user_query`)
    - The retrieved knowledge base context (`retrieved_contexts_str`) as a system message
    - The assistant's final response (`total_stream_content`)

    If `optimized_queries` are provided, they are appended to the system message with
    a "--- Optimized Queries ---" separator for visibility and downstream auditing.

    This function ensures all three messages are inserted into the database within a
    single transaction. If any part of the insertion fails, the entire transaction
    is rolled back to maintain data consistency.

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
        optimized_queries (list[str] | None, optional): A list of optimized queries.
        use_reranker: Whether reranking was used.

    Raises:
        ValueError: if connection is not provided.
    """
    created_at = datetime.now(timezone.utc)

    system_content = retrieved_contexts_str
    if optimized_queries:
        system_content += "\n\n--- Optimized Queries ---\n" + "\n".join(
            optimized_queries
        )

    # Each record represents a message to insert with fixed schema mapping
    created_by_system = "system"
    created_by_assistant = "assistant"
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
            "created_by": created_by_system,
        },
        {
            "id": assistant_msg_id,
            "message_type": "assistant",
            "content": total_stream_content,
            "created_by": created_by_assistant,
        },
    ]
    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            for record in records:
                db.execute(
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

        logger.info(
            f"[insert_three_records] Inserted round successfully: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )

    except Exception:
        logger.exception(
            f"[insert_three_records] insert_three_records failed: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise


def generate_queries_from_history(
    user_id: str,
    session_id: str,
    messages_list: list[dict[str, str]],
    system_instructions: str,
) -> list[str]:
    """
    Generate multiple search queries from a given conversation history.

    This function takes the existing conversation (messages_list) and appends
    final instructions (system_instructions) to request a JSON array of objects
    like: [{"query": "..."}]. The model's response is parsed into Python objects,
    and a list of query strings is returned.

    Args:
        user_id (str): ID of the user (used for logging).
        session_id (str): ID of the session (used for logging).
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
    try:
        raw_text = response.choices[0].message.content.strip()
        logger.info(
            f"[generate_queries_from_history] Generated queries: user_id={user_id}, session_id={session_id}"
        )
    except (IndexError, AttributeError) as exc:
        logger.exception(
            f"[generate_queries_from_history] Failed to extract text from OpenAI response: user_id={user_id}, session_id={session_id}"
        )
        raise ValueError("OpenAI response missing expected content") from exc
    except Exception as exc:
        logger.exception(
            f"[generate_queries_from_history] OpenAI API call failed: user_id={user_id}, session_id={session_id}"
        )
        raise ValueError("Failed to call language model") from exc

    # 4) Parse the JSON output (expected: [{"query": "..."}])
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            f"[generate_queries_from_history] Failed to parse JSON from model response: user_id={user_id}, session_id={session_id}"
        )
        logger.debug(
            f"Raw response from model for user_id={user_id}, session_id={session_id}: {raw_text}",
            exc_info=True,
        )
        raise ValueError(f"Failed to parse JSON from model: {raw_text}") from exc

    if not isinstance(parsed, list):
        logger.warning(
            f"[generate_queries_from_history] Expected JSON list, got {type(parsed)}: {raw_text} for user_id={user_id}, session_id={session_id}"
        )
        raise ValueError(f"Expected a JSON list, got: {type(parsed)}")

    # 5) Collect queries
    queries = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            logger.warning(
                f"[generate_queries_from_history] Invalid type at index {i}: expected dict, got {type(item)} for user_id={user_id}, session_id={session_id}"
            )
            raise ValueError(f"Expected dict at index {i}, got: {type(item)}")
        if "query" not in item:
            logger.warning(
                f"[generate_queries_from_history] Missing 'query' key at index {i}: {item} for user_id={user_id}, session_id={session_id}"
            )
            raise ValueError(f"Missing 'query' key at index {i} in: {item}")
        queries.append(item["query"])

    return queries


def generate_session_title(
    query: str,
    user_id: str,
    session_id: str,
    client: OpenAI | AzureOpenAI,
    model_name: str,
    model_type: Literal["openai", "azure"],
) -> str:
    """
    Generates a concise session title from the user's query using a language model.

    Args:
        query (str): The user's initial query string.
        user_id (str): The user ID for logging and traceability.
        session_id (str): The session ID for logging and traceability.
        client (OpenAI | AzureOpenAI): The OpenAI-compatible client instance.
        model_name (str): The model name (OpenAI) or deployment ID (Azure).
        model_type (Literal["openai", "azure"]): Indicates which backend is in use.

    Returns:
        str: A concise session title (up to 15 characters) in Japanese.

    Raises:
        ValueError: If the API call fails or the response is invalid.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that generates concise session titles. "
                "Based on the user's first query, provide a title of at most "
                "15 characters in Japanese."
            ),
        },
        {"role": "user", "content": query},
    ]

    response: ChatCompletion = call_llm_sync_with_handling(
        client=client,
        model=model_name,
        messages=messages,
        user_id=user_id,
        session_id=session_id,
        temperature=0.0,
        stream=False,
        model_type=model_type,
    )

    try:
        content = response.choices[0].message.content.strip()
        if not content:
            logger.exception(
                f"[generate_session_title] OpenAI returned empty title for user_id={user_id}, session_id={session_id}"
            )
            raise ValueError("OpenAI returned empty title content")
        logger.info(
            f"[generate_session_title] Title generated"
            f"for user_id={user_id}, session_id={session_id}, query='{query}'"
        )
        return content
    except (IndexError, AttributeError) as exc:
        logger.exception(
            f"[generate_session_title] Failed to extract title for user_id={user_id}, session_id={session_id}"
        )
        raise ValueError("OpenAI response missing expected content") from exc


async def generate_session_title_async(
    query: str,
    user_id: str,
    session_id: str,
    client: AsyncOpenAI | AsyncAzureOpenAI,
    model_name: str,
    model_type: Literal["openai", "azure"],
) -> str:
    """
    Generates a concise session title from the user's query using a language model.

    Args:
        query (str): The user's initial query string.
        user_id (str): The user ID for logging and traceability.
        session_id (str): The session ID for logging and traceability.
        client (AsyncOpenAI | AsyncAzureOpenAI): The OpenAI-compatible async client instance.
        model_name (str): The model name (OpenAI) or deployment ID (Azure).
        model_type (Literal["openai", "azure"]): Indicates which backend is in use.

    Returns:
        str: A concise session title (up to 15 characters) in Japanese.

    Raises:
        ValueError: If the API call fails or the response is invalid.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that generates concise session titles. "
                "Based on the user's first query, provide a title of at most "
                "15 characters in Japanese."
            ),
        },
        {"role": "user", "content": query},
    ]

    response: ChatCompletion = await call_llm_async_with_handling(
        client=client,
        model=model_name,
        messages=messages,
        user_id=user_id,
        session_id=session_id,
        temperature=0.0,
        stream=False,
        model_type=model_type,
    )

    try:
        content = response.choices[0].message.content.strip()
        if not content:
            logger.exception(
                f"[generate_session_title_async] OpenAI returned empty title for user_id={user_id}, session_id={session_id}"
            )
            raise ValueError("OpenAI returned empty title content")
        logger.info(
            f"[generate_session_title_async] Title generated"
            f"for user_id={user_id}, session_id={session_id}, query='{query}'"
        )
        return content
    except (IndexError, AttributeError) as exc:
        logger.exception(
            f"[generate_session_title_async] Failed to extract title for user_id={user_id}, session_id={session_id}"
        )
        raise ValueError("OpenAI response missing expected content") from exc


def stream_chat_completion(
    *,
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
    client: OpenAI | AzureOpenAI,
    model_name: str,
    model_type: Literal["openai", "azure"],
    optimized_queries: list[str] | None = None,
    use_reranker: bool = False,
) -> Generator[str, None, None]:
    """
    Generates a streaming response for the user's query by calling the LLM, then
    inserts three records into a database once the streaming completes.

    This function yields chunks of text to the client via Server-Sent Events (SSE)
    and accumulates the entire assistant response locally. After the final chunk
    is yielded, it performs a database insertion for user/system/assistant
    messages, including relevant metadata.
    It also attempts to generate a session title if it's the first round.

    Args:
        user_id (str): The ID of the user initiating the query.
        session_id (str): The ID of the chat session.
        app_name (str): The name of the calling application.
        round_id (int): The round number of the current interaction.
        user_msg_id (str): The UUID for the user's message.
        system_msg_id (str): The UUID for the system message.
        assistant_msg_id (str): The UUID for the assistant's message.
        messages (list[dict]): The list of chat messages for the prompt.
            [ { "role": "user"/"assistant"/"system", "content": "..." }, ... ]
        retrieved_contexts_str (str): The optional context string to inject (e.g., from RAG).
        rag_mode (str): The retrieval-augmented generation mode ("No RAG", etc.).
        client (OpenAI | AzureOpenAI): OpenAI or Azure client.
        model_name (str): The model name or deployment ID.
        model_type (Literal["openai", "azure"]): Indicates which backend is being used.
        optimized_queries (list[str] | None, optional): Optimized queries used for context retrieval.
        use_reranker (bool, optional): Whether a reranker was applied during retrieval.

    Yields:
        str: A text chunk in Server-Sent Events format (e.g., "data: ...\\n\\n").

    Raises:
        Exception: If streaming fails or DB insertion fails, a generic exception is raised,
            and a fallback "[error]" message is streamed instead.
    """
    accumulated_content = ""  # Will hold the entire assistant response
    rerank_model = None  # If you want to pass something else, do so
    logger.info(
        f"[stream_chat_completion] Start: user_id={user_id}, session_id={session_id}, round_id={round_id}"
    )

    if rag_mode == "No RAG":
        openai_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please answer in Japanese.",
            }
        ]
    else:
        openai_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. In your responses, please "
                    "include the file names and page numbers that serve as "
                    "the basis for your statements.  Please answer in Japanese."
                ),
            },
            {"role": "system", "content": f"Context: {retrieved_contexts_str}"},
        ]
    openai_messages.extend(messages)

    try:
        response_stream = call_llm_sync_with_handling(
            client=client,
            model=model_name,
            messages=openai_messages,
            user_id=user_id,
            session_id=session_id,
            stream=True,
            temperature=0.7,
            model_type=model_type,
        )

        for content_chunk in cast(Generator[str, None, None], response_stream):
            accumulated_content += content_chunk
            yield f"data: {json.dumps({'data': content_chunk}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'data': '[DONE]'}, ensure_ascii=False)}\n\n"

        user_query = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

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
            llm_model=model_name,
            rerank_model=rerank_model,
            rag_mode=rag_mode,
            optimized_queries=optimized_queries,
            use_reranker=use_reranker,
        )

        if round_id == 0:
            try:
                with get_database_client(DB_TYPE, db_pool) as db:
                    db.execute(
                        "SELECT session_name FROM sessions WHERE session_id = %s",
                        (session_id,),
                    )
                    result = db.fetchone()
                    if result is None:
                        logger.warning(
                            f"[stream_chat_completion] Session ID not found: user_id={user_id}, session_id={session_id}"
                        )
                        return
                    elif result[0] == "Untitled Session":
                        try:
                            new_title = generate_session_title(
                                query=user_query,
                                user_id=user_id,
                                session_id=session_id,
                                client=client,
                                model_name=model_name,
                                model_type=model_type,
                            )
                            db.execute(
                                """
                                UPDATE sessions SET session_name = %s, updated_at = %s, updated_by = %s
                                WHERE session_id = %s
                                """,
                                (
                                    new_title,
                                    datetime.now(timezone.utc),
                                    user_id,
                                    session_id,
                                ),
                            )
                        except Exception:
                            logger.warning(
                                f"[stream_chat_completion] Failed to generate session title: user_id={user_id}, session_id={session_id}"
                            )
            except Exception:
                logger.exception(
                    f"[stream_chat_completion] Failed to update session title: user_id={user_id}, session_id={session_id}"
                )

        logger.info(
            f"[stream_chat_completion] Done streaming: user_id={user_id}, session_id={session_id}, round_id={round_id}, total_chars={len(accumulated_content)}"
        )

    except Exception:
        logger.exception(
            f"[stream_chat_completion] Error: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        yield f"data: {json.dumps({'error': 'Internal server error occurred'}, ensure_ascii=False)}\n\n"


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
    logger.info(
        f"[list_sessions] Fetching sessions for user_id={user_id}, app_name={app_name}"
    )

    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            db.execute(
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
            rows = db.fetchall()

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

    except Exception:
        logger.exception(
            f"[list_sessions] Failed to fetch sessions: user_id={user_id}, app_name={app_name}"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions.")


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

    Args:
        user_id: ID of the user.
        app_name: Name of the application.
        session_id: Target session UUID.

    Returns:
        list[Message]: Messages in the session (user and assistant only).
    """
    logger.info(
        f"[list_session_queries] Querying messages: user_id={user_id}, app_name={app_name}, session_id={session_id}"
    )

    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            db.execute(
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
            rows = db.fetchall()

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

    except Exception:
        logger.exception(
            f"[list_session_queries] Failed to list session queries: user_id={user_id}, app_name={app_name}, session_id={session_id}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session queries."
        )


@app.put("/users/{user_id}/apps/{app_name}/sessions/{session_id}")
async def create_session(
    user_id: str,
    app_name: str,
    session_id: str,
    session_data: SessionCreate = Body(...),
):
    """
    Creates a new session record in the sessions table.

    This endpoint is typically used to register a new chat session
    for a specific user and application. The session ID is provided
    by the client, not auto-generated.

    Args:
        user_id (str): ID of the user creating the session.
        app_name (str): Name of the application context.
        session_id (str): Unique identifier for the session (UUID format).
        session_data (SessionCreate): JSON body containing:
            - session_name (str): Human-readable title of the session.
            - is_private_session (bool): Whether the session is private.
            - is_deleted (bool): Logical deletion flag.

    Returns:
        dict: A JSON object indicating success or failure. On success:
            {
                "status": "ok",
                "detail": "Session created successfully."
            }

    Raises:
        HTTPException: If a session with the same ID already exists (409),
                       or if database insertion fails (500).
    """
    logger.info(
        f"[create_session] Creating session: user_id={user_id}, app_name={app_name}, session_id={session_id}, "
        f"session_name='{session_data.session_name}', "
        f"is_private_session={session_data.is_private_session}, "
        f"is_deleted={session_data.is_deleted}"
    )

    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            db.execute(
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

        logger.info(
            f"[create_session] Session created: user_id={user_id}, session_id={session_id}, name={session_data.session_name}"
        )
        return {"status": "ok", "detail": "Session created successfully."}

    except errors.UniqueViolation:
        logger.warning(
            f"[create_session] Session already exists: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=409, detail="Session already exists.")

    except Exception:
        logger.exception(
            f"[create_session] Failed to create session: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=500, detail="Internal server error.")


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
        session_id (str): The UUID of the session to be updated.
        update_data (SessionUpdate): Fields to update (name, flags, etc.).

    Returns:
        JSONResponse: Success response message.

    Raises:
        HTTPException: 404 if session not found, 500 on internal DB error.
    """
    logger.info(
        f"[patch_session_info] PATCH session info: user_id={user_id}, session_id={session_id}, "
        f"session_name='{update_data.session_name}', "
        f"is_private_session={update_data.is_private_session}, "
        f"is_deleted={update_data.is_deleted}"
    )

    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            db.execute(
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
            if db.rowcount == 0:
                logger.warning(
                    f"[patch_session_info] Session not found: user_id={user_id}, session_id={session_id}"
                )
                raise HTTPException(status_code=404, detail="Session not found")

        logger.info(
            f"[patch_session_info] Session updated: user_id={user_id}, session_id={session_id}, name={update_data.session_name}"
        )
        return JSONResponse({"status": "ok", "detail": "Session updated successfully."})

    except HTTPException as e:
        logger.warning(
            f"[patch_session_info] Client error while updating session: user_id={user_id}, session_id={session_id}, "
            f"{e.status_code} - {e.detail}"
        )
        raise
    except Exception:
        logger.exception(
            f"[patch_session_info] Failed to update session: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=500, detail="Failed to update session.")


@app.delete("/sessions/{session_id}/rounds/{round_id}")
async def delete_round(session_id: str, round_id: int, payload: DeleteRoundPayload):
    """
    Marks all messages in a specific round as deleted.

    Args:
        session_id (str): The UUID of the session.
        round_id (int): The round number to delete.
        payload (DeleteRoundRequest): Contains user_id as deleted_by.

    Returns:
        JSONResponse: A response indicating successful logical deletion.

    Raises:
        HTTPException: 404 if no messages were found, 500 on internal DB error.
    """
    user_id = payload.deleted_by

    logger.info(
        f"[delete_round] DELETE round (logical): user_id={user_id}, session_id={session_id}, round_id={round_id}"
    )

    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            db.execute(
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
                    user_id,
                    datetime.now(timezone.utc),
                    user_id,
                    session_id,
                    round_id,
                ),
            )

            if db.rowcount == 0:
                logger.warning(
                    f"[delete_round] No active messages found to delete: user_id={user_id}, "
                    f"session_id={session_id}, round_id={round_id}"
                )
                raise HTTPException(status_code=404, detail="No messages to delete")

        logger.info(
            f"[delete_round] Round marked as deleted: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        return JSONResponse({"status": "ok", "detail": "Messages logically deleted."})

    except Exception:
        logger.exception(
            f"[delete_round] Failed to logically delete round: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise HTTPException(status_code=500, detail="Failed to delete round.")


@app.patch("/llm_outputs/{llm_output_id}")
async def patch_feedback(llm_output_id: str, payload: PatchFeedbackPayload):
    """
    Patch feedback for a specific LLM output message in the messages table.

    Args:
        llm_output_id (str): The UUID of the assistant message.
        payload (PatchFeedbackPayload): Contains feedback type and reason.

    Returns:
        JSONResponse: Confirmation message.

    Raises:
        HTTPException: 404 if the message was not found, 500 on DB failure.
    """
    logger.info(
        f"[patch_feedback] Patching feedback: user_id={payload.user_id}, session_id={payload.session_id}, "
        f"id={llm_output_id}, feedback={payload.feedback}, reason={payload.reason}"
    )

    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            db.execute(
                """
                UPDATE messages
                SET feedback = %s,
                    feedback_reason = %s,
                    feedback_at = %s,
                    updated_at = %s,
                    updated_by = %s
                WHERE id = %s
                    AND session_id = %s
                    AND message_type = 'assistant'
                """,
                (
                    payload.feedback,
                    payload.reason,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    payload.user_id,
                    llm_output_id,
                    payload.session_id,
                ),
            )
            if db.rowcount == 0:
                logger.warning(
                    f"[patch_feedback] LLM output not found: id={llm_output_id}, user_id={payload.user_id}, session_id={payload.session_id}"
                )
                raise HTTPException(status_code=404, detail="LLM output not found")

        logger.info(
            f"[patch_feedback] Feedback patched: id={llm_output_id}, feedback={payload.feedback}, reason={payload.reason}, user_id={payload.user_id}, session_id={payload.session_id}"
        )

        return JSONResponse({"status": "ok", "msg": "Feedback patched successfully."})

    except HTTPException as e:
        logger.warning(
            f"[patch_feedback] Client error in feedback patch: status_code={e.status_code}, detail={e.detail}, "
            f"user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}"
        )
        raise

    except Exception:
        logger.exception(
            f"[patch_feedback] Failed to patch feedback: user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}"
        )
        raise HTTPException(status_code=500, detail="Failed to patch feedback")


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
        logger.exception("[handle_query] Failed to parse query request")
        raise HTTPException(status_code=400, detail="Invalid query payload")

    user_query = ""
    for msg in reversed(messages_list):
        if msg["role"] == "user":
            user_query = msg["content"]
            break

    if rag_mode == "No RAG":
        retrieved_contexts_str = ""
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
                    client=client,
                    model_name=MODEL_NAME if OPENAI_TYPE == "openai" else DEPLOYMENT_ID,
                    model_type=OPENAI_TYPE,
                    optimized_queries=None,
                    use_reranker=use_reranker,
                ),
                media_type="text/event-stream",
            )
        finally:
            pass
            # put_connection(conn)

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
                user_id=user_id,
                session_id=session_id,
                messages_list=messages_list,
                system_instructions=instructions,
            )
            logger.info(
                f"[handle_query] Optimized queries from LLM: {optimized_queries}"
            )
        except ValueError as exc:
            logger.warning(
                f"[handle_query] Failed to generate optimized queries. Using fallback. Error: {exc}"
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
            logger.info(f"[handle_query] Searching in ChromaDB for query: {query}")
            partial_chroma = chroma_repo.search(
                query, top_k=top_k_for_chroma, query_embeddings=embedded_query
            )
            search_results["chroma"].extend(partial_chroma)

        if use_bm25 and bm25_repo:
            logger.info(f"[handle_query] Searching in BM25 for query: {query}")
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
                client=client,
                model_name=MODEL_NAME if OPENAI_TYPE == "openai" else DEPLOYMENT_ID,
                model_type=OPENAI_TYPE,
                optimized_queries=optimized_queries,
                use_reranker=use_reranker,
            ),
            media_type="text/event-stream",
        )
    except Exception:
        logger.exception(
            f"[handle_query] Streaming failed: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise HTTPException(status_code=500, detail="Internal server error.")


# %%
