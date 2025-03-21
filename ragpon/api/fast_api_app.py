# %%
# FastAPI side
import json
import logging
from typing import Generator

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ragpon import (
    BaseDocument,
    BM25Repository,
    ChromaDBEmbeddingAdapter,
    ChromaDBRepository,
    Config,
    Document,
    RuriLargeEmbedderCTranslate2,
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

config = Config(
    config_file=r"D:\Users\AtsushiSuzuki\OneDrive\デスクトップ\test\ragpon\ragpon\examples\sample_config.yml"
)
config.set(
    "DATABASES.BM25_PATH",
    "D:\\Users\\AtsushiSuzuki\\OneDrive\\デスクトップ\\test\\ragpon\\ragpon\\examples\\db\\bm25",
)
config.set(
    "DATABASES.CHROMADB_FOLDER_PATH",
    "D:\\Users\\AtsushiSuzuki\\OneDrive\\デスクトップ\\test\\ragpon\\ragpon\\examples\\db",
)

embedder = ChromaDBEmbeddingAdapter(RuriLargeEmbedderCTranslate2(config=config))

# Initialize logger
logger = get_library_logger(__name__)

# logging settings for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


chroma_repo = ChromaDBRepository(
    collection_name="pdf_collection",
    embed_func=embedder,
    metadata_class=BaseDocument,
    result_class=Document,
    similarity="cosine",
    connection_mode="http",
    folder_path=None,
    http_url="localhost",
    port=8007,
)

bm25_repo = BM25Repository(
    db_path=config.get("DATABASES.BM25_PATH"),
    schema=BaseDocument,
    result_class=Document,
    tokenizer=SudachiTokenizer(),
)

client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE = create_openai_client()


def mock_insert_three_records(
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
    Mocks inserting three records (user/system/assistant) into the DB.
    In reality, you'd do a single transaction with three INSERT statements
    or something similar.

    Args:
        user_id (str): The user ID.
        session_id (str): The session ID.
        app_name (str): The app name.
        round_id (int): The round number.
        user_msg_id (str): ID for the user message.
        system_msg_id (str): ID for the system message (if any).
        assistant_msg_id (str): ID for the assistant message.
        user_query (str): The user's query content.
        retrieved_contexts_str (str): The retrieved context string from RAG.
        total_stream_content (str): The complete assistant's streamed response.
        llm_model (str): The model name used (e.g., GPT-4).
        rerank_model (str | None): The reranker model, if any (None here).
        rag_mode (str): The mode used for generating queries from RAG results.
        optimized_queries (list[str] | None): The optimized queries, if any.
    """
    logger.info("[MOCK INSERT] Storing 3 records in the DB with the following data:")
    logger.info(
        "  user_id=%s, session_id=%s, app_name=%s, round_id=%d",
        user_id,
        session_id,
        app_name,
        round_id,
    )
    logger.info(
        "  user_msg_id=%s, system_msg_id=%s, assistant_msg_id=%s",
        user_msg_id,
        system_msg_id,
        assistant_msg_id,
    )
    logger.info("  user_query=%s", user_query)
    logger.info("  retrieved_contexts_str=%s", retrieved_contexts_str)
    logger.info("  total_stream_content=%s", total_stream_content)
    logger.info(
        "  llm_model=%s, rerank_model=%s, rag_mode=%s",
        llm_model,
        rerank_model,
        rag_mode,
    )
    logger.info("  optimized_queries=%s", str(optimized_queries))
    logger.info("  use_reranker=%s", use_reranker)


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
        kwargs["deployment_id"] = DEPLOYMENT_ID

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
        use_reanker (bool, optional): Whether to use a reranker model. Defaults to False.

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
                {"role": "system", "content": "You are a helpful assistant."},
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
                        "the basis for your statements..."
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
            kwargs["deployment_id"] = DEPLOYMENT_ID

        response = client.chat.completions.create(**kwargs)

        # (A) Yield chunks to the client and accumulate them
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                accumulated_content += content
                yield f"data: {content}\n\n"

        user_query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        # (B) Once streaming is finished, do the mock DB insert
        mock_insert_three_records(
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
        )

    except Exception as e:
        yield f"data: Error during streaming: {str(e)}\n\n"


def build_context_string(rag_results: dict) -> str:
    """
    Convert the dictionary of RAG results into a text block
    that includes relevant metadata and passages.
    """
    lines = []
    for db_type in ["chroma", "bm25"]:
        for doc in rag_results[db_type]:
            # Access doc.base_document.text, doc.base_document.doc_id, doc.metadata, etc.
            # lines.append(f"DocID: {doc.base_document.doc_id}")
            lines.append(f"File: {doc.base_document.metadata['file_path']}")
            lines.append(f"Page: {doc.base_document.metadata['page_number']}")
            lines.append(f"Text: {doc.enhanced_text}")
            lines.append("---")

    return "\n".join(lines)


@app.get("/users/{user_id}/apps/{app_name}/sessions")
async def list_sessions(user_id: str, app_name: str) -> list[dict]:
    """
    Returns a list of sessions for the specified user and application.

    This function currently returns mock data that matches the shape
    of SessionData. In a real-world implementation, you would query
    your database or other persistent storage for the sessions
    belonging to the given user and application.

    Args:
        user_id (str): The ID of the user.
        app_name (str): The name of the application.

    Returns:
        list[dict]: A list of session objects. Each session object
        contains the keys "session_id", "session_name", and
        "is_private_session".
    """
    logger.info(f"Fetching sessions for user_id={user_id}, app_name={app_name}")
    # For now, return mock data that matches the shape of SessionData
    mock_sessions = [
        {
            "session_id": "1234",
            "session_name": "The First session",
            "is_private_session": False,
        },
        {
            "session_id": "5678",
            "session_name": "Session 5678",
            "is_private_session": False,
        },
        {
            "session_id": "9999",
            "session_name": "Newest session 9999",
            "is_private_session": True,
        },
    ]
    return mock_sessions


@app.get(
    "/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries",
    response_model=list[Message],
)
async def list_session_queries(
    user_id: str, app_name: str, session_id: str
) -> list[Message]:
    """
    Retrieve the conversation history for the specified session from mock data.

    Args:
        user_id (str): The ID of the user who owns the session.
        app_name (str): The name of the application.
        session_id (str): The ID of the session to retrieve messages from.

    Returns:
        list[Message]: A list of messages, each represented by a Message dataclass.
    """
    logger.info(
        f"Retrieving queries for user_id={user_id}, app_name={app_name}, session_id={session_id}"
    )

    # Mocked session history data (replace with actual DB or logic).
    # For demonstration, we do a simple if-else to pick which messages to return.
    # In reality, you'd query your DB or another data source.
    if session_id == "1234":
        return [
            Message(
                role="user",
                content="Hi, how can I use this system (session 1234)?",
                id="usr-1234-1",
                round_id=0,
                is_deleted=False,
            ),
            Message(
                role="assistant",
                content="Hello! You can ask me anything in 1234.",
                id="ast-1234-1",
                round_id=0,
                is_deleted=False,
            ),
            Message(
                role="user",
                content="この質問は捨てられましたか？",
                id="usr-1234-2",
                round_id=1,
                is_deleted=True,
            ),
            Message(
                role="assistant",
                content="この回答が見えていたら失敗です。",
                id="ast-1234-2",
                round_id=1,
                is_deleted=True,
            ),
            Message(
                role="user",
                content="Got it. Any advanced tips for session 1234?",
                id="usr-1234-3",
                round_id=2,
                is_deleted=False,
            ),
            Message(
                role="assistant",
                content="Yes, here are advanced tips...",
                id="ast-1234-3",
                round_id=2,
                is_deleted=False,
            ),
        ]
    elif session_id == "5678":
        return [
            Message(
                role="user",
                content="Hello from session 5678! (Round 1)",
                id="usr-5678-1",
                round_id=0,
                is_deleted=False,
            ),
            Message(
                role="assistant",
                content="Hi! This is the 5678 conversation. (Round 1)",
                id="ast-5678-1",
                round_id=0,
                is_deleted=False,
            ),
            Message(
                role="user",
                content="Let's discuss something else in 5678. (Round 2)",
                id="usr-5678-2",
                round_id=1,
                is_deleted=False,
            ),
            Message(
                role="assistant",
                content="Sure, here's more about 5678. (Round 2)",
                id="ast-5678-2",
                round_id=1,
                is_deleted=False,
            ),
            Message(
                role="user",
                content="Any final points for 5678? (Round 3)",
                id="usr-5678-3",
                round_id=2,
                is_deleted=False,
            ),
            Message(
                role="assistant",
                content="Yes, final remarks on 5678... (Round 3)",
                id="ast-5678-3",
                round_id=2,
                is_deleted=False,
            ),
        ]
    elif session_id == "9999":
        return [
            Message(
                role="user",
                content="Session 9999: RAG testing.",
                id="usr-9999-1",
                round_id=0,
                is_deleted=False,
            ),
            Message(
                role="assistant",
                content="Sure, let's test RAG in session 9999.",
                id="ast-9999-1",
                round_id=0,
                is_deleted=False,
            ),
            Message(
                role="user",
                content="アルプスの少女ハイジが好きです。",
                id="usr-9999-2",
                round_id=1,
                is_deleted=False,
            ),
            Message(
                role="assistant",
                content=(
                    "「アルプスの少女ハイジ」は、スイスのアルプス山脈を舞台にした"
                    "心温まる物語ですね..."
                ),
                id="ast-9999-2",
                round_id=1,
                is_deleted=False,
            ),
        ]
    else:
        # Return an empty list if the session_id is unrecognized
        return []


@app.put("/users/{user_id}/sessions/{session_id}")
async def create_session(
    user_id: str, session_id: str, session_data: SessionCreate = Body(...)
):
    """
    Mock endpoint to create a new session. Prints out the received fields instead of actually creating a new session.

    Args:
        user_id (str): The user ID associated with the session.
        session_id (str): The ID of the new session.
        session_data (SessionUpdate): The body payload containing the new session name, privacy flag, and deletion flag.

    Returns:
        dict: A simple confirmation response indicating success.
    """
    logger.info(
        f"Received PUT for user_id={user_id}, session_id={session_id} "
        f"with session_name='{session_data.session_name}', "
        f"is_private_session={session_data.is_private_session}, "
        f"is_deleted={session_data.is_deleted}"
    )

    # Here you'd normally create a new session in your DB:

    return {"status": "ok", "detail": "Mock creation successful (no DB logic)."}


@app.patch("/users/{user_id}/sessions/{session_id}")
async def patch_session_info(
    user_id: str,
    session_id: str,
    update_data: SessionUpdate = Body(...),
):
    """
    Mock endpoint to receive session update data. Prints out the received
    fields instead of actually updating a database.

    Args:
        user_id (str): The user ID associated with the session.
        session_id (str): The ID of the session to be updated.
        update_data (SessionUpdate): The body payload containing the new
            session name, privacy flag, and deletion flag.

    Returns:
        dict: A simple confirmation response indicating success.
    """
    logger.info(
        f"Received PATCH for user_id={user_id}, session_id={session_id} "
        f"with session_name='{update_data.session_name}', "
        f"is_private_session={update_data.is_private_session}, "
        f"is_deleted={update_data.is_deleted}"
    )

    # Here you'd normally update your DB record:
    #   e.g. db.update_session(session_id, update_data.session_name, ...)

    return {"status": "ok", "detail": "Mock update successful (no DB logic)."}


@app.delete("/sessions/{session_id}/rounds/{round_id}")
async def delete_round(session_id: str, round_id: str, payload: DeleteRoundPayload):
    # Here we just log or print, but in real code you would update the DB
    return JSONResponse(
        {"status": "ok", "msg": "Round deleted (mock DB logic pending)"}
    )


@app.patch("/llm_outputs/{llm_output_id}")
async def patch_feedback(llm_output_id: str, payload: PatchFeedbackPayload):
    # Here we just log or print, but in real code you would update the DB
    return JSONResponse(
        {"status": "ok", "msg": "Feedback patched (mock DB logic pending)"}
    )


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
            ),
            media_type="text/event-stream",
        )

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
        top_k_for_chroma = 20 // (len(optimized_queries))
        top_k_for_bm25 = 8 // (len(optimized_queries))
        enhance_num_brefore = 2
        enhance_num_after = 3

        if chroma_repo:
            embedded_query = embedder(query)
            logger.info(f"Searching in ChromaDB for query: {query}")
            partial_chroma = chroma_repo.search(
                query, top_k=top_k_for_chroma, query_embeddings=embedded_query
            )
            search_results["chroma"].extend(partial_chroma)

        if bm25_repo:
            logger.info(f"Searching in BM25 for query: {query}")
            partial_bm25 = bm25_repo.search(query, top_k=top_k_for_bm25)
            search_results["bm25"].extend(partial_bm25)

    enhanced_results = {}
    enhanced_results["chroma"] = chroma_repo.enhance(
        search_results["chroma"],
        num_before=enhance_num_brefore,
        num_after=enhance_num_after,
    )
    enhanced_results["bm25"] = chroma_repo.enhance(
        search_results["bm25"],
        num_before=enhance_num_brefore,
        num_after=enhance_num_after,
    )

    # please write rerank process
    if use_reranker:
        pass

    retrieved_contexts_str = build_context_string(enhanced_results)

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
        ),
        media_type="text/event-stream",
    )
