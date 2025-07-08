# %%
# FastAPI side
import json
import logging
import os
import re
from datetime import datetime, timezone
from json import dumps
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Literal, NoReturn, cast
from uuid import UUID

from fastapi import Body, FastAPI, HTTPException, Request, status
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
from ragpon.apps.chat_domain import (
    CreateSessionWithLimitRequest,
    DeleteRoundPayload,
    Message,
    PatchFeedbackPayload,
    SessionCreate,
    SessionData,
    SessionUpdate,
    SessionUpdateWithCheckRequest,
)
from ragpon.apps.fastapi.db.db_session import get_database_client
from ragpon.apps.fastapi.openai.client_init import (
    call_llm_async_with_handling,
    call_llm_sync_with_handling,
    create_async_openai_client,
    create_openai_client,
)
from ragpon.tokenizer import SudachiTokenizer

# Initialize logger
logger = get_library_logger(__name__)

# Global (other) logger level
other_level_str = os.getenv("RAGPON_OTHER_LOG_LEVEL", "WARNING").upper()
other_level = getattr(logging, other_level_str, logging.WARNING)

# RAGPON-specific logger level
app_level_str = os.getenv("RAGPON_APP_LOG_LEVEL", "INFO").upper()
app_level = getattr(logging, app_level_str, logging.INFO)


# logging settings for debugging
logging.basicConfig(
    level=other_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set INFO level logging specifically for the ragpon.apps.fastapi package
logging.getLogger("ragpon.apps.fastapi").setLevel(app_level)

# Log the resolved log levels
logger.info(
    f"[Startup] RAGPON_APP_LOG_LEVEL resolved to {logging.getLevelName(app_level)}"
)
logger.info(
    f"[Startup] RAGPON_OTHER_LOG_LEVEL resolved to {logging.getLevelName(other_level)}"
)

app = FastAPI()

MAX_CHUNK_LOG_LEN = 300
DB_TYPE = "postgres"
MAX_TOP_K = 30

SYSTEM_PROMPT_NO_RAG = "You are a helpful assistant. Please answer in Japanese."

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a helpful assistant. Please answer in Japanese.\n"
    "You MUST always cite the reference materials that support your statements using their RAG Rank.\n"
    "Use the format: [[RAG_RANK=number]].\n"
    "This format is REQUIRED so the system can later extract references.\n"
    "If multiple sources are used, include all relevant RAG Rank values like [[RAG_RANK=1]], [[RAG_RANK=2]].\n"
    "Do NOT include doc_id or semantic distance.\n"
    "Every factual statement based on retrieved content MUST include the RAG Rank.\n"
    "Example:\n"
    "1. 強化されたテキストは、検索文脈を明確にするのに役立ちます [[RAG_RANK=5]]。\n"
    "2. 様々なテキストを扱えることで柔軟なインプットを可能にしています [[RAG_RANK=8]]。\n"
    "Be concise and accurate in your response."
)

OPTIMIZED_QUERY_INSTRUCTION = """
Your task is to reconstruct the user's request from the conversation above so it can effectively retrieve the most relevant documents using vector search in **Japanese**. 
Currently, only vector search is used (not BM25+), so it is essential that your reformulated queries maximize semantic relevance.

Please focus on generating queries that will help answer the **final user question** in the conversation. You may refer to the context of the entire conversation to understand the user's intent, but your output should support answering the final question directly and concretely.

If needed, split the request into multiple queries. You must respond **only** with strictly valid JSON, containing an array of objects, each with a single 'query' key. No extra text or explanation is allowed. For example:
[
    {"query": "Example query 1"},
    {"query": "Example query 2"}
]
If only one query is sufficient, you may include just one object. Ensure that each query captures the user's intent clearly and naturally, using language that enhances the effectiveness of vector-based retrieval.
"""

try:
    client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE = create_openai_client()
    logger.info(
        f"[Startup] OpenAI client initialized: model={MODEL_NAME}, deployment={DEPLOYMENT_ID}, type={OPENAI_TYPE}"
    )
except Exception:
    logger.exception("[Startup] Failed to initialize OpenAI client during startup")
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
    logger.info("[Startup] PostgreSQL connection pool initialized")
except Exception:
    logger.exception("[Startup] Failed to initialize PostgreSQL connection pool")
    raise

base_path = Path(__file__).parent
try:
    config = Config(config_file=base_path / "config" / "sample_config.yml")
    logger.info("[Startup] Config file loaded successfully")
except Exception:
    logger.exception("[Startup] Failed to load configuration file")
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
    logger.info("[Startup] Embedder initialized successfully")
except Exception:
    logger.exception("[Startup] Failed to initialize embedder")
    raise

logger.info(f"[Startup] use_chromadb = {use_chromadb}, use_bm25 = {use_bm25}")

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
        logger.info("[Startup] ChromaDB repository initialized")
    except Exception:
        logger.exception("[Startup] Failed to initialize ChromaDB repository")
        raise
else:
    chroma_repo = None
    logger.warning("[Startup] ChromaDB initialization is skipped (disabled by config)")

if use_bm25:
    try:
        bm25_repo = BM25Repository(
            db_path=config.get("DATABASES.BM25_PATH"),
            schema=BaseDocument,
            result_class=Document,
            tokenizer=SudachiTokenizer(),
        )
        logger.info("[Startup] BM25 repository initialized")
    except Exception:
        logger.exception("[Startup] Failed to initialize BM25 repository")
        raise
else:
    bm25_repo = None
    logger.warning("[Startup] BM25_DB initialization is skipped (disabled by config)")

# Load retrieval-related parameters from config with validation and logging
try:
    # NOTE: config.config is the raw dict loaded from YAML;
    retrieval_cfg = config.config.get("RETRIEVAL", {})
    logger.debug(f"[Startup] Loaded RETRIEVAL config: {retrieval_cfg}")

    DEFAULT_TOP_K_CHROMA = int(retrieval_cfg.get("TOP_K_CHROMADB", 12))
    DEFAULT_TOP_K_BM25 = int(retrieval_cfg.get("TOP_K_BM25", 4))
    DEFAULT_ENHANCE_BEFORE = int(retrieval_cfg.get("ENHANCE_NUM_BEFORE", 1))
    DEFAULT_ENHANCE_AFTER = int(retrieval_cfg.get("ENHANCE_NUM_AFTER", 1))

    # Validate top_k values must be positive
    if DEFAULT_TOP_K_CHROMA <= 0 or DEFAULT_TOP_K_BM25 <= 0:
        raise ValueError("TOP_K values must be positive integers")

    # Warn if values are too large (e.g., might slow down retrieval)
    if DEFAULT_TOP_K_CHROMA > MAX_TOP_K:
        logger.warning(
            f"[Startup] TOP_K_CHROMADB={DEFAULT_TOP_K_CHROMA} may be too large and impact performance"
        )
    if DEFAULT_TOP_K_BM25 > MAX_TOP_K:
        logger.warning(
            f"[Startup] TOP_K_BM25={DEFAULT_TOP_K_BM25} may be too large and impact performance"
        )

    logger.info(
        f"[Startup] Retrieval parameters loaded: "
        f"TOP_K_CHROMADB={DEFAULT_TOP_K_CHROMA}, "
        f"TOP_K_BM25={DEFAULT_TOP_K_BM25}, "
        f"ENHANCE_BEFORE={DEFAULT_ENHANCE_BEFORE}, "
        f"ENHANCE_AFTER={DEFAULT_ENHANCE_AFTER}"
    )
except (ValueError, TypeError) as exc:
    logger.exception(
        f"[Startup] Failed to parse retrieval parameters from config: {retrieval_cfg}"
    )
    raise RuntimeError("Invalid retrieval configuration values") from exc


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
                "content": SYSTEM_PROMPT_NO_RAG,
            }
        ]
    else:
        openai_messages = [
            {
                "role": "system",
                "content": (SYSTEM_PROMPT_WITH_CONTEXT),
            },
            {"role": "user", "content": retrieved_contexts_str},
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

        context_rows = parse_context_string_to_structured(
            content=retrieved_contexts_str, user_id=user_id, session_id=session_id
        )

        # Extract referenced RAG ranks from assistant content
        referenced_ranks = extract_referenced_rag_ranks(
            text=accumulated_content, user_id=user_id, session_id=session_id
        )

        # Filter *and* preserve the order in which ranks were referenced
        rank_order: dict[int, int] = {r: i for i, r in enumerate(referenced_ranks)}
        filtered_context_rows = sorted(
            (row for row in context_rows if row["rag_rank"] in rank_order),
            key=lambda r: rank_order[r["rag_rank"]],
        )

        logger.debug(
            f"[stream_chat_completion] Referenced RAG ranks for user_id={user_id}, session_id={session_id}: {referenced_ranks}"
        )
        logger.debug(
            f"[stream_chat_completion] Filtered context rows for user_id={user_id}, session_id={session_id}: {len(filtered_context_rows)} items"
        )

        yield f"data: {json.dumps({'system_context_rows': filtered_context_rows}, ensure_ascii=False)}\n\n"

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
    user_id: str,
    session_id: str,
) -> str:
    """
    Convert the dictionary of RAG results into a text block
    that includes relevant metadata and passages.
    Args:
        rag_results (dict[str, list[Document]]):
            Mapping from backend name ("chroma" or "bm25") to list of docs.
        use_chromadb (bool): Whether chromadb results should be included.
        use_bm25 (bool): Whether bm25 results should be included.
        user_id (str): ID of the user for logging context.
        session_id (str): Session ID for logging context.

    Returns:
        str: Multiline string with file, page, and text from each doc.

    Raises:
        Exception: If any unexpected error occurs during the context string
        construction. Per-document errors are logged and skipped, but a
        failure in the outer loop or formatting may still raise.
    """
    # Determine which backends to include based on feature flags
    db_types: list[str] = []
    if use_chromadb:
        db_types.append("chroma")
    if use_bm25:
        db_types.append("bm25")

    lines: list[str] = []
    try:
        for db_type in db_types:
            for i, doc in enumerate(rag_results.get(db_type, [])):
                try:
                    doc_id = doc.base_document.doc_id
                    distance = doc.base_document.distance
                    # file_path = doc.base_document.metadata.get("file_path", "unknown")
                    # page_number = doc.base_document.metadata.get(
                    #     "page_number", "unknown"
                    # )
                    enhanced_text = doc.enhanced_text or ""

                    # Append metadata and enhanced text
                    lines.append(f"RAG Rank: {i}")
                    lines.append(f"doc_id: {doc_id}")
                    lines.append(f"semantic distance: {distance}")
                    # lines.append(f"File: {file_path}")
                    # lines.append(f"Page: {page_number}")
                    lines.append(f"Text: {enhanced_text}")

                except Exception as e:
                    logger.warning(
                        f"[build_context_string] Failed to process document from {db_type}: {e}; "
                        f"user_id={user_id}, session_id={session_id}"
                    )

        max_len = 300
        joined_lines = "\n".join(lines)

        if len(joined_lines) <= max_len * 2:
            trimmed = joined_lines
        else:
            trimmed = (
                f"{joined_lines[:max_len]}... [truncated] ...{joined_lines[-max_len:]}"
            )

        logger.info(
            f"[build_context_string] Context built: user_id={user_id}, session_id={session_id}, total_lines={len(joined_lines)}"
        )
        logger.debug(
            f"[build_context_string] Context built: user_id={user_id}, session_id={session_id}, lines=\n{trimmed}"
        )
        return joined_lines

    except Exception:
        logger.exception(
            f"[build_context_string] Unexpected error while building context string: user_id={user_id}, session_id={session_id}"
        )
        return ""


def parse_context_string_to_structured(
    content: str,
    user_id: str,
    session_id: str,
) -> list[dict[str, str | float | int]]:
    """
    Parses a previously generated context string into structured RAG entries.

    This function is used to reverse-engineer the structured format when only the
    stringified context (stored in DB) is available.

    Args:
        content (str): Retrieved context string in fixed textual format.
        user_id (str): The user ID for logging and traceability.
        session_id (str): The session ID for logging and traceability.

    Returns:
        list[dict[str, str | float | int]]: List of structured entries with rag_rank, doc_id, semantic_distance, and text.

    Raises:
        Exception: If parsing fails, returns partial results and logs errors.
    """

    results: list[dict[str, str | float | int]] = []
    try:
        context_only = content.split("\n\n--- Optimized Queries ---\n")[0]

        pattern = re.compile(
            r"RAG Rank: (?P<rank>\d+)\n"
            r"doc_id: (?P<doc_id>.+?)\n"
            r"semantic distance: (?P<distance>.+?)\n"
            r"Text: (?P<text>.*?)(?=\nRAG Rank:|\Z)",
            re.DOTALL,
        )

        for match in pattern.finditer(context_only):
            results.append(
                {
                    "rag_rank": int(match.group("rank")),
                    "doc_id": match.group("doc_id").strip(),
                    "semantic_distance": float(match.group("distance")),
                    "text": match.group("text").strip(),
                }
            )
    except Exception as exc:
        logger.warning(
            f"[parse_context_string_to_structured] Failed to parse context string: user_id={user_id}, session_id={session_id}, error={exc}"
        )

    return results


def extract_referenced_rag_ranks(text: str, user_id: str, session_id: str) -> list[int]:
    """
    Extracts all unique RAG_RANK references from a given string,
    in the order they first appear.

    Example: "...これは重要です [[RAG_RANK=7]] または [[RAG_RANK=3]] ... [[RAG_RANK=3]] ..."
    will return [7, 3]

    Args:
        text (str): LLM response or assistant message.
        user_id (str): The user ID for logging and traceability.
        session_id (str): The session ID for logging and traceability.

    Returns:
        list[int]: List of unique RAG rank numbers in the order they appear.
    """
    try:
        seen = set()
        result: list[int] = []
        for match in re.findall(r"\[\[RAG_RANK=(\d+)\]\]", text):
            rank = int(match)
            if rank not in seen:
                seen.add(rank)
                result.append(rank)
        return result
    except Exception as exc:
        logger.warning(
            f"[extract_referenced_rag_ranks] Failed to extract ranks: user_id={user_id}, session_id={session_id}, error={exc}"
        )
        return []


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
        f"[list_sessions] Fetching sessions: user_id={user_id}, app_name={app_name}"
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

        logger.info(
            f"[list_sessions] Retrieved {len(sessions)} sessions for user_id={user_id}"
        )

        return sessions

    except Exception:
        logger.exception(
            f"[list_sessions] Failed to fetch sessions: user_id={user_id}, app_name={app_name}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session list from the database."
        )


@app.get(
    "/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries",
    response_model=list[Message],
)
async def list_session_queries(
    user_id: str, app_name: str, session_id: str
) -> list[Message]:
    """
    Retrieve user, assistant, and system messages for a specific session.
    If system message is present, its content will be updated to structured
    JSON of referenced RAG context rows.

    Args:
        user_id: ID of the user.
        app_name: Name of the application.
        session_id: Target session UUID.

    Returns:
        Messages in the session, structured and ordered by round_id.

    Raises:
        HTTPException: If the query fails due to database issues or unexpected errors.
    """
    logger.info(
        f"[list_session_queries] Start querying messages: user_id={user_id}, app_name={app_name}, session_id={session_id}"
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
                  AND message_type IN ('user', 'assistant', 'system')
                ORDER BY round_id ASC
                """,
                (user_id, app_name, session_id),
            )
            rows = db.fetchall()

        # Build intermediate structure by round_id
        round_raws: dict[int, dict[str, object]] = {}

        for row in rows:
            raw_dict = {
                "id": str(row[0]),
                "round_id": row[1],
                "role": row[2],
                "content": row[3],
                "is_deleted": row[4],
            }
            round_raws.setdefault(row[1], {})[row[2]] = raw_dict

        results: list[Message] = []

        for round_id in sorted(round_raws.keys()):
            round_data = round_raws[round_id]

            # Append user and assistant directly
            if "user" in round_data:
                results.append(Message(**round_data["user"]))

            if "assistant" in round_data:
                results.append(Message(**round_data["assistant"]))

            # If system message exists, modify it based on assistant content
            if "system" in round_data:
                system_msg = round_data["system"]
                try:
                    original = round_data["system"]
                    raw_context = str(original["content"])
                    context_rows = parse_context_string_to_structured(
                        content=raw_context, user_id=user_id, session_id=session_id
                    )

                    assistant_content = str(
                        round_data.get("assistant", {}).get("content", "")
                    )
                    referenced_ranks = extract_referenced_rag_ranks(
                        text=assistant_content, user_id=user_id, session_id=session_id
                    )

                    rank_order: dict[int, int] = {
                        r: i for i, r in enumerate(referenced_ranks)
                    }
                    filtered = sorted(
                        (row for row in context_rows if row["rag_rank"] in rank_order),
                        key=lambda r: rank_order[r["rag_rank"]],
                    )

                    original["content"] = dumps(filtered, ensure_ascii=False, indent=2)

                except Exception as exc:
                    logger.warning(
                        f"[list_session_queries] Failed to transform system context for user_id={user_id}, round_id={round_id}: {exc}"
                    )
                    system_msg["content"] = "[]"
                results.append(Message(**system_msg))

        logger.info(
            f"[list_session_queries] Retrieved {len(results)} messages "
            f"for user_id={user_id}, session_id={session_id}"
        )

        return results

    except Exception:
        logger.exception(
            f"[list_session_queries] Failed to fetch messages: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session queries from the database.",
        )


@app.post(
    "/users/{user_id}/apps/{app_name}/sessions/create_with_limit",
    status_code=status.HTTP_201_CREATED,
)
def create_session_with_limit(
    user_id: str,
    app_name: str,
    payload: CreateSessionWithLimitRequest,
):
    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            # BEGIN TRANSACTION
            db.execute(
                """
                SELECT session_id, session_name, is_private_session, created_at
                FROM sessions
                WHERE user_id = %s AND app_name = %s AND is_deleted = FALSE
                FOR UPDATE NOWAIT;
                """,
                (user_id, app_name),
            )

            session_rows = db.fetchall()

            # Step 2: Compute last_touched_at using messages
            session_infos = []
            for row in session_rows:
                session_id, session_name, is_private, created_at = row
                db.execute(
                    """
                    SELECT MAX(created_at)
                    FROM messages
                    WHERE user_id = %s AND session_id = %s
                      AND message_type = 'user'
                      AND is_deleted = FALSE
                    """,
                    (user_id, session_id),
                )
                last_user_query_at = db.fetchone()[0]
                last_touched_at = last_user_query_at or created_at
                session_infos.append(
                    {
                        "session_id": session_id,
                        "last_touched_at": last_touched_at,
                    }
                )

            try:
                known_ids_as_uuid = sorted(UUID(k) for k in payload.known_session_ids)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid UUID in known_session_ids."
                )

            current_ids = sorted(s["session_id"] for s in session_infos)

            if set(current_ids) != set(known_ids_as_uuid):
                logger.error(f"{set(current_ids)=}")
                logger.error(f"{set(known_ids_as_uuid)=}")

                raise HTTPException(
                    status_code=409,
                    detail="Known session IDs do not match server-side session state.",
                )

            new_session_id = payload.new_session_data.session_id
            new_session_name = payload.new_session_data.session_name
            is_private = payload.new_session_data.is_private_session
            now = datetime.now(timezone.utc)

            if len(current_ids) > 10:
                raise HTTPException(
                    status_code=500,
                    detail="Too many non-deleted sessions. Server state invalid.",
                )
            elif len(current_ids) == 10:

                try:
                    delete_target_uuid = UUID(payload.delete_target_session_id)
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail="Invalid delete_target_session_id"
                    )

                oldest = min(session_infos, key=lambda s: s["last_touched_at"])
                if delete_target_uuid != oldest["session_id"]:
                    raise HTTPException(
                        status_code=409,
                        detail="Provided delete_target_session_id is not the oldest session.",
                    )

                db.execute(
                    """
                    UPDATE sessions
                    SET is_deleted = true,
                        updated_at = %s,
                        updated_by = %s,
                        deleted_by = %s
                    WHERE user_id = %s AND session_id = %s;
                    """,
                    (now, user_id, user_id, user_id, oldest["session_id"]),
                )

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
                    new_session_id,
                    app_name,
                    new_session_name,
                    is_private,
                    now,
                    user_id,
                    None,
                    None,
                    False,
                    None,
                ),
            )

            return {
                "message": "Session created successfully",
                "session_id": new_session_id,
            }

    except HTTPException:
        raise

    except Exception:
        logger.exception("[create_session_with_limit] Unexpected server error")
        raise HTTPException(status_code=500, detail="Unexpected server error")


@app.patch(
    "/users/{user_id}/apps/{app_name}/sessions/{session_id}/update_with_check",
    status_code=status.HTTP_200_OK,
)
def update_session_with_check(
    user_id: str,
    app_name: str,
    session_id: str,
    payload: SessionUpdateWithCheckRequest,
):
    """
    Updates a session only if the current database values match the client's expected values.

    This endpoint uses pessimistic locking to prevent concurrent modifications.

    Args:
        user_id (str): The user ID.
        app_name (str): The application name.
        session_id (str): The session ID to update.
        payload (SessionUpdateWithCheckRequest): The new and old session values.

    Returns:
        dict: Confirmation message with session ID.

    Raises:
        HTTPException (404): If the session does not exist or is already deleted.
        HTTPException (409): If the session's current state does not match the client-side state.
        HTTPException (500): If an unexpected server error occurs during processing.
    """
    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            db.execute(
                """
                SELECT session_name, is_private_session, is_deleted
                FROM sessions
                WHERE user_id = %s AND app_name = %s AND session_id = %s AND is_deleted = FALSE
                FOR UPDATE NOWAIT;
                """,
                (user_id, app_name, session_id),
            )
            row = db.fetchone()
            if row is None:
                logger.warning(
                    "[update_session_with_check] Session not found for user_id={user_id}, session_id={session_id}."
                )
                raise HTTPException(status_code=404, detail="Session not found")

            current_name, current_private, current_is_deleted = row

            if (
                payload.before_session_name != current_name
                or payload.before_is_private_session != current_private
                or payload.before_is_deleted != current_is_deleted  # almost nonsense
            ):
                logger.warning(
                    "[update_session_with_check] Session conflict detected for user_id={user_id}, session_id={session_id}."
                )
                raise HTTPException(
                    status_code=409,
                    detail="Session was modified by another client. Please reload.",
                )

            now = datetime.now(timezone.utc)
            db.execute(
                """
                UPDATE sessions
                SET session_name = %s,
                    is_private_session = %s,
                    is_deleted = %s,
                    updated_at = %s,
                    updated_by = %s
                WHERE user_id = %s AND app_name = %s AND session_id = %s
                """,
                (
                    payload.after_session_name,
                    payload.after_is_private_session,
                    payload.after_is_deleted,
                    now,
                    user_id,
                    user_id,
                    app_name,
                    session_id,
                ),
            )

            return {"message": "Session updated successfully", "session_id": session_id}

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "[update_session_with_check] Unexpected error for user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=500, detail="Unexpected server error")


@app.delete("/sessions/{session_id}/rounds/{round_id}")
async def delete_round(session_id: str, round_id: int, payload: DeleteRoundPayload):
    """
    Marks all messages in a specific round as deleted.

    Args:
        session_id (str): The UUID of the session.
        round_id (int): The round number to delete.
        payload (DeleteRoundPayload): Contains user_id as deleted_by.

    Returns:
        JSONResponse: A response indicating successful logical deletion.

    Raises:
        HTTPException:
            - 404: If no messages were found to delete (already deleted or non-existent).
            - 500: On unexpected database or system errors.
    """
    user_id = payload.deleted_by

    logger.info(
        f"[delete_round] DELETE round (logical): user_id={user_id}, session_id={session_id}, round_id={round_id}"
    )

    try:
        now = datetime.now(timezone.utc)
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
                    now,
                    user_id,
                    session_id,
                    round_id,
                ),
            )

            if db.rowcount == 0:
                logger.warning(
                    f"[delete_round] No active messages found to delete: "
                    f"user_id={user_id}, session_id={session_id}, round_id={round_id}"
                )
                raise HTTPException(status_code=404, detail="No messages to delete")

        logger.info(
            f"[delete_round] Round marked as deleted: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        return JSONResponse({"status": "ok", "detail": "Messages logically deleted."})

    except HTTPException:
        raise

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
        f"[patch_feedback] Attempting feedback patch: user_id={payload.user_id}, session_id={payload.session_id}, "
        f"llm_output_id={llm_output_id}, feedback={payload.feedback}, reason={payload.reason}"
    )

    try:
        now = datetime.now(timezone.utc)
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
                    now,
                    now,
                    payload.user_id,
                    llm_output_id,
                    payload.session_id,
                ),
            )
            if db.rowcount == 0:
                logger.warning(
                    f"[patch_feedback] Feedbacked LLM output not found: user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}"
                )
                raise HTTPException(status_code=404, detail="LLM output not found")

        logger.info(
            f"[patch_feedback] Feedback successfully patched: user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}"
        )

        return JSONResponse({"status": "ok", "msg": "Feedback patched successfully."})

    except HTTPException as e:
        logger.warning(
            f"[patch_feedback] Client error: "
            f"user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}, status_code={e.status_code}, detail={e.detail}"
        )
        raise

    except Exception:
        logger.exception(
            f"[patch_feedback] Unexpected failure: user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}"
        )
        raise HTTPException(status_code=500, detail="Failed to patch feedback")


@app.post("/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries")
async def handle_query(
    user_id: str, app_name: str, session_id: str, request: Request
) -> StreamingResponse:
    """
    Handles a user query request, performs optional retrieval-augmented generation (RAG),
    and streams the assistant's response using Server-Sent Events (SSE).

    This endpoint supports two modes:
    - "No RAG": Directly sends user messages to the LLM.
    - "RAG (Optimized Query)": Generates optimized queries, retrieves relevant documents,
      enhances them with surrounding context, and streams the final LLM response.

    The function performs validation, optimized query generation, document search (ChromaDB/BM25),
    context enhancement, and response streaming. It also logs all important events.

    Args:
        user_id (str): ID of the user issuing the query.
        app_name (str): Name of the calling application.
        session_id (str): Unique session identifier for the conversation.
        request (Request): The incoming POST request, including query data.

    Returns:
        StreamingResponse: A server-sent event stream containing the assistant's response.

    Raises:
        HTTPException: On JSON parsing error, missing user message, search/enhancement failure,
        or unexpected internal server error.
    """
    try:
        data = await request.json()
    except Exception:
        logger.exception(
            f"[handle_query] Failed to parse query request for user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=400, detail="Invalid query payload")

    # Extract fields after successful parsing
    user_msg_id = data.get("user_msg_id", "")
    system_msg_id = data.get("system_msg_id", "")
    assistant_msg_id = data.get("assistant_msg_id", "")
    round_id = data.get("round_id", 0)
    messages_list = data.get("messages", [])  # an array of {role, content}
    rag_mode = data.get("rag_mode", "RAG (Optimized Query)")
    use_reranker = data.get("use_reranker", False)

    user_query = ""
    for msg in reversed(messages_list):
        if msg.get("role") == "user":
            user_query = msg["content"]
            break

    if not user_query:
        logger.warning(
            f"[handle_query] No user message found in messages_list for user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            status_code=500, detail="No user message found in request. Cannot proceed."
        )

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
        except Exception:
            logger.exception(
                f"[handle_query] Streaming failed for No RAG Mode: user_id={user_id}, session_id={session_id}, round_id={round_id}"
            )
            raise HTTPException(status_code=500, detail="Internal server error.")

    if rag_mode == "RAG (Optimized Query)":
        # Example usage of generate_queries_from_history, if needed:
        instructions = OPTIMIZED_QUERY_INSTRUCTION

        try:
            optimized_queries = generate_queries_from_history(
                user_id=user_id,
                session_id=session_id,
                messages_list=messages_list,
                system_instructions=instructions,
            )
            logger.info(
                f"[handle_query] Optimized queries from LLM for user_id={user_id}, session_id={session_id}, round_id={round_id}"
            )
            logger.debug(
                f"[handle_query] Optimized queries from LLM for user_id={user_id}, session_id={session_id}, round_id={round_id}: {optimized_queries}"
            )
        except ValueError as exc:
            logger.warning(
                f"[handle_query] Failed to generate optimized queries, fallback to user query: user_id={user_id}, session_id={session_id}, round_id={round_id}, error={exc}"
            )
            optimized_queries = [user_query]
        except Exception as exc:
            logger.warning(
                f"[handle_query] Unexpected Error while generating optimized queries, fallback to user query: user_id={user_id}, session_id={session_id}, round_id={round_id}, error={exc}"
            )
            optimized_queries = [user_query]
    else:
        # RAG (Standard) or fallback mode
        optimized_queries = [user_query]

    search_results = {
        "chroma": [],
        "bm25": [],
    }

    for query in optimized_queries:
        top_k_for_chroma = DEFAULT_TOP_K_CHROMA // (len(optimized_queries))
        top_k_for_bm25 = DEFAULT_TOP_K_BM25 // (len(optimized_queries))
        enhance_num_before = DEFAULT_ENHANCE_BEFORE
        enhance_num_after = DEFAULT_ENHANCE_AFTER

        if use_chromadb and chroma_repo:
            try:
                embedded_query = embedder(query)
                logger.info(
                    f"[handle_query] Searching in ChromaDB for user_id={user_id}, session_id={session_id}"
                )
                logger.debug(
                    f"[handle_query] Searching in ChromaDB for user_id={user_id}, session_id={session_id}, query={query}"
                )
                partial_chroma = chroma_repo.search(
                    query, top_k=top_k_for_chroma, query_embeddings=embedded_query
                )
                search_results["chroma"].extend(partial_chroma)
            except Exception as e:
                logger.exception(
                    f"[handle_query] ChromaDB search failed for query='{query}': user_id={user_id}, session_id={session_id}"
                )

        if use_bm25 and bm25_repo:
            try:
                logger.info(
                    f"[handle_query] Searching in BM25 for user_id={user_id}, session_id={session_id}"
                )
                logger.debug(
                    f"[handle_query] Searching in BM25 for user_id={user_id}, session_id={session_id}, query={query}"
                )
                partial_bm25 = bm25_repo.search(query, top_k=top_k_for_bm25)
                search_results["bm25"].extend(partial_bm25)
            except Exception as e:
                logger.exception(
                    f"[handle_query] BM25 search failed for query='{query}': user_id={user_id}, session_id={session_id}"
                )

    if not search_results["chroma"] and not search_results["bm25"]:
        logger.error(
            f"[handle_query] No search results found from either ChromaDB or BM25: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            status_code=500,
            detail="No documents were retrieved from either ChromaDB or BM25 search.",
        )

    # TODO: add Try/Except for each search to handle individual failures
    enhanced_results = {}
    if use_chromadb and chroma_repo:
        try:
            enhanced_results["chroma"] = chroma_repo.enhance(
                search_results["chroma"],
                num_before=enhance_num_before,
                num_after=enhance_num_after,
            )
        except Exception:
            logger.exception(
                f"[handle_query] ChromaDB enhancement failed: user_id={user_id}, session_id={session_id}"
            )

    if use_bm25 and chroma_repo:
        try:
            enhanced_results["bm25"] = chroma_repo.enhance(
                search_results["bm25"],
                num_before=enhance_num_before,
                num_after=enhance_num_after,
            )
        except Exception:
            logger.exception(
                f"[handle_query] BM25 enhancement failed: user_id={user_id}, session_id={session_id}"
            )

    if not enhanced_results.get("chroma") and not enhanced_results.get("bm25"):
        logger.error(
            f"[handle_query] No enhanced results available from either ChromaDB or BM25: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            status_code=500,
            detail="Document retrieval succeeded, but context enhancement failed for both ChromaDB and BM25.",
        )

    # please write rerank process
    if use_reranker:
        pass

    retrieved_contexts_str = build_context_string(
        enhanced_results,
        use_chromadb=use_chromadb,
        use_bm25=use_bm25,
        user_id=user_id,
        session_id=session_id,
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
