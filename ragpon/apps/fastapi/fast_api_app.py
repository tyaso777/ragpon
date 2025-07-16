# %%
# FastAPI side
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from json import dumps
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Generator, Literal, NoReturn, cast
from uuid import UUID

from fastapi import Body, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from psycopg2 import errors
from psycopg2.errors import UniqueViolation
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
    MessageListResponse,
    PatchFeedbackPayload,
    RagModeEnum,
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
MAX_OPTIMIZED_QUERIES = 3  # "split the request into **1 to 3 queries**" is written in OPTIMIZED_QUERY_INSTRUCTION

DEFAULT_MESSAGE_LIMIT: int = 10  # initial fetch size

MAX_MESSAGE_LIMIT: int = 1000  # hard-cap to avoid accidental huge queries
# NOTE: Must be **≥ Streamlit-side MAX_MESSAGE_LIMIT (100)** so the server
#       never rejects a limit value the client may legally send.

# Factor to compensate for collisions when multiple queries are searched.
OVER_FETCH_FACTOR: int = 2

SYSTEM_PROMPT_NO_RAG = "You are a helpful assistant. Please answer in Japanese."

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a helpful assistant. Please answer in Japanese.\n"
    "If your answer is based on relevant documents, you MUST always cite the reference materials that support your statements using their RAG Rank.\n"
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

If needed, split the request into **1 to 3 queries**. You must respond **only** with strictly valid JSON, containing an array of objects, each with a single 'query' key. No extra text or explanation is allowed. For example:
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


def insert_placeholder_user_message(
    *,
    db_type: str,
    pool: SimpleConnectionPool,
    user_msg_id: str,
    user_id: str,
    session_id: str,
    app_name: str,
    round_id: int,
) -> None:
    """Insert a *placeholder* user-message row (``is_deleted=True``).

    This pre-registration guarantees that only one tab can claim a given
    ``(user_id, session_id, round_id, 'user')`` slot:

    Args:
        db_type: Database type identifier (e.g., ``"postgres"``).
        pool: Psycopg2 connection pool.
        user_msg_id: UUID for the user message row.
        user_id: Authenticated user ID.
        session_id: Parent chat-session UUID.
        app_name: Application name for telemetry.
        round_id: Incremental round number inside the session.

    Raises:
        HTTPException:
            * ``409 CONFLICT`` – another tab already inserted the same
              *placeholder* row.
            * ``500 INTERNAL SERVER ERROR`` – unexpected DB failure.
    """
    created_at = datetime.now(timezone.utc)

    try:
        with get_database_client(db_type, pool) as db:
            db.execute(
                """
                INSERT INTO messages (
                    id, round_id, user_id, session_id, app_name,
                    content, message_type,
                    created_at, created_by, is_deleted
                )
                VALUES (%s, %s, %s, %s, %s,
                        '', 'user',
                        %s, %s, TRUE)
                """,
                (
                    user_msg_id,
                    round_id,
                    user_id,
                    session_id,
                    app_name,
                    created_at,
                    user_id,  # created_by
                ),
            )
        logger.debug(
            "[insert_placeholder_user_message] Placeholder inserted: "
            f"user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
    except UniqueViolation:
        logger.error(
            "[insert_placeholder_user_message] Duplicate placeholder detected: "
            f"user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        # Abort downstream processing for this request.
        raise HTTPException(
            status_code=409,
            detail=(
                "Another session is already processing this round_id. "
                "Please retry or wait for the existing response."
            ),
        ) from None
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[insert_placeholder_user_message] Database error while inserting placeholder for "
            f"user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to pre-register user message.",
        ) from exc


def finalize_user_round(
    *,
    db_type: str,
    pool: SimpleConnectionPool,
    user_msg_id: str,
    system_msg_id: str,
    assistant_msg_id: str,
    round_id: int,
    user_id: str,
    session_id: str,
    app_name: str,
    user_query: str,
    system_content: str,
    assistant_content: str,
    llm_model: str,
    rag_mode: RagModeEnum,
    rerank_model: str | None,
    use_reranker: bool,
    optimized_queries: list[str] | None,
) -> None:
    """Complete a chat round in a single transaction.

    Steps:
        1. UPDATE the placeholder *user* row (un-delete & fill metadata).
        2. INSERT the *system* and *assistant* rows.

    Args:
        db_type: Identifier of the target DB backend (e.g., ``"postgres"``).
        pool: Psycopg2 connection pool.
        user_msg_id: UUID of the placeholder user-message row.
        system_msg_id: UUID for the system row to insert.
        assistant_msg_id: UUID for the assistant row to insert.
        round_id: Sequential round number inside this chat session.
        user_id: Authenticated user ID.
        session_id: Parent session UUID.
        app_name: Application name (for multi-app deployments).
        user_query: Final user prompt text.
        system_content: RAG context string (before query optimisation block).
        assistant_content: Full LLM answer (already accumulated).
        llm_model: Model identifier used for this response.
        rag_mode : Retrieval mode applied; see :class:`RagModeEnum`.
        rerank_model: Optional reranker model identifier.
        use_reranker: ``True`` if a reranker was applied.
        optimized_queries: Optional list of query-rewrite strings.

    Raises:
        psycopg2.DatabaseError: Unhandled DB errors bubble up; the caller
            should map these to HTTP 500.
    """
    created_at = datetime.now(timezone.utc)

    system_content_full: str = system_content
    if optimized_queries:
        system_content_full += "\n\n--- Optimized Queries ---\n" + "\n".join(
            optimized_queries
        )

    with get_database_client(db_type, pool) as db:
        # STEP-A: revive + fill user row
        db.execute(
            """
            UPDATE messages
            SET
                is_deleted   = FALSE,
                content      = %s,
                updated_at   = %s,
                llm_model    = %s,
                use_reranker = %s,
                rerank_model = %s,
                rag_mode     = %s
            WHERE id = %s
            """,
            (
                user_query,
                created_at,
                llm_model,
                use_reranker,
                rerank_model,
                rag_mode.value,
                user_msg_id,
            ),
        )

        # STEP-B: insert system & assistant rows
        db.execute(
            """
            INSERT INTO messages (
                id, round_id, user_id, session_id, app_name,
                content, message_type,
                created_at, created_by,
                is_deleted,
                llm_model, use_reranker, rerank_model,
                rag_mode
            )
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, 'system',
                 %s, 'system',
                 FALSE,
                 %s, %s, %s,
                 %s),
                (%s, %s, %s, %s, %s,
                 %s, 'assistant',
                 %s, 'assistant',
                 FALSE,
                 %s, %s, %s,
                 %s)
            """,
            (
                # system row
                system_msg_id,
                round_id,
                user_id,
                session_id,
                app_name,
                system_content_full,
                created_at,
                llm_model,
                use_reranker,
                rerank_model,
                rag_mode.value,
                # assistant row
                assistant_msg_id,
                round_id,
                user_id,
                session_id,
                app_name,
                assistant_content,
                created_at,
                llm_model,
                use_reranker,
                rerank_model,
                rag_mode.value,
            ),
        )


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

    # Limit the number of optimized queries to avoid overload or abuse
    queries = queries[:MAX_OPTIMIZED_QUERIES]

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


def stream_and_persist_chat_response(
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
    rag_mode: RagModeEnum,
    client: OpenAI | AzureOpenAI,
    model_name: str,
    model_type: Literal["openai", "azure"],
    optimized_queries: list[str] | None = None,
    use_reranker: bool = False,
) -> Generator[str, None, None]:
    """Stream an LLM reply and persist chat messages.

    Workflow:
        1. Calls the LLM in streaming mode and yields SSE chunks to the client.
        2. Accumulates the full assistant response locally.
        3. After streaming completes, finalises the placeholder *user* row
           created earlier and inserts the *system* and *assistant* rows,
           including metadata (model name, RAG mode, etc.).

    Args:
        user_id: Authenticated user ID.
        session_id: Chat-session UUID.
        app_name: Name of the calling application.
        round_id: Sequential round number inside the session.
        user_msg_id: UUID for the placeholder *user* message row.
        system_msg_id: UUID for the new *system* message row.
        assistant_msg_id: UUID for the new *assistant* message row.
        messages: Prompt messages already sent in this round
            (each as ``{"role": "...", "content": "..."}``).
        retrieved_contexts_str: Optional RAG context to prepend.
        rag_mode: Retrieval mode applied; see :class:`RagModeEnum`.
        client: OpenAI or Azure client instance.
        model_name: Model (or deployment) name.
        model_type: Backend type, ``"openai"`` or ``"azure"``.
        optimized_queries: Re-written queries produced during retrieval.
        use_reranker: ``True`` if a reranker was applied.

    Yields:
        str: Raw SSE text chunks (``"data: …\\n\\n"``).

    Raises:
        Exception: Any unexpected error during streaming or DB persistence.
            A fallback ``"data: {\"error\": …}\\n\\n"`` chunk is yielded
            before the exception propagates.
    """

    rerank_model = None  # If you want to pass something else, do so
    logger.info(
        f"[stream_and_persist_chat_response] Start: user_id={user_id}, session_id={session_id}, round_id={round_id}"
    )

    openai_messages = build_openai_messages(
        rag_mode=rag_mode,
        retrieved_contexts_str=retrieved_contexts_str,
        messages=messages,
        system_prompt_no_rag=SYSTEM_PROMPT_NO_RAG,
        system_prompt_with_context=SYSTEM_PROMPT_WITH_CONTEXT,
    )

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

        # accumulate_llm_response
        assistant_parts: list[str] = []
        for content_chunk in cast(Generator[str, None, None], response_stream):
            assistant_parts.append(content_chunk)
            yield format_sse_chunk({"data": content_chunk})

        assistant_response = "".join(assistant_parts)

        yield from yield_context_info(
            user_id=user_id,
            session_id=session_id,
            retrieved_contexts_str=retrieved_contexts_str,
            assistant_response=assistant_response,
        )

        user_query = extract_latest_user_message(
            messages_list=messages, user_id=user_id, session_id=session_id
        )

        finalize_user_round(
            db_type=DB_TYPE,
            pool=db_pool,
            user_msg_id=user_msg_id,
            system_msg_id=system_msg_id,
            assistant_msg_id=assistant_msg_id,
            round_id=round_id,
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
            user_query=user_query,
            system_content=retrieved_contexts_str,
            assistant_content=assistant_response,
            llm_model=model_name,
            rag_mode=rag_mode,
            rerank_model=rerank_model,
            use_reranker=use_reranker,
            optimized_queries=optimized_queries,
        )

        yield format_sse_chunk(data={"data": "[DONE]"})

        logger.info(
            f"[stream_and_persist_chat_response] Done streaming: user_id={user_id}, session_id={session_id}, round_id={round_id}, total_chars={len(assistant_response)}"
        )

    except Exception:
        logger.exception(
            f"[stream_and_persist_chat_response] Error: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        yield format_sse_chunk(data={"error": "Internal server error occurred"})


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
    response_model=MessageListResponse,
)
async def list_session_queries(
    user_id: str,
    app_name: str,
    session_id: str,
    limit: int = Query(
        DEFAULT_MESSAGE_LIMIT,
        ge=1,
        le=MAX_MESSAGE_LIMIT,
        description="Maximum number of messages (newest-first) to return.",
    ),
) -> MessageListResponse:
    """
    Retrieve user, assistant, and system messages for a specific session.
    If system message is present, its content will be updated to structured
    JSON of referenced RAG context rows.

    Args:
        user_id: ID of the user.
        app_name: Name of the application.
        session_id: Target session UUID.
        limit: Max number of rows to return (defaults to DEFAULT_MESSAGE_LIMIT).

    Returns:
        MessageListResponse: An envelope with a chronological list of messages
        and a flag indicating whether older rounds remain on the server.

    Returns:
    MessageListResponse: Up to *limit* newest **rounds** of chat history
    (each round can contain multiple Message objects).

    Raises:
        HTTPException: If the query fails due to database issues or unexpected errors.
    """
    logger.info(
        f"[list_session_queries] Start querying messages: user_id={user_id}, app_name={app_name}, session_id={session_id}"
    )

    try:
        with get_database_client(DB_TYPE, db_pool) as db:
            # DENSE_RANK() assigns 1 to the latest round, 2 to the next, …;
            # filtering rk ≤ limit keeps exactly limit rounds.
            extra_limit: int = limit + 1
            db.execute(
                """
                SELECT id, round_id, message_type, content, is_deleted
                FROM (
                    SELECT  *,
                            DENSE_RANK() OVER (ORDER BY round_id DESC) AS rk
                    FROM    messages
                    WHERE   user_id      = %s
                    AND   app_name     = %s
                    AND   session_id   = %s
                    AND   message_type IN ('user', 'assistant', 'system')
                ) ranked
                WHERE rk <= %s               -- keep only the newest *limit* rounds
                ORDER BY round_id ASC
                """,
                (user_id, app_name, session_id, extra_limit),
            )

            rows: list[tuple] = db.fetchall()

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

        unique_rounds: list[int] = sorted({m.round_id for m in results})
        has_more: bool = len(unique_rounds) > limit
        if has_more:
            # Keep only the newest *limit* round_ids
            keep_rounds: set[int] = set(unique_rounds[-limit:])
            results = [m for m in results if m.round_id in keep_rounds]

        unique_rounds: int = len({m.round_id for m in results})  # distinct rounds

        logger.info(
            "[list_session_queries] limit_rounds=%d → returned %d rounds (%d messages) "
            "(has_more=%s, user_id=%s, session_id=%s)",
            limit,
            unique_rounds,
            len(results),
            has_more,
            user_id,
            session_id,
        )

        return MessageListResponse(messages=results, has_more=has_more)

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


# === handle_query() support functions ===
@dataclass(slots=True)
class QueryRequest:
    """Validated payload passed through the handle_query pipeline."""

    user_msg_id: str
    system_msg_id: str
    assistant_msg_id: str
    round_id: int
    messages: list[dict]
    rag_mode: RagModeEnum
    use_reranker: bool


def _parse_request(data: dict, user_id: str, session_id: str) -> QueryRequest:
    """Parse and validate the incoming request body.

    The function normalizes the raw JSON payload into a strongly typed
    ``QueryRequest`` instance. It performs basic type coercion (e.g.,
    converting IDs to ``str`` and ``round_id`` to ``int``) and ensures
    that the ``messages`` field is a list. Any malformed or missing
    field triggers an :class:`fastapi.HTTPException` with status code 400,
    allowing the caller to surface a clear client-side error.

    Args:
        data (dict): The JSON-decoded request body sent by the client.
        user_id (str): The ID of the requesting user; used for contextual
            logging only.
        session_id (str): The chat session ID; used for contextual logging
            only.

    Returns:
        QueryRequest: A validated, normalized data object ready for the
        downstream chat-handling pipeline.

    Raises:
        HTTPException: If required fields are absent, have the wrong type,
        or fail coercion (e.g., non-integer ``round_id`` or non-list
        ``messages``).
    """
    try:
        return QueryRequest(
            user_msg_id=str(data["user_msg_id"]),
            system_msg_id=str(data["system_msg_id"]),
            assistant_msg_id=str(data["assistant_msg_id"]),
            round_id=int(data["round_id"]),
            messages=list(data["messages"]),
            rag_mode=RagModeEnum(data.get("rag_mode", RagModeEnum.OPTIMIZED.value)),
            use_reranker=bool(data.get("use_reranker", False)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.exception(
            "[_parse_request] Failed to parse query request for user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=400, detail="Invalid query payload") from exc


def _build_retrieval_queries(
    *, req: QueryRequest, user_query: str, user_id: str, session_id: str
) -> list[str]:
    """Return optimised search queries, or fall back to the raw user query.

    Parameters
    ----------
    req :
        Parsed :class:`QueryRequest` carrying ``rag_mode`` and message
        history.
    user_query :
        Latest user prompt.
    user_id :
        Authenticated user ID.
    session_id :
        Chat-session UUID.

    Returns
    -------
    list[str]
        • If ``rag_mode`` is **not** :pyattr:`RagModeEnum.OPTIMIZED`,
          returns ``[user_query]``.
        • Otherwise, returns the list produced by
          :func:`generate_queries_from_history`, or the fallback on
          failure.
    """
    if req.rag_mode is not RagModeEnum.OPTIMIZED:
        logger.info(
            "[_build_retrieval_queries] Non-optimised path: "
            "user_id=%s, session_id=%s, round_id=%s, user_query_len=%d",
            user_id,
            session_id,
            req.round_id,
            len(user_query),
        )
        return [user_query]
    try:
        queries: list[str] = generate_queries_from_history(
            user_id=user_id,
            session_id=session_id,
            messages_list=req.messages,
            system_instructions=OPTIMIZED_QUERY_INSTRUCTION,
        )

        logger.debug(
            "[_build_retrieval_queries] Optimised path: "
            "user_id=%s, session_id=%s, round_id=%s, "
            "user_query_len=%d chars → %d queries (%s)",
            user_id,
            session_id,
            req.round_id,
            len(user_query),
            len(queries),
            ", ".join(f"len={len(q)} chars" for q in queries),
        )
        return queries

    except Exception as exc:
        logger.warning(
            "[_build_retrieval_queries] Fallback to user query: "
            "user_id=%s, session_id=%s, round_id=%s, user_query_len=%d, error=%s",
            user_id,
            session_id,
            req.round_id,
            len(user_query),
            exc,
        )
        return [user_query]


def extract_latest_user_message(
    messages_list: list[dict], user_id: str, session_id: str
) -> str:
    """
    Extracts the most recent user message content from a list of message dictionaries.

    Args:
        messages_list (list[dict]): List of chat messages with 'role' and 'content' fields.
        user_id (str): ID of the user (for logging purposes).
        session_id (str): Session ID (for logging purposes).

    Returns:
        str: The latest message content from the user.

    Raises:
        HTTPException: If no user message is found in the list.
            Returns HTTP 400 (Bad Request), as this indicates a client-side input error.
    """
    for msg in reversed(messages_list):
        if msg.get("role") == "user":
            return msg.get("content", "")
    logger.warning(
        f"[extract_latest_user_message] No user message found for user_id={user_id}, session_id={session_id}"
    )
    raise HTTPException(status_code=400, detail="No user message found in request.")


def build_openai_messages(
    rag_mode: RagModeEnum,
    retrieved_contexts_str: str,
    messages: list[dict],
    system_prompt_no_rag: str,
    system_prompt_with_context: str,
) -> list[dict]:
    """
    Constructs the message list for OpenAI API based on rag_mode.

    Args:
        rag_mode (RagModeEnum): Retrieval strategy to apply.  See :class:`RagModeEnum`.
        retrieved_contexts_str (str): Context string for RAG.
        messages (list[dict]): Base chat messages.
        system_prompt_no_rag (str): System prompt for non-RAG mode.
        system_prompt_with_context (str): System prompt for RAG mode.

    Returns:
        list[dict]: The complete list of messages to send to the LLM.
    """
    system_prompt = (
        system_prompt_no_rag
        if rag_mode is RagModeEnum.NO_RAG
        else system_prompt_with_context
    )
    pre_messages = (
        [{"role": "system", "content": system_prompt}]
        if rag_mode is RagModeEnum.NO_RAG
        else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": retrieved_contexts_str},
        ]
    )
    return pre_messages + messages


def format_sse_chunk(data: dict) -> str:
    """Formats a dictionary as a Server-Sent Event (SSE) chunk."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def yield_context_info(
    user_id: str,
    session_id: str,
    retrieved_contexts_str: str,
    assistant_response: str,
) -> Generator[str, None, None]:
    """Yields structured context info based on referenced ranks from assistant output."""
    context_rows = parse_context_string_to_structured(
        content=retrieved_contexts_str, user_id=user_id, session_id=session_id
    )
    # Extract referenced RAG ranks from assistant content
    referenced_ranks = extract_referenced_rag_ranks(
        text=assistant_response, user_id=user_id, session_id=session_id
    )
    # Filter *and* preserve the order in which ranks were referenced
    rank_order = {r: i for i, r in enumerate(referenced_ranks)}
    filtered_rows = sorted(
        (r for r in context_rows if r["rag_rank"] in rank_order),
        key=lambda r: rank_order[r["rag_rank"]],
    )
    logger.debug(
        f"[yield_context_info] Referenced RAG ranks for user_id={user_id}, session_id={session_id}: {referenced_ranks}"
    )
    logger.debug(
        f"[yield_context_info] Filtered context rows for user_id={user_id}, session_id={session_id}: {len(filtered_rows)} items"
    )
    yield format_sse_chunk({"system_context_rows": filtered_rows})


def maybe_generate_session_title(
    *,
    user_id: str,
    session_id: str,
    round_id: int,
    user_query: str,
    client: OpenAI | AzureOpenAI,
    model_name: str,
    model_type: Literal["openai", "azure"],
    db_type: str,
    db_pool: Any,
) -> None:
    """
    Attempts to generate a session title for the initial round of a session.

    If the session title is still the default ("Untitled Session") and it's round 0,
    this function calls an LLM to generate a better session title and updates the DB.

    Args:
        user_id (str): The ID of the user.
        session_id (str): The ID of the session.
        round_id (int): The round number.
        user_query (str): The user's input used for generating the title.
        client (OpenAI | AzureOpenAI): The LLM client.
        model_name (str): The name of the model.
        model_type (Literal["openai", "azure"]): The type of model.
        db_type (str): The database type identifier.
        db_pool (Any): The database connection pool.
    """
    if round_id != 0:
        return

    try:
        with get_database_client(db_type, db_pool) as db:
            db.execute(
                "SELECT session_name FROM sessions WHERE session_id = %s", (session_id,)
            )
            result = db.fetchone()
            if not result:
                logger.warning(
                    f"[maybe_generate_session_title] Session not found: user_id={user_id}, session_id={session_id}"
                )
                return
            elif result[0] != "Untitled Session":
                return

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
                    (new_title, datetime.now(timezone.utc), user_id, session_id),
                )
            except Exception:
                logger.warning(
                    f"[maybe_generate_session_title] Failed to generate session title: user_id={user_id}, session_id={session_id}"
                )
    except Exception:
        logger.exception(
            f"[maybe_generate_session_title] Failed to update session title: user_id={user_id}, session_id={session_id}"
        )


def _deduplicate_by_doc_id(docs: list[Document]) -> list[Document]:
    """Remove duplicates keeping the smallest semantic-distance one.

    Args:
        docs: List of retrieved ``Document`` objects.

    Returns:
        list[Document]: Deduplicated list.
    """
    unique: dict[str, Document] = {}
    for doc in docs:
        doc_id: str = doc.base_document.doc_id
        dist: float = doc.base_document.distance
        logger.debug(
            "dedup candidate doc_id=%s distance=%.4f",  # no f-string here
            doc_id,
            dist,
        )
        # keep the closer (smaller distance) hit if we have seen this id already
        if (prev := unique.get(doc_id)) is None or dist < prev.base_document.distance:
            unique[doc_id] = doc
    return list(unique.values())


def _dedup_sort_trim(
    docs: list[Document],
    *,
    top_k: int,
) -> list[Document]:
    """Return up to ``top_k`` documents, unique and sorted by distance.

    Args:
        docs: Raw hits (may contain duplicates).
        top_k: Maximum unique items to return.

    Returns:
        Deduplicated and sorted list (len ≤ top_k).
    """
    uniques: list[Document] = _deduplicate_by_doc_id(docs)
    uniques.sort(key=lambda d: d.base_document.distance)
    return uniques[:top_k]


def _search_repositories(
    *,
    queries: list[str],
    chroma_repo: ChromaDBRepository | None,
    bm25_repo: BM25Repository | None,
    embedder: Callable[[str], list[float]],
    top_k_chroma: int,
    top_k_bm25: int,
    over_fetch_factor: int,
    user_id: str,
    session_id: str,
    round_id: int,
) -> dict[str, list[Document]]:
    """Search ChromaDB and/or BM25 and merge results.

    Args:
        queries: Already-optimised search strings.
        chroma_repo: Initialised ChromaDB repository, or ``None`` to skip.
        bm25_repo: Initialised BM25 repository, or ``None`` to skip.
        embedder: Callable that converts a query to embeddings (for ChromaDB).
        top_k_chroma: Maximum number of ChromaDB neighbours **per call**.
            The helper divides this by ``len(queries)`` and enforces
            ``>= 1`` so at least one result is requested per query.
        top_k_bm25: Same as ``top_k_chroma`` but for the BM25 repository.
        over_fetch_factor: Raw-hit multiplier to compensate for duplicates.
        user_id: User ID for contextual logging only.
        session_id: Session ID for contextual logging only.
        round_id: Round number for contextual logging only.

    Returns:
        ``{"chroma": [...], "bm25": [...]}`` — each list may be empty.
    """
    results: dict[str, list[Document]] = {"chroma": [], "bm25": []}
    per_query_k_chroma = max(1, top_k_chroma * over_fetch_factor // len(queries))
    per_query_k_bm25 = max(1, top_k_bm25 * over_fetch_factor // len(queries))

    for q in queries:
        if chroma_repo:
            try:
                results["chroma"] += chroma_repo.search(
                    q,
                    top_k=per_query_k_chroma,
                    query_embeddings=embedder(q),
                )
            except Exception:
                logger.exception(
                    f"[_search_repositories] Chroma failure: "
                    f"user_id={user_id}, session_id={session_id}, round_id={round_id}"
                )
        if bm25_repo:
            try:
                results["bm25"] += bm25_repo.search(q, top_k=per_query_k_bm25)
            except Exception:
                logger.exception(
                    f"[_search_repositories] BM25 failure:"
                    f"user_id={user_id}, session_id={session_id}, round_id={round_id}"
                )

    len_chroma_before: int = len(results["chroma"])
    results["chroma"] = _dedup_sort_trim(results["chroma"], top_k=top_k_chroma)
    len_bm25_before: int = len(results["bm25"])
    results["bm25"] = _dedup_sort_trim(results["bm25"], top_k=top_k_bm25)

    # ─── just for logging ─────────────────────────────────────────────────────
    chroma_dupes: int = len_chroma_before - len(results["chroma"])
    if chroma_dupes:
        logger.debug(
            f"[_search_repositories] Chroma duplicates removed: {chroma_dupes} "
            f"(user_id={user_id}, session_id={session_id}, round_id={round_id})"
        )

    bm25_dupes: int = len_bm25_before - len(results["bm25"])
    if bm25_dupes:
        logger.debug(
            f"[_search_repositories] BM25 duplicates removed: {bm25_dupes} "
            f"(user_id={user_id}, session_id={session_id}, round_id={round_id})"
        )

    logger.debug(
        "[_search_repositories] Final unique counts — Chroma: %d, BM25: %d "
        "(user_id=%s, session_id=%s, round_id=%d)",
        len(results["chroma"]),
        len(results["bm25"]),
        user_id,
        session_id,
        round_id,
    )

    return results


def _enhance_results(
    *,
    search_results: dict[str, list[Document]],
    chroma_repo: ChromaDBRepository | None,
    num_before: int,
    num_after: int,
    user_id: str,
    session_id: str,
) -> dict[str, list[Document]]:
    """Expand each hit by surrounding context lines / chunks.

    The helper delegates to ``chroma_repo.enhance`` for both Chroma and
    BM25 results (the latter are passed straight through the same API).

    Args:
        search_results: Output from the retrieval step —
            ``{"chroma": [...], "bm25": [...]}``.
        chroma_repo: Initialised ChromaDB repository or ``None`` to skip.
        num_before: Number of neighbouring chunks *before* each hit.
        num_after:  Number of neighbouring chunks *after* each hit.
        user_id: User ID for contextual logging only.
        session_id: Session ID for contextual logging only.

    Returns:
        dict[str, list[Document]]: Enhanced results using the same keys as
        *search_results*. Lists may be empty if a repository is disabled,
        empty, or the enhancement call fails.

    Notes:
        All exceptions thrown by the repository are caught and logged; the
        function never raises.
    """
    enhanced: dict[str, list[Document]] = {}

    if search_results["chroma"] and chroma_repo:
        try:
            enhanced["chroma"] = chroma_repo.enhance(
                search_results["chroma"],
                num_before=DEFAULT_ENHANCE_BEFORE,
                num_after=DEFAULT_ENHANCE_AFTER,
            )
        except Exception:
            logger.exception(
                f"[enhance] Chroma enhancement failed:"
                f"user_id={user_id}, session_id={session_id}"
            )

    if search_results["bm25"] and chroma_repo:
        try:
            enhanced["bm25"] = chroma_repo.enhance(
                search_results["bm25"],
                num_before=DEFAULT_ENHANCE_BEFORE,
                num_after=DEFAULT_ENHANCE_AFTER,
            )
        except Exception:
            logger.exception(
                f"[enhance] BM25 enhancement failed "
                f"(user_id={user_id}, session_id={session_id})"
            )

    return enhanced


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

    req: QueryRequest = _parse_request(
        data=await request.json(), user_id=user_id, session_id=session_id
    )
    user_query: str = extract_latest_user_message(req.messages, user_id, session_id)

    logger.info(
        f"[handle_query] ⇢ user_id={user_id}, session_id={session_id}, "
        f"round_id={req.round_id}"
    )

    insert_placeholder_user_message(
        db_type=DB_TYPE,
        pool=db_pool,
        user_msg_id=req.user_msg_id,
        user_id=user_id,
        session_id=session_id,
        app_name=app_name,
        round_id=req.round_id,
    )

    maybe_generate_session_title(
        user_id=user_id,
        session_id=session_id,
        round_id=req.round_id,
        user_query=user_query,
        client=client,
        model_name=MODEL_NAME if OPENAI_TYPE == "openai" else DEPLOYMENT_ID,
        model_type=OPENAI_TYPE,
        db_type=DB_TYPE,
        db_pool=db_pool,
    )

    if req.rag_mode is RagModeEnum.NO_RAG:
        logger.info(
            "[handle_query] Streaming in NO_RAG mode: "
            "user_id=%s, session_id=%s, round_id=%s",
            user_id,
            session_id,
            req.round_id,
        )

        return StreamingResponse(
            stream_and_persist_chat_response(
                user_id=user_id,
                session_id=session_id,
                app_name=app_name,
                round_id=req.round_id,
                user_msg_id=req.user_msg_id,
                system_msg_id=req.system_msg_id,
                assistant_msg_id=req.assistant_msg_id,
                messages=req.messages,
                retrieved_contexts_str="",
                rag_mode=req.rag_mode,
                client=client,
                model_name=MODEL_NAME if OPENAI_TYPE == "openai" else DEPLOYMENT_ID,
                model_type=OPENAI_TYPE,
                optimized_queries=None,
                use_reranker=req.use_reranker,
            ),
            media_type="text/event-stream",
        )

    queries: list[str] = _build_retrieval_queries(
        req=req, user_query=user_query, user_id=user_id, session_id=session_id
    )

    search_results = _search_repositories(
        queries=queries,
        chroma_repo=chroma_repo if use_chromadb else None,
        bm25_repo=bm25_repo if use_bm25 else None,
        embedder=embedder,
        top_k_chroma=DEFAULT_TOP_K_CHROMA,
        top_k_bm25=DEFAULT_TOP_K_BM25,
        over_fetch_factor=OVER_FETCH_FACTOR,
        user_id=user_id,
        session_id=session_id,
        round_id=req.round_id,
    )

    if not (search_results["chroma"] or search_results["bm25"]):
        logger.error(
            f"[handle_query] No search results found from either ChromaDB or BM25: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            500, "No documents were retrieved from either ChromaDB or BM25."
        )

    enhanced = _enhance_results(
        search_results=search_results,
        chroma_repo=chroma_repo,
        num_before=DEFAULT_ENHANCE_BEFORE,
        num_after=DEFAULT_ENHANCE_AFTER,
        user_id=user_id,
        session_id=session_id,
    )

    if not (enhanced.get("chroma") or enhanced.get("bm25")):
        logger.error(
            f"[handle_query] No enhanced results available from either ChromaDB or BM25: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            500,
            detail="Document retrieval succeeded, but context enhancement failed for both ChromaDB and BM25.",
        )

    # please write rerank process
    if req.use_reranker:
        pass

    ctx_str: str = build_context_string(
        enhanced,
        use_chromadb=use_chromadb,
        use_bm25=use_bm25,
        user_id=user_id,
        session_id=session_id,
    )

    return StreamingResponse(
        stream_and_persist_chat_response(
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
            round_id=req.round_id,
            user_msg_id=req.user_msg_id,
            system_msg_id=req.system_msg_id,
            assistant_msg_id=req.assistant_msg_id,
            messages=req.messages,
            retrieved_contexts_str=ctx_str,
            rag_mode=req.rag_mode,
            client=client,
            model_name=MODEL_NAME if OPENAI_TYPE == "openai" else DEPLOYMENT_ID,
            model_type=OPENAI_TYPE,
            optimized_queries=queries,
            use_reranker=req.use_reranker,
        ),
        media_type="text/event-stream",
    )
