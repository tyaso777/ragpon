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
from mysql.connector.pooling import MySQLConnectionPool
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion
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
from ragpon.apps.fastapi.db.db_errors import (
    DatabaseConflictError,
    DatabaseError,
    DatabaseQueryError,
    DatabaseUnavailableError,
)
from ragpon.apps.fastapi.db.db_session import get_database_client
from ragpon.apps.fastapi.openai.client_init import (
    call_llm_async_with_handling,
    call_llm_sync_with_handling,
    create_async_openai_client,
    create_openai_client,
)
from ragpon.apps.fastapi.prompts.prompts import (
    get_optimized_query_instruction,
    get_system_prompt_no_rag,
    get_system_prompt_with_context,
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

# Determine log file path and console logging setting from environment variables
log_path_str: str | None = os.getenv("RAGPON_LOG_PATH")
console_log_str: str = os.getenv("RAGPON_CONSOLE_LOG", "True")
console_log: bool = console_log_str.lower() in ("true", "1", "yes")

# Prepare logging handlers
handlers: list[logging.Handler] = []

if log_path_str:
    # Ensure the directory exists
    log_path = Path(log_path_str)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # File handler for log output
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handlers.append(file_handler)

# Add console handler if enabled or if no other handlers are configured
if console_log or not handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handlers.append(stream_handler)

# Configure root logger with the assembled handlers
logging.basicConfig(level=other_level, handlers=handlers)

# Set INFO level logging specifically for the ragpon.apps.fastapi package
logging.getLogger("ragpon.apps.fastapi").setLevel(app_level)

# Log the resolved log levels
logger.debug(
    f"[Startup] RAGPON_APP_LOG_LEVEL resolved to {logging.getLevelName(app_level)}"
)
logger.debug(
    f"[Startup] RAGPON_OTHER_LOG_LEVEL resolved to {logging.getLevelName(other_level)}"
)

app = FastAPI()

MAX_CHUNK_LOG_LEN = 300
# DB_TYPE = "postgres"
DB_TYPE = "mysql"
MAX_TOP_K = 30
MAX_OPTIMIZED_QUERIES = 3  # "split the request into **1 to 3 queries**" is written in OPTIMIZED_QUERY_INSTRUCTION

DEFAULT_MESSAGE_LIMIT: int = 10  # initial fetch size

MAX_MESSAGE_LIMIT: int = 1000  # hard-cap to avoid accidental huge queries
# NOTE: Must be **≥ Streamlit-side MAX_MESSAGE_LIMIT (100)** so the server
#       never rejects a limit value the client may legally send.

# Factor to compensate for collisions when multiple queries are searched.
OVER_FETCH_FACTOR: int = 2

SYSTEM_PROMPT_NO_RAG = get_system_prompt_no_rag()
SYSTEM_PROMPT_WITH_CONTEXT = get_system_prompt_with_context()
OPTIMIZED_QUERY_INSTRUCTION = get_optimized_query_instruction()

try:
    client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE = create_openai_client()
    logger.debug(
        f"[Startup] OpenAI client initialized: model={MODEL_NAME}, deployment={DEPLOYMENT_ID}, type={OPENAI_TYPE}"
    )
except Exception:
    logger.exception("[Startup] Failed to initialize OpenAI client during startup")
    raise


def _get_env(name: str) -> str:
    """Fetch an environment variable or raise if it's not set."""
    try:
        return os.environ[name]
    except KeyError as e:
        raise RuntimeError(f"Required environment variable '{name}' is not set") from e


try:
    if DB_TYPE == "postgres":
        db_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=32,
            host="postgres",
            dbname="postgres",
            user="postgres",
            password="postgres123",
        )
        logger.debug("[Startup] PostgreSQL connection pool initialized")
    elif DB_TYPE == "mysql":
        db_pool = MySQLConnectionPool(
            pool_name=_get_env("MYSQL_POOL_NAME"),
            pool_size=int(_get_env("MYSQL_POOL_SIZE")),
            host=_get_env("MYSQL_HOST"),
            port=int(_get_env("MYSQL_PORT")),
            user=_get_env("MYSQL_USER"),
            password=_get_env("MYSQL_PASSWORD"),
            database=_get_env("MYSQL_DATABASE"),
            autocommit=_get_env("MYSQL_AUTOCOMMIT").lower() in ("true", "1", "yes"),
            charset=_get_env("MYSQL_CHARSET"),
        )
        logger.debug("[Startup] MySQL connection pool initialised")
except Exception:
    logger.exception("[Startup] Failed to initialize PostgreSQL connection pool")
    raise

base_path = Path(__file__).parent
try:
    config = Config(config_file=base_path / "config" / "sample_config.yml")
    logger.debug("[Startup] Config file loaded successfully")
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
    logger.debug("[Startup] Embedder initialized successfully")
except Exception:
    logger.exception("[Startup] Failed to initialize embedder")
    raise

logger.debug(f"[Startup] use_chromadb = {use_chromadb}, use_bm25 = {use_bm25}")

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
            http_url=_get_env("CHROMADB_HOST"),  # "chromadb",
            port=8007,
        )
        logger.debug("[Startup] ChromaDB repository initialized")
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
        logger.debug("[Startup] BM25 repository initialized")
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

    logger.debug(
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
    except DatabaseConflictError as exc:
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
        ) from exc
    except DatabaseError as exc:
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
        DatabaseError: If any database-related error occurs (e.g., connection failure, constraint violation). The caller should map these to HTTP 500.
    """
    created_at = datetime.now(timezone.utc)

    system_content_full: str = system_content
    if optimized_queries:
        system_content_full += "\n\n--- Optimized Queries ---\n" + "\n".join(
            optimized_queries
        )

    try:
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
    except DatabaseError:
        logger.exception(
            "[finalize_user_round] Database error during finalization: "
            f"user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise


def generate_queries_from_history(
    user_id: str,
    session_id: str,
    messages_list: list[dict[str, str]],
    system_instructions: str,
    client: OpenAI | AzureOpenAI,
    model_name: str,
    model_type: Literal["openai", "azure"],
) -> list[str]:
    """Return retrieval queries extracted via function-calling.

    The model is forced to call a single function `emit_queries` whose schema
    guarantees valid JSON. We then parse that tool call, sanitize, de-duplicate,
    and cap the number of queries.

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
        client (OpenAI | AzureOpenAI): OpenAI-compatible client instance.
        model_name (str): Model name or Azure deployment ID.
        model_type (Literal['openai','azure']): Backend type indicator.

    Returns:
        list[str]: A list of query strings extracted from the model's JSON output.

    Raises:
        ValueError: If the model does not return the required tool call or if
            the tool-call arguments are not valid JSON matching the schema.
    """
    # 1) Put the system guardrails FIRST for maximum authority.
    final_messages = [
        {"role": "system", "content": system_instructions}
    ] + messages_list

    # 2) Function (tool) schema to force a well-typed JSON payload.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "emit_queries",
                "description": "Return retrieval queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    # 3) Call the LLM with tools enabled and force the specific function.
    response: ChatCompletion = call_llm_sync_with_handling(
        client=client,
        model=model_name,
        messages=final_messages,
        user_id=user_id,
        session_id=session_id,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=False,
        model_type=model_type,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "emit_queries"}},
    )

    # 4) Locate the correct tool call by name (not just index 0).
    tool_calls = response.choices[0].message.tool_calls or []
    emit_call = next(
        (tc for tc in tool_calls if tc.function.name == "emit_queries"), None
    )
    if emit_call is None:
        logger.error(
            "[generate_queries_from_history] No 'emit_queries' tool call: "
            "user_id=%s, session_id=%s, tool_calls=%s",
            user_id,
            session_id,
            [tc.function.name for tc in tool_calls],
        )
        raise ValueError("Expected tool call 'emit_queries' but got none")

    # 5) Parse and validate arguments; fail closed on invalid JSON.
    try:
        args = json.loads(emit_call.function.arguments)
    except json.JSONDecodeError as exc:
        logger.error(
            "[generate_queries_from_history] JSON parse error from tool call: "
            "user_id=%s, session_id=%s, err=%s, raw=%s",
            user_id,
            session_id,
            exc,
            emit_call.function.arguments[:500],
        )
        raise ValueError("Malformed JSON in tool call arguments") from exc

    items = cast(list[dict[str, str]], args.get("items", []))

    # 6) Sanitize: strip, drop empties, preserve order, dedupe, and cap length.
    seen: set[str] = set()
    queries: list[str] = []
    for it in items:
        q = (it.get("query") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        queries.append(q)
        if len(queries) >= MAX_OPTIMIZED_QUERIES:
            break

    logger.info(
        "[generate_queries_from_history] Generated %d query(ies): user_id=%s, session_id=%s",
        len(queries),
        user_id,
        session_id,
    )

    return queries


# def generate_queries_from_history(
#     user_id: str,
#     session_id: str,
#     messages_list: list[dict[str, str]],
#     system_instructions: str,
#     client: OpenAI | AzureOpenAI,
#     model_name: str,
#     model_type: Literal["openai", "azure"],
# ) -> list[str]:
#     """
#     Generate multiple search queries from a given conversation history.

#     This function takes the existing conversation (messages_list) and appends
#     final instructions (system_instructions) to request a JSON array of objects
#     like: [{"query": "..."}]. The model's response is parsed into Python objects,
#     and a list of query strings is returned.

#     Args:
#         user_id (str): ID of the user (used for logging).
#         session_id (str): ID of the session (used for logging).
#         messages_list (list[dict[str, str]]):
#             The conversation so far, e.g.:
#             [
#               {"role": "user", "content": "User's question 1"},
#               {"role": "assistant", "content": "Assistant's answer 1"},
#               ...
#             ]
#         system_instructions (str):
#             The final system message that instructs the model to produce
#             multiple queries in JSON format, for example:
#             "You are a helpful assistant. Return an array of objects with
#              a 'query' key in each."
#         client (OpenAI | AzureOpenAI): OpenAI-compatible client instance.
#         model_name (str): Model name or Azure deployment ID.
#         model_type (Literal['openai','azure']): Backend type indicator.

#     Returns:
#         list[str]: A list of query strings extracted from the model's JSON output.

#     Raises:
#         ValueError: If the JSON returned by the model is malformed or missing
#             the expected structure.
#     """
#     # Build the messages for the ChatCompletion request
#     #    We'll append a final system message at the end
#     final_messages = messages_list + [
#         {"role": "system", "content": system_instructions}
#     ]

#     try:
#         response: ChatCompletion = call_llm_sync_with_handling(
#             client=client,
#             model=model_name,
#             messages=final_messages,
#             user_id=user_id,
#             session_id=session_id,
#             temperature=0.0,
#             stream=False,
#             model_type=model_type,
#         )
#         raw_text = response.choices[0].message.content.strip()
#         logger.info(
#             f"[generate_queries_from_history] Generated queries using OpenAI response: user_id={user_id}, session_id={session_id}"
#         )
#     except (IndexError, AttributeError) as exc:
#         logger.exception(
#             f"[generate_queries_from_history] Failed to extract text from OpenAI response: user_id={user_id}, session_id={session_id}"
#         )
#         raise ValueError("OpenAI response missing expected content") from exc
#     except Exception as exc:
#         logger.exception(
#             f"[generate_queries_from_history] OpenAI API call failed: user_id={user_id}, session_id={session_id}"
#         )
#         raise ValueError("Failed to call language model") from exc

#     # Parse the JSON output (expected: [{"query": "..."}])
#     try:
#         parsed = json.loads(raw_text)
#     except json.JSONDecodeError as exc:
#         logger.warning(
#             f"[generate_queries_from_history] Failed to parse JSON from model response: user_id={user_id}, session_id={session_id}, len(raw_text)={len(raw_text)}"
#         )
#         raise ValueError(f"Failed to parse JSON from model") from exc

#     if not isinstance(parsed, list):
#         logger.warning(
#             f"[generate_queries_from_history] Expected JSON list, got {type(parsed)}: {raw_text} for user_id={user_id}, session_id={session_id}"
#         )
#         raise ValueError(f"Expected a JSON list, got: {type(parsed)}")

#     # Collect queries
#     queries = []
#     for i, item in enumerate(parsed):
#         if not isinstance(item, dict):
#             logger.warning(
#                 f"[generate_queries_from_history] Invalid type at index {i}: expected dict, got {type(item)} for user_id={user_id}, session_id={session_id}"
#             )
#             raise ValueError(f"Expected dict at index {i}, got: {type(item)}")
#         if "query" not in item:
#             logger.warning(
#                 f"[generate_queries_from_history] Missing 'query' key at index {i}: {item} for user_id={user_id}, session_id={session_id}"
#             )
#             raise ValueError(f"Missing 'query' key at index {i} in: {item}")
#         queries.append(item["query"])

#     # Limit the number of optimized queries to avoid overload or abuse
#     queries = queries[:MAX_OPTIMIZED_QUERIES]

#     return queries


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
            f"[generate_session_title] Title generated using OpenAI response for user_id={user_id}, session_id={session_id}, len(query)='{len(query)}'"
        )
        return content
    except (IndexError, AttributeError) as exc:
        logger.exception(
            f"[generate_session_title] Failed to extract title using OpenAI response for user_id={user_id}, session_id={session_id}"
        )
        raise ValueError("OpenAI response missing expected content") from exc


# async def generate_session_title_async(
#     query: str,
#     user_id: str,
#     session_id: str,
#     client: AsyncOpenAI | AsyncAzureOpenAI,
#     model_name: str,
#     model_type: Literal["openai", "azure"],
# ) -> str:
#     """
#     Generates a concise session title from the user's query using a language model.

#     Args:
#         query (str): The user's initial query string.
#         user_id (str): The user ID for logging and traceability.
#         session_id (str): The session ID for logging and traceability.
#         client (AsyncOpenAI | AsyncAzureOpenAI): The OpenAI-compatible async client instance.
#         model_name (str): The model name (OpenAI) or deployment ID (Azure).
#         model_type (Literal["openai", "azure"]): Indicates which backend is in use.

#     Returns:
#         str: A concise session title (up to 15 characters) in Japanese.

#     Raises:
#         ValueError: If the API call fails or the response is invalid.
#     """
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are an assistant that generates concise session titles. "
#                 "Based on the user's first query, provide a title of at most "
#                 "15 characters in Japanese."
#             ),
#         },
#         {"role": "user", "content": query},
#     ]

#     response: ChatCompletion = await call_llm_async_with_handling(
#         client=client,
#         model=model_name,
#         messages=messages,
#         user_id=user_id,
#         session_id=session_id,
#         temperature=0.0,
#         stream=False,
#         model_type=model_type,
#     )

#     try:
#         content = response.choices[0].message.content.strip()
#         if not content:
#             logger.exception(
#                 f"[generate_session_title_async] OpenAI returned empty title for user_id={user_id}, session_id={session_id}"
#             )
#             raise ValueError("OpenAI returned empty title content")
#         logger.info(
#             f"[generate_session_title_async] Title generated"
#             f"for user_id={user_id}, session_id={session_id}, len(query)='{len(query)}'"
#         )
#         return content
#     except (IndexError, AttributeError) as exc:
#         logger.exception(
#             f"[generate_session_title_async] Failed to extract title for user_id={user_id}, session_id={session_id}"
#         )
#         raise ValueError("OpenAI response missing expected content") from exc


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
    logger.debug(
        f"[stream_and_persist_chat_response] Starting stream_and_persist_chat_response: user_id={user_id}, session_id={session_id}, round_id={round_id}"
    )

    openai_messages = build_openai_messages(
        rag_mode=rag_mode,
        retrieved_contexts_str=retrieved_contexts_str,
        messages=messages,
        system_prompt_no_rag=SYSTEM_PROMPT_NO_RAG,
        system_prompt_with_context=SYSTEM_PROMPT_WITH_CONTEXT,
    )

    logger.debug(
        "[stream_and_persist_chat_response] Built OpenAI messages: user_id=%s, session_id=%s, round_id=%s, message_count=%d",
        user_id,
        session_id,
        round_id,
        len(openai_messages),
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

        logger.debug(
            "[stream_and_persist_chat_response] LLM stream generator acquired: user_id=%s, session_id=%s, round_id=%s",
            user_id,
            session_id,
            round_id,
        )

        # accumulate_llm_response
        assistant_parts: list[str] = []
        for content_chunk in cast(Generator[str, None, None], response_stream):
            assistant_parts.append(content_chunk)
            yield format_sse_chunk({"data": content_chunk})

        assistant_response = "".join(assistant_parts)

        logger.debug(
            "[stream_and_persist_chat_response] Yielding context info start: user_id=%s, session_id=%s, round_id=%s",
            user_id,
            session_id,
            round_id,
        )

        yield from yield_context_info(
            user_id=user_id,
            session_id=session_id,
            retrieved_contexts_str=retrieved_contexts_str,
            assistant_response=assistant_response,
        )

        logger.debug(
            "[stream_and_persist_chat_response] Yielding context info done: user_id=%s, session_id=%s, round_id=%s",
            user_id,
            session_id,
            round_id,
        )

        user_query = extract_latest_user_message(
            messages_list=messages, user_id=user_id, session_id=session_id
        )

        logger.debug(
            "[stream_and_persist_chat_response] Extracted latest user message: user_id=%s, session_id=%s, round_id=%s",
            user_id,
            session_id,
            round_id,
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
            f"[stream_and_persist_chat_response] Finished stream_and_persist_chat_response: user_id={user_id}, session_id={session_id}, round_id={round_id}, total_chars={len(assistant_response)}"
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
                    enhanced_text = doc.enhanced_text or ""

                    # Append metadata and enhanced text
                    lines.append(f"RAG Rank: {i}")
                    lines.append(f"doc_id: {doc_id}")
                    lines.append(f"semantic distance: {distance}")
                    lines.append(f"Text: {enhanced_text}")

                except Exception as e:
                    logger.warning(
                        f"[build_context_string] Failed to process document from {db_type}: {e}; "
                        f"user_id={user_id}, session_id={session_id}"
                    )

        joined_lines = "\n".join(lines)

        logger.debug(
            f"[build_context_string] Context built: user_id={user_id}, session_id={session_id}, total_chars={len(joined_lines)}"
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
    """Parse context blocks fast and deterministically (no heavy regex).

    Expected block shape (repeated):
        RAG Rank: <int>
        doc_id: <str>
        semantic distance: <float>
        Text: <multiline...until next 'RAG Rank:' or EOF>
        # Note: The first text chunk may be on the same line as 'Text:'.

    This parser:
      * Ignores everything after the optional '--- Optimized Queries ---' divider.
      * Is robust to extra blank lines / incidental noise between fields.
      * Parses in a single pass (O(n)).
      * Tolerates missing/invalid fields (uses defaults instead of raising).

    Args:
      content: Raw context string stored in the system message.
      user_id: For logging/tracing (unused here).
      session_id: For logging/tracing (unused here).

    Returns:
      List of dict rows with keys: "rag_rank", "doc_id", "semantic_distance", "text".
    """
    # Fast exit if no obvious header.
    if "RAG Rank:" not in content:
        return []

    # Cut off after divider if present (be tolerant about surrounding newlines).
    div_idx = content.find("--- Optimized Queries ---")
    if div_idx != -1:
        content = content[:div_idx]

    # Normalize newlines and split once.
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")
    n = len(lines)

    def _is_rank_header(line: str) -> bool:
        """True if line begins a new 'RAG Rank: <int>' block."""
        s = line.lstrip()
        if not s.startswith("RAG Rank:"):
            return False
        rest = s[len("RAG Rank:") :].strip()
        return bool(rest) and rest[0].isdigit()

    def _safe_int(s: str, default: int = 0) -> int:
        try:
            return int(s)
        except Exception:
            return default

    def _safe_float(s: str, default: float = 0.0) -> float:
        try:
            return float(s)
        except Exception:
            return default

    results: list[dict[str, str | float | int]] = []
    i = 0

    while i < n:
        # Seek next header
        while i < n and not _is_rank_header(lines[i]):
            i += 1
        if i >= n:
            break

        # Parse "RAG Rank: <int>"
        rank_line = lines[i].lstrip()
        rank_str = rank_line.split(":", 1)[1].strip()
        rag_rank = _safe_int(rank_str, default=0)
        i += 1

        # Holders
        doc_id: str = ""
        semantic_distance: float = 0.0
        text_lines: list[str] = []

        # Scan header fields until we hit "Text:" or the next header/EOF
        while i < n and not _is_rank_header(lines[i]):
            line = lines[i].lstrip()

            if line.startswith("doc_id:"):
                doc_id = line.split(":", 1)[1].strip()
                i += 1
                continue

            if line.startswith("semantic distance:"):
                dist_str = line.split(":", 1)[1].strip()
                semantic_distance = _safe_float(dist_str, default=0.0)
                i += 1
                continue

            if line.startswith("Text:"):
                # 1) Capture any inline text on this same line
                #    e.g., "Text: foo bar" → first chunk = "foo bar"
                #    (If empty, we just start from next lines.)
                inline = line.split(":", 1)[1]
                inline = inline.lstrip()  # keep leading spaces minimal
                if inline:
                    text_lines.append(inline)

                # 2) Then collect following lines until the next header or EOF
                i += 1
                while i < n and not _is_rank_header(lines[i]):
                    text_lines.append(lines[i])
                    i += 1
                break  # end of this block

            # Unknown line in header area → skip
            i += 1

        results.append(
            {
                "rag_rank": rag_rank,
                "doc_id": doc_id,
                "semantic_distance": semantic_distance,
                "text": "\n".join(text_lines).strip(),
            }
        )

    return results


# def parse_context_string_to_structured(
#     content: str,
#     user_id: str,
#     session_id: str,
# ) -> list[dict[str, str | float | int]]:
#     """
#     Parses a previously generated context string into structured RAG entries.

#     This function is used to reverse-engineer the structured format when only the
#     stringified context (stored in DB) is available. If parsing fails, the function
#     logs a warning and returns any partially parsed results instead of raising an exception.

#     Args:
#         content (str): Retrieved context string in fixed textual format.
#         user_id (str): The user ID for logging and traceability.
#         session_id (str): The session ID for logging and traceability.

#     Returns:
#         list[dict[str, str | float | int]]: List of structured entries with rag_rank, doc_id, semantic_distance, and text.

#     """

#     results: list[dict[str, str | float | int]] = []
#     try:
#         context_only = content.split("\n\n--- Optimized Queries ---\n")[0]
#         pattern = re.compile(
#             r"RAG Rank: (?P<rank>\d+)\n"
#             r"doc_id: (?P<doc_id>.+?)\n"
#             r"semantic distance: (?P<distance>.+?)\n"
#             r"Text: (?P<text>.*?)(?=\nRAG Rank:|\Z)",
#             re.DOTALL,
#         )

#         for match in pattern.finditer(context_only):
#             results.append(
#                 {
#                     "rag_rank": int(match.group("rank")),
#                     "doc_id": match.group("doc_id").strip(),
#                     "semantic_distance": float(match.group("distance")),
#                     "text": match.group("text").strip(),
#                 }
#             )
#     except Exception as exc:
#         logger.warning(
#             f"[parse_context_string_to_structured] Failed to parse context string: user_id={user_id}, session_id={session_id}, error={exc}"
#         )

#     return results


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
    logger.debug(
        f"[list_sessions] Starting list_sessions: user_id={user_id}, app_name={app_name}"
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

        logger.debug(
            f"[list_sessions] Finished list_sessions: user_id={user_id}, app_name={app_name}, session_count={len(sessions)}"
        )

        return sessions

    except Exception as exc:
        logger.exception(
            f"[list_sessions] Failed to fetch sessions: user_id={user_id}, app_name={app_name}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session list from the database."
        ) from exc


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
    logger.debug(
        f"[list_session_queries] Starting list_session_queries: user_id={user_id}, app_name={app_name}, session_id={session_id}"
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

        logger.debug(
            "[list_session_queries] Finished list_session_queries: user_id=%s, app_name=%s, session_id=%s, "
            "limit_rounds=%d → returned %d rounds (%d messages), has_more=%s",
            user_id,
            app_name,
            session_id,
            limit,
            unique_rounds,
            len(results),
            has_more,
        )

        return MessageListResponse(messages=results, has_more=has_more)

    except Exception as exc:
        logger.exception(
            f"[list_session_queries] Failed to fetch messages: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session queries from the database.",
        ) from exc


@app.post(
    "/users/{user_id}/apps/{app_name}/sessions/create_with_limit",
    status_code=status.HTTP_201_CREATED,
)
def create_session_with_limit(
    user_id: str,
    app_name: str,
    payload: CreateSessionWithLimitRequest,
):
    """
    Create a new session with validation of existing sessions and enforcement of session limits.

    This endpoint ensures that:
    - Existing sessions for the user and app are locked and validated for consistency.
    - At most 10 non-deleted sessions exist; the oldest may be deleted to make room.
    - A new session is inserted after validations pass.

    Args:
        user_id (str): ID of the user.
        app_name (str): Name of the application.
        payload (CreateSessionWithLimitRequest): Request payload with new session data,
            known session IDs, and optional delete target.

    Returns:
        dict: Response containing a success message and the created session ID.

    Raises:
        HTTPException (400): If UUIDs are invalid or inputs are malformed.
        HTTPException (409): If client-side session state is inconsistent with the server.
        HTTPException (500): If the session count exceeds limit or a DB error occurs.
    """
    try:
        logger.debug(
            f"[create_session_with_limit] Starting create_session_with_limit: user_id={user_id}, app_name={app_name}"
        )
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
                if DB_TYPE == "postgresql":
                    known_ids_as_uuid = sorted(
                        UUID(k) for k in payload.known_session_ids
                    )
                elif DB_TYPE == "mysql":
                    known_ids_as_uuid = sorted(k for k in payload.known_session_ids)
                else:
                    logger.error(
                        f"[create_session_with_limit] Unsupported DB_TYPE: {DB_TYPE} "
                        f"for user_id={user_id}, app_name={app_name}"
                    )
                    raise HTTPException(
                        status_code=500, detail="Unsupported database type"
                    )
            except ValueError:
                logger.exception(
                    f"[create_session_with_limit] Invalid UUID in known_session_ids: user_id={user_id}, app_name={app_name}"
                )
                raise HTTPException(
                    status_code=400, detail="Invalid UUID in known_session_ids."
                )

            current_ids = sorted(s["session_id"] for s in session_infos)

            if set(current_ids) != set(known_ids_as_uuid):
                logger.error(
                    "[create_session_with_limit] Session ID mismatch: user_id=%s, app_name=%s, server_session_ids=%s, client_known_session_ids=%s",
                    user_id,
                    app_name,
                    current_ids,
                    known_ids_as_uuid,
                )

                raise HTTPException(
                    status_code=409,
                    detail="Known session IDs do not match server-side session state.",
                )

            new_session_id = payload.new_session_data.session_id
            new_session_name = payload.new_session_data.session_name
            is_private = payload.new_session_data.is_private_session
            now = datetime.now(timezone.utc)

            if len(current_ids) > 10:
                logger.error(
                    f"[create_session_with_limit] Too many non-deleted sessions: user_id={user_id}, len(current_ids)={len(current_ids)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail="Too many non-deleted sessions. Server state invalid.",
                )
            elif len(current_ids) == 10:

                if DB_TYPE == "postgresql":
                    try:
                        delete_target_uuid = UUID(payload.delete_target_session_id)
                    except ValueError as exc:
                        logger.exception(
                            f"[create_session_with_limit] Invalid delete_target_session_id: user_id={user_id}, session_id={payload.delete_target_session_id}"
                        )
                        raise HTTPException(
                            status_code=400, detail="Invalid delete_target_session_id"
                        ) from exc
                elif DB_TYPE == "mysql":
                    delete_target_uuid = payload.delete_target_session_id

                oldest = min(session_infos, key=lambda s: s["last_touched_at"])
                if delete_target_uuid != oldest["session_id"]:
                    logger.error(
                        f"[create_session_with_limit] Mismatch in delete target: user_id={user_id}, "
                        f"expected={oldest['session_id']}, got={delete_target_uuid}"
                    )
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

            logger.debug(
                f"[create_session_with_limit] Finished create_session_with_limit: user_id={user_id}, app_name={app_name}, session_id={new_session_id}"
            )

            return {
                "message": "Session created successfully",
                "session_id": new_session_id,
            }

    except HTTPException as http_exc:
        logger.warning(
            f"[create_session_with_limit] HTTPException raised: status_code={http_exc.status_code}, "
            f"detail={http_exc.detail}, user_id={user_id}, app_name={app_name}"
        )
        raise

    except DatabaseError as exc:
        logger.exception(
            f"[create_session_with_limit] Database error: user_id={user_id}, app_name={app_name}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to create session due to a database error.",
        ) from exc

    except Exception as exc:
        logger.exception(
            f"[create_session_with_limit] Unexpected error: user_id={user_id}, app_name={app_name}"
        )
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error",
        ) from exc


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
        logger.debug(
            f"[update_session_with_check] Starting update_session_with_check: user_id={user_id}, app_name={app_name}, session_id={session_id}"
        )
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
                    f"[update_session_with_check] Session not found for user_id={user_id}, session_id={session_id}."
                )
                raise HTTPException(status_code=404, detail="Session not found")

            current_name, current_private, current_is_deleted = row

            if (
                payload.before_session_name != current_name
                or payload.before_is_private_session != current_private
                or payload.before_is_deleted != current_is_deleted
            ):
                logger.warning(
                    f"[update_session_with_check] Session conflict detected for user_id={user_id}, session_id={session_id}."
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

            logger.debug(
                f"[update_session_with_check] Finished update_session_with_check: user_id={user_id}, app_name={app_name}, session_id={session_id}"
            )

            return {"message": "Session updated successfully", "session_id": session_id}

    except HTTPException as http_exc:
        logger.warning(
            f"[update_session_with_check] HTTPException raised: status_code={http_exc.status_code}, "
            f"detail={http_exc.detail}, user_id={user_id}, session_id={session_id}"
        )
        raise
    except DatabaseError as exc:
        logger.exception(
            f"[update_session_with_check] DB error: user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to update session."
        ) from exc
    except Exception as exc:
        logger.exception(
            "[update_session_with_check] Unexpected error for user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=500, detail="Unexpected server error") from exc


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
    logger.debug(
        f"[delete_round] Starting delete_round: user_id={user_id}, session_id={session_id}, round_id={round_id}"
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

        logger.debug(
            f"[delete_round] Finished delete_round: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )

        return JSONResponse({"status": "ok", "detail": "Messages logically deleted."})

    except HTTPException as http_exc:
        logger.warning(
            f"[delete_round] HTTPException raised: status_code={http_exc.status_code}, "
            f"detail={http_exc.detail}, user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise
    except Exception as exc:
        logger.exception(
            f"[delete_round] Failed to logically delete round: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise HTTPException(status_code=500, detail="Failed to delete round.") from exc


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

    logger.debug(
        f"[patch_feedback] Starting patch_feedback: user_id={payload.user_id}, session_id={payload.session_id}, "
        f"llm_output_id={llm_output_id}, feedback={payload.feedback}"
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

        logger.debug(
            f"[patch_feedback] Finished patch_feedback: user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}"
        )

        return JSONResponse({"status": "ok", "msg": "Feedback patched successfully."})

    except HTTPException as e:
        logger.warning(
            f"[patch_feedback] Client error: "
            f"user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}, status_code={e.status_code}, detail={e.detail}"
        )
        raise

    except Exception as exc:
        logger.exception(
            f"[patch_feedback] Unexpected failure: user_id={payload.user_id}, session_id={payload.session_id}, llm_output_id={llm_output_id}"
        )
        raise HTTPException(status_code=500, detail="Failed to patch feedback") from exc


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
            f"[_parse_request] Failed to parse query request for user_id={user_id}, session_id={session_id}"
        )
        raise HTTPException(status_code=400, detail="Invalid query payload") from exc


def _build_retrieval_queries(
    *,
    req: QueryRequest,
    user_query: str,
    user_id: str,
    session_id: str,
    client: OpenAI | AzureOpenAI,
    model_name: str,
    model_type: Literal["openai", "azure"],
) -> list[str]:
    """Generate optimized retrieval queries or fall back to the original user query.

    This function determines whether to return a single user query string or
    generate multiple optimized queries based on the RAG mode. If the RAG mode
    is not OPTIMIZED, it returns the original user query as a single-element list.
    Otherwise, it attempts to generate optimized search queries from message history.
    If optimization fails or results in an empty query list, it also falls back to
    the original user query.

    Args:
        req (QueryRequest): Parsed request payload, including RAG mode and message history.
        user_query (str): The latest user prompt to be used for document retrieval.
        user_id (str): Authenticated user ID (used for contextual logging).
        session_id (str): The chat session UUID (used for contextual logging).

    Returns:
        list[str]: A list of search queries to use for retrieval.
            - If `req.rag_mode` is not OPTIMIZED, returns [user_query].
            - If optimization succeeds, returns the generated query list.
            - If optimization fails or results in an empty list, returns [user_query].
    """
    if req.rag_mode is not RagModeEnum.OPTIMIZED:
        logger.debug(
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
            client=client,
            model_name=model_name,
            model_type=model_type,
        )

        if not queries:
            logger.warning(
                "[_build_retrieval_queries] Empty queries from history: user_id=%s, session_id=%s, round_id=%s",
                user_id,
                session_id,
                req.round_id,
            )
            return [user_query]

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
        logger.exception(
            "[_build_retrieval_queries] Fallback to user query: "
            "user_id=%s, session_id=%s, round_id=%s, user_query_len=%d",
            user_id,
            session_id,
            req.round_id,
            len(user_query),
        )
        return [user_query]


def extract_latest_user_message(
    messages_list: list[dict], user_id: str, session_id: str
) -> str:
    """
    Extract the most recent user message content from a list of message dictionaries.

    This function scans messages in reverse order to find the latest message from the user
    (i.e., where 'role' == 'user'), and returns its content. If the content is missing,
    empty, or the message is not found, it raises an HTTP 400 error to indicate a client issue.

    Args:
        messages_list (list[dict]): List of chat messages containing 'role' and 'content' fields.
        user_id (str): ID of the user, used for contextual logging.
        session_id (str): Session ID, used for contextual logging.

    Returns:
        str: The content of the most recent user message.

    Raises:
        HTTPException: If no valid user message is found. Returns HTTP 400 to indicate
        a bad client request.
    """
    for msg in reversed(messages_list):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            break
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
    result = pre_messages + messages

    if not all("role" in m and "content" in m for m in messages):
        logger.warning(
            "[build_openai_messages] Some message(s) missing 'role' or 'content'"
        )

    return result


def format_sse_chunk(data: dict) -> str:
    """Formats a dictionary as a Server-Sent Event (SSE) chunk.

    Args:
        data (dict): Data to be sent as part of an SSE message.

    Returns:
        str: Formatted SSE string.
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def yield_context_info(
    user_id: str,
    session_id: str,
    retrieved_contexts_str: str,
    assistant_response: str,
) -> Generator[str, None, None]:
    """
    Yield structured context rows used in the assistant response, based on referenced RAG ranks.

    This function parses the retrieved context string into structured rows, then extracts
    only those rows that are actually referenced in the assistant's output.

    Args:
        user_id (str): The user ID, used for logging.
        session_id (str): The session ID, used for logging.
        retrieved_contexts_str (str): Concatenated context string passed to the LLM.
        assistant_response (str): Assistant's full response, used to extract referenced ranks.

    Yields:
        str: A Server-Sent Event (SSE) chunk containing the filtered context rows.
    """
    logger.debug(
        "[yield_context_info] Starting: user_id=%s, session_id=%s, "
        "len(retrieved_contexts_str)=%d, len(assistant_response)=%d",
        user_id,
        session_id,
        len(retrieved_contexts_str),
        len(assistant_response),
    )
    try:
        logger.debug(
            "[yield_context_info] Calling parse_context_string_to_structured: "
            "user_id=%s, session_id=%s",
            user_id,
            session_id,
        )
        context_rows = parse_context_string_to_structured(
            content=retrieved_contexts_str, user_id=user_id, session_id=session_id
        )
        logger.debug(
            "[yield_context_info] parse_context_string_to_structured completed: "
            "user_id=%s, session_id=%s, len(context_rows)=%d",
            user_id,
            session_id,
            len(context_rows),
        )

        # Extract referenced RAG ranks from assistant content
        logger.debug(
            "[yield_context_info] Calling extract_referenced_rag_ranks: "
            "user_id=%s, session_id=%s",
            user_id,
            session_id,
        )
        referenced_ranks = extract_referenced_rag_ranks(
            text=assistant_response, user_id=user_id, session_id=session_id
        )
        logger.debug(
            "[yield_context_info] extract_referenced_rag_ranks completed: "
            "user_id=%s, session_id=%s, referenced_ranks=%s",
            user_id,
            session_id,
            referenced_ranks,
        )

        if not referenced_ranks:
            logger.debug(
                "[yield_context_info] No referenced RAG ranks found: user_id=%s, session_id=%s",
                user_id,
                session_id,
            )
        # Filter *and* preserve the order in which ranks were referenced
        rank_order = {r: i for i, r in enumerate(referenced_ranks)}
        filtered_rows = sorted(
            (r for r in context_rows if r.get("rag_rank") in rank_order),
            key=lambda r: rank_order[r["rag_rank"]],
        )
        logger.debug(
            "[yield_context_info] user_id=%s, session_id=%s, referenced_ranks=%s, len(filtered_rows)=%d",
            user_id,
            session_id,
            referenced_ranks,
            len(filtered_rows),
        )
        yield format_sse_chunk({"system_context_rows": filtered_rows})
    except Exception:
        logger.exception(
            "[yield_context_info] Failed to yield context info: user_id=%s, session_id=%s",
            user_id,
            session_id,
        )
        yield format_sse_chunk({"system_context_rows": []})


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
                    f"[maybe_generate_session_title] Session not found: user_id={user_id}, session_id={session_id}, round_id={round_id}"
                )
                return

            if result[0] != "Untitled Session":
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
                logger.exception(
                    f"[maybe_generate_session_title] Failed to generate session title: user_id={user_id}, session_id={session_id}, round_id={round_id}"
                )
    except Exception:
        logger.exception(
            f"[maybe_generate_session_title] Failed to update session title: user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )


def _deduplicate_by_doc_id(docs: list[Document]) -> list[Document]:
    """Remove duplicates keeping the smallest semantic-distance one.

    Args:
        docs: List of retrieved ``Document`` objects.

    Returns:
        list[Document]: Deduplicated list, one Document per doc_id with the smallest distance.
    """
    unique: dict[str, Document] = {}

    for doc in docs:
        doc_id: str = doc.base_document.doc_id
        dist: float = doc.base_document.distance

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

    This function deduplicates input documents by doc_id, then sorts them by semantic
    distance and returns the top_k items.

    Args:
        docs: Raw hits (may contain duplicates).
        top_k: Maximum unique items to return.

    Returns:
        Deduplicated and sorted list (len ≤ top_k).
    """
    uniques: list[Document] = _deduplicate_by_doc_id(docs)
    uniques.sort(key=lambda d: d.base_document.distance)

    logger.debug(
        "[_dedup_sort_trim] Input=%d, Deduped=%d, Returning=%d",
        len(docs),
        len(uniques),
        min(len(uniques), top_k),
    )

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
    """Expand each hit by surrounding context lines/chunks with graceful fallback.

    The helper delegates to ``chroma_repo.enhance`` for both Chroma and
    BM25 results (the latter are passed straight through the same API).

    Strategy:
      1) Try bulk enhancement for each source (Chroma / BM25) as before.
      2) If bulk fails, enhance per document with directional fallbacks:
         - Try (num_before, num_after)
         - If that fails, try (0, num_after)     # give up on BEFORE
         - If that fails, try (num_before, 0)    # give up on AFTER
      3) If all fail, keep the doc as-is but ensure `enhanced_text` is base text.
      4) Never raise; always return a dict with lists (possibly base-text-only).

    Args:
        search_results: Retrieval output, e.g., {"chroma": [...], "bm25": [...]}.
        chroma_repo: Initialized ChromaDB repository or None to skip enhancement.
        num_before: Number of neighboring chunks before each hit.
        num_after: Number of neighboring chunks after each hit.
        user_id: For contextual logging.
        session_id: For contextual logging.

    Returns:
        dict[str, list[Document]]: Enhanced results keyed like `search_results`.
    """
    enhanced: dict[str, list[Document]] = {}

    def _ensure_enhanced_text(doc: Document) -> None:
        """Guarantee `doc.enhanced_text` is usable.

        Uses `doc.base_document.text` if `enhanced_text` is None or empty.
        Sets empty string as a last resort.
        Idempotent.
        """
        doc_id: str = doc.base_document.doc_id
        if isinstance(doc.enhanced_text, str) and doc.enhanced_text.strip() != "":
            logger.debug(
                "[enhance.ensure] Keep existing enhanced_text "
                "(user_id=%s, session_id=%s, doc_id=%s)",
                user_id,
                session_id,
                doc_id,
            )
            return
        doc.enhanced_text = doc.base_document.text or ""
        logger.debug(
            "[enhance.ensure] Set enhanced_text from base text "
            "(user_id=%s, session_id=%s, doc_id=%s, base_len=%d)",
            user_id,
            session_id,
            doc_id,
            len(doc.enhanced_text),
        )

    def _try_enhance_one(doc: Document) -> Document:
        """Enhance one doc with directional fallbacks.

        Tries full (before/after), then disables BEFORE, then disables AFTER.
        If all fail, returns doc with `enhanced_text` set to base text.
        """
        doc_id: str = doc.base_document.doc_id

        if not chroma_repo:
            logger.debug(
                "[enhance.per-doc] Enhancement disabled; using base text "
                "(user_id=%s, session_id=%s, doc_id=%s)",
                user_id,
                session_id,
                doc_id,
            )
            _ensure_enhanced_text(doc)
            return doc

        # Full attempt
        try:
            return chroma_repo.enhance(doc, num_before=num_before, num_after=num_after)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(
                "[enhance.per-doc] Full enhance failed "
                "(user_id=%s, session_id=%s, doc_id=%s, before=%d, after=%d, err=%s)",
                user_id,
                session_id,
                doc_id,
                num_before,
                num_after,
                e,
            )

        # Disable BEFORE
        if num_before > 0:
            try:
                return chroma_repo.enhance(doc, num_before=0, num_after=num_after)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning(
                    "[enhance.per-doc] Retry without BEFORE failed "
                    "(user_id=%s, session_id=%s, doc_id=%s, after=%d, err=%s)",
                    user_id,
                    session_id,
                    doc_id,
                    num_after,
                    e,
                )

        # Disable AFTER
        if num_after > 0:
            try:
                return chroma_repo.enhance(doc, num_before=num_before, num_after=0)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning(
                    "[enhance.per-doc] Retry without AFTER failed "
                    "(user_id=%s, session_id=%s, doc_id=%s, before=%d, err=%s)",
                    user_id,
                    session_id,
                    doc_id,
                    num_before,
                    e,
                )

        # Give up on enhancement; keep the doc usable
        logger.warning(
            "[enhance.per-doc] Enhancement abandoned; using base text "
            "(user_id=%s, session_id=%s, doc_id=%s)",
            user_id,
            session_id,
            doc_id,
        )
        _ensure_enhanced_text(doc)
        return doc

    def _bulk_then_per_doc(docs: list[Document], label: str) -> list[Document]:
        """Try bulk enhance; fall back to per-doc with graceful degradation.

        Args:
            docs: Documents to enhance.
            label: Source label (e.g., 'Chroma' or 'BM25') for logging.

        Returns:
            Enhanced docs; never raises.
        """
        if not docs:
            return []
        if not chroma_repo:
            # Enhancement disabled; ensure downstream has usable text
            for d in docs:
                _ensure_enhanced_text(d)
            logger.debug(
                "[enhance.bulk] Enhancement disabled for %s; ensured base text "
                "(user_id=%s, session_id=%s, count=%d)",
                label,
                user_id,
                session_id,
                len(docs),
            )
            return docs

        # 1) Bulk attempt (fast path, original behavior)
        try:
            result: list[Document] = chroma_repo.enhance(
                docs, num_before=num_before, num_after=num_after  # type: ignore[arg-type]
            )
            logger.debug(
                "[enhance.bulk] %s bulk enhancement succeeded "
                "(user_id=%s, session_id=%s, before=%d, after=%d, count=%d)",
                label,
                user_id,
                session_id,
                num_before,
                num_after,
                len(result),
            )
            return result
        except Exception:
            logger.exception(
                "[enhance.bulk] %s bulk enhancement failed; falling back per-doc "
                "(user_id=%s, session_id=%s, before=%d, after=%d)",
                label,
                user_id,
                session_id,
                num_before,
                num_after,
            )

        # 2) Per-doc fallback with directional retries
        out: list[Document] = []
        for d in docs:
            try:
                out.append(_try_enhance_one(d))
            except Exception as e:  # Extra defensive guard
                logger.warning(
                    "[enhance.bulk->per-doc] Unexpected error; using base text "
                    "(user_id=%s, session_id=%s, source=%s, doc_id=%s, err=%s)",
                    user_id,
                    session_id,
                    label,
                    d.base_document.doc_id,
                    e,
                )
                _ensure_enhanced_text(d)
                out.append(d)
        logger.debug(
            "[enhance.bulk->per-doc] Completed with fallbacks "
            "(user_id=%s, session_id=%s, source=%s, count=%d)",
            user_id,
            session_id,
            label,
            len(out),
        )
        return out

    # Process each source independently; never raise
    chroma_docs = search_results.get("chroma", [])
    if chroma_docs:
        logger.debug(
            "[enhance.entry] Start Chroma enhancement "
            "(user_id=%s, session_id=%s, count=%d, before=%d, after=%d)",
            user_id,
            session_id,
            len(chroma_docs),
            num_before,
            num_after,
        )
        enhanced["chroma"] = _bulk_then_per_doc(chroma_docs, "Chroma")

    bm25_docs = search_results.get("bm25", [])
    if bm25_docs:
        logger.debug(
            "[enhance.entry] Start BM25 enhancement "
            "(user_id=%s, session_id=%s, count=%d, before=%d, after=%d)",
            user_id,
            session_id,
            len(bm25_docs),
            num_before,
            num_after,
        )
        enhanced["bm25"] = _bulk_then_per_doc(bm25_docs, "BM25")

    return enhanced


# def _enhance_results(
#     *,
#     search_results: dict[str, list[Document]],
#     chroma_repo: ChromaDBRepository | None,
#     num_before: int,
#     num_after: int,
#     user_id: str,
#     session_id: str,
# ) -> dict[str, list[Document]]:
#     """Expand each hit by surrounding context lines / chunks.

#     The helper delegates to ``chroma_repo.enhance`` for both Chroma and
#     BM25 results (the latter are passed straight through the same API).

#     Args:
#         search_results: Output from the retrieval step —
#             ``{"chroma": [...], "bm25": [...]}``.
#         chroma_repo: Initialised ChromaDB repository or ``None`` to skip.
#         num_before: Number of neighbouring chunks *before* each hit.
#         num_after:  Number of neighbouring chunks *after* each hit.
#         user_id: User ID for contextual logging only.
#         session_id: Session ID for contextual logging only.

#     Returns:
#         dict[str, list[Document]]: Enhanced results using the same keys as
#         *search_results*. Lists may be empty if a repository is disabled,
#         empty, or the enhancement call fails.

#     Notes:
#         All exceptions thrown by the repository are caught and logged; the
#         function never raises.
#     """
#     enhanced: dict[str, list[Document]] = {}

#     if search_results["chroma"] and chroma_repo:
#         try:
#             enhanced["chroma"] = chroma_repo.enhance(
#                 search_results["chroma"],
#                 num_before=num_before,
#                 num_after=num_after,
#             )
#         except Exception:
#             logger.exception(
#                 f"[enhance] Chroma enhancement failed: "
#                 f"user_id={user_id}, session_id={session_id}"
#             )

#     if search_results["bm25"] and chroma_repo:
#         try:
#             enhanced["bm25"] = chroma_repo.enhance(
#                 search_results["bm25"],
#                 num_before=num_before,
#                 num_after=num_after,
#             )
#         except Exception:
#             logger.exception(
#                 f"[enhance] BM25 enhancement failed: "
#                 f"user_id={user_id}, session_id={session_id}"
#             )

#     return enhanced


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

    resolved_model_name = MODEL_NAME if OPENAI_TYPE == "openai" else DEPLOYMENT_ID

    logger.debug(
        f"[handle_query] Starting handle_query: user_id={user_id}, session_id={session_id}, round_id={req.round_id}"
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
        model_name=resolved_model_name,
        model_type=OPENAI_TYPE,
        db_type=DB_TYPE,
        db_pool=db_pool,
    )

    if req.rag_mode is RagModeEnum.NO_RAG:
        logger.debug(
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
                model_name=resolved_model_name,
                model_type=OPENAI_TYPE,
                optimized_queries=None,
                use_reranker=req.use_reranker,
            ),
            media_type="text/event-stream",
        )

    queries: list[str] = _build_retrieval_queries(
        req=req,
        user_query=user_query,
        user_id=user_id,
        session_id=session_id,
        client=client,
        model_name=resolved_model_name,
        model_type=OPENAI_TYPE,
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

    # TODO: Apply reranker to enhance the retrieved documents before building context.
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
            model_name=resolved_model_name,
            model_type=OPENAI_TYPE,
            optimized_queries=queries,
            use_reranker=req.use_reranker,
        ),
        media_type="text/event-stream",
    )
