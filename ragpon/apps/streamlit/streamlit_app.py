import json
import logging
import os
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Final

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ragpon._utils.logging_helper import get_library_logger
from ragpon.apps.chat_domain import Message, RagModeEnum, RoleEnum, SessionData


@dataclass(frozen=True)
class Labels:
    APP_TITLE: str = "📘 規程・マニュアル等検索アプリ"
    # Session creation
    CREATE_SESSION: str = "🆕 新しいセッションを作成"
    SESSION_NAME: str = "📛 セッション名"
    IS_PRIVATE: str = "🙈 非公開セッションですか？"
    SUBMIT: str = "作成"
    CANCEL: str = "キャンセル"
    # Session management
    EDIT_SESSION: str = "✏️ セッションを編集する"
    DELETE_SESSION: str = "🗑️ このセッションを削除する"
    UPDATE: str = "更新"
    # confirmation messages for session and round deletion
    CONFIRM_DELETION: str = "⚠️ 本当に削除しますか？"
    YES_DELETE: str = "✅ はい、削除します"
    NO_CANCEL: str = "❌ キャンセルします"
    # Sidebar sections
    SESSION_LIST: str = "## 👉セッション一覧"
    SELECT_SESSION: str = "セッションを選択してください："
    RAG_MODE_SECTION: str = "## 🔍社内情報の検索方法の設定"
    CHOOSE_RAG_MODE: str = "検索モードを選択してください："
    RAG_MODE_HELP_TITLE: str = "💡 検索モードとは？"
    RAG_MODE_HELP: str = (
        "**Pro Mode**：AI がこれまでの会話内容をもとにクエリ（資料検索用の文章）を自動生成し、資料を検索します。検索結果をふまえてAI があなたの質問に回答します。\n\n"
        "**Standard**：あなたが入力した文章をそのままクエリとして利用し、資料を検索します。検索結果をふまえて AI があなたの質問に回答します。\n\n"
        "**No RAG**：資料は使わず、AI 自身の知識だけであなたの質問に回答します。"
    )
    RERANKER_SECTION: str = "## 🔀リランカーの使用"
    CHOOSE_RERANKER: str = "リランカーを使用しますか？"
    YES: str = "はい"
    NO: str = "いいえ"
    # Chat input
    CHAT_INPUT_PLACEHOLDER: str = "ここに質問を入力してください..."
    # Feedback
    FEEDBACK_PROMPT: str = "📝 フィードバックを入力"
    FEEDBACK_REASON: str = (
        "フィードバックの理由をご記入ください（任意・最大{max_feedback_length}文字）"
    )
    SUBMIT_FEEDBACK: str = "✅ フィードバックを送信"
    CANCEL_FEEDBACK: str = "❌ フィードバックをキャンセル"
    FEEDBACK_GOOD: str = "😊"
    FEEDBACK_BAD: str = "😞"
    # Load more
    LOAD_MORE: str = "追加で10件表示"
    LOAD_MORE_HELP: str = "さらに古いラウンドを表示"
    # Context display
    VIEW_SOURCES: str = "📚 回答に使用された情報を見る"
    # Deleted message
    DELETED_MESSAGE_NOTICE: str = "🗑️ このメッセージは削除されました。"


CONTACT_ADMIN: str = "問題が継続する場合は管理者に連絡してください。"

RECONNECTION_ACTION: str = "時間をおいて再接続してください。"
RETRY_ACTION: str = "しばらく待ってから再度お試しください。"
REFRESH_ACTION: str = "F5 キーを押して画面を更新してください。"

RECONNECTION_INSTRUCTIONS: str = f"{RECONNECTION_ACTION}{CONTACT_ADMIN}"
RETRY_INSTRUCTIONS: str = f"{RETRY_ACTION}{CONTACT_ADMIN}"
REFRESH_INSTRUCTIONS: str = f"{REFRESH_ACTION}{CONTACT_ADMIN}"


@dataclass(frozen=True)
class ErrorLabels:
    ACCESS_DENIED: str = "⚠️ アクセス権限がありません。管理者にご確認ください。"
    APP_INITIALIZATION_FAILED: str = (
        f"⚠️ アプリの初期化に失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )
    # Session operations
    SESSION_LIST_LOAD_FAILED: str = (
        f"⚠️ セッション一覧の読み込みに失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )
    SESSION_INDEX_RESOLVE_FAILED: str = (
        f"⚠️ セッションの初期選択処理に失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )
    AUTO_SESSION_CREATION_FAILED: str = (
        f"⚠️ 初期セッションの作成に失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )
    AUTO_SESSION_REGISTER_FAILED: str = (
        f"⚠️ 初期セッションの登録に失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )
    SESSION_SELECTION_FAILED: str = (
        f"⚠️ セッションの選択時にエラーが発生しました。{RECONNECTION_INSTRUCTIONS}"
    )
    SESSION_RADIO_RENDER_FAILED: str = (
        f"⚠️ セッション一覧の表示に失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )
    MESSAGE_HISTORY_LOAD_FAILED: str = (
        f"⚠️ セッション履歴の読み込みに失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )
    MESSAGE_DISPLAY_FAILED: str = (
        f"⚠️ メッセージの表示に失敗しました。セッション履歴の読み込みに失敗した可能性があります。{RECONNECTION_INSTRUCTIONS}"
    )
    STREAMING_BACKEND_ERROR: str = (
        f"⚠️ サーバーからの応答中にエラーが発生しました。{RETRY_INSTRUCTIONS}"
    )
    SESSION_CREATION_FAILED: str = (
        f"⚠️ セッションの作成に失敗しました。{RECONNECTION_INSTRUCTIONS}"
    )

    # Feedback
    FEEDBACK_SUBMISSION_FAILED: str = (
        f"⚠️ フィードバックの送信に失敗しました。{RETRY_INSTRUCTIONS}"
    )
    MESSAGE_DELETION_FAILED: str = (
        f"⚠️ メッセージの削除に失敗しました。{RETRY_INSTRUCTIONS}"
    )
    MESSAGE_HISTORY_APPEND_FAILED: str = (
        f"⚠️ 過去のメッセージの読み込みに失敗しました。{RETRY_INSTRUCTIONS}"
    )
    LLM_RESPONSE_FORMAT_ERROR: str = (
        f"⚠️ アシスタントの応答が予期しない形式だったため、処理できませんでした。{RETRY_INSTRUCTIONS}"
    )
    LLM_RESPONSE_MALFORMED: str = (
        f"⚠️ アシスタントの応答が不正な形式だったため、処理できませんでした。{RETRY_INSTRUCTIONS}"
    )
    # HTTP Errors
    SESSION_CREATION_HTTP_500: str = (
        f"⚠️ サーバー内部エラーが発生しました。セッションが一部作成された可能性があります。{REFRESH_INSTRUCTIONS}"
    )
    SESSION_CREATION_HTTP_409: str = (
        f"⚠️ セッション状態の不整合が検出されました。{REFRESH_INSTRUCTIONS}"
    )
    SESSION_CREATION_HTTP_UNEXPECTED: str = (
        f"⚠️ 予期しないHTTPエラーが発生しました。{RECONNECTION_INSTRUCTIONS}"
    )
    SESSION_EDIT_HTTP_404: str = (
        f"⚠️ このセッションは存在しないか、すでに削除されています。{REFRESH_INSTRUCTIONS}"
    )
    SESSION_EDIT_HTTP_409: str = (
        f"⚠️ このセッションは他のタブで内容が更新された可能性があります。{REFRESH_INSTRUCTIONS}"
    )
    SESSION_EDIT_HTTP_500: str = (
        f"⚠️ セッションの変更中にサーバーエラーが発生しました。{RECONNECTION_INSTRUCTIONS}"
    )
    # Generic
    GENERIC_UNEXPECTED_ERROR: str = (
        f"⚠️ 予期しないエラーが発生しました。{RECONNECTION_INSTRUCTIONS}"
    )
    SESSION_CREATION_UNEXPECTED_ERROR: str = (
        f"⚠️ セッションの作成に失敗しました。ネットワークに問題がある可能性があります。{RECONNECTION_INSTRUCTIONS}"
    )
    MESSAGE_SUBMISSION_UNEXPECTED_ERROR: str = (
        f"⚠️ メッセージの送信中に予期しないエラーが発生しました。{RECONNECTION_INSTRUCTIONS}"
    )
    MESSAGE_DELETION_UNEXPECTED_ERROR: str = (
        f"⚠️ メッセージの削除中に予期しないエラーが発生しました。{RECONNECTION_INSTRUCTIONS}"
    )
    FEEDBACK_SUBMISSION_UNEXPECTED_ERROR: str = (
        f"⚠️ フィードバック送信中に予期しないエラーが発生しました。{RECONNECTION_INSTRUCTIONS}"
    )


@dataclass(frozen=True)
class WarningLabels:
    NO_SESSION_SELECTED: str = "⚠️ 現在選択されているセッションがありません。"
    SESSION_LIMIT_REACHED: str = (
        "⚠️ セッションの上限（{max_count}）に達しました。新規セッション作成時に最も古いセッションが削除されます。"
    )
    CONFIRM_DELETION_PROMPT: str = "本当にこの応答を削除してよろしいですか？"
    NO_CONTEXT: str = "⚠️ 関連する情報が見つかりませんでした。"
    PARSE_SYSTEM_MESSAGE_FAILED: str = (
        "⚠️ 関連情報の表示に失敗しました（形式が正しくない可能性があります）。"
    )
    RENDER_SYSTEM_MESSAGE_FAILED: str = "⚠️ 関連情報の表示中にエラーが発生しました。"


@dataclass(frozen=True)
class UiLockLabels:
    CREATING_SESSION: str = "⏳ セッションを作成中..."
    UPDATING_SESSION: str = "⏳ セッション情報を更新中..."
    DELETING_SESSION: str = "⏳ セッションを削除中..."
    DELETING_ROUND: str = "⏳ 応答を削除中..."
    SENDING_MESSAGE: str = "⏳ メッセージを送信中..."
    LOADING_HISTORY: str = "⏳ 履歴を読み込み中..."
    SUBMITTING_FEEDBACK: str = "⏳ フィードバックを送信中..."


LABELS = Labels()
ERROR_LABELS = ErrorLabels()
WARNING_LABELS = WarningLabels()
UI_LOCK_LABELS = UiLockLabels()


@dataclass
class DevTestConfig:
    simulate_delay_seconds: int = 0
    simulate_llm_config_error: bool = False
    simulate_llm_timeout: bool = False
    simulate_backend_failure: bool = False
    simulate_db_failure: bool = False
    simulate_session_autocreate_failure: bool = False
    simulate_llm_stream_error: bool = False
    simulate_llm_invalid_json: bool = False
    simulate_unexpected_exception: bool = False


EMPLOYEE_CLASS_ALLOWED_IDS: Final[set[str]] = {
    "70",
    "50",
    "80",
    "99",
    "60",
    "90",
    "40",
    "45",
    "65",
    "86",
}

# Role priority for stable, explicit ordering
ROLE_ORDER: Final[dict[str, int]] = {
    "user": 0,
    "assistant": 1,
    "system": 2,
}

RAG_MODE_OPTIONS: list[str] = RagModeEnum.list()

DEFAULT_MESSAGE_LIMIT: int = 10  # initial + increment size
MAX_MESSAGE_LIMIT: int = 100  # client-side sliding window cap
# NOTE: Must be **≥ Streamlit-side MAX_MESSAGE_LIMIT (100)** so the server
#       never rejects a limit value the client may legally send.
MAX_FEEDBACK_LENGTH = 500

# Prevent overly long queries from breaking the system
MAX_CHAT_INPUT_LENGTH: int = 1000

APP_NAME: str = "search_regulations"
BPC_API_CONTAINER_NAME: str = os.getenv("BPC_API_CONTAINER_NAME", "ragpon-fastapi")
SERVER_URL: str = f"http://{BPC_API_CONTAINER_NAME}:8006"

# Number of messages to keep in the chat with the assistant
NUM_OF_PREV_MSG_WITH_LLM: int = 6

USE_INACTIVITY_REDIRECT: bool = os.getenv(
    "USE_INACTIVITY_REDIRECT", "true"
).lower() in ("true", "1", "yes")
DEFAULT_TIMEOUT_SECONDS: int = int(
    os.getenv("RAGPON_TIMEOUT_SECONDS", "1800")
)  # default: 30 min
DEFAULT_INTERVAL_MS: int = int(
    os.getenv("RAGPON_INTERVAL_MS", "60_000")
)  # default: 1 min
DEFAULT_REDIRECT_URL: str = os.getenv("RAGPON_REDIRECT_URL", "https://example.com")

# Set root logger level from environment (default: WARNING)
other_level_str = os.getenv("RAGPON_OTHER_LOG_LEVEL", "WARNING").upper()
other_level = getattr(logging, other_level_str, logging.WARNING)

# Set app-specific logger level from environment (default: INFO)
app_level_str = os.getenv("RAGPON_APP_LOG_LEVEL", "INFO").upper()
app_level = getattr(logging, app_level_str, logging.INFO)


# Determine log file path and console logging setting
log_path_str: str | None = os.getenv("RAGPON_LOG_PATH")
console_log_str: str = os.getenv("RAGPON_CONSOLE_LOG", "True")
console_log: bool = console_log_str.lower() in ("true", "1", "yes")

# Initialize Streamlit app logger
logger = logging.getLogger("ragpon.apps.streamlit")

if not logger.handlers:  # avoid duplicate handlers on reload
    # Remove any existing handlers at root
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    # Prepare handler list
    handlers: list[logging.Handler] = []

    if log_path_str:
        # Ensure the log directory exists
        log_path = Path(log_path_str)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler for persistent logs
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        # file_handler.setLevel(other_level)
        file_handler.setLevel(app_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Add console handler if enabled or if no other handlers
    if console_log or not handlers:
        stream_handler = logging.StreamHandler()
        # stream_handler.setLevel(other_level)
        stream_handler.setLevel(app_level)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(stream_handler)

    # Apply configuration
    logging.basicConfig(level=other_level, handlers=handlers)

    # Configure this app’s logger
    logger.setLevel(app_level)

    for handler in handlers:
        logger.addHandler(handler)

    logger.propagate = False


# If you create a session with this name in the Streamlit app, a debug mode is activated.
DEBUG_SESSION_TRIGGER = "__DEBUG_MODE__"


def last_n_non_deleted(messages: list[Message], n: int) -> list[Message]:
    """Returns the last n non-deleted messages in chronological order.

    This function traverses the list of messages from newest to oldest, filters
    out any messages that have `is_deleted=True`, and then takes up to `n` items.
    It reverses that subset so the final result is oldest-to-newest.

    Example:
        Suppose you have 6 messages: M1, M2, M3, M4, M5, M6 (in ascending order).
        Let M4 and M5 be `is_deleted=True`. If you call:
            last_n_non_deleted(messages, 3)
        you might get [M2, M3, M6] in ascending order, ignoring the deleted
        messages and only returning 3 of the most recent non-deleted ones.

    Args:
        messages (list[Message]): The full conversation history, assumed to be
            in ascending chronological order (oldest first).
        n (int): The maximum number of non-deleted messages to return.

    Returns:
        list[Message]: Up to n messages (in ascending chronological order)
        that have `is_deleted=False`. If there are fewer than n non-deleted
        messages total, all of them are returned.
    """
    reversed_filtered = (m for m in reversed(messages) if not m.is_deleted)
    newest_non_deleted = list(islice(reversed_filtered, n))
    newest_non_deleted.reverse()
    return newest_non_deleted


def _last_confirmed_round_id(msgs: list[Message]) -> int:
    """Return the greatest round_id.

    Args:
        msgs: In-memory list of ``Message`` objects currently shown in the UI.

    Returns:
        The highest ``round_id`` found in the list of messages.
        Returns **-1** if no messages exist yet.
    """
    rids = [m.round_id for m in msgs]
    return max(rids) if rids else -1


#################################
# API Calls
#################################


def fetch_session_ids(
    server_url: str,
    user_id: str,
    app_name: str,
) -> list[SessionData]:
    """
    Fetch a list of session data from the FastAPI server via GET:
      GET /users/{user_id}/apps/{app_name}/sessions

    Each session is expected to have:
        {
            "session_id": str,
            "session_name": str,
            "is_private_session": bool
        }

    Args:
        server_url (str): The base URL of the backend server.
        user_id (str): The user ID for which to fetch sessions.
        app_name (str): The name of the application.

    Returns:
        list[SessionData]: A list of SessionData objects.

    Raises:
        requests.RequestException: If the HTTP request fails.
    """
    endpoint = f"{server_url}/users/{user_id}/apps/{app_name}/sessions"
    logger.debug(
        f"[fetch_session_ids] GET {endpoint}, user_id={user_id}, app={app_name}"
    )
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception(
            f"[fetch_session_ids] Failed to fetch sessions from {endpoint}, user_id={user_id}"
        )
        raise

    # Expecting a JSON array of objects
    data = response.json()

    summarized = [
        {
            "session_id": s["session_id"][:8] + "...",
            "is_private": bool(s["is_private_session"]),
            "last_touched": s["last_touched_at"],
        }
        for s in data
    ]
    logger.debug(
        f"[fetch_session_ids] Received response for user_id={user_id}: {summarized}"
    )

    # Convert each JSON object into a SessionData instance
    session_list = []
    for item in data:
        session_list.append(
            SessionData(
                session_id=item["session_id"],
                session_name=item["session_name"],
                is_private_session=item["is_private_session"],
                last_touched_at=datetime.fromisoformat(
                    item["last_touched_at"]
                ).astimezone(timezone.utc),
            )
        )
    logger.debug(
        f"[fetch_session_ids] Loaded {len(session_list)} sessions for user_id={user_id}"
    )
    return session_list


def fetch_session_history(
    server_url: str,
    user_id: str,
    app_name: str,
    session_id: str,
    limit: int = DEFAULT_MESSAGE_LIMIT,
) -> tuple[list[Message], bool]:
    """Fetch a slice of chat history from FastAPI.

    Args:
        server_url: Base URL of the backend (e.g. ``"http://localhost:8006"``).
        user_id: ID of the signed-in user.
        app_name: Name of the application.
        session_id: Session UUID whose history is requested.
        limit: Max number of newest rounds to retrieve.
        timeout: HTTP timeout in seconds.

    Returns:
        tuple[list[Message], bool]: Parsed messages (oldest → newest) and the
        ``has_more`` flag.

    Raises:
        requests.HTTPError: If the server responds with a non-2xx status.
        requests.RequestException: For network-level errors and timeouts.
        ValueError: If the JSON payload is malformed.

    """
    endpoint = (
        f"{server_url}/users/{user_id}/apps/{app_name}/sessions/"
        f"{session_id}/queries?limit={limit}"
    )

    response = requests.get(endpoint)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"Invalid JSON returned from {endpoint} for user_id={user_id}, session_id={session_id}: {exc}"
        ) from exc

    messages: list[Message] = [Message(**m) for m in payload["messages"]]
    has_more: bool = bool(payload["has_more"])
    return messages, has_more


def post_query_to_fastapi(
    server_url: str,
    user_id: str,
    app_name: str,
    session_id: str,
    messages_list: list[dict],
    user_msg_id: str,
    system_msg_id: str,
    assistant_msg_id: str,
    round_id: int,
    rag_mode: RagModeEnum,
    use_reranker: bool,
) -> requests.Response:
    """
    Sends a POST request to FastAPI's RAG+LLM endpoint with stream=True.

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID.
        app_name (str): The name of the application.
        session_id (str): The session ID for which the query is posted.
        messages_list (list[dict]): A list of messages, each with {"role", "content"}.
        user_msg_id (str): The UUID for the user's message.
        system_msg_id (str): The UUID for the system message.
        assistant_msg_id (str): The UUID for the assistant's message.
        round_id (int): The round number computed on the client side.
        rag_mode (RagModeEnum): Retrieval mode to apply. See :class:`RagModeEnum` for options.
        use_reranker (bool): Whether to use the reranker.

    Returns:
        requests.Response: A *streaming* Response object.
            • The caller must check ``response.status_code`` (it can be
              2xx *or* 4xx/5xx).
            • For 409 Conflict, the UI shows “Press F5 to refresh”.

    Raises:
        requests.RequestException: Network-level errors (connection,
            DNS, timeout).  HTTP error codes are **not** raised here.
    """
    endpoint = (
        f"{server_url}/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries"
    )

    # NOTE: I deleted file option. If it's necessary, use multipart/form-data for file uploads.
    payload = {
        "messages": messages_list,
        "user_msg_id": user_msg_id,
        "system_msg_id": system_msg_id,
        "assistant_msg_id": assistant_msg_id,
        "round_id": round_id,
        "rag_mode": rag_mode.value,
        "use_reranker": use_reranker,
    }

    logger.debug(
        f"[post_query_to_fastapi] Posting query to endpoint={endpoint} for user_id={user_id}, session_id={session_id}, round_id={round_id}"
    )

    try:
        response = requests.post(endpoint, json=payload, stream=True)
        logger.debug(
            f"[post_query_to_fastapi] POST succeeded for user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        return response
    except requests.RequestException as e:
        logger.exception(
            f"[post_query_to_fastapi] POST failed for user_id={user_id}, session_id={session_id}, round_id={round_id}"
        )
        raise


def put_session_info(
    server_url: str,
    user_id: str,
    app_name: str,
    session_id: str,
    session_name: str,
    is_private_session: bool,
    is_deleted: bool = False,
) -> None:
    """
    Sends a PUT request to create or replace session info in the backend.

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID.
        session_id (str): The session ID to be created or replaced.
        session_name (str): The new session name.
        is_private_session (bool): The new is_private_session value.
        is_deleted (bool): Whether the session is being marked as deleted.

    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    endpoint = f"{server_url}/users/{user_id}/apps/{app_name}/sessions/{session_id}"
    payload = {
        "session_name": session_name,
        "is_private_session": is_private_session,
        "is_deleted": is_deleted,
    }

    response = requests.put(endpoint, json=payload)
    response.raise_for_status()


def update_session_info_with_check(
    server_url: str,
    user_id: str,
    app_name: str,
    session_id: str,
    before_name: str,
    before_is_private: bool,
    before_is_deleted: bool,
    after_name: str,
    after_is_private: bool,
    after_is_deleted: bool,
) -> None:
    """
    Sends a session update request to the FastAPI server with before/after values for pessimistic locking.

    Args:
        server_url (str): The base URL of the FastAPI backend.
        user_id (str): The ID of the user performing the update.
        app_name (str): The name of the application the session belongs to.
        session_id (str): The ID of the session to update.
        before_name (str): The expected name of the session before the update.
        before_is_private (bool): The expected privacy status before the update.
        before_is_deleted (bool): The expected deletion status before the update.
        after_name (str): The new name for the session.
        after_is_private (bool): The new privacy setting.
        after_is_deleted (bool): The new deletion flag.

    Raises:
        requests.HTTPError: If the request fails with a non-200 response.
    """
    url = f"{server_url}/users/{user_id}/apps/{app_name}/sessions/{session_id}/update_with_check"

    payload = {
        "before_session_name": before_name,
        "before_is_private_session": before_is_private,
        "before_is_deleted": before_is_deleted,
        "after_session_name": after_name,
        "after_is_private_session": after_is_private,
        "after_is_deleted": after_is_deleted,
    }

    response = requests.patch(url, json=payload)
    if not response.ok:
        response.raise_for_status()


def post_create_session_with_limit(
    server_url: str,
    user_id: str,
    app_name: str,
    new_session_id: str,
    session_name: str,
    is_private: bool,
    known_session_ids: list[str],
    delete_target_session_id: str | None = None,
) -> str:
    """
    Calls the new session creation API with session count checks and consistency validation.

    Args:
        server_url (str): Backend URL
        user_id (str): User ID
        app_name (str): Application name
        new_session_id (str): New session UUID
        session_name (str): Session name
        is_private (bool): Whether session is private
        known_session_ids (list[str]): List of session UUIDs (non-deleted)
        delete_target_session_id (str | None): ID of session to delete if needed

    Returns:
        str: The created session_id

    Raises:
        requests.exceptions.HTTPError: If the server returns an HTTP error status.
        requests.exceptions.RequestException: For network or connection issues.
    """
    endpoint = (
        f"{server_url}/users/{user_id}/apps/{app_name}/sessions/create_with_limit"
    )
    payload = {
        "new_session_data": {
            "session_id": new_session_id,
            "session_name": session_name,
            "is_private_session": is_private,
        },
        "known_session_ids": known_session_ids,
    }
    if delete_target_session_id:
        payload["delete_target_session_id"] = delete_target_session_id
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()["session_id"]

    except requests.exceptions.HTTPError as http_err:
        # Log detailed error with status and body
        status_code = http_err.response.status_code
        error_body = http_err.response.text
        logger.exception(
            f"[post_create_session_with_limit] HTTPError: status={status_code}, user_id={user_id}, app_name={app_name}, body={error_body}"
        )
        raise

    except requests.exceptions.RequestException as req_err:
        logger.exception(
            f"[post_create_session_with_limit] RequestException: user_id={user_id}, app_name={app_name}, error={req_err}"
        )
        raise


def delete_round(
    server_url: str, session_id: str, round_id: str, deleted_by: str
) -> None:
    """
    Calls the FastAPI endpoint to delete a round (logical delete).

    Args:
        server_url (str): The backend server URL.
        session_id (str): The session ID.
        round_id (str): The unique round ID to delete.
        deleted_by (str): The user ID performing the deletion.
    """
    endpoint = f"{server_url}/sessions/{session_id}/rounds/{round_id}"
    payload = {"is_deleted": True, "deleted_by": deleted_by}
    # Make a DELETE request with a JSON body
    response = requests.delete(endpoint, json=payload)
    response.raise_for_status()


def patch_feedback(
    server_url: str,
    llm_output_id: str,
    feedback: str,
    reason: str | None,
    user_id: str,
    session_id: str,
) -> None:
    """
    Calls the FastAPI endpoint to patch feedback for a given LLM output ID.

    Args:
        server_url (str): The backend server URL.
        llm_output_id (str): The ID of the LLM output to patch feedback on.
        feedback (str): "good" or "bad" feedback type.
        reason (str): The user's explanation or comment for the feedback.
        user_id (str): The user submitting the feedback.
        session_id (str): The session associated with the feedback.

    Raises:
        HTTPError: If the PATCH request fails (non-2xx response).

    """
    endpoint = f"{server_url}/llm_outputs/{llm_output_id}"
    payload = {
        "feedback": feedback,
        "reason": reason,
        "user_id": user_id,
        "session_id": session_id,
    }
    # Make a PATCH request with a JSON body
    response = requests.patch(endpoint, json=payload)
    response.raise_for_status()


#################################
# Utility / Helper Functions
#################################


def check_access_control(user_id: str, employee_class_id: str) -> None:
    """
    Check whether the given employee_class_id is allowed to access the app.

    If the ID is not in the allowed set, an error message is shown and execution is halted.

    Args:
        user_id (str): The ID of the current user (for logging).
        employee_class_id (str): The employee class ID of the current user.

    Side Effects:
        - Logs an access denial event with user ID.
        - Displays an error message via Streamlit.
        - Calls st.stop() to terminate the app flow if access is denied.
    """
    if employee_class_id not in EMPLOYEE_CLASS_ALLOWED_IDS:
        logger.warning(
            f"[check_access_control] Access denied for user_id={user_id}, employee_class_id={employee_class_id}"
        )
        st.error(ERROR_LABELS.ACCESS_DENIED)
        st.stop()


def is_debug_session_active() -> bool:
    sessions = st.session_state.get("session_ids", [])
    return any(s.session_name == DEBUG_SESSION_TRIGGER for s in sessions)


def inject_header_css(app_title: str) -> None:
    """Inject CSS to fix the Streamlit header and display a centered title.

    This makes the header fixed at the top, adds a custom title via a ::before
    pseudo-element, and shifts the page content down so nothing is hidden.

    Args:
        app_title: The title string to render in the header.
    """
    st.markdown(
        f"""
        <style>
        header.stAppHeader {{
            position: fixed !important;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
            background-color: #f9f9f9;
            border-bottom: 1px solid #ccc;
        }}

        header.stAppHeader:before {{
            content: "{app_title}";
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            top: 0.75rem;
            display: block;
            font-size: 1.8rem;
            font-weight: 600;
            color: #333;
            font-family: 'Segoe UI', sans-serif;
            white-space: nowrap;
        }}

        [data-testid="stAppViewContainer"] > .main > .block-container {{
            padding-top: 3rem;
            padding-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hide_streamlit_deploy_button() -> None:
    """Hide the Streamlit deploy button in the app header."""
    st.markdown(
        """
        <style>
            .stAppDeployButton { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hide_streamlit_menu() -> None:
    """Hide the Streamlit hamburger menu in the app header."""
    st.markdown(
        """
        <style>
            #MainMenu { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_ui_locking() -> bool:
    """Ensure UI locking state is initialized and display a warning if locked.

    This sets default session_state values for UI locking. If the UI is locked,
    a warning with the lock reason is shown in the sidebar.

    Returns:
        bool: True if UI is locked, False otherwise.
    """
    st.session_state.setdefault("is_ui_locked", False)
    st.session_state.setdefault("ui_lock_reason", "")

    is_locked: bool = st.session_state["is_ui_locked"]
    if is_locked:
        st.sidebar.warning(st.session_state["ui_lock_reason"])
    return is_locked


def initialize_session_state(user_id: str) -> None:
    """Initialize session state variables for the current user.

    Args:
        user_id: The ID of the user to store in session_state.
    """
    st.session_state.setdefault("current_session", None)
    st.session_state.setdefault("show_edit_form", False)
    st.session_state.setdefault("session_histories", {})
    st.session_state.setdefault("server_has_more", {})
    st.session_state.setdefault("show_create_form", False)
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = user_id
        logger.info(
            f"[initialize_session_state] user_id={user_id} has logged in to the app."
        )


def record_user_touch(*, user_id: str) -> None:
    """
    Record the current UTC time as the last moment of user interaction or
    app load and log it.

    Args:
        user_id: Authenticated user identifier, added to the log message.
    """
    st.session_state["user_touch_app_last_at"] = datetime.now(timezone.utc)
    logger.debug(
        "[record_user_touch] user_id=%s set user_touch_app_last_at=%s",
        user_id,
        st.session_state["user_touch_app_last_at"],
    )


def inject_redirect_js(*, timeout_seconds: int, target_url: str, user_id: str) -> None:
    """
    Inject a <meta http-equiv="refresh"> tag to redirect after inactivity.

    Args:
        timeout_seconds: Idle period before redirect (seconds).
        target_url: Absolute URL of the destination page.
        user_id: Authenticated user identifier, passed through for logging.
    """
    last_touch: datetime = st.session_state.get(
        "user_touch_app_last_at", datetime.now(timezone.utc)
    )
    remaining = max(
        0,
        timeout_seconds
        - int((datetime.now(timezone.utc) - last_touch).total_seconds()),
    )
    remaining_ms: int = remaining * 1000

    if remaining == 0:
        logger.info(
            "[inject_redirect_js] user_id=%s idle timeout exceeded; "
            "redirecting immediately → %s",
            user_id,
            target_url,
        )
        # If the timeout is already exceeded, redirect immediately
        st.markdown(
            f'<meta http-equiv="refresh" content="{remaining};url={target_url}" />',
            unsafe_allow_html=True,
        )
    else:
        logger.debug(
            "[inject_redirect_js] user_id=%s scheduling redirect in %d s (%d ms) → %s",
            user_id,
            remaining,
            remaining_ms,
            target_url,
        )


def setup_inactivity_redirect(
    *, user_id: str, timeout_seconds: int, target_url: str, interval_ms: int = 1000
) -> None:
    """
    Drive periodic reruns, reset the inactivity timer on user actions,
    and inject/refresh the redirect script.

    Args:
        user_id: Authenticated user identifier for logging.
        timeout_seconds: Seconds of inactivity before redirect.
        target_url: URL to navigate to when idle timeout triggers.
        interval_ms: Milliseconds between automatic reruns.
    """
    # Auto-refresh to drive reruns
    refresh_count = st_autorefresh(interval=interval_ms, key="inactivity_check")

    # Determine if this is a user-triggered run by comparing to previous count
    prev_count = st.session_state.get("inac_prev_refresh_count")
    # On initial load, prev_count is None -> treat as user interaction
    is_user_run = (prev_count is None) or (refresh_count == prev_count)

    if is_user_run:
        # Reset inactivity timer
        record_user_touch(user_id=user_id)

    # Store current count for next comparison
    st.session_state["inac_prev_refresh_count"] = refresh_count

    # Inject or update redirect script
    inject_redirect_js(
        timeout_seconds=timeout_seconds, target_url=target_url, user_id=user_id
    )


def render_dev_test_settings() -> None:
    """
    Renders the developer testing options in the sidebar if debug mode is active.
    Allows toggling simulated error states and delays.
    """
    if not is_debug_session_active():
        return

    with st.sidebar.expander("🛠️ Developer Settings", expanded=False):
        dev_cfg: DevTestConfig = st.session_state.get(
            "dev_test_config", DevTestConfig()
        )

        delay_sec = st.slider(
            "⏳ Simulate Delay (seconds)",
            min_value=0,
            max_value=10,
            key="dev_delay_sec",
            value=dev_cfg.simulate_delay_seconds,
        )
        llm_config_error = st.checkbox(
            "❌ Simulate LLM Config Error",
            key="dev_llm_config_error",
            value=dev_cfg.simulate_llm_config_error,
        )
        llm_timeout = st.checkbox(
            "⏱️ Simulate LLM Timeout",
            key="dev_llm_timeout",
            value=dev_cfg.simulate_llm_timeout,
        )
        backend_failure = st.checkbox(
            "📡 Simulate Backend Failure",
            key="dev_backend_failure",
            value=dev_cfg.simulate_backend_failure,
        )
        db_failure = st.checkbox(
            "🧩 Simulate DB Failure",
            key="dev_db_failure",
            value=dev_cfg.simulate_db_failure,
        )

        session_autocreate_failure = st.checkbox(
            "🚫 Simulate Auto Session Creation Failure",
            key="dev_session_autocreate_failure",
            value=dev_cfg.simulate_session_autocreate_failure,
        )

        llm_stream_error = st.checkbox(
            "📶 Simulate LLM Stream Error (bad structure)",
            key="dev_llm_stream_error",
            value=dev_cfg.simulate_llm_stream_error,
        )

        llm_invalid_json = st.checkbox(
            "🧨 Simulate Invalid JSON Chunk",
            key="dev_llm_invalid_json",
            value=dev_cfg.simulate_llm_invalid_json,
        )

        unexpected_exception = st.checkbox(
            "🧪 Simulate Unexpected Exception",
            key="dev_unexpected_exception",
            value=dev_cfg.simulate_unexpected_exception,
        )

        # Always rebuild from primitive widget state
        st.session_state["dev_test_config"] = DevTestConfig(
            simulate_delay_seconds=st.session_state["dev_delay_sec"],
            simulate_llm_config_error=st.session_state["dev_llm_config_error"],
            simulate_llm_timeout=st.session_state["dev_llm_timeout"],
            simulate_backend_failure=st.session_state["dev_backend_failure"],
            simulate_db_failure=st.session_state["dev_db_failure"],
            simulate_session_autocreate_failure=st.session_state[
                "dev_session_autocreate_failure"
            ],
            simulate_llm_stream_error=st.session_state["dev_llm_stream_error"],
            simulate_llm_invalid_json=st.session_state["dev_llm_invalid_json"],
            simulate_unexpected_exception=st.session_state["dev_unexpected_exception"],
        )


def sync_dev_test_config() -> None:
    """Re-build DevTestConfig from the primitive widget keys."""
    st.session_state["dev_test_config"] = DevTestConfig(
        simulate_delay_seconds=st.session_state.get("dev_delay_sec", 0),
        simulate_llm_config_error=st.session_state.get("dev_llm_config_error", False),
        simulate_llm_timeout=st.session_state.get("dev_llm_timeout", False),
        simulate_backend_failure=st.session_state.get("dev_backend_failure", False),
        simulate_db_failure=st.session_state.get("dev_db_failure", False),
        simulate_session_autocreate_failure=st.session_state.get(
            "dev_session_autocreate_failure", False
        ),
        simulate_llm_stream_error=st.session_state.get("dev_llm_stream_error", False),
        simulate_llm_invalid_json=st.session_state.get("dev_llm_invalid_json", False),
    )


def show_sidebar_error_message(user_id: str) -> None:
    """
    Displays any error message stored in st.session_state['error_message'] in the sidebar,
    and logs the message using logger.warning. After displaying, the message is deleted.

    Args:
        user_id (str): The user ID associated with the error message.
    """
    error_msg = st.session_state.pop("error_message", None)
    if error_msg:
        logger.warning(
            f"[show_sidebar_error_message] Showing deferred error message for user_id={user_id}: {error_msg}"
        )
        st.sidebar.error(error_msg)


def show_chat_error_message(user_id: str) -> None:
    """
    Displays any error message stored in st.session_state['chat_error_message']
    in the assistant's chat area, and logs the message using logger.warning.
    After displaying, the message is deleted.

    Args:
        user_id (str): The user ID associated with the error message.
    """
    error_msg = st.session_state.pop("chat_error_message", None)
    if error_msg:
        logger.warning(
            f"[show_chat_error_message] Showing chat error message for user_id={user_id}: {error_msg}"
        )
        # with st.chat_message("assistant"):
        st.error(error_msg)


def load_session_ids(server_url: str, user_id: str, app_name: str) -> list[SessionData]:
    """
    Load the list of session metadata from the server and cache it in Streamlit's session state.

    This function initializes the session list the first time it is called, by fetching
    data from the backend server. If already loaded, it simply returns the cached list.

    Args:
        server_url (str): The base URL of the backend server.
        user_id (str): The user ID whose sessions should be loaded.
        app_name (str): The name of the application associated with the sessions.

    Returns:
        list[SessionData]: A list of active session metadata objects.
    """
    try:
        # Ensure that session_ids is initialized in the session state.
        if "session_ids" not in st.session_state:
            st.session_state["session_ids"] = fetch_session_ids(
                server_url=server_url, user_id=user_id, app_name=app_name
            )
        return st.session_state["session_ids"]
    except Exception:
        logger.exception(
            f"[load_session_ids] Failed to fetch sessions for user_id={user_id}"
        )
        st.error(ERROR_LABELS.SESSION_LIST_LOAD_FAILED)
        st.stop()


def render_create_session_form(
    server_url: str, user_id: str, app_name: str, disabled_ui: bool
) -> None:
    """
    Render the UI and logic for creating a new session in the sidebar.

    This function shows a form in the Streamlit sidebar to allow the user to create
    a new session. If the session limit is reached, it automatically deletes the oldest
    sessions to make room for a new one. The session creation process is deferred and
    handled only when the user submits the form. UI is locked during the process to prevent
    concurrent interactions.

    Args:
        server_url (str): The base URL of the backend server.
        user_id (str): The ID of the current user.
        app_name (str): The name of the application managing the sessions.
        disabled_ui (bool): If True, disables all input elements in the form.

    Side Effects:
        - Modifies `st.session_state` to store session info.
        - Updates UI and session list.
        - Logs user interactions and error events.
        - Displays warnings, errors, or success messages in the sidebar.
    """

    MAX_SESSION_COUNT = 10
    current_session_count = len(st.session_state.get("session_ids", []))

    # --- Create Form ---
    if not st.session_state["show_create_form"]:
        if st.sidebar.button(
            LABELS.CREATE_SESSION, key="open_create_button", disabled=disabled_ui
        ):
            st.session_state["show_create_form"] = True
            st.rerun()

    if st.session_state["show_create_form"]:
        with st.sidebar.expander(LABELS.CREATE_SESSION, expanded=True):

            if current_session_count >= MAX_SESSION_COUNT:
                st.warning(
                    WARNING_LABELS.SESSION_LIMIT_REACHED.format(
                        max_count=MAX_SESSION_COUNT
                    )
                )

            new_session_name: str = st.text_input(
                LABELS.SESSION_NAME,
                value="Untitled Session",
                max_chars=30,
                key="create_session_name",
                disabled=disabled_ui,
            )
            new_session_is_private: bool = st.radio(
                LABELS.IS_PRIVATE,
                options=[True, False],
                key="create_is_private",
                disabled=disabled_ui,
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    LABELS.SUBMIT, key="finalize_create_button", disabled=disabled_ui
                ):
                    st.session_state["is_ui_locked"] = True
                    st.session_state["ui_lock_reason"] = UI_LOCK_LABELS.CREATING_SESSION
                    st.session_state["pending_create_session"] = {
                        "name": new_session_name,
                        "is_private": new_session_is_private,
                    }
                    st.session_state["show_create_form"] = False
                    st.rerun()

            with col2:
                if st.button(
                    LABELS.CANCEL, key="cancel_create_button", disabled=disabled_ui
                ):
                    for k in ("create_session_name", "create_is_private"):
                        if k in st.session_state:
                            del st.session_state[k]
                    st.session_state["show_create_form"] = False
                    st.rerun()

    # --- Handle deferred session creation ---
    if "pending_create_session" in st.session_state:
        data = st.session_state.pop("pending_create_session")

        # --- Developer test hooks ---
        dev_cfg: DevTestConfig = st.session_state.get(
            "dev_test_config", DevTestConfig()
        )

        try:
            logger.debug(
                f"[render_create_session_form] Creating session via API: "
                f"user_id={user_id}, private={data['is_private']}"
            )

            # --- Developer test hooks ---
            # Optional simulated delay
            if dev_cfg.simulate_delay_seconds > 0:
                time.sleep(dev_cfg.simulate_delay_seconds)

            # Optional simulated backend failure
            if dev_cfg.simulate_backend_failure:
                raise requests.exceptions.RequestException("Simulated backend failure")

            # Optional simulated unexpected exception
            if dev_cfg.simulate_unexpected_exception:
                raise RuntimeError(
                    "[DevTest] Simulated unexpected exception during session creation"
                )

            new_session_id = str(uuid.uuid4())

            known_ids = [s.session_id for s in st.session_state["session_ids"]]

            delete_target_id = None
            if len(known_ids) >= MAX_SESSION_COUNT:
                oldest = min(
                    st.session_state["session_ids"], key=lambda s: s.last_touched_at
                )
                delete_target_id = oldest.session_id

            _ = post_create_session_with_limit(
                server_url=server_url,
                user_id=user_id,
                app_name=app_name,
                new_session_id=new_session_id,
                session_name=data["name"],
                is_private=data["is_private"],
                known_session_ids=known_ids,
                delete_target_session_id=delete_target_id,
            )

            new_session = SessionData(
                session_id=new_session_id,
                session_name=data["name"],
                is_private_session=data["is_private"],
                last_touched_at=datetime.now(timezone.utc),  # UI上の一時的な最新時刻
            )

            st.session_state["session_ids"] = [
                s
                for s in st.session_state["session_ids"]
                if s.session_id != delete_target_id
            ] + [new_session]
            st.session_state["current_session"] = new_session

            logger.debug(
                f"[render_create_session_form] Session created successfully: "
                f"user_id={user_id}, session_id={new_session_id}"
            )

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            detail = http_err.response.json().get("detail", str(http_err))

            if status_code == 500:
                logger.error(
                    f"[render_create_session_form] Server error for user_id={user_id}: {detail}"
                )
                st.session_state["error_message"] = (
                    ERROR_LABELS.SESSION_CREATION_HTTP_500
                )

            elif status_code == 409:
                logger.warning(
                    f"[render_create_session_form] Conflict error for user_id={user_id}: {detail}"
                )
                st.session_state["error_message"] = (
                    ERROR_LABELS.SESSION_CREATION_HTTP_409
                )
            else:
                logger.exception(
                    f"[render_create_session_form] Unexpected HTTP error for user_id={user_id}"
                )
                st.session_state["error_message"] = (
                    f"{ERROR_LABELS.SESSION_CREATION_HTTP_UNEXPECTED}（HTTP {status_code}）"
                )

        except requests.exceptions.RequestException as exc:
            logger.exception(
                f"[render_create_session_form] API request failed for user_id={user_id}"
            )
            st.session_state["error_message"] = ERROR_LABELS.SESSION_CREATION_FAILED

        except Exception:
            logger.exception(
                f"[render_create_session_form] Unexpected error for user_id={user_id}"
            )
            st.session_state["error_message"] = (
                ERROR_LABELS.SESSION_CREATION_UNEXPECTED_ERROR
            )

        finally:
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()


def render_session_list(
    user_id: str, app_name: str, server_url: str, disabled_ui: bool
) -> SessionData:
    """
    Display the list of sessions in the sidebar and return the selected session.

    This function renders a radio button listing all available sessions and
    updates the current session when the selection changes. It disables
    session switching if UI operations are locked.

    Args:
        user_id (str): The ID of the current user.
        app_name (str): The name of the application context.
        server_url (str): The base URL of the FastAPI server.
        disabled_ui (bool): If True, disables the session selection UI during ongoing operations.

    Returns:
        SessionData: The session selected by the user in the sidebar.
    """

    st.sidebar.write(LABELS.SESSION_LIST)

    # If no session has been chosen yet, default to the first in the list
    if st.session_state["current_session"] is None:
        if len(st.session_state["session_ids"]) == 0:
            # Automatically create a new session if none exist
            new_session_id = str(uuid.uuid4())
            default_session_name = "Untitled Session"
            is_private = True
            try:
                # Developer mode: Simulate failure on automatic session creation
                dev_cfg: DevTestConfig = st.session_state.get(
                    "dev_test_config", DevTestConfig()
                )
                if dev_cfg.simulate_session_autocreate_failure:
                    raise requests.exceptions.RequestException(
                        "[DevTest] Simulated failure during auto session creation"
                    )

                # Register the session on the FastAPI side
                _ = post_create_session_with_limit(
                    server_url=server_url,
                    user_id=user_id,
                    app_name=app_name,
                    new_session_id=new_session_id,
                    session_name=default_session_name,
                    is_private=is_private,
                    known_session_ids=[],
                    delete_target_session_id=None,
                )

            except requests.exceptions.RequestException as exc:
                logger.exception(
                    f"[render_session_list] Failed to register session: user_id={user_id}"
                )
                st.sidebar.error(ERROR_LABELS.AUTO_SESSION_REGISTER_FAILED)
                st.stop()
            except Exception:
                logger.exception(
                    f"[render_session_list] Failed to create session: user_id={user_id}"
                )
                st.sidebar.error(ERROR_LABELS.AUTO_SESSION_CREATION_FAILED)
                st.stop()

            new_session = SessionData(
                session_id=new_session_id,
                session_name=default_session_name,
                is_private_session=is_private,
                last_touched_at=datetime.now(timezone.utc),
            )

            st.session_state["session_ids"].append(new_session)
            st.session_state["current_session"] = new_session
            st.rerun()
        else:
            # Default to the most recent session
            st.session_state["current_session"] = st.session_state["session_ids"][-1]

    # Radio button to choose an existing session
    # 1) Sort sessions newest-first
    sorted_sessions: list[SessionData] = sorted(
        st.session_state["session_ids"],
        key=lambda x: x.last_touched_at,
        reverse=True,
    )
    # 2) Determine default index to match current_session
    current_idx: int = 0
    try:
        current_session = st.session_state["current_session"]
        for idx, session in enumerate(sorted_sessions):
            if session.session_id == current_session.session_id:
                current_idx = idx
                break
    except Exception:
        logger.exception(
            f"[render_session_list] Failed to determine default session index: user_id={user_id}"
        )
        st.sidebar.error(ERROR_LABELS.SESSION_INDEX_RESOLVE_FAILED)
        st.stop()

    try:
        # 3-3) Define callback to update current_session when selection changes
        def _on_session_change() -> None:
            """Update the current_session when the user selects a different session."""
            try:
                st.session_state["current_session"] = st.session_state[
                    "unique_session_radio"
                ]
            except Exception:
                logger.exception(
                    f"[render_session_list] Failed to update current_session during radio selection: user_id={user_id}"
                )
                st.sidebar.error(ERROR_LABELS.SESSION_SELECTION_FAILED)
                st.stop()

        # 3-4) Render radio with explicit index and callback
        selected_session_data: SessionData = st.sidebar.radio(
            LABELS.SELECT_SESSION,
            options=sorted_sessions,
            index=current_idx,
            format_func=lambda x: (
                f"{x.session_name} (Private)"
                if x.is_private_session
                else x.session_name
            ),
            key="unique_session_radio",
            on_change=_on_session_change,
            disabled=disabled_ui,
        )
    except Exception:
        logger.exception(
            f"[render_session_list] Failed to render session radio: user_id={user_id}"
        )
        st.sidebar.error(ERROR_LABELS.SESSION_RADIO_RENDER_FAILED)
        st.stop()

    # Update the current session in session_state
    st.session_state["current_session"] = selected_session_data
    selected_session_id: str = selected_session_data.session_id

    # If we haven't loaded this session before, fetch from the server
    if selected_session_id not in st.session_state["session_histories"]:
        try:
            msgs, has_more = fetch_session_history(
                server_url=server_url,
                user_id=user_id,
                app_name=app_name,
                session_id=selected_session_id,
                limit=DEFAULT_MESSAGE_LIMIT,
            )
            st.session_state["session_histories"][selected_session_id] = msgs
            st.session_state["server_has_more"][selected_session_id] = has_more
        except Exception:
            logger.exception(
                f"[render_session_list] Failed to fetch session history: user_id={user_id}, session_id={selected_session_id}"
            )
            st.sidebar.error(ERROR_LABELS.MESSAGE_HISTORY_LOAD_FAILED)
            st.stop()

    return selected_session_data


def render_edit_session_form(
    user_id: str, app_name: str, server_url: str, disabled_ui: bool
) -> None:
    """Display the edit/delete form for the currently selected session.

    Args:
        user_id (str): The ID of the current user.
        server_url (str): The base URL of the FastAPI server.
        disabled_ui (bool): If True, disables user interactions for all input widgets during ongoing operations.


    Side Effects:
        - Locks the UI during update/delete operations.
        - Updates or deletes the session by calling `patch_session_info`.
        - Updates session state and reruns Streamlit.
    """
    # Display which session is active
    st.write(f"**Current Session**: {st.session_state['current_session']}")

    if not st.session_state["show_edit_form"]:
        if st.sidebar.button(
            LABELS.EDIT_SESSION, key="open_edit_button", disabled=disabled_ui
        ):
            st.session_state["show_edit_form"] = True
            st.rerun()

    if st.session_state["show_edit_form"]:
        with st.sidebar.expander(LABELS.EDIT_SESSION, expanded=True):
            current_session = st.session_state.get("current_session")
            if current_session is None:
                st.warning(WARNING_LABELS.NO_SESSION_SELECTED)
                return

            current_name: str = current_session.session_name
            current_is_private: bool = current_session.is_private_session
            current_is_deleted: bool = False

            edited_session_name: str = st.text_input(
                LABELS.SESSION_NAME,
                value=current_name,
                max_chars=30,
                key="edit_session_name",
                disabled=disabled_ui,
            )
            edited_is_private: bool = st.radio(
                LABELS.IS_PRIVATE,
                options=[True, False],
                index=0 if current_is_private else 1,
                key="edit_is_private",
                disabled=disabled_ui,
            )

            delete_this_session: bool = st.checkbox(
                LABELS.DELETE_SESSION, key="delete_session"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button(LABELS.UPDATE, key="update_session", disabled=disabled_ui):
                    st.session_state["is_ui_locked"] = True
                    st.session_state["ui_lock_reason"] = UI_LOCK_LABELS.UPDATING_SESSION
                    st.session_state["pending_edit_action"] = {
                        "delete": delete_this_session,
                        "session_id": current_session.session_id,
                        "before_name": current_name,
                        "before_private": current_is_private,
                        "before_deleted": current_is_deleted,
                        "after_name": edited_session_name,
                        "after_private": edited_is_private,
                        "after_deleted": delete_this_session,
                    }
                    st.session_state["show_edit_form"] = False
                    st.rerun()

            with col2:
                if st.button(
                    LABELS.CANCEL, key="cancel_edit_button", disabled=disabled_ui
                ):
                    for k in ("edit_session_name", "edit_is_private", "delete_session"):
                        if k in st.session_state:
                            del st.session_state[k]
                    st.session_state["show_edit_form"] = False
                    st.rerun()

    # --- Perform the pending edit action after rerun ---
    if "pending_edit_action" in st.session_state:
        action = st.session_state.pop("pending_edit_action")
        try:
            logger.debug(
                f"[render_edit_session_form] Processing session update request: "
                f"user_id={user_id}, action={'DELETE' if action['delete'] else 'UPDATE'}, "
                f"session_id={action['session_id']}, len(after_name)='{len(action['after_name'])}', "
                f"private={action['after_private']}"
            )

            # --- Developer test hooks ---
            dev_cfg: DevTestConfig = st.session_state.get(
                "dev_test_config", DevTestConfig()
            )

            if dev_cfg.simulate_delay_seconds > 0:
                time.sleep(dev_cfg.simulate_delay_seconds)

            if dev_cfg.simulate_backend_failure:
                raise requests.exceptions.ConnectionError(
                    "Simulated backend failure during session edit"
                )

            if dev_cfg.simulate_unexpected_exception:
                raise RuntimeError(
                    "[DevTest] Simulated unexpected exception during session modification"
                )

            update_session_info_with_check(
                server_url=server_url,
                user_id=user_id,
                app_name=app_name,
                session_id=action["session_id"],
                before_name=action["before_name"],
                before_is_private=action["before_private"],
                before_is_deleted=action["before_deleted"],
                after_name=action["after_name"],
                after_is_private=action["after_private"],
                after_is_deleted=action["after_deleted"],
            )

            logger.debug(
                f"[render_edit_session_form] Session '{action['session_id']}' updated successfully for user_id={user_id}, session_id={action['session_id']}, deletion={action['delete']}"
            )

            for s in st.session_state["session_ids"]:
                if s.session_id == action["session_id"]:
                    s.session_name = action["after_name"]
                    s.is_private_session = action["after_private"]
                    # s.is_deleted = action["after_deleted"]
                    break

            if action["after_deleted"]:
                st.session_state["session_ids"] = [
                    s
                    for s in st.session_state["session_ids"]
                    if s.session_id != action["session_id"]
                ]

                if (
                    st.session_state["current_session"]
                    and st.session_state["current_session"].session_id
                    == action["session_id"]
                ):
                    st.session_state["current_session"] = None

        except requests.HTTPError as http_err:
            status = http_err.response.status_code
            if status == 404:
                # HTTPException (404): Session does not exist or is already deleted.
                logger.warning(
                    f"[render_edit_session_form] user_id={user_id} - Session not found (deleted or missing): session_id={action['session_id']}"
                )
                st.session_state["error_message"] = ERROR_LABELS.SESSION_EDIT_HTTP_404
            elif status == 409:
                # HTTPException (409): Session state conflict.
                logger.warning(
                    f"[render_edit_session_form] user_id={user_id} - Session conflict detected for session_id={action['session_id']}"
                )
                st.session_state["error_message"] = ERROR_LABELS.SESSION_EDIT_HTTP_409
            else:
                # HTTPException (500): Unexpected server error.
                logger.error(
                    f"[render_edit_session_form] user_id={user_id} - Server error ({status}) while modifying session_id={action['session_id']}"
                )
                st.session_state["error_message"] = (
                    f"Server error occurred. Please try again later."
                )

        except Exception as e:
            logger.exception(
                f"[render_edit_session_form] Unexpected error during session modification for session '{action['session_id']}', user_id={user_id}"
            )
            st.session_state["error_message"] = ERROR_LABELS.SESSION_EDIT_HTTP_500
        finally:
            st.session_state["show_edit_form"] = False
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()


def get_current_session_messages(user_id: str) -> tuple[str, list[Message]]:
    """Retrieve the current session ID and its sorted messages.

    Handles errors by showing a user-facing message and stopping the app.

    Args:
        user_id (str): The user ID, used for logging context.

    Returns:
        tuple[str, list[Message]]: A tuple of (current_session_id, sorted messages list).
    """
    try:
        current_session = st.session_state["current_session"]
        current_session_id = current_session.session_id
        messages = st.session_state["session_histories"][current_session_id]
        messages.sort(
            key=lambda m: (
                m.round_id,
                ROLE_ORDER.get(m.role, 99),  # unknown roles sink to the end
            )
        )
        return current_session_id, messages
    except Exception:
        logger.exception(
            f"[get_current_session_messages] Failed to retrieve or sort messages: "
            f"user_id={user_id}, session_id={locals().get('current_session_id', 'unknown')}"
        )
        # Show an error in the UI and stop execution
        st.error(ERROR_LABELS.MESSAGE_DISPLAY_FAILED)
        st.stop()


def render_load_more_button(
    *,
    session_id: str,
    server_url: str,
    user_id: str,
    app_name: str,
    messages: list[Message],
    disabled_ui: bool,
) -> None:
    """Render the “Load 10 more” button and handle its click.

    Args:
        session_id: UUID of the session currently displayed.
        server_url: Base URL of the FastAPI backend.
        user_id: ID of the signed-in user.
        app_name: Name of the application (passed through to the backend).
        messages: Cached conversation for the session (oldest → newest).
        disabled_ui (bool): If ``True``, disables the button (UI lock is active).

    Side Effects:
        * Locks the UI while a fetch is in progress.
        * Updates ``st.session_state["session_histories"][session_id]`` on success.
        * Stores a user-visible error in ``st.session_state["chat_error_message"]``
          on failure.
        * Always triggers ``st.rerun()`` to refresh the UI.
    """
    current_rounds: int = len({m.round_id for m in messages})
    server_has_more = st.session_state["server_has_more"].get(session_id, True)
    if current_rounds >= MAX_MESSAGE_LIMIT or not server_has_more:
        return

    new_limit = min(current_rounds + DEFAULT_MESSAGE_LIMIT, MAX_MESSAGE_LIMIT)

    if st.button(LABELS.LOAD_MORE, key="load_more_button", disabled=disabled_ui):
        st.session_state["is_ui_locked"] = True
        st.session_state["ui_lock_reason"] = UI_LOCK_LABELS.LOADING_HISTORY
        st.session_state["pending_load_more"] = {
            "session_id": session_id,
            "limit": new_limit,
        }
        st.rerun()

    if pending := st.session_state.pop("pending_load_more", None):
        sid: str = pending["session_id"]
        limit: int = pending["limit"]

        # Optional: show spinner during fetch
        with st.spinner("Loading older rounds…"):
            try:
                new_msgs, has_more = fetch_session_history(
                    server_url=server_url,
                    user_id=user_id,
                    app_name=app_name,
                    session_id=sid,
                    limit=limit,
                )
                st.session_state["session_histories"][sid] = new_msgs
                st.session_state["server_has_more"][sid] = has_more
                logger.debug(
                    "[load_more] Loaded history up to limit=%d (has_more=%s) "
                    "for session_id=%s, user_id=%s",
                    limit,
                    has_more,
                    sid,
                    user_id,
                )
            except Exception:
                logger.exception(
                    "[load_more] Failed to load more rounds: session_id=%s, user_id=%s",
                    sid,
                    user_id,
                )
                st.session_state["chat_error_message"] = (
                    ERROR_LABELS.MESSAGE_HISTORY_APPEND_FAILED
                )
            finally:
                # Always rerun so that UI (button disable, new messages) updates
                st.session_state["is_ui_locked"] = False
                st.session_state["ui_lock_reason"] = ""
                st.rerun()


def render_system_context_rows(
    rows: list[dict[str, Any]],
    user_id: str,
    session_id: str,
) -> None:
    """
    Render system context rows with error handling.

    Args:
        rows (list[dict[str, Any]]): Parsed context rows to be displayed.
        user_id (str): User ID for logging.
        session_id (str): Session ID for logging.

    Side Effects:
        - Renders context rows in Markdown.
        - Falls back to warning if display fails.
    """
    try:
        if isinstance(rows, list) and rows:
            with st.expander(LABELS.VIEW_SOURCES):
                for row in rows:
                    raw_text = row.get("text", "").strip()
                    lines = raw_text.splitlines()
                    quoted_text = "\n".join(f"> {line}" for line in lines)

                    st.markdown(
                        f"**RAG Rank:** {row.get('rag_rank', '-')}\n"
                        f"**Doc ID:** {row.get('doc_id', '')}\n"
                        f"**Semantic Distance:** {float(row.get('semantic_distance', 0.0)):.4f}\n\n"
                        f"{quoted_text}"
                    )
        else:
            st.caption(WARNING_LABELS.NO_CONTEXT)
    except Exception:
        st.caption(WARNING_LABELS.RENDER_SYSTEM_MESSAGE_FAILED)
        logger.exception(
            f"[render_system_context_rows] Failed to render rows: user_id={user_id}, session_id={session_id}"
        )


def render_chat_messages(
    messages: list[Message],
    server_url: str,
    current_session_id: str,
    user_id: str,
    disabled_ui: bool,
) -> None:
    """
    Renders the existing conversation messages, including
    Trash/Good/Bad buttons for assistant messages.

    Args:
        messages (list[Message]): The list of messages in the conversation.
        server_url (str): The base URL of the FastAPI server.
        current_session_id (str): The current session's ID.
        user_id (str): The user ID for the current user.
        disabled_ui (bool): If True, disables feedback and deletion buttons during ongoing operations.

    Side Effects:
        - Displays each message via st.chat_message
        - If user clicks Trash, calls delete_round(...)
          and marks relevant messages as deleted in local state
        - If user clicks Good/Bad, sets st.session_state["feedback_form_id"]
          and st.session_state["feedback_form_type"]
    """
    displayed_round_ids: set[int] = set()
    logger.debug(
        f"[render_chat_messages] Rendering chat for user_id={user_id}, session_id={current_session_id} with {len(messages)} messages"
    )

    dev_cfg: DevTestConfig = st.session_state.get("dev_test_config", DevTestConfig())

    for msg in messages:
        if msg.is_deleted:
            if msg.role == "user":
                st.caption(LABELS.DELETED_MESSAGE_NOTICE)
            continue

        if msg.role in {"user", "assistant"}:
            with st.chat_message(msg.role):
                st.write(msg.content)
        elif msg.role == "system":
            try:
                rows = json.loads(msg.content)
                if not isinstance(rows, list):
                    raise ValueError("System message is not a list.")
            except Exception:
                st.caption(WARNING_LABELS.PARSE_SYSTEM_MESSAGE_FAILED)
                logger.exception(
                    f"[render_chat_messages] Failed to parse system message content: user_id={user_id}, session_id={current_session_id}"
                )
            else:
                render_system_context_rows(rows, user_id, current_session_id)

        # For assistant messages, show the row of Trash/Good/Bad
        if msg.role == "assistant":
            # Only show one trash button per round
            if msg.round_id not in displayed_round_ids:
                displayed_round_ids.add(msg.round_id)

            col_trash, col_good, col_bad = st.columns([1, 1, 1])

            # Trash icon button
            if col_trash.button(
                "🗑️",
                key=f"confirm_delete_button_{msg.round_id}",
                help="Delete this round",
                disabled=disabled_ui,
            ):
                logger.debug(
                    f"[render_chat_messages] Trash button clicked for round_id={msg.round_id} by user_id={user_id}"
                )
                st.session_state["confirm_delete_round_id"] = msg.round_id

            # If delete was requested, show confirmation prompt
            if st.session_state.get("confirm_delete_round_id") == msg.round_id:
                with st.expander(LABELS.CONFIRM_DELETION, expanded=True):
                    st.warning(WARNING_LABELS.CONFIRM_DELETION_PROMPT)
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button(
                            LABELS.YES_DELETE, key=f"confirm_yes_{msg.round_id}"
                        ):
                            logger.debug(
                                f"[render_chat_messages] Confirmed deletion for round_id={msg.round_id} by user_id={user_id}"
                            )
                            st.session_state["is_ui_locked"] = True
                            st.session_state["ui_lock_reason"] = (
                                UI_LOCK_LABELS.DELETING_ROUND
                            )
                            st.session_state["pending_delete_round_id"] = msg.round_id
                            st.session_state["pending_delete_user_id"] = user_id
                            st.session_state["confirm_delete_round_id"] = None
                            st.rerun()
                    with col_cancel:
                        if st.button(
                            LABELS.NO_CANCEL, key=f"confirm_no_{msg.round_id}"
                        ):
                            st.session_state["confirm_delete_round_id"] = None
                            st.rerun()

            # Good button
            if col_good.button("😊", key=f"good_{msg.id}", disabled=disabled_ui):
                logger.debug(
                    f"[render_chat_messages] GOOD feedback for message_id={msg.id} by user_id={user_id}"
                )
                st.session_state["feedback_form_id"] = msg.id
                st.session_state["feedback_form_type"] = "good"
                st.rerun()

            # Bad button
            if col_bad.button("😞", key=f"bad_{msg.id}", disabled=disabled_ui):
                logger.debug(
                    f"[render_chat_messages] BAD feedback for message_id={msg.id} by user_id={user_id}"
                )
                st.session_state["feedback_form_id"] = msg.id
                st.session_state["feedback_form_type"] = "bad"
                st.rerun()

            # Inline feedback form if active
            if st.session_state.get("feedback_form_id") == msg.id:
                with st.expander(LABELS.FEEDBACK_PROMPT, expanded=True):
                    feedback_reason = st.text_area(
                        LABELS.FEEDBACK_REASON.format(
                            max_feedback_length=MAX_FEEDBACK_LENGTH
                        ),
                        key="feedback_reason",
                        max_chars=MAX_FEEDBACK_LENGTH,
                        disabled=disabled_ui,
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            LABELS.SUBMIT_FEEDBACK,
                            key="submit_feedback",
                            disabled=disabled_ui,
                        ):
                            logger.debug(
                                f"[render_chat_messages] Submitting feedback for message_id={msg.id}, by user_id={user_id}, "
                                f"type={st.session_state['feedback_form_type']}, "
                                f"len(reason)='{len(feedback_reason)}'"
                            )
                            st.session_state["is_ui_locked"] = True
                            st.session_state["ui_lock_reason"] = (
                                UI_LOCK_LABELS.SUBMITTING_FEEDBACK
                            )
                            st.session_state["pending_feedback"] = {
                                "llm_output_id": msg.id,
                                "feedback_type": st.session_state["feedback_form_type"],
                                "reason": feedback_reason,
                            }
                            st.rerun()
                    with col2:
                        if st.button(
                            LABELS.CANCEL_FEEDBACK,
                            key="cancel_feedback",
                            disabled=disabled_ui,
                        ):
                            st.session_state["feedback_form_id"] = None
                            st.session_state["feedback_form_type"] = None
                            st.rerun()

    # Handle deletion
    if (
        st.session_state.get("pending_delete_round_id") is not None
        and st.session_state.get("pending_delete_user_id") is not None
    ):
        round_id = st.session_state.pop("pending_delete_round_id")
        deleted_by_user = st.session_state.pop("pending_delete_user_id")
        try:
            if dev_cfg.simulate_backend_failure:
                raise requests.exceptions.RequestException(
                    "Simulated backend failure (delete_round)"
                )
            if dev_cfg.simulate_unexpected_exception:
                raise RuntimeError(
                    "[DevTest] Simulated unexpected exception in delete_round"
                )

            delete_round(
                server_url=server_url,
                session_id=current_session_id,
                round_id=round_id,
                deleted_by=deleted_by_user,
            )
            logger.debug(
                f"[render_chat_messages] Deleted round_id={round_id} by user={deleted_by_user}"
            )
            for m in messages:
                if m.round_id == round_id:
                    m.is_deleted = True
            st.session_state["session_histories"][current_session_id] = messages
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"[render_chat_messages] RequestException while deleting round_id={round_id} for user_id={user_id}: {e}"
            )
            st.session_state["chat_error_message"] = (
                ERROR_LABELS.MESSAGE_DELETION_FAILED
            )
        except Exception:
            logger.exception(
                f"[render_chat_messages] Unexpected error while deleting round_id={round_id} for user_id={user_id}"
            )
            st.session_state["chat_error_message"] = (
                ERROR_LABELS.MESSAGE_DELETION_UNEXPECTED_ERROR
            )
        finally:
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()

    # Handle feedback submission
    if "pending_feedback" in st.session_state:
        try:
            pending = st.session_state.pop("pending_feedback")
            logger.debug(
                f"[render_chat_messages] Feedback submitted: message_id={pending['llm_output_id']} by user_id={user_id}, "
                f"type={pending['feedback_type']}, len(reason)='{len(pending['reason'])}'"
            )

            if dev_cfg.simulate_backend_failure:
                raise requests.exceptions.RequestException(
                    "Simulated backend failure (patch_feedback)"
                )
            if dev_cfg.simulate_unexpected_exception:
                raise RuntimeError(
                    "[DevTest] Simulated unexpected exception in feedback submission"
                )

            patch_feedback(
                server_url=server_url,
                llm_output_id=pending["llm_output_id"],
                feedback=pending["feedback_type"],
                reason=pending["reason"],
                user_id=user_id,
                session_id=current_session_id,
            )
            st.session_state["feedback_form_id"] = None
            st.session_state["feedback_form_type"] = None
        except requests.exceptions.RequestException as e:
            logger.exception(
                f"[render_chat_messages] RequestException while submitting feedback for message_id={pending['llm_output_id']} by user_id={user_id}"
            )
            st.session_state["chat_error_message"] = (
                ERROR_LABELS.FEEDBACK_SUBMISSION_FAILED
            )
            st.session_state["feedback_form_id"] = None
            st.session_state["feedback_form_type"] = None
        except Exception:
            logger.exception(
                f"[render_chat_messages] Unexpected error while submitting feedback for message_id={pending['llm_output_id']} by user_id={user_id}"
            )
            st.session_state["chat_error_message"] = (
                ERROR_LABELS.FEEDBACK_SUBMISSION_UNEXPECTED_ERROR
            )
            st.session_state["feedback_form_id"] = None
            st.session_state["feedback_form_type"] = None
        finally:
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()


def render_user_chat_input(
    messages: list[Message],
    server_url: str,
    user_id: str,
    app_name: str,
    current_session_id: str,
    num_of_prev_msg_with_llm: int,
    disabled_ui: bool,
) -> None:
    """
    Displays a radio button to select a RAG usage mode, a chat input box,
    and if the user enters text, sends it to the backend with the chosen RAG mode
    and streams the assistant's response. All UI inputs are disabled if disabled_ui is True.

    Args:
        messages (list[Message]): The current conversation messages.
        server_url (str): The base URL of the FastAPI server.
        user_id (str): The ID of the current user.
        app_name (str): The name of the application.
        current_session_id (str): The session ID for which messages are posted.
        num_of_prev_msg_with_llm (int): Number of previous messages to send to the LLM.
        disabled_ui (bool): If True, disables input widgets during ongoing operations.

    Side Effects:
        - Displays a radio button to pick how to use RAG (or not).
        - Displays a radio button to pick whether to use the reranker.
        - If the user enters text, appends a user message to `messages`.
        - Calls post_query_to_fastapi(...) to get a streaming response.
        - Streams partial responses and appends final assistant message.
        - Potentially reruns the app after completion.
    """

    # 1) Let user pick the RAG usage mode.
    st.sidebar.write(LABELS.RAG_MODE_SECTION)

    rag_mode_label: str = st.sidebar.radio(
        LABELS.CHOOSE_RAG_MODE,
        options=RAG_MODE_OPTIONS,
        index=0,
        key="rag_mode_radio",
        disabled=disabled_ui,
    )
    rag_mode: RagModeEnum = RagModeEnum(rag_mode_label)

    with st.sidebar.expander(LABELS.RAG_MODE_HELP_TITLE, expanded=False):
        st.markdown(LABELS.RAG_MODE_HELP)

    # reranker option
    # st.sidebar.write(LABELS["reranker_section"])
    # use_reranker: bool = st.sidebar.radio(
    #     label="Choose whether to use the Reranker:",
    #     options=[False],  # Only one option for now
    #     format_func=lambda x: "Yes" if x else "No",
    #     index=0,  # Default to "No"
    #     key="use_reranker_radio",
    #     disabled=disabled_ui,
    # )
    use_reranker: bool = False

    # 2) Provide a chat input box for the user to type their query.
    user_input: str = st.chat_input(
        LABELS.CHAT_INPUT_PLACEHOLDER,
        disabled=disabled_ui,
        max_chars=MAX_CHAT_INPUT_LENGTH,
        key="user_chat_input",
    )

    if user_input:
        st.session_state["is_ui_locked"] = True
        st.session_state["ui_lock_reason"] = UI_LOCK_LABELS.SENDING_MESSAGE
        st.session_state["pending_user_input"] = user_input
        st.rerun()

    if st.session_state.get("pending_user_input"):
        try:
            dev_cfg: DevTestConfig = st.session_state.get(
                "dev_test_config", DevTestConfig()
            )

            user_input = st.session_state.pop("pending_user_input")
            logger.debug(
                f"[render_user_chat_input] Query submitted by user_id={user_id}, session_id={current_session_id}, len(user_input)={len(user_input)}"
            )

            if dev_cfg.simulate_unexpected_exception:
                raise RuntimeError(
                    "[DevTest] Simulated unexpected exception in user chat input"
                )

            # 2a) Build the short array of recent non-deleted messages
            last_msgs = last_n_non_deleted(
                messages=messages, n=num_of_prev_msg_with_llm
            )
            messages_to_send = [
                {"role": m.role, "content": m.content}
                for m in last_msgs
                if m.role in {"user", "assistant"}
            ]

            # 2b) Compute the next round_id
            #     Next round_id is “last confirmed + 1”.
            #     This prevents gaps caused by failed duplicates.
            last_round_id: int = _last_confirmed_round_id(messages)
            new_round_id: int = last_round_id + 1

            # 2c) Generate UUIDs for user/system/assistant messages
            user_msg_id = str(uuid.uuid4())
            system_msg_id = str(uuid.uuid4())
            assistant_msg_id = str(uuid.uuid4())

            # 2d) Add the user's message locally
            user_msg = Message(
                role=RoleEnum.USER,
                content=user_input,
                id=user_msg_id,
                round_id=new_round_id,
                is_deleted=False,
            )
            messages.append(user_msg)
            messages_to_send.append({"role": "user", "content": user_input})

            # Display the user message immediately
            with st.chat_message("user"):
                st.write(user_input)

            messages_summary = [
                {
                    "role": (
                        m["role"].value
                        if hasattr(m["role"], "value")
                        else str(m["role"])
                    ),
                    "len(msg)": len(m["content"]),
                }
                for m in messages_to_send
            ]

            logger.debug(
                f"[render_user_chat_input] Sending query to FastAPI: "
                f"user_id={user_id}, session_id={current_session_id}, round_id={new_round_id}, rag_mode={rag_mode}, reranker={use_reranker}, messages_summary={messages_summary}"
            )

            # 3) Post to FastAPI (streaming)
            try:
                if dev_cfg.simulate_backend_failure:
                    raise requests.exceptions.RequestException(
                        "Simulated backend failure (post_query_to_fastapi)"
                    )

                response = post_query_to_fastapi(
                    server_url=server_url,
                    user_id=user_id,
                    app_name=app_name,
                    session_id=current_session_id,
                    messages_list=messages_to_send,
                    user_msg_id=user_msg_id,
                    system_msg_id=system_msg_id,
                    assistant_msg_id=assistant_msg_id,
                    round_id=new_round_id,
                    rag_mode=rag_mode,
                    use_reranker=use_reranker,
                )

                # ✨  Handle non-200 status codes early
                if response.status_code == 409:
                    # Another tab beat us; advise user to refresh
                    logger.exception(
                        "[render_user_chat_input] FastAPI returned 409 Conflict — "
                        f"session is stale. user_id={user_id}, session_id={current_session_id}, round_id={new_round_id}"
                    )
                    st.error(ERROR_LABELS.SESSION_EDIT_HTTP_409)
                    st.stop()

                # Raise for any other HTTP error (4xx/5xx)
                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                logger.exception(
                    f"[render_user_chat_input] Failed to post query to FastAPI for user_id={user_id}, session_id={current_session_id}, round_id={new_round_id}"
                )
                st.session_state["chat_error_message"] = (
                    ERROR_LABELS.MESSAGE_SUBMISSION_UNEXPECTED_ERROR
                )
                return

            logger.debug(
                f"[render_user_chat_input] Streaming response started for user_id={user_id}, session_id={current_session_id}, round_id={new_round_id}"
            )

            # 4) Stream partial assistant responses
            # Initialize buffer for streaming
            buf = ""
            partial_message_text = ""
            SSE_DATA_PREFIX = "data: "
            stream_chunk_index = 0  # For dev testing, to inject errors
            system_context_rows = None

            with st.chat_message("assistant"):
                placeholder = st.empty()

                for chunk in response.iter_content(decode_unicode=True):
                    # Inject errors for dev testing
                    if stream_chunk_index == 0:
                        if dev_cfg.simulate_llm_invalid_json:
                            chunk = "data: {invalid json}\n\n"
                        elif dev_cfg.simulate_llm_stream_error:
                            chunk = 'data: {"data": 123}\n\n'  # JSON is valid but structurally wrong
                        stream_chunk_index += 1

                    buf += chunk

                    while "\n\n" in buf:
                        event, buf = buf.split("\n\n", 1)
                        if not event.startswith(SSE_DATA_PREFIX):
                            continue

                        json_str = event[len(SSE_DATA_PREFIX) :]

                        try:
                            payload = json.loads(json_str)
                            if "data" in payload:
                                data = payload["data"]
                                if not isinstance(data, str):
                                    logger.warning(
                                        f"[render_user_chat_input] Invalid assistant response structure for user_id={user_id}: data is not string ({type(data)})"
                                    )
                                    st.session_state["chat_error_message"] = (
                                        ERROR_LABELS.LLM_RESPONSE_FORMAT_ERROR
                                    )
                                    return
                                if data == "[DONE]":
                                    break
                                partial_message_text += data
                                placeholder.markdown(
                                    partial_message_text, unsafe_allow_html=False
                                )
                            elif "system_context_rows" in payload:
                                system_context_rows = payload["system_context_rows"]
                            elif "error" in payload:
                                logger.warning(
                                    f"[render_user_chat_input] Assistant backend returned error: {payload['error']}"
                                )
                                st.session_state["chat_error_message"] = (
                                    ERROR_LABELS.STREAMING_BACKEND_ERROR
                                )
                                return
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[render_user_chat_input] Invalid JSON in chunk for user_id={user_id}: {json_str}"
                            )
                            st.session_state["chat_error_message"] = (
                                ERROR_LABELS.LLM_RESPONSE_MALFORMED
                            )
                            return

                # Display system context rows outside chat message
                if isinstance(system_context_rows, list):
                    render_system_context_rows(
                        system_context_rows, user_id, current_session_id
                    )
                else:
                    logger.warning(
                        f"[render_user_chat_input] system_context_rows is not a list: type={type(system_context_rows)}, user_id={user_id}, session_id={current_session_id}"
                    )
                    st.caption(WARNING_LABELS.PARSE_SYSTEM_MESSAGE_FAILED)

            # 5) Save final assistant message
            assistant_msg = Message(
                role=RoleEnum.ASSISTANT,
                content=partial_message_text,
                id=assistant_msg_id,
                round_id=new_round_id,
                is_deleted=False,
            )
            messages.append(assistant_msg)

            if isinstance(system_context_rows, list):
                system_msg = Message(
                    role=RoleEnum.SYSTEM,
                    content=json.dumps(
                        system_context_rows, ensure_ascii=False, indent=2
                    ),
                    id=system_msg_id,
                    round_id=new_round_id,
                    is_deleted=False,
                )
                messages.append(system_msg)

            # 6) If it was the very first query, reload all sessions
            if new_round_id == 0:
                # Force re-fetch so the updated title from the backend is shown
                st.session_state["session_ids"] = fetch_session_ids(
                    server_url=server_url, user_id=user_id, app_name=app_name
                )
                # Update current_session to include the new title
                for sess in st.session_state["session_ids"]:
                    if sess.session_id == current_session_id:
                        st.session_state["current_session"] = sess
                        break

            # 7) Refresh sidebar ordering locally
            now = datetime.now(timezone.utc)
            for s in st.session_state["session_ids"]:
                if s.session_id == current_session_id:
                    s.last_touched_at = now
                    break
            st.session_state["session_ids"].sort(key=lambda x: x.last_touched_at)
        except Exception:
            logger.exception(
                f"[render_user_chat_input] Unexpected error in main block for user_id={user_id}, session_id={current_session_id}"
            )
            st.session_state["chat_error_message"] = (
                ERROR_LABELS.GENERIC_UNEXPECTED_ERROR
            )
        finally:
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()


#################################
# Streamlit App
#################################


def main(user_id: str, employee_class_id: str) -> None:
    """
    Launches the main Streamlit interface for the multi-session RAG+LLM application.

    This function initializes UI components and session state, renders chat history,
    and handles user interaction including session creation, editing, and feedback.
    The `user_id` must be authenticated beforehand, either via SAML or a development override.

    Args:
        user_id (str): Unique identifier of the authenticated user.
            In production, this is obtained from a SAML cookie.
            In development, this may be passed via an environment variable.
        employee_class_id (str): The employee class ID of the user, used for access control.
    """

    st.markdown(
        """
        <meta http-equiv="Content-Language" content="ja">
        <script>
        document.documentElement.lang = 'ja';
        </script>
        """,
        unsafe_allow_html=True,
    )

    try:
        hide_streamlit_deploy_button()
        hide_streamlit_menu()
        inject_header_css(app_title=LABELS.APP_TITLE)
        check_access_control(user_id=user_id, employee_class_id=employee_class_id)
        disabled_ui = setup_ui_locking()
        initialize_session_state(user_id=user_id)
        if USE_INACTIVITY_REDIRECT:
            setup_inactivity_redirect(
                user_id=user_id,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                target_url=DEFAULT_REDIRECT_URL,
                interval_ms=DEFAULT_INTERVAL_MS,
            )
    except Exception:
        logger.exception("[Main] Unexpected error during app setup.")
        st.error(ERROR_LABELS.APP_INITIALIZATION_FAILED)
        st.stop()

    # Step 1.5: Show global error messages at the top of sidebar
    show_sidebar_error_message(user_id=user_id)

    # Step 2: Fetch list of sessions
    load_session_ids(server_url=SERVER_URL, user_id=user_id, app_name=APP_NAME)

    # Step 2.5.1 ── Sync DevTestConfig from widget state
    # (Runs only when the special debug session "__DEBUG_MODE__" exists.)
    sync_dev_test_config()

    # Step 2.5.2 ── Render the Developer Settings panel
    # (Visible only in "__DEBUG_MODE__".)
    render_dev_test_settings()

    # Step 3: Sidebar creation form
    render_create_session_form(
        server_url=SERVER_URL,
        user_id=user_id,
        app_name=APP_NAME,
        disabled_ui=disabled_ui,
    )

    # Step 4: Sidebar session list
    render_session_list(
        user_id=user_id,
        app_name=APP_NAME,
        server_url=SERVER_URL,
        disabled_ui=disabled_ui,
    )

    # Step 5: Sidebar edit/delete form
    render_edit_session_form(
        user_id=user_id,
        app_name=APP_NAME,
        server_url=SERVER_URL,
        disabled_ui=disabled_ui,
    )

    # Step 6: Display conversation messages
    current_session_id, messages = get_current_session_messages(user_id=user_id)

    render_load_more_button(
        session_id=current_session_id,
        server_url=SERVER_URL,
        user_id=user_id,
        app_name=APP_NAME,
        messages=messages,
        disabled_ui=disabled_ui,
    )

    render_chat_messages(
        messages=messages,
        server_url=SERVER_URL,
        current_session_id=current_session_id,
        user_id=user_id,
        disabled_ui=disabled_ui,
    )

    # Step 6.5: Show any chat error messages
    show_chat_error_message(user_id=user_id)

    # Step 7: Handle new user input
    render_user_chat_input(
        messages=messages,
        server_url=SERVER_URL,
        user_id=user_id,
        app_name=APP_NAME,
        current_session_id=current_session_id,
        num_of_prev_msg_with_llm=NUM_OF_PREV_MSG_WITH_LLM,
        disabled_ui=disabled_ui,
    )


if __name__ == "__main__":
    use_saml = os.getenv("USE_SAML", "true").lower() == "true"

    if use_saml:
        if (
            "session_token" not in st.context.cookies
            or "session_data" not in st.context.cookies
        ):
            from ragpon.apps.streamlit.common.common_saml import login

            login()
        else:
            attribute_json = json.loads(
                urllib.parse.unquote(st.context.cookies["session_data"])
            )
            user_id = attribute_json["employeeNumber"][0]
            employee_class_id = attribute_json["employee_class_id"][0]
            main(user_id, employee_class_id)
    else:
        user_id = os.getenv("DEV_USER_ID", "test_user5")
        employee_class_id = os.getenv("DEV_EMPLOYEE_CLASS_ID", "test_user5")
        main(user_id, employee_class_id)
