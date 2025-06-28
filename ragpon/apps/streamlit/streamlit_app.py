import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice

import requests
import streamlit as st

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.chat import Message, SessionData


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


# Initialize logger
logger = get_library_logger(__name__)

# logging settings for debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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
    logger.info(
        f"[fetch_session_ids] Fetching sessions for user={user_id}, app={app_name}"
    )
    logger.debug(f"[fetch_session_ids] GET {endpoint}")
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.exception(
            f"[fetch_session_ids] Failed to fetch sessions from {endpoint}"
        )
        raise

    # Expecting a JSON array of objects
    data = response.json()
    logger.debug(f"[fetch_session_ids] Received response: {data}")

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
    logger.info(f"[fetch_session_ids] Loaded {len(session_list)} sessions")
    return session_list


def fetch_session_history(
    server_url: str, user_id: str, app_name: str, session_id: str
) -> list[Message]:
    """
    Fetches the conversation history for a given session from FastAPI.

    Args:
        server_url (str): The URL of the backend server (e.g. "http://localhost:8006").
        user_id (str): The ID of the user who owns the session.
        app_name (str): The name of the application.
        session_id (str): The unique identifier of the session.

    Returns:
        list[Message]: A list of Message objects representing the conversation history.
    """
    endpoint = (
        f"{server_url}/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries"
    )
    response = requests.get(endpoint)
    response.raise_for_status()
    data = response.json()  # This should be a list of dicts

    # Convert each dict to your local Message domain model
    messages: list[Message] = []
    for item in data:
        msg = Message(
            role=item["role"],
            content=item["content"],
            id=item["id"],
            round_id=item["round_id"],
            is_deleted=item["is_deleted"],
        )
        messages.append(msg)

    return messages


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
    rag_mode: str,
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
        rag_mode (str): How to use RAG. One of:
            "RAG (Optimized Query)", "RAG (Standard)", or "No RAG".
            Defaults to "RAG (Optimized Query)".
        use_reranker (bool): Whether to use the reranker.

    Returns:
        requests.Response: A streaming Response object from the server.

    Raises:
        requests.RequestException: If the HTTP request fails.
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
        "rag_mode": rag_mode,
        "use_reranker": use_reranker,
    }

    logger.info(
        f"[post_query_to_fastapi] Posting query for user={user_id}, session={session_id}, round={round_id}"
    )
    logger.debug(f"[post_query_to_fastapi] POST {endpoint}")
    logger.debug(f"[post_query_to_fastapi] Payload: {payload}")

    try:
        response = requests.post(endpoint, json=payload, stream=True)
        response.raise_for_status()
        logger.info("[post_query_to_fastapi] POST succeeded")
        return response
    except requests.RequestException as e:
        logger.exception("[post_query_to_fastapi] POST failed")
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


def patch_session_info(
    server_url: str,
    user_id: str,
    session_id: str,
    session_name: str,
    is_private_session: bool,
    is_deleted: bool = False,
) -> None:
    """
    Sends a PATCH request to update session info in the backend.

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID.
        session_id (str): The session ID to be updated.
        session_name (str): The new session name.
        is_private_session (bool): The new is_private_session value.
        is_deleted (bool): Whether the session is being marked as deleted.

    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    endpoint = f"{server_url}/users/{user_id}/sessions/{session_id}"
    payload = {
        "session_name": session_name,
        "is_private_session": is_private_session,
        "is_deleted": is_deleted,
    }

    # Perform a real PATCH request
    response = requests.patch(endpoint, json=payload)
    response.raise_for_status()  # Raise an error if the request was unsuccessful


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


def is_debug_session_active() -> bool:
    sessions = st.session_state.get("session_ids", [])
    return any(s.session_name == DEBUG_SESSION_TRIGGER for s in sessions)


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


def show_sidebar_error_message() -> None:
    """
    Displays any error message stored in st.session_state['error_message'] in the sidebar,
    and logs the message using logger.warning. After displaying, the message is deleted.
    """
    error_msg = st.session_state.pop("error_message", None)
    if error_msg:
        logger.warning(f"[UI] Showing deferred error message: {error_msg}")
        st.sidebar.error(error_msg)


def show_chat_error_message() -> None:
    """
    Displays any error message stored in st.session_state['chat_error_message']
    in the assistant's chat area, and logs the message using logger.warning.
    After displaying, the message is deleted.
    """
    error_msg = st.session_state.pop("chat_error_message", None)
    if error_msg:
        logger.warning(f"[UI][Chat] Showing chat error message: {error_msg}")
        # with st.chat_message("assistant"):
        st.error(error_msg)


def load_session_ids(server_url: str, user_id: str, app_name: str) -> list[SessionData]:
    """
    Loads the list of sessions from the server and returns them as a list of SessionData.
    This was originally inline in the main() function, but refactored for clarity.
    """
    # Ensure that session_ids is initialized in the session state.
    if "session_ids" not in st.session_state:
        st.session_state["session_ids"] = fetch_session_ids(
            server_url, user_id, app_name
        )
    return st.session_state["session_ids"]


def render_create_session_form(
    server_url: str, user_id: str, app_name: str, disabled_ui: bool
) -> None:
    """
    Renders the UI for creating a new session in the sidebar, and handles the
    logic of creating the session via put_session_info(). Disables interactive UI elements
    during ongoing operations when disabled_ui is True.

    Args:
        server_url (str): The base URL for the server.
        user_id (str): The user ID for whom the session is being created.
        app_name (str): The application name under which the session is managed.
        disabled_ui (bool): If True, disables user interactions for all input widgets during ongoing operations.
    """

    MAX_SESSION_COUNT = 10
    current_session_count = len(st.session_state.get("session_ids", []))

    if current_session_count >= MAX_SESSION_COUNT:
        st.sidebar.write(
            f"🚫 Session limit reached ({MAX_SESSION_COUNT}). Delete existing sessions to create new ones."
        )
        return

    with st.sidebar.expander(
        "🆕 Create New Session", expanded=st.session_state["show_create_form"]
    ):
        create_label = (
            "🆕Cancel Create Session"
            if st.session_state["show_create_form"]
            else "🆕Create New Session"
        )
        if st.button(create_label, key="toggle_create_button", disabled=disabled_ui):
            st.session_state["show_create_form"] = not st.session_state[
                "show_create_form"
            ]
            st.rerun()

        if st.session_state["show_create_form"]:
            new_session_name: str = st.text_input(
                "📛Session Name",
                value="Untitled Session",
                max_chars=30,
                key="create_session_name",
                disabled=disabled_ui,
            )
            new_session_is_private: bool = st.radio(
                "🙈Is Private?",
                options=[True, False],
                key="create_is_private",
                disabled=disabled_ui,
            )

            if st.button("Create", key="finalize_create_button", disabled=disabled_ui):
                # Lock the UI and schedule creation
                st.session_state["is_ui_locked"] = True
                st.session_state["ui_lock_reason"] = "Creating new session..."
                st.session_state["pending_create_session"] = {
                    "name": new_session_name,
                    "is_private": new_session_is_private,
                }
                st.rerun()

    # --- Handle deferred session creation ---
    if "pending_create_session" in st.session_state:
        data = st.session_state.pop("pending_create_session")

        try:
            logger.info(
                f"[render_create_session_form] User '{user_id}' requested session creation: "
                f"name='{data['name']}', private={data['is_private']}"
            )

            # --- Developer test hooks ---
            dev_cfg: DevTestConfig = st.session_state.get(
                "dev_test_config", DevTestConfig()
            )

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

            # Real logic
            new_session_id: str = str(uuid.uuid4())
            put_session_info(
                server_url=server_url,
                user_id=user_id,
                app_name=app_name,
                session_id=new_session_id,
                session_name=data["name"],
                is_private_session=data["is_private"],
            )
            # Update session list and current session
            new_session = SessionData(
                session_id=new_session_id,
                session_name=data["name"],
                is_private_session=data["is_private"],
                last_touched_at=datetime.now(timezone.utc),
            )
            st.session_state["session_ids"].append(new_session)
            st.session_state["current_session"] = new_session
            st.session_state["show_create_form"] = False

            logger.info(
                f"[render_create_session_form] Session created successfully: "
                f"id={new_session_id}, name='{data['name']}'"
            )

        except requests.exceptions.RequestException as exc:
            logger.exception(
                f"[render_create_session_form] Failed to create session '{data['name']}'"
            )
            st.session_state["error_message"] = f"Failed to create a new session: {exc}"

        except Exception:
            logger.exception(
                "[render_create_session_form] Unexpected error during session creation"
            )
            st.session_state["error_message"] = (
                "An unexpected error occurred during session creation."
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

    st.sidebar.write("## 👉Session List")

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
                put_session_info(
                    server_url=server_url,
                    user_id=user_id,
                    app_name=app_name,
                    session_id=new_session_id,
                    session_name=default_session_name,
                    is_private_session=is_private,
                )
            except requests.exceptions.RequestException as exc:
                st.sidebar.error(f"Failed to register default session to server: {exc}")
                st.stop()
            except Exception as e:
                # Catch simulated or unexpected errors like RuntimeError
                st.sidebar.error(f"Auto session creation failed: {e}")
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
    current_session = st.session_state["current_session"]
    for idx, session in enumerate(sorted_sessions):
        if session.session_id == current_session.session_id:
            current_idx = idx
            break
    # 3) Render radio with explicit index
    # 3-1) Sort sessions newest-first
    sorted_sessions: list[SessionData] = sorted(
        st.session_state["session_ids"],
        key=lambda x: x.last_touched_at,
        reverse=True,
    )
    # 3-2) Determine default index to match current_session
    current_idx: int = 0
    current_session = st.session_state["current_session"]
    for idx, session in enumerate(sorted_sessions):
        if session.session_id == current_session.session_id:
            current_idx = idx
            break

    # 3-3) Define callback to update current_session when selection changes
    def _on_session_change() -> None:
        """Update the current_session when the user selects a different session."""
        st.session_state["current_session"] = st.session_state["unique_session_radio"]

    # 3-4) Render radio with explicit index and callback
    selected_session_data: SessionData = st.sidebar.radio(
        "Choose a session:",
        options=sorted_sessions,
        index=current_idx,
        format_func=lambda x: (
            f"{x.session_name} (Private)" if x.is_private_session else x.session_name
        ),
        key="unique_session_radio",
        on_change=_on_session_change,
        disabled=disabled_ui,
    )

    # Update the current session in session_state
    st.session_state["current_session"] = selected_session_data
    selected_session_id: str = selected_session_data.session_id

    # If we haven't loaded this session before, fetch from the server
    if selected_session_id not in st.session_state["session_histories"]:
        st.session_state["session_histories"][selected_session_id] = (
            fetch_session_history(
                server_url=server_url,
                user_id=user_id,
                app_name=app_name,
                session_id=selected_session_id,
            )
        )
    return selected_session_data


def render_edit_session_form(user_id: str, server_url: str, disabled_ui: bool) -> None:
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

    # Edit Session button (single button for both edit and delete)
    with st.sidebar.expander(
        "✏️Manage Selected Session",
        expanded=st.session_state.get("show_edit_form", False),
    ):
        edit_label = (
            "✏️Cancel Edit"
            if st.session_state.get("show_edit_form", False)
            else "✏️Edit Session"
        )

        toggle_edit_button: bool = st.button(
            edit_label, key="toggle_edit_button", disabled=disabled_ui
        )

        if toggle_edit_button:
            st.session_state["show_edit_form"] = not st.session_state.get(
                "show_edit_form", False
            )
            st.rerun()

        # Show or hide the edit form
        if st.session_state.get("show_edit_form", False):

            current_session = st.session_state["current_session"]
            if current_session is None:
                st.warning("No session is currently selected.")
                return

            current_name: str = current_session.session_name
            current_is_private: bool = current_session.is_private_session

            edited_session_name: str = st.text_input(
                "📛Session Name",
                value=current_name,
                max_chars=30,
                key="edit_session_name",
                disabled=disabled_ui,
            )
            edited_is_private: bool = st.radio(
                "🙈Is Private?",
                options=[True, False],
                index=0 if current_is_private else 1,
                key="edit_is_private",
                disabled=disabled_ui,
            )

            delete_this_session: bool = st.checkbox("🗑️", key="delete_session")

            if st.button("Update", key="update_session", disabled=disabled_ui):
                # Lock UI and store intent
                st.session_state["is_ui_locked"] = True
                st.session_state["ui_lock_reason"] = (
                    "Updating or deleting the session..."
                )
                st.session_state["pending_edit_action"] = {
                    "delete": delete_this_session,
                    "session_id": current_session.session_id,
                    "new_name": edited_session_name,
                    "new_privacy": edited_is_private,
                }
                st.rerun()

    # --- Perform the pending edit action after rerun ---
    if "pending_edit_action" in st.session_state:
        action = st.session_state.pop("pending_edit_action")
        try:
            logger.info(
                f"[render_edit_session_form] User '{user_id}' requested to "
                f"{'DELETE' if action['delete'] else 'UPDATE'} session '{action['session_id']}' "
                f"with name='{action['new_name']}', private={action['new_privacy']}"
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

            if action["delete"]:
                patch_session_info(
                    server_url=server_url,
                    user_id=user_id,
                    session_id=action["session_id"],
                    session_name=action["new_name"],
                    is_private_session=action["new_privacy"],
                    is_deleted=True,
                )
                logger.info(
                    f"[render_edit_session_form] Session '{action['session_id']}' deleted successfully"
                )
                st.session_state["session_ids"] = [
                    s
                    for s in st.session_state["session_ids"]
                    if s.session_id != action["session_id"]
                ]
                if (
                    st.session_state["current_session"].session_id
                    == action["session_id"]
                ):
                    st.session_state["current_session"] = None
            else:
                patch_session_info(
                    server_url=server_url,
                    user_id=user_id,
                    session_id=action["session_id"],
                    session_name=action["new_name"],
                    is_private_session=action["new_privacy"],
                    is_deleted=False,
                )
                logger.info(
                    f"[render_edit_session_form] Session '{action['session_id']}' updated successfully "
                    f"with name='{action['new_name']}', private={action['new_privacy']}"
                )
                for s in st.session_state["session_ids"]:
                    if s.session_id == action["session_id"]:
                        s.session_name = action["new_name"]
                        s.is_private_session = action["new_privacy"]
                        break
        except requests.RequestException:
            logger.exception(
                f"[render_edit_session_form] Failed to modify session '{action['session_id']}'"
            )
            st.session_state["error_message"] = (
                f"Failed to modify session: {action['session_id']}"
            )
        except Exception as e:
            logger.exception(
                f"[render_edit_session_form] Unexpected error during session modification for session '{action['session_id']}'"
            )
            st.session_state["error_message"] = (
                f"Unexpected error occurred while modifying session: {e}"
            )
        finally:
            st.session_state["show_edit_form"] = False
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()


def render_chat_messages(
    messages: list[Message],
    server_url: str,
    session_id_for_display: str,
    user_id: str,
    disabled_ui: bool,
) -> None:
    """
    Renders the existing conversation messages, including
    Trash/Good/Bad buttons for assistant messages.

    Args:
        messages (list[Message]): The list of messages in the conversation.
        server_url (str): The base URL of the FastAPI server.
        session_id_for_display (str): The current session's ID.
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
    logger.info(
        f"[render_chat_messages] Rendering chat for session_id={session_id_for_display} with {len(messages)} messages"
    )

    dev_cfg: DevTestConfig = st.session_state.get("dev_test_config", DevTestConfig())

    for msg in messages:
        if msg.is_deleted:
            continue

        with st.chat_message(msg.role):
            st.write(msg.content)

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
                logger.info(
                    f"[render_chat_messages] Trash button clicked for round_id={msg.round_id} by user={user_id}"
                )
                st.session_state["confirm_delete_round_id"] = msg.round_id

            # If delete was requested, show confirmation prompt
            if st.session_state.get("confirm_delete_round_id") == msg.round_id:
                with st.expander("⚠️ Confirm Deletion", expanded=True):
                    st.warning("Are you sure you want to delete this round?")
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button(
                            "✅ Yes, delete", key=f"confirm_yes_{msg.round_id}"
                        ):
                            logger.info(
                                f"[render_chat_messages] Confirmed deletion for round_id={msg.round_id} by user={user_id}"
                            )
                            st.session_state["is_ui_locked"] = True
                            st.session_state["ui_lock_reason"] = (
                                "Deleting assistant response..."
                            )
                            st.session_state["pending_delete_round_id"] = msg.round_id
                            st.session_state["pending_delete_user_id"] = user_id
                            st.session_state["confirm_delete_round_id"] = None
                            st.rerun()
                    with col_cancel:
                        if st.button("❌ Cancel", key=f"confirm_no_{msg.round_id}"):
                            st.session_state["confirm_delete_round_id"] = None
                            st.rerun()

            # Good button
            if col_good.button("😊", key=f"good_{msg.id}", disabled=disabled_ui):
                logger.info(
                    f"[render_chat_messages] GOOD feedback for message_id={msg.id} by user={user_id}"
                )
                st.session_state["feedback_form_id"] = msg.id
                st.session_state["feedback_form_type"] = "good"
                st.rerun()

            # Bad button
            if col_bad.button("😞", key=f"bad_{msg.id}", disabled=disabled_ui):
                logger.info(
                    f"[render_chat_messages] BAD feedback for message_id={msg.id} by user={user_id}"
                )
                st.session_state["feedback_form_id"] = msg.id
                st.session_state["feedback_form_type"] = "bad"
                st.rerun()

            # Inline feedback form if active
            if st.session_state.get("feedback_form_id") == msg.id:
                with st.expander("📝 Provide Feedback", expanded=True):
                    feedback_reason = st.text_area(
                        "Reason (optional)", key="feedback_reason", disabled=disabled_ui
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "✅Submit Feedback",
                            key="submit_feedback",
                            disabled=disabled_ui,
                        ):
                            logger.info(
                                f"[render_chat_messages] Submitting feedback for message_id={msg.id} by user={user_id}"
                                f"type={st.session_state['feedback_form_type']} "
                                f"reason='{feedback_reason}'"
                            )
                            st.session_state["is_ui_locked"] = True
                            st.session_state["ui_lock_reason"] = (
                                "Submitting feedback..."
                            )
                            st.session_state["pending_feedback"] = {
                                "llm_output_id": msg.id,
                                "feedback_type": st.session_state["feedback_form_type"],
                                "reason": feedback_reason,
                            }
                            st.rerun()
                    with col2:
                        if st.button(
                            "❌Cancel Feedback",
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
                session_id=session_id_for_display,
                round_id=round_id,
                deleted_by=deleted_by_user,
            )
            logger.info(
                f"[render_chat_messages] Deleted round_id={round_id} by user={deleted_by_user}"
            )
            for m in messages:
                if m.round_id == round_id:
                    m.is_deleted = True
            st.session_state["session_histories"][session_id_for_display] = messages
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"[render_chat_messages] RequestException while deleting round_id={round_id}: {e}"
            )
            st.session_state["chat_error_message"] = (
                "Failed to delete the message. Please try again."
            )
        except Exception:
            logger.exception(
                f"[render_chat_messages] Unexpected error while deleting round_id={round_id}"
            )
            st.session_state["chat_error_message"] = (
                "An unexpected error occurred during deletion. Please contact support if the issue persists."
            )
        finally:
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()

    # Handle feedback submission
    if "pending_feedback" in st.session_state:
        try:
            pending = st.session_state.pop("pending_feedback")
            logger.info(
                f"[render_chat_messages] Feedback submitted: message_id={pending['llm_output_id']} by user={user_id}"
                f"type={pending['feedback_type']} reason='{pending['reason']}'"
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
                session_id=session_id_for_display,
            )
            st.session_state["feedback_form_id"] = None
            st.session_state["feedback_form_type"] = None
        except requests.exceptions.RequestException as e:
            logger.exception(
                f"[render_chat_messages] RequestException while submitting feedback for message_id={pending['llm_output_id']} by user={user_id}"
            )
            st.session_state["chat_error_message"] = (
                "Failed to submit feedback. Please try again."
            )
            st.session_state["feedback_form_id"] = None
            st.session_state["feedback_form_type"] = None
        except Exception:
            logger.exception(
                f"[render_chat_messages] Unexpected error while submitting feedback for message_id={pending['llm_output_id']} by user={user_id}"
            )
            st.session_state["chat_error_message"] = (
                "An unexpected error occurred during feedback submission."
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
    session_id_for_display: str,
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
        session_id_for_display (str): The session ID for which messages are posted.
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
    st.sidebar.write("## 🔍RAG Mode")
    rag_mode: str = st.sidebar.radio(
        "Choose RAG mode:",
        options=["RAG (Optimized Query)", "RAG (Standard)", "No RAG"],
        index=0,  # default to the first option
        key="rag_mode_radio",
        disabled=disabled_ui,
    )

    st.sidebar.write("## 🔀Use Reranker")
    use_reranker: bool = st.sidebar.radio(
        label="Choose whether to use the Reranker:",
        options=[False],  # Only one option for now
        format_func=lambda x: "Yes" if x else "No",
        index=0,  # Default to "No"
        key="use_reranker_radio",
        disabled=disabled_ui,
    )

    # 2) Provide a chat input box for the user to type their query.
    user_input: str = st.chat_input("Type your query here...", disabled=disabled_ui)

    if user_input:
        st.session_state["is_ui_locked"] = True
        st.session_state["ui_lock_reason"] = "Sending message to assistant..."
        st.session_state["pending_user_input"] = user_input
        st.rerun()

    if st.session_state.get("pending_user_input"):
        try:
            dev_cfg: DevTestConfig = st.session_state.get(
                "dev_test_config", DevTestConfig()
            )

            user_input = st.session_state.pop("pending_user_input")
            logger.info(
                f"[render_user_chat_input] User '{user_id}' submitted query: {user_input}"
            )

            if dev_cfg.simulate_unexpected_exception:
                raise RuntimeError(
                    "[DevTest] Simulated unexpected exception in user chat input"
                )

            # 2a) Build the short array of recent non-deleted messages
            last_msgs = last_n_non_deleted(messages, num_of_prev_msg_with_llm)
            messages_to_send = [
                {"role": m.role, "content": m.content} for m in last_msgs
            ]

            # 2b) Compute the next round_id
            if len(messages) > 0:
                last_round_id = max(m.round_id for m in messages)
                new_round_id: int = last_round_id + 1
            else:
                new_round_id: int = 0

            # 2c) Generate UUIDs for user/system/assistant messages
            user_msg_id = str(uuid.uuid4())
            system_msg_id = str(uuid.uuid4())
            assistant_msg_id = str(uuid.uuid4())

            # 2d) Add the user's message locally
            user_msg = Message(
                role="user",
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

            logger.info(
                f"[render_user_chat_input] Sending query to FastAPI: "
                f"session_id={session_id_for_display}, round_id={new_round_id}, rag_mode={rag_mode}, reranker={use_reranker}"
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
                    session_id=session_id_for_display,
                    messages_list=messages_to_send,
                    user_msg_id=user_msg_id,
                    system_msg_id=system_msg_id,
                    assistant_msg_id=assistant_msg_id,
                    round_id=new_round_id,
                    rag_mode=rag_mode,
                    use_reranker=use_reranker,
                )
            except requests.exceptions.RequestException as e:
                logger.exception(
                    "[render_user_chat_input] Failed to post query to FastAPI"
                )
                st.session_state["chat_error_message"] = (
                    f"Sorry, something went wrong while sending your message. Please try again."
                )
                return

            logger.info("[render_user_chat_input] Streaming response started")

            # 4) Stream partial assistant responses
            # Initialize buffer for streaming
            buf = ""
            partial_message_text = ""
            SSE_DATA_PREFIX = "data: "
            stream_chunk_index = 0  # For dev testing, to inject errors
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
                    logger.debug(f"Chunk: {chunk}")

                    while "\n\n" in buf:
                        event, buf = buf.split("\n\n", 1)
                        if not event.startswith(SSE_DATA_PREFIX):
                            continue

                        json_str = event[len(SSE_DATA_PREFIX) :]
                        try:
                            payload = json.loads(event[len(SSE_DATA_PREFIX) :])
                            data = payload.get("data")
                            if not isinstance(data, str):
                                logger.warning(
                                    f"[render_user_chat_input] Invalid assistant response structure: data is not string ({type(data)})"
                                )
                                st.session_state["chat_error_message"] = (
                                    "The assistant response could not be processed due to unexpected format."
                                )
                                return
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[render_user_chat_input] Invalid JSON in chunk: {json_str}"
                            )
                            st.session_state["chat_error_message"] = (
                                "The assistant response was malformed and could not be processed."
                            )
                            return
                        if data == "[DONE]":
                            break

                        partial_message_text += data
                        placeholder.markdown(
                            partial_message_text, unsafe_allow_html=False
                        )

            # 5) Save final assistant message
            assistant_msg = Message(
                role="assistant",
                content=partial_message_text,
                id=assistant_msg_id,
                round_id=new_round_id,
                is_deleted=False,
            )
            messages.append(assistant_msg)

            # 6) If it was the very first query, reload all sessions
            if new_round_id == 0:
                # Force re-fetch so the updated title from the backend is shown
                st.session_state["session_ids"] = fetch_session_ids(
                    server_url, user_id, app_name
                )
                # Update current_session to include the new title
                for sess in st.session_state["session_ids"]:
                    if sess.session_id == session_id_for_display:
                        st.session_state["current_session"] = sess
                        break

            # 7) Refresh sidebar ordering locally
            now = datetime.now(timezone.utc)
            for s in st.session_state["session_ids"]:
                if s.session_id == session_id_for_display:
                    s.last_touched_at = now
                    break
            st.session_state["session_ids"].sort(key=lambda x: x.last_touched_at)
        except Exception as e:
            logger.exception("[render_user_chat_input] Unexpected error in main block")
            st.session_state["chat_error_message"] = (
                "An unexpected error occurred. Please try again."
            )
        finally:
            st.session_state["is_ui_locked"] = False
            st.session_state["ui_lock_reason"] = ""
            st.rerun()


#################################
# Streamlit App
#################################


def main() -> None:
    """
    Main Streamlit application for demonstrating multi-session RAG+LLM
    with an ability to delete (is_deleted) a round via a trash button.
    """
    st.title("RAG + LLM Streamlit App")

    # Show global error messages at the top of sidebar
    show_sidebar_error_message()

    # Initialize session state variables for UI locking
    st.session_state.setdefault("is_ui_locked", False)
    st.session_state.setdefault("ui_lock_reason", "")

    disabled_ui = st.session_state["is_ui_locked"]

    if disabled_ui:
        st.sidebar.warning(f"⏳ {st.session_state['ui_lock_reason']}")

    # Step 1: Initialize session state
    if "current_session" not in st.session_state:
        st.session_state["current_session"] = None
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "test_user3"  # Mock user
    if "show_edit_form" not in st.session_state:
        st.session_state["show_edit_form"] = False
    if "session_histories" not in st.session_state:
        st.session_state["session_histories"] = {}  # { session_id: [messages], ... }
    if "show_create_form" not in st.session_state:
        st.session_state["show_create_form"] = False

    user_id: str = st.session_state["user_id"]
    app_name: str = "search_regulations"
    server_url: str = "http://ragpon-fastapi:8006"  # fixed server URL
    num_of_prev_msg_with_llm: int = (
        6  # Number of messages to keep in the chat with the assistant
    )

    # Step 2: Fetch list of sessions
    load_session_ids(server_url=server_url, user_id=user_id, app_name=app_name)

    # Step 2.5.1 ── Sync DevTestConfig from widget state
    # (Runs only when the special debug session "__DEBUG_MODE__" exists.)
    sync_dev_test_config()

    # Step 2.5.2 ── Render the Developer Settings panel
    # (Visible only in "__DEBUG_MODE__".)
    render_dev_test_settings()

    # Step 3: Sidebar creation form
    render_create_session_form(
        server_url=server_url,
        user_id=user_id,
        app_name=app_name,
        disabled_ui=disabled_ui,
    )

    # Step 4: Sidebar session list
    render_session_list(
        user_id=user_id,
        app_name=app_name,
        server_url=server_url,
        disabled_ui=disabled_ui,
    )

    # Step 5: Sidebar edit/delete form
    render_edit_session_form(
        user_id=user_id, server_url=server_url, disabled_ui=disabled_ui
    )

    # Step 6: Display conversation messages
    session_id_for_display = st.session_state["current_session"].session_id
    messages: list[Message] = st.session_state["session_histories"][
        session_id_for_display
    ]
    messages.sort(key=lambda m: (m.round_id, 0 if m.role == "user" else 1))

    render_chat_messages(
        messages=messages,
        server_url=server_url,
        session_id_for_display=session_id_for_display,
        user_id=user_id,
        disabled_ui=disabled_ui,
    )

    # Step 6.5: Show any chat error messages
    show_chat_error_message()

    # Step 7: Handle new user input
    render_user_chat_input(
        messages=messages,
        server_url=server_url,
        user_id=user_id,
        app_name=app_name,
        session_id_for_display=session_id_for_display,
        num_of_prev_msg_with_llm=num_of_prev_msg_with_llm,
        disabled_ui=disabled_ui,
    )


if __name__ == "__main__":
    main()
