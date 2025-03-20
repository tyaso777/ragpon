import uuid
from itertools import islice

import requests
import streamlit as st

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.chat import Message, SessionData

# Initialize logger
logger = get_library_logger(__name__)


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
# Mock or Real API Calls
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
    """
    endpoint = f"{server_url}/users/{user_id}/apps/{app_name}/sessions"
    response = requests.get(endpoint)
    response.raise_for_status()

    # Expecting a JSON array of objects
    data = response.json()

    # Convert each JSON object into a SessionData instance
    session_list = []
    for item in data:
        session_list.append(
            SessionData(
                session_id=item["session_id"],
                session_name=item["session_name"],
                is_private_session=item["is_private_session"],
            )
        )

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

    Returns:
        requests.Response: A streaming Response object from the server.
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
    }

    response = requests.post(endpoint, json=payload, stream=True)
    response.raise_for_status()
    return response


def put_session_info(
    server_url: str,
    user_id: str,
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
    endpoint = f"{server_url}/users/{user_id}/sessions/{session_id}"
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
    server_url: str, llm_output_id: str, feedback: str, reason: str
) -> None:
    """
    Calls the FastAPI endpoint to patch feedback for a given LLM output ID.

    Args:
        server_url (str): The backend server URL.
        llm_output_id (str): The ID of the LLM output to patch feedback on.
        feedback (str): "good" or "bad" feedback type.
        reason (str): The user's explanation or comment for the feedback.
    """
    endpoint = f"{server_url}/llm_outputs/{llm_output_id}"
    payload = {"feedback": feedback, "reason": reason}
    # Make a PATCH request with a JSON body
    response = requests.patch(endpoint, json=payload)
    response.raise_for_status()


#################################
# Utility / Helper Functions
#################################


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


def show_create_session_form(server_url: str, user_id: str, app_name: str) -> None:
    """
    Renders the UI for creating a new session in the sidebar, and handles the
    logic of creating the session via put_session_info().

    Args:
        server_url (str): The base URL for the server.
        user_id (str): The user ID for whom the session is being created.
        app_name (str): The application name under which the session is managed.
    """
    st.sidebar.write("## Create New Session")
    # Decide the toggle label
    if st.session_state["show_create_form"]:
        create_label = "Cancel Create Session"
    else:
        create_label = "Create New Session"
    # Toggle button for showing/hiding the create session form
    toggle_create_button: bool = st.sidebar.button(
        create_label, key="toggle_create_button"
    )
    if toggle_create_button:
        st.session_state["show_create_form"] = not st.session_state["show_create_form"]
        st.rerun()

    # If the user has toggled the create form on, display it
    if st.session_state["show_create_form"]:
        # Input fields for new session
        new_session_name: str = st.sidebar.text_input(
            "Session Name", value="No title", key="create_session_name"
        )
        new_session_is_private: bool = st.sidebar.radio(
            "Is Private?", options=[True, False], key="create_is_private"
        )

        # Button to finalize session creation
        if st.sidebar.button("Create", key="finalize_create_button"):
            # Generate a new session ID
            new_session_id: str = str(uuid.uuid4())
            try:
                put_session_info(
                    server_url=server_url,
                    user_id=user_id,
                    session_id=new_session_id,
                    session_name=new_session_name,
                    is_private_session=new_session_is_private,
                )

                # Update local session state
                st.session_state["session_ids"].append(
                    SessionData(
                        session_id=new_session_id,
                        session_name=new_session_name,
                        is_private_session=new_session_is_private,
                    )
                )

                # Switch to the newly created session
                st.session_state["current_session"] = SessionData(
                    session_id=new_session_id,
                    session_name=new_session_name,
                    is_private_session=new_session_is_private,
                )

                # Hide the create form
                st.session_state["show_create_form"] = False

                # Rerun to refresh the UI
                st.rerun()

            except requests.exceptions.RequestException as exc:
                st.error(f"Failed to create a new session: {exc}")


#################################
# Streamlit App
#################################


def main() -> None:
    """
    Main Streamlit application for demonstrating multi-session RAG+LLM
    with an ability to delete (is_deleted) a round via a trash button.
    """
    st.title("RAG + LLM with Multiple Sessions (Mock)")

    # 1) Initialize session_state items
    if "current_session" not in st.session_state:
        st.session_state["current_session"] = None
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "test_user"  # Mock user
    if "show_edit_form" not in st.session_state:
        st.session_state["show_edit_form"] = False
    if "session_histories" not in st.session_state:
        st.session_state["session_histories"] = {}  # { session_id: [messages], ... }
    if "show_create_form" not in st.session_state:
        st.session_state["show_create_form"] = False

    user_id: str = st.session_state["user_id"]
    app_name: str = "search_regulations"
    server_url: str = "http://localhost:8006"  # fixed server URL
    num_of_prev_msg_with_llm: int = (
        6  # Number of messages to keep in the chat with the assistant
    )

    # 2) Fetch list of sessions from the server.
    load_session_ids(server_url=server_url, user_id=user_id, app_name=app_name)

    # 3) Sidebar: pick a session or create a new one
    show_create_session_form(server_url=server_url, user_id=user_id, app_name=app_name)

    st.sidebar.write("## Session List")

    # If no session has been chosen yet, default to the first in the list
    if st.session_state["current_session"] is None:
        if len(st.session_state["session_ids"]) == 0:
            # Automatically create a new session if none exist
            new_session_id = str(uuid.uuid4())
            default_session_name = "Default Session"
            is_private = True

            st.session_state["session_ids"].append(
                SessionData(
                    session_id=new_session_id,
                    session_name=default_session_name,
                    is_private_session=is_private,
                )
            )
            st.session_state["current_session"] = SessionData(
                session_id=new_session_id,
                session_name=default_session_name,
                is_private_session=is_private,
            )
            st.rerun()
        else:
            st.session_state["current_session"] = st.session_state["session_ids"][-1]

    # Radio button to choose an existing session
    selected_session_data: SessionData = st.sidebar.radio(
        "Choose a session:",
        st.session_state["session_ids"][::-1],
        format_func=lambda x: (
            f"{x.session_name} (Private)" if x.is_private_session else x.session_name
        ),
        key="unique_session_radio",
    )

    # If the form is not shown, assume we're selecting an existing session
    st.session_state["current_session"] = selected_session_data
    selected_session_id: str = selected_session_data.session_id

    # If we haven't loaded this session before, fetch from the server
    if selected_session_id not in st.session_state["session_histories"]:
        history: list[dict] = fetch_session_history(
            server_url=server_url,
            user_id=user_id,
            app_name=app_name,
            session_id=selected_session_id,
        )
        st.session_state["session_histories"][selected_session_id] = history

    # Now point a local variable to the chosen session's messages
    messages: list[dict] = st.session_state["session_histories"][selected_session_id]

    # Display which session is active
    st.write(f"**Current Session**: {st.session_state['current_session']}")

    # Edit Session button (single button for both edit and delete)
    st.sidebar.write("## Manage Selected Session")

    if st.session_state["show_edit_form"]:
        edit_label = "Cancel Edit"
    else:
        edit_label = "Edit Session"

    toggle_edit_button: bool = st.sidebar.button(edit_label, key="toggle_edit_button")

    if toggle_edit_button:
        st.session_state["show_edit_form"] = not st.session_state["show_edit_form"]
        st.rerun()  # so the UI updates immediately

    if st.session_state["show_edit_form"]:
        # Show a form to edit session name, privacy, or delete
        st.sidebar.write("### Edit or Delete Session")

        current_name: str = selected_session_data.session_name
        current_is_private: bool = selected_session_data.is_private_session

        edited_session_name: str = st.sidebar.text_input(
            "Session Name",
            value=current_name,
            key="edit_session_name",
        )
        edited_is_private: bool = st.sidebar.radio(
            "Is Private?",
            options=[True, False],
            index=0 if current_is_private else 1,
            key="edit_is_private",
        )

        # "Delete this session?" as a checkbox
        delete_this_session: bool = st.sidebar.checkbox(
            "Delete this session?", key="delete_session"
        )

        if st.sidebar.button("Update", key="update_session"):
            if delete_this_session:
                # Perform delete
                patch_session_info(
                    server_url=server_url,
                    user_id=user_id,
                    session_id=selected_session_data.session_id,
                    session_name=edited_session_name,
                    is_private_session=edited_is_private,
                    is_deleted=True,
                )
                # Remove from local list
                st.session_state["session_ids"] = [
                    s
                    for s in st.session_state["session_ids"]
                    if s.session_id != selected_session_data.session_id
                ]
                if (
                    st.session_state["current_session"].session_id
                    == selected_session_data.session_id
                ):
                    st.session_state["current_session"] = None
            else:
                # Perform update (no delete)
                patch_session_info(
                    server_url=server_url,
                    user_id=user_id,
                    session_id=selected_session_data.session_id,
                    session_name=edited_session_name,
                    is_private_session=edited_is_private,
                    is_deleted=False,
                )
                # Update local data
                for s in st.session_state["session_ids"]:
                    if s.session_id == selected_session_data.session_id:
                        s.session_name = edited_session_name
                        s.is_private_session = edited_is_private
                        break

            st.session_state["show_edit_form"] = False
            st.rerun()

    # 4) Show the existing conversation messages
    #    We'll display them in order, but note each message has a "round_id".
    #    If the role is user, we add a trash button to delete that round.
    session_id_for_display = st.session_state["current_session"].session_id

    # We'll track which round_ids we've displayed (user+assistant pairs).
    displayed_round_ids: set[str] = set()

    for msg in messages:
        if msg.is_deleted:
            continue
        # Only create the chat message block once per message
        with st.chat_message(msg.role):
            st.write(msg.content)
        # If this is a user message, display the trash button for that round
        if msg.role == "assistant":
            # To avoid showing multiple trash buttons for the same round,
            # only show it once.
            if msg.round_id not in displayed_round_ids:
                displayed_round_ids.add(msg.round_id)
            # We'll show three buttons: Trash, Good, Bad
            col_trash, col_good, col_bad = st.columns([1, 1, 1])

            # Trash icon button
            if col_trash.button(
                "🗑️", key=f"delete_{msg.round_id}", help="Delete this round"
            ):
                delete_round(
                    server_url=server_url,
                    session_id=session_id_for_display,
                    round_id=msg.round_id,
                    deleted_by=user_id,
                )
                # 2) Remove user+assistant messages with this round_id locally
                for m in messages:
                    if m.round_id == msg.round_id:
                        m.is_deleted = True
                st.session_state["session_histories"][session_id_for_display] = messages
                st.rerun()

            # Good button
            if col_good.button("Good", key=f"good_{msg.id}"):
                st.session_state["feedback_form_id"] = msg.id
                st.session_state["feedback_form_type"] = "good"
                st.rerun()

            # Bad button
            if col_bad.button("Bad", key=f"bad_{msg.id}"):
                st.session_state["feedback_form_id"] = msg.id
                st.session_state["feedback_form_type"] = "bad"
                st.rerun()

    # Check if we have a pending feedback form
    if (
        "feedback_form_id" in st.session_state
        and st.session_state["feedback_form_id"] is not None
    ):
        st.write("### Provide feedback")
        feedback_reason = st.text_area("Reason (optional)", key="feedback_reason")

        if st.button("Submit Feedback", key="submit_feedback"):
            llm_output_id = st.session_state["feedback_form_id"]
            feedback_type = st.session_state["feedback_form_type"]  # "good" or "bad"
            reason_text = feedback_reason

            # 1) Call our mock patch function
            patch_feedback(
                server_url=server_url,
                llm_output_id=llm_output_id,
                feedback=feedback_type,
                reason=reason_text,
            )

            # 2) Reset the form
            st.session_state["feedback_form_id"] = None
            st.session_state["feedback_form_type"] = None
            st.rerun()

        # If you want a Cancel button for feedback:
        if st.button("Cancel Feedback", key="cancel_feedback"):
            st.session_state["feedback_form_id"] = None
            st.session_state["feedback_form_type"] = None
            st.rerun()

    # 5) Chat input at the bottom to continue conversation
    user_input: str = st.chat_input("Type your query here...")
    if user_input:

        # 0) Build the array of dicts for the last num_of_prev_msg_with_llm or all messages
        last_msgs = last_n_non_deleted(messages, num_of_prev_msg_with_llm)
        messages_to_send = [
            {"role": msg.role, "content": msg.content} for msg in last_msgs
        ]

        # 1) Compute the next round_id
        if len(messages) > 0:
            last_round_id = max(msg.round_id for msg in messages)
            new_round_id: str = last_round_id + 1
        else:
            new_round_id: str = 0

        # 2) Generate UUIDs for user/system/assistant messages
        user_msg_id = str(uuid.uuid4())
        system_msg_id = str(uuid.uuid4())
        assistant_msg_id = str(uuid.uuid4())

        # 3) Add the user's message to local state
        user_msg: Message = Message(
            role="user",
            content=user_input,
            id=user_msg_id,
            round_id=new_round_id,
            is_deleted=False,
        )
        messages.append(user_msg)
        messages_to_send.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        # 4) Post to FastAPI (streaming)
        try:
            response: requests.Response = post_query_to_fastapi(
                server_url=server_url,
                user_id=st.session_state["user_id"],
                app_name=app_name,
                session_id=session_id_for_display,
                messages_list=messages_to_send,
                user_msg_id=user_msg_id,
                system_msg_id=system_msg_id,
                assistant_msg_id=assistant_msg_id,
                round_id=new_round_id,
            )
        except requests.exceptions.RequestException as e:
            # If request fails, show error from assistant
            with st.chat_message("assistant"):
                st.error(f"Request failed: {e}")
            return

        # 5) Stream partial assistant responses
        partial_message_text: str = ""
        with st.chat_message("assistant"):
            assistant_msg_placeholder = st.empty()
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    chunk: str = line[len("data: ") :]
                    partial_message_text += chunk
                    assistant_msg_placeholder.write(partial_message_text)

        # 6) Save final assistant message
        assistant_msg: Message = Message(
            role="assistant",
            content=partial_message_text,
            id=assistant_msg_id,
            round_id=new_round_id,
            is_deleted=False,
        )
        messages.append(assistant_msg)
        st.rerun()


if __name__ == "__main__":
    main()
