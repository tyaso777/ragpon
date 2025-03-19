import uuid
from dataclasses import dataclass
from enum import Enum
from itertools import islice

import requests
import streamlit as st
from pydantic import BaseModel, Field

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)


class RoleEnum(str, Enum):
    """Allowed roles for a chat message."""

    user = "user"
    assistant = "assistant"


class SessionData(BaseModel):
    """
    Represents session information, including an ID, name, and privacy setting.
    """

    session_id: str
    session_name: str
    is_private_session: bool


class Message(BaseModel):
    """
    Represents a single chat message with role, content, message ID, and round ID.

    Attributes:
        role (RoleEnum): The role of the sender, either "user" or "assistant".
        content (str): The message content.
        id (str): A unique identifier for the message. (alias: id)
        round_id (int): The round number in the conversation.
        is_deleted (bool): Indicates if the message is deleted. Defaults to False.
    """

    role: RoleEnum = Field(
        ..., description="Role of the sender. Either 'user' or 'assistant'."
    )
    content: str = Field(..., description="The content of the message.")
    id: str = Field(..., alias="id", description="Unique identifier for the message.")
    round_id: int = Field(..., description="The round number of the conversation.")
    is_deleted: bool = Field(
        default=False,
        description="Indicates if the message has been marked as deleted.",
    )

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?",
                "id": "6c72abf7-d494-4f7b-a383-40f5d3233726",
                "round_id": 0,
                "is_deleted": False,
            }
        }


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


def mock_fetch_session_history(
    server_url: str, user_id: str, app_name: str, session_id: str
) -> list[Message]:
    """
    Simulate fetching the conversation history for a given session.
    Replace with a real GET request when implemented:
      GET /users/{user_id}/apps/{app_name}/sessions/{session_id}/queries

    Args:
        server_url (str): The URL of the backend server (not used in this mock).
        user_id (str): The ID of the user who owns the session (not used in this mock).
        app_name (str): The name of the application (not used in this mock).
        session_id (str): The unique identifier of the session to retrieve history for.

    Returns:
        list[Message]: A list of Message objects representing the conversation history.
    """

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
                content="「アルプスの少女ハイジ」は、スイスのアルプス山脈を舞台にした心温まる物語ですね。ハイジの純粋さや自然への愛、友人との絆が描かれていて、多くの人に愛されています。特に、山の美しい風景や、彼女が祖父と過ごす場面は印象的です。あなたの好きなキャラクターやエピソードはありますか？",
                id="ast-9999-2",
                round_id=1,
                is_deleted=False,
            ),
        ]
    else:
        return []


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


def mock_patch_session_info(
    server_url: str,
    user_id: str,
    session_id: str,
    session_name: str,
    is_private_session: bool,
    is_deleted: bool = False,
) -> None:
    """
    Simulate (or actually do) a PATCH request to update session info.

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID.
        session_id (str): The session ID to be updated.
        session_name (str): The new session name.
        is_private_session (bool): The new is_private_session value.
        is_deleted (bool): Whether the session is being marked as deleted.
    """
    endpoint = f"{server_url}/users/{user_id}/sessions/{session_id}"
    payload = {
        "session_name": session_name,
        "is_private_session": is_private_session,
        "is_deleted": is_deleted,
    }

    # Here we simulate a patch call, but in a real scenario:
    # response = requests.patch(endpoint, json=payload)
    # response.raise_for_status()

    print(f"[MOCK PATCH] endpoint={endpoint}, payload={payload}")


def mock_delete_round(
    server_url: str, session_id: str, round_id: str, deleted_by: str
) -> None:
    """
    Simulate deleting a round from the conversation history in the backend.
    In reality you'd call:
      DELETE /sessions/{session_id}/rounds/{round_id}
    with a JSON body: {"is_deleted": True, "deleted_by": <user_id>}

    Args:
        server_url (str): The backend server URL.
        session_id (str): The session ID.
        round_id (str): The unique round ID to delete.
        deleted_by (str): The user ID performing the deletion.
    """
    endpoint = f"{server_url}/sessions/{session_id}/rounds/{round_id}"
    payload = {"is_deleted": True, "deleted_by": deleted_by}
    print(f"[MOCK DELETE] endpoint={endpoint}, payload={payload}")


def mock_patch_feedback(llm_output_id: str, feedback: str, reason: str) -> None:
    """
    Simulate sending feedback to FastAPI:
      PATCH /llm_outputs/{id}
    Body: { "feedback": "good"|"bad", "reason": "..." }

    Args:
        llm_output_id (str): The unique ID of the LLM output (msg["id"]).
        feedback (str): "good" or "bad".
        reason (str): The user's explanation or comment.
    """
    endpoint = f"/llm_outputs/{llm_output_id}"
    payload = {"feedback": feedback, "reason": reason}
    print(f"[MOCK PATCH] {endpoint}, payload={payload}")


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
    N: int = 6  # Number of messages to keep in the UI

    # 2) Fetch list of sessions from the server (mocked)
    # Ensure that session_ids is initialized
    if "session_ids" not in st.session_state:
        st.session_state["session_ids"] = fetch_session_ids(
            server_url, user_id, app_name
        )

    # 3) Sidebar: pick a session or create a new one
    st.sidebar.write("## Create New Session")
    # Button to show the "Create New Session" form
    if st.session_state["show_create_form"]:
        create_label = "Cancel Create Session"
    else:
        create_label = "Create New Session"
    toggle_create_button: bool = st.sidebar.button(
        create_label, key="toggle_create_button"
    )
    if toggle_create_button:
        st.session_state["show_create_form"] = not st.session_state["show_create_form"]
        st.rerun()

    # Show the create form if the user clicked "Create New Session"
    messages: list[dict] = []
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
        history: list[dict] = mock_fetch_session_history(
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
                mock_patch_session_info(
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
                mock_patch_session_info(
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
                mock_delete_round(
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
            mock_patch_feedback(llm_output_id, feedback_type, reason_text)

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

        # 0) Build the array of dicts for the last N or all messages
        last_msgs = last_n_non_deleted(messages, N)
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
