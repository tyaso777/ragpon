import uuid
from dataclasses import dataclass

import requests
import streamlit as st


@dataclass
class SessionData:
    """
    Represents session information, including an ID, name, and privacy setting.
    """

    session_id: str
    session_name: str
    is_private_session: bool


#################################
# Mock or Real API Calls
#################################


def mock_fetch_session_ids(
    server_url: str, user_id: str, app_name: str
) -> list[SessionData]:
    """
    Simulate fetching a list of session data from the server.
    Each session is represented as [session_id, session_name, is_private_session].

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID for which to fetch sessions.
        app_name (str): The name of the application.

    Returns:
        list[SessionData]: A list of SessionData objects.
    """
    return [
        SessionData(
            session_id="1234",
            session_name="The First session",
            is_private_session=False,
        ),
        SessionData(
            session_id="5678", session_name="Session 5678", is_private_session=False
        ),
        SessionData(
            session_id="9999",
            session_name="Newest session 9999",
            is_private_session=True,
        ),
    ]


def mock_fetch_session_history(
    server_url: str, user_id: str, app_name: str, session_id: str
) -> list[dict]:
    """
    Simulate fetching the conversation history for a given session.
    Replace with a real GET request when implemented:
      GET /users/{user_id}/apps/{app_name}/sessions/{session_id}/queries

    Now each dict has keys:
    - role (str)
    - content (str)
    - id (str): an internal message ID
    - round_id (str): to group user+assistant pairs

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID associated with the session.
        app_name (str): The name of the application.
        session_id (str): The session ID for which to fetch conversation history.

    Returns:
        list[dict]: A list of messages with "role", "content", "id", and "round_id".
    """


def mock_fetch_session_history(
    server_url: str, user_id: str, app_name: str, session_id: str
) -> list[dict]:
    """
    Simulate fetching the conversation history for a given session.
    Now each dict has:
        - role (str): "user" or "assistant"
        - content (str)
        - id (str): a unique message identifier
        - round_id (int): an integer representing the round (1-based)
    """
    if session_id == "1234":
        return [
            # Round 1
            {
                "role": "user",
                "content": "Hi, how can I use this system (session 1234)?",
                "id": "usr-1234-1",
                "round_id": 1,
            },
            {
                "role": "assistant",
                "content": "Hello! You can ask me anything in 1234.",
                "id": "ast-1234-1",
                "round_id": 1,
            },
            # Round 2
            {
                "role": "user",
                "content": "Could you explain more features for 1234?",
                "id": "usr-1234-2",
                "round_id": 2,
            },
            {
                "role": "assistant",
                "content": "Sure, here are some more features...",
                "id": "ast-1234-2",
                "round_id": 2,
            },
            # Round 3
            {
                "role": "user",
                "content": "Got it. Any advanced tips for session 1234?",
                "id": "usr-1234-3",
                "round_id": 3,
            },
            {
                "role": "assistant",
                "content": "Yes, here are advanced tips...",
                "id": "ast-1234-3",
                "round_id": 3,
            },
        ]

    elif session_id == "5678":
        return [
            # Round 1
            {
                "role": "user",
                "content": "Hello from session 5678! (Round 1)",
                "id": "usr-5678-1",
                "round_id": 1,
            },
            {
                "role": "assistant",
                "content": "Hi! This is the 5678 conversation. (Round 1)",
                "id": "ast-5678-1",
                "round_id": 1,
            },
            # Round 2
            {
                "role": "user",
                "content": "Let's discuss something else in 5678. (Round 2)",
                "id": "usr-5678-2",
                "round_id": 2,
            },
            {
                "role": "assistant",
                "content": "Sure, here's more about 5678. (Round 2)",
                "id": "ast-5678-2",
                "round_id": 2,
            },
            # Round 3
            {
                "role": "user",
                "content": "Any final points for 5678? (Round 3)",
                "id": "usr-5678-3",
                "round_id": 3,
            },
            {
                "role": "assistant",
                "content": "Yes, final remarks on 5678... (Round 3)",
                "id": "ast-5678-3",
                "round_id": 3,
            },
        ]
    elif session_id == "9999":
        return [
            {
                "role": "user",
                "content": "Session 9999: RAG testing.",
                "id": "usr-9999-1",
                "round_id": 1,
            },
            {
                "role": "assistant",
                "content": "Sure, let's test RAG in session 9999.",
                "id": "ast-9999-1",
                "round_id": 1,
            },
            {
                "role": "user",
                "content": "Second user message for 9999.",
                "id": "usr-9999-2",
                "round_id": 2,
            },
            {
                "role": "assistant",
                "content": "Second assistant reply for 9999.",
                "id": "ast-9999-2",
                "round_id": 2,
            },
        ]
    else:
        return []


def post_query_to_fastapi(
    server_url: str, user_id: str, app_name: str, session_id: str, user_query: str
) -> requests.Response:
    """
    Simulate (or actually do) a POST to FastAPI's RAG+LLM endpoint:
      POST /users/{user_id}/apps/{app_name}/sessions/{session_id}/queries
    with stream=True to get partial responses.

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID.
        app_name (str): The name of the application.
        session_id (str): The session ID for which the query is posted.
        user_query (str): The user's query text to be processed by the LLM.

    Returns:
        requests.Response: A streaming Response object from the server.
    """
    endpoint = (
        f"{server_url}/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries"
    )
    payload = {"query": user_query, "file": None, "is_private_session": False}

    # Make a streaming POST request
    response = requests.post(endpoint, json=payload, stream=True)
    response.raise_for_status()
    return response  # We'll handle the streaming in the main code


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
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
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

    # 2) Fetch list of sessions from the server (mocked)
    # Ensure that session_ids is initialized
    if "session_ids" not in st.session_state:
        st.session_state["session_ids"] = mock_fetch_session_ids(
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
            st.session_state["messages"] = []

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
            st.session_state["messages"] = []
            st.rerun()
        else:
            st.session_state["current_session"] = st.session_state["session_ids"][-1]
            st.session_state["messages"] = mock_fetch_session_history(
                server_url=server_url,
                user_id=user_id,
                app_name=app_name,
                session_id=st.session_state["current_session"].session_id,
            )

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
            server_url, user_id, app_name, selected_session_id
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
                    st.session_state["messages"] = []
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
        # Only create the chat message block once per message
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
        # If this is a user message, display the trash button for that round
        if msg["role"] == "assistant":
            # To avoid showing multiple trash buttons for the same round,
            # only show it once.
            if msg["round_id"] not in displayed_round_ids:
                displayed_round_ids.add(msg["round_id"])
                # [CHANGE] Add a trash icon button to delete user+assistant pair
                delete_key: str = f"delete_{msg['round_id']}"
                if st.button("🗑️", key=delete_key, help="Delete this round"):
                    # 1) Call mock_delete_round
                    mock_delete_round(
                        server_url=server_url,
                        session_id=session_id_for_display,
                        round_id=msg["round_id"],
                        deleted_by=user_id,
                    )
                    # 2) Remove user+assistant messages with this round_id locally
                    updated_msgs: list[dict] = [
                        m for m in messages if m["round_id"] != msg["round_id"]
                    ]
                    st.session_state["session_histories"][
                        session_id_for_display
                    ] = updated_msgs
                    st.rerun()

    # 5) Chat input at the bottom to continue conversation
    user_input: str = st.chat_input("Type your query here...")
    if user_input:
        # We generate a new round_id for this user+assistant pair
        new_round_id: str = str(uuid.uuid4())

        # TODO: new_round_idは本当はpost_query_to_fastapiの中で生成されるべき

        # (A) Add user message to local state
        user_msg: dict = {
            "role": "user",
            "content": user_input,
            "id": f"usr-{new_round_id}",
            "round_id": new_round_id,
        }
        messages.append(user_msg)

        with st.chat_message("user"):
            st.write(user_input)

        # (B) Send the query to FastAPI (streaming)
        try:
            response: requests.Response = post_query_to_fastapi(
                server_url=server_url,
                user_id=st.session_state["user_id"],
                app_name=app_name,
                session_id=session_id_for_display,
                user_query=user_input,
            )
        except requests.exceptions.RequestException as e:
            # If request fails, show error from assistant
            with st.chat_message("assistant"):
                st.error(f"Request failed: {e}")
            return

        # (C) Stream partial assistant responses
        partial_message_text: str = ""
        with st.chat_message("assistant"):
            assistant_msg_placeholder = st.empty()
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    chunk: str = line[len("data: ") :]
                    partial_message_text += chunk
                    assistant_msg_placeholder.write(partial_message_text)

        # (D) Save final assistant message
        assistant_msg: dict = {
            "role": "assistant",
            "content": partial_message_text,
            "id": f"ast-{new_round_id}",
            "round_id": new_round_id,
        }
        messages.append(assistant_msg)
        st.rerun()


if __name__ == "__main__":
    main()
