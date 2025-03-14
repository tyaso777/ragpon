import uuid

import requests
import streamlit as st

#################################
# Mock or Real API Calls
#################################


def mock_fetch_session_ids(server_url: str, user_id: str, app_name: str) -> list[list]:
    """
    Simulate fetching a list of session data from the server.
    Each session is represented as [session_id, session_name, is_private_session].

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID for which to fetch sessions.
        app_name (str): The name of the application.

    Returns:
        list[list]: A list of sessions. Each session is
            [session_id (str), session_name (str), is_private_session (bool)].
    """
    return [
        ["1234", "Session 1234", False],
        ["5678", "Session 5678", False],
        ["9999", "Session 9999", True],
    ]


def mock_fetch_session_history(
    server_url: str, user_id: str, app_name: str, session_id: str
) -> list[dict]:
    """
    Simulate fetching the conversation history for a given session.
    Replace with a real GET request when implemented:
      GET /users/{user_id}/apps/{app_name}/sessions/{session_id}/queries

    Args:
        server_url (str): The URL of the backend server.
        user_id (str): The user ID associated with the session.
        app_name (str): The name of the application.
        session_id (str): The session ID for which to fetch conversation history.

    Returns:
        list[dict]: A list of messages. Each dict has keys "role" and "content".
    """
    if session_id == "1234":
        return [
            {
                "role": "user",
                "content": "Hi, how can I use this system (session 1234)?",
            },
            {"role": "assistant", "content": "Hello! You can ask me anything in 1234."},
        ]
    elif session_id == "5678":
        return [
            {"role": "user", "content": "Hello from session 5678!"},
            {"role": "assistant", "content": "Hi! This is the 5678 conversation."},
        ]
    elif session_id == "9999":
        return [
            {"role": "user", "content": "Session 9999: RAG testing."},
            {"role": "assistant", "content": "Sure, let's test RAG in session 9999."},
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


#################################
# Streamlit App
#################################


def main():
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

    user_id = st.session_state["user_id"]
    app_name = "search_regulations"
    server_url = "http://localhost:8006"  # Fixed server URL

    # Ensure that session_ids is initialized
    if "session_ids" not in st.session_state:
        st.session_state["session_ids"] = mock_fetch_session_ids(
            server_url, user_id, app_name
        )

    # 2) Fetch list of sessions from the server (mocked)
    session_ids = st.session_state["session_ids"]

    # 3) Sidebar: pick a session or create a new one

    # Create form visibility flag
    if "show_create_form" not in st.session_state:
        st.session_state["show_create_form"] = False

    # Button to show the "Create New Session" form
    create_button_clicked = st.sidebar.button("Create New Session")
    if create_button_clicked:
        st.session_state["show_create_form"] = True

    st.sidebar.write("## Session List")

    # Radio button to choose an existing session
    selected_session_data = st.sidebar.radio(
        "Choose a session:",
        session_ids,
        format_func=lambda x: x[1],  # x[1] is the session_name
        key="unique_session_radio",
    )

    # If no session has been chosen yet, default to the first in the list
    if st.session_state["current_session"] is None:
        st.session_state["current_session"] = session_ids[0]
        st.session_state["messages"] = mock_fetch_session_history(
            server_url, user_id, app_name, session_ids[0]
        )

    # Show the create form if the user clicked "Create New Session"
    if st.session_state["show_create_form"]:
        # Input fields for new session
        new_session_name = st.sidebar.text_input("Session Name", value="No title")
        new_session_is_private = st.sidebar.radio("Is Private?", options=[True, False])

        # Button to finalize session creation
        if st.sidebar.button("Create"):
            # Generate a new session ID
            new_session_id = str(uuid.uuid4())
            # Insert [session_id, session_name, is_private_session] into session_ids
            st.session_state["session_ids"].insert(
                0, [new_session_id, new_session_name, new_session_is_private]
            )
            # Switch to the newly created session
            st.session_state["current_session"] = new_session_id
            st.session_state["messages"] = []

            # Hide the create form
            st.session_state["show_create_form"] = False

            # Rerun to refresh the UI
            st.rerun()
    else:
        # If the form is not shown, assume we're selecting an existing session
        st.session_state["current_session"] = selected_session_data[0]
        st.session_state["messages"] = mock_fetch_session_history(
            server_url, user_id, app_name, selected_session_data[0]
        )

    # Display which session is active
    st.write(f"**Current Session**: {st.session_state['current_session']}")

    # Edit Session button (single button for both edit and delete)
    st.sidebar.write("## Manage Selected Session")
    edit_button_clicked = st.sidebar.button("Edit Session")

    if edit_button_clicked:
        st.session_state["show_edit_form"] = True

    if st.session_state["show_edit_form"]:
        # Show a form to edit session name, privacy, or delete
        st.sidebar.write("### Edit or Delete Session")

        current_name = selected_session_data[1]
        current_is_private = selected_session_data[2]

        edited_session_name = st.sidebar.text_input("Session Name", value=current_name)
        edited_is_private = st.sidebar.radio(
            "Is Private?", options=[True, False], index=0 if current_is_private else 1
        )

        # "Delete this session?" as a checkbox
        delete_this_session = st.sidebar.checkbox("Delete this session?")

        if st.sidebar.button("Update"):
            if delete_this_session:
                # Perform delete
                mock_patch_session_info(
                    server_url,
                    user_id,
                    selected_session_data[0],
                    edited_session_name,
                    edited_is_private,
                    is_deleted=True,
                )
                # Remove from local list
                st.session_state["session_ids"] = [
                    s
                    for s in st.session_state["session_ids"]
                    if s[0] != selected_session_data[0]
                ]
                if st.session_state["current_session"] == selected_session_data[0]:
                    st.session_state["current_session"] = None
                    st.session_state["messages"] = []
            else:
                # Perform update (no delete)
                mock_patch_session_info(
                    server_url,
                    user_id,
                    selected_session_data[0],
                    edited_session_name,
                    edited_is_private,
                    is_deleted=False,
                )
                # Update local data
                for s in st.session_state["session_ids"]:
                    if s[0] == selected_session_data[0]:
                        s[1] = edited_session_name
                        s[2] = edited_is_private
                        break

            st.session_state["show_edit_form"] = False
            st.rerun()

    # 4) Show the existing conversation messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 5) Chat input at the bottom to continue conversation
    user_input = st.chat_input("Type your query here...")
    if user_input:
        # (A) Add user message to local state
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # (B) Send the query to FastAPI (streaming)
        try:
            response = post_query_to_fastapi(
                server_url,
                st.session_state["user_id"],
                app_name,
                st.session_state["current_session"],
                user_input,
            )
        except requests.exceptions.RequestException as e:
            # If request fails, show error from assistant
            with st.chat_message("assistant"):
                st.error(f"Request failed: {e}")
            return

        # (C) Stream partial assistant responses
        partial_message_text = ""
        with st.chat_message("assistant"):
            assistant_msg_placeholder = st.empty()
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    chunk = line[len("data: ") :]
                    partial_message_text += chunk
                    assistant_msg_placeholder.write(partial_message_text)

        # (D) Save final assistant message
        st.session_state["messages"].append(
            {"role": "assistant", "content": partial_message_text}
        )


if __name__ == "__main__":
    main()
