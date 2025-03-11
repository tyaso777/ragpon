import requests
import streamlit as st


def main():
    st.title("Search Regulations (Mock Auth with Chat UI)")

    # 1) Initialize conversation storage
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 2) Mock a user ID if there's no real auth
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "test_user"  # Placeholder

    # 3) Fixed app name and FastAPI server URL
    app_name = "search_regulations"
    server_url = "http://localhost:8006"

    # 4) Optional session ID input
    session_id = st.text_input("Session ID", "1234")

    # 5) Display the conversation so far
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 6) Provide a chat input box at the bottom
    user_input = st.chat_input("Type your query here...")

    if user_input:
        # (A) Show the user's message in the conversation
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # (B) Send the user message to your FastAPI endpoint, streaming the response
        endpoint = f"{server_url}/users/{st.session_state['user_id']}/apps/{app_name}/sessions/{session_id}/queries"
        payload = {"query": user_input, "file": None, "is_private_session": False}

        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # If the request fails, show an error bubble from the assistant
            with st.chat_message("assistant"):
                st.error(f"Request failed: {e}")
            return

        # (C) Stream the assistant's response line by line
        partial_message_text = ""
        with st.chat_message("assistant"):
            # We use a placeholder to update the text progressively
            assistant_msg_placeholder = st.empty()

            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    chunk = line[len("data: ") :]
                    partial_message_text += chunk
                    assistant_msg_placeholder.write(partial_message_text)

        # (D) After streaming completes, save the full assistant message
        st.session_state["messages"].append(
            {"role": "assistant", "content": partial_message_text}
        )


if __name__ == "__main__":
    main()
