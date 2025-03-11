import json

import requests
import streamlit as st


def main():
    st.title("FastAPI (RAG + LLM) Streaming Test")

    # Enter the URL of the FastAPI server
    # For example, if running locally: "http://localhost:8000"
    server_url = st.text_input("FastAPI Server URL", "http://localhost:8006")

    # Specify the user ID, app name, and session ID (can be changed as needed)
    user_id = st.text_input("User ID", "test_user")
    app_name = st.text_input("App Name", "my_app")
    session_id = st.text_input("Session ID", "1234")

    # Text area for the query (user_query)
    user_query = st.text_area("Query", "Hello, how are you?")

    if st.button("Send Query"):
        if not server_url:
            st.error("Please enter the FastAPI server URL.")
            return

        # Construct the endpoint
        # Example: http://localhost:8000/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries
        endpoint = f"{server_url}/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries"

        # Create the JSON request body
        payload = {"query": user_query, "file": None, "is_private_session": False}

        # Use stream=True to receive the response in a streaming manner
        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            return

        # Read the response line by line, parsing the SSE (Server-Sent Events) format "data: ..."
        st.write("### Response (Streaming):")
        output_placeholder = st.empty()  # Placeholder to display the output

        accumulated_text = ""  # Accumulate the received text for display

        for line in response.iter_lines(decode_unicode=True):
            if line:
                # With SSE, lines like "data: ..." will be returned repeatedly
                if line.startswith("data: "):
                    chunk = line[len("data: ") :]  # Remove the "data: " prefix
                    # Accumulate the received string
                    accumulated_text += chunk
                    # Update the display in real time
                    output_placeholder.write(accumulated_text)

        # The streaming response has ended
        st.success("Streaming completed.")


if __name__ == "__main__":
    main()
