import json
import uuid

from locust import HttpUser, between, task


class ChatLoadTestUser(HttpUser):
    # Simulates realistic time between user actions
    wait_time = between(1, 3)

    def on_start(self):
        """
        Initializes each simulated user with unique session/user IDs.
        """
        self.user_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.app_name = "chat-test"
        self.headers = {"Content-Type": "application/json"}
        # Create session explicitly
        self.client.put(
            f"/users/{self.user_id}/apps/{self.app_name}/sessions/{self.session_id}",
            data=json.dumps({"session_name": "locust-test-session"}),
            headers=self.headers,
            timeout=10,
        )

    @task
    def chat_completion_stream(self):
        """
        Simulates a chat completion request to the correct FastAPI endpoint.
        """
        payload = {
            "round_id": 0,
            "user_msg_id": str(uuid.uuid4()),
            "system_msg_id": str(uuid.uuid4()),
            "assistant_msg_id": str(uuid.uuid4()),
            "messages": [
                {
                    "role": "user",
                    "content": "江戸時代と明治時代の日本の社会構造を300自程度で比較してください。",
                }
            ],
            "retrieved_contexts_str": "",
            "rag_mode": "No RAG",
            "optimized_queries": [],
            "use_reranker": False,
        }

        endpoint = f"/users/{self.user_id}/apps/{self.app_name}/sessions/{self.session_id}/queries"

        with self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=self.headers,
            stream=True,
            timeout=60,
            catch_response=True,
        ) as response:
            try:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        if decoded_line.endswith("[DONE]"):
                            break
                response.success()
            except Exception as e:
                response.failure(f"Stream read failed: {e}")
