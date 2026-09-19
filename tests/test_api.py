import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from src.api import (
    MODEL_NAME,
    ChatCompletionRequest,
    ChatMessage,
    chat_completions,
    health,
    models,
)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.api_key_patch = patch("src.api.API_KEY", "local-demo-key")
        self.api_key_patch.start()

    def tearDown(self):
        self.api_key_patch.stop()

    def test_health_and_models(self):
        self.assertEqual(asyncio.run(health()), {"status": "ok"})
        listed = asyncio.run(models())["data"]
        self.assertEqual([model["id"] for model in listed], [MODEL_NAME])

    @patch("src.api.run_agent")
    def test_chat_completion(self, run_agent):
        run_agent.return_value = {
            "answer": "Check the database pool.",
            "route": "tools",
            "intent": "database",
            "trace_id": "trace-123",
        }
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Database timeout")]
        )
        body = asyncio.run(chat_completions(request, "Bearer local-demo-key"))
        self.assertEqual(body["choices"][0]["message"]["content"], "Check the database pool.")
        self.assertEqual(body["system_metadata"]["trace_id"], "trace-123")

    def test_rejects_invalid_requests(self):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")]
        )
        with self.assertRaises(HTTPException) as unauthorized:
            asyncio.run(chat_completions(request, None))
        self.assertEqual(unauthorized.exception.status_code, 401)

    @patch("src.api.run_agent")
    def test_streaming_completion(self, run_agent):
        run_agent.return_value = {"answer": "Streaming works."}
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
        )

        async def collect():
            response = await chat_completions(request, "Bearer local-demo-key")
            return "".join([chunk async for chunk in response.body_iterator])

        body = asyncio.run(collect())
        self.assertIn("Streaming works.", body)
        self.assertIn("data: [DONE]", body)


if __name__ == "__main__":
    unittest.main()
