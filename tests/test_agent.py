import unittest
from unittest.mock import Mock, patch

from src.agent import graph, run_agent


class AgentGraphTests(unittest.TestCase):
    @patch("src.agent._langfuse_handler", return_value=None)
    @patch("src.agent.graph.invoke")
    def test_run_returns_trace_and_latency(self, invoke, _handler):
        invoke.side_effect = lambda state, config: {**state, "answer": "ok"}
        result = run_agent("hello")
        self.assertTrue(result["trace_id"])
        self.assertGreaterEqual(result["latency_ms"], 0)

    @patch("src.agent.run_qa_model", return_value="45 minutes")
    @patch("src.agent.hybrid_router", return_value={"route": "qa", "confidence": 1.0})
    def test_qa_route(self, _router, _qa):
        result = graph.invoke({"user_message": "According to the docs, how long do access tokens last?"})
        self.assertEqual(result["route"], "qa")
        self.assertEqual(result["answer"], "45 minutes")
        self.assertTrue(result["context"])

    @patch("src.agent.run_support_model", return_value="Check the database pool.")
    @patch("src.agent.hybrid_router", return_value={
        "route": "tools", "intent": "database", "confidence": 0.95
    })
    def test_tools_then_support(self, _router, _support):
        result = graph.invoke({"user_message": "The database shows ERROR pool timeout"})
        self.assertEqual(result["route"], "tools")
        self.assertTrue(result["tool_results"])
        self.assertEqual(result["answer"], "Check the database pool.")

    @patch("src.agent.run_support_model", return_value="Troubleshooting answer")
    @patch("src.agent.hybrid_router", return_value={
        "route": "support_specialist", "confidence": 0.9
    })
    def test_support_route(self, _router, _support):
        result = graph.invoke({"user_message": "Help me troubleshoot the deployment"})
        self.assertEqual(result["answer"], "Troubleshooting answer")

    @patch("src.agent.escalate_to_human")
    @patch("src.agent.hybrid_router", return_value={"route": "escalate", "confidence": 1.0})
    def test_escalation_route(self, _router, escalate):
        escalate.invoke.return_value = {"ok": True, "ticket": {"ticket_id": 99}}
        result = graph.invoke({"user_message": "Production down with data loss"})
        self.assertTrue(result["escalate"])
        self.assertIn("99", result["answer"])


if __name__ == "__main__":
    unittest.main()
