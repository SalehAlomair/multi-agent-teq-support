import json
import unittest
from unittest.mock import patch

from src.routing.router_a import baseline_router, rule_first
from src.routing.router_b import (
    build_router_prompt,
    hybrid_router,
    validate_router_response,
)


class RouterTests(unittest.TestCase):
    def test_escalation_rule_has_priority(self):
        result = rule_first("Production down after suspected data loss")
        self.assertEqual(result["route"], "escalate")
        self.assertEqual(result["source"], "rule")

    @patch("src.routing.router_a.classifier_route")
    def test_classifier_mapping_and_confidence_policy(self, classifier_route):
        classifier_route.return_value = {
            "intent": "package",
            "confidence": 0.91,
            "source": "classifier",
        }
        result = baseline_router("Which package version is installed?")
        self.assertEqual(result["route"], "tools")
        self.assertFalse(result["needs_fallback"])

    def test_llm_prompt_contains_two_examples(self):
        prompt = build_router_prompt("Why is deployment failing?")
        self.assertEqual([item["role"] for item in prompt].count("assistant"), 2)
        self.assertEqual(prompt[-1]["content"], "Why is deployment failing?")

    def test_llm_response_is_strictly_validated(self):
        valid = {"route": "qa", "confidence": 0.9}
        result = validate_router_response(json.dumps(valid))
        self.assertEqual(result["route"], "qa")
        self.assertEqual(result["source"], "llm")

        with self.assertRaises(ValueError):
            validate_router_response("```json\n{}\n```")
        with self.assertRaises(ValueError):
            validate_router_response(json.dumps({**valid, "unexpected": True}))

    @patch("src.routing.router_a.classifier_route")
    def test_hybrid_uses_llm_for_low_confidence(self, classifier_route):
        classifier_route.return_value = {
            "intent": "network",
            "confidence": 0.4,
            "source": "classifier",
        }

        def generate(_messages):
            return json.dumps(
                {"route": "support_specialist", "confidence": 0.88}
            )

        result = hybrid_router("Connection drops sometimes", generate)
        self.assertEqual(result["route"], "support_specialist")
        self.assertEqual(result["source"], "llm")


if __name__ == "__main__":
    unittest.main()
