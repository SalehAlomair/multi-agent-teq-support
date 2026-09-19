import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tools import ALL_TOOLS, TOOL_BY_NAME
from src.tools import domain


class DomainToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.database_patch = patch.object(domain, "SUPPORT_DB_PATH", root / "support.db")
        self.uploads_patch = patch.object(domain, "UPLOADS_ROOT", root / "uploads")
        self.database_patch.start()
        self.uploads_patch.start()
        domain.UPLOADS_ROOT.mkdir()
        (domain.UPLOADS_ROOT / "service.log").write_text(
            "INFO service started\nERROR pool_timeout while acquiring connection\n"
        )

    def tearDown(self):
        self.uploads_patch.stop()
        self.database_patch.stop()
        self.temp_directory.cleanup()

    def test_exposes_all_thirteen_contracts(self):
        expected = {
            "knowledge_base_search", "ticket_search", "ticket_create",
            "system_health_check", "log_analyzer", "documentation_search",
            "package_lookup", "sql_query", "calculator", "file_search",
            "web_search", "escalate_to_human", "diagnostic_runbook",
        }
        self.assertEqual({item.name for item in ALL_TOOLS}, expected)
        self.assertEqual(set(TOOL_BY_NAME), expected)
        for item in ALL_TOOLS:
            self.assertTrue(item.description)
            self.assertIsNotNone(item.args_schema)

    def test_search_tools_return_ranked_structured_results(self):
        kb = domain.knowledge_base_search.invoke({"query": "database connection pool timeout"})
        docs = domain.documentation_search.invoke({"query": "Bearer authentication"})
        web = domain.web_search.invoke({"query": "Kubernetes troubleshooting"})
        self.assertEqual(kb["results"][0]["source_id"], "kb-db-001")
        self.assertEqual(docs["results"][0]["source_id"], "docs-api-auth")
        self.assertTrue(web["mock"])
        self.assertIn("kubernetes.io", web["results"][0]["url"])

    def test_ticket_create_search_sql_and_escalation(self):
        created = domain.ticket_create.invoke({
            "title": "Proxy failure",
            "description": "Requests fail through the proxy.",
            "priority": "high",
        })
        found = domain.ticket_search.invoke({"query": "Proxy"})
        selected = domain.sql_query.invoke({
            "query": "SELECT id, title, status FROM tickets WHERE id = %d" % created["ticket_id"]
        })
        escalation = domain.escalate_to_human.invoke({
            "reason": "suspected data loss",
            "evidence": "Audit counts differ after migration.",
        })
        self.assertTrue(created["ok"])
        self.assertEqual(found["tickets"][0]["title"], "Proxy failure")
        self.assertEqual(selected["rows"][0]["id"], created["ticket_id"])
        self.assertTrue(escalation["escalated"])

    def test_sql_is_read_only_and_allowlisted(self):
        self.assertTrue(domain.sql_query.invoke({"query": "SELECT COUNT(*) AS count FROM tickets"})["ok"])
        self.assertFalse(domain.sql_query.invoke({"query": "DELETE FROM tickets"})["ok"])
        self.assertFalse(domain.sql_query.invoke({"query": "SELECT * FROM sqlite_master"})["ok"])
        self.assertFalse(domain.sql_query.invoke({"query": "SELECT * FROM tickets; DROP TABLE tickets"})["ok"])
        self.assertFalse(domain.sql_query.invoke({"query": "SELECT random() FROM tickets"})["ok"])

    def test_calculator_is_safe(self):
        self.assertEqual(domain.calculator.invoke({"expression": "(12 + 8) / 4"})["result"], 5)
        self.assertFalse(domain.calculator.invoke({"expression": "__import__('os')"})["ok"])
        self.assertFalse(domain.calculator.invoke({"expression": "2 ** 999"})["ok"])

    def test_health_logs_package_and_files(self):
        health = domain.system_health_check.invoke({"service": "database"})
        logs = domain.log_analyzer.invoke({"log_text": "WARN slow\nERROR failed\nINFO done"})
        package = domain.package_lookup.invoke({"package_name": "transformers", "version": "5.17.0"})
        files = domain.file_search.invoke({"query": "pool_timeout"})
        traversal = domain.file_search.invoke({"query": "secret", "path": "../"})
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(logs["error_count"], 1)
        self.assertTrue(package["version_available"])
        self.assertEqual(files["matches"][0]["path"], "service.log")
        self.assertFalse(traversal["ok"])

    def test_diagnostic_runbook_calls_multiple_tools(self):
        result = domain.diagnostic_runbook.invoke({
            "issue_type": "database",
            "service": "database",
            "symptom": "ERROR pool timeout",
        })
        self.assertEqual(result["status"], "diagnostics_complete")
        self.assertEqual(
            [step["step"] for step in result["steps"]],
            ["health_check", "log_analysis", "prior_tickets"],
        )


if __name__ == "__main__":
    unittest.main()
