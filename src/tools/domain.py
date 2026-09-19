"""The 13 local tools required by the technical-support system."""

import ast
import operator
import re
import sqlite3
from datetime import datetime, timezone

from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.paths import DATA_ROOT


SUPPORT_DB_PATH = DATA_ROOT / "tools" / "support.db"
UPLOADS_ROOT = DATA_ROOT / "uploads"

KB = [
    ("kb-auth-001", "Access tokens expire after 45 minutes. Refresh-token lifetime is not specified."),
    ("kb-db-001", "A pool_timeout and high utilization indicate pressure, not proof of a connection leak."),
    ("kb-deploy-001", "Preserve logs and snapshots before rollback. Escalate suspected data loss."),
]

DOCS = [
    ("docs-api-auth", "Send the access token in the Authorization header using the Bearer scheme."),
    ("docs-backup", "Create a verified snapshot before a production database migration."),
    ("docs-health", "A deployment is ready after the API and database health checks pass."),
]

PACKAGES = {
    "transformers": ["5.17.0", "5.16.0", "4.57.6"],
    "torch": ["2.14.0", "2.13.0"],
    "peft": ["0.21.0", "0.20.0"],
}

HEALTH = {
    "api": {"status": "healthy", "latency_ms": 42},
    "database": {"status": "degraded", "connections_pct": 91},
    "gpu-worker": {"status": "healthy", "gpu_utilization": 74},
}


def _search(query: str, records: list[tuple[str, str]], limit: int) -> list[dict]:
    """Small TF-IDF search shared by the KB and documentation tools."""
    if not query.strip():
        return []
    matrix = TfidfVectorizer().fit_transform([text for _, text in records] + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [
        {"source_id": records[index][0], "text": records[index][1], "score": round(float(score), 4)}
        for index, score in ranked[:limit]
        if score > 0
    ]


def _connect() -> sqlite3.Connection:
    """Open the ticket database and create its small lab dataset once."""
    SUPPORT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SUPPORT_DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            resolution TEXT,
            created_at TEXT NOT NULL
        )"""
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(tickets)")}
    if "created_at" not in columns:
        connection.execute(
            "ALTER TABLE tickets ADD COLUMN created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )
    if connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
        created_at = datetime.now(timezone.utc).isoformat()
        connection.executemany(
            """INSERT INTO tickets(
                title, description, priority, status, resolution, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            [
                ("API returns 422", "Schema validation failed.", "medium", "resolved", "Fixed the payload.", created_at),
                ("Database pool timeout", "Connections reached 95%.", "high", "resolved", "Fixed slow queries.", created_at),
                ("GPU worker using CPU", "CUDA is visible.", "medium", "open", None, created_at),
            ],
        )
        connection.commit()
    return connection


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _calculate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("Exponent is too large")
        return OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")


@tool
def knowledge_base_search(query: str, limit: int = 3) -> dict:
    """Return the most relevant trusted internal KB passages."""
    results = _search(query, KB, limit)
    return {"ok": True, "results": results, "count": len(results)}


@tool
def ticket_search(query: str, limit: int = 5) -> dict:
    """Search previous support tickets in the local SQLite database."""
    connection = _connect()
    rows = connection.execute(
        """SELECT id, title, priority, status, resolution FROM tickets
        WHERE title LIKE ? OR description LIKE ? ORDER BY id DESC LIMIT ?""",
        (f"%{query}%", f"%{query}%", limit),
    ).fetchall()
    connection.close()
    tickets = [dict(row) for row in rows]
    return {"ok": True, "tickets": tickets, "count": len(tickets)}


@tool
def ticket_create(title: str, description: str, priority: str = "medium") -> dict:
    """Create a structured support ticket in the local SQLite database."""
    if priority not in {"low", "medium", "high", "critical"}:
        return {"ok": False, "error": "Invalid priority"}
    connection = _connect()
    cursor = connection.execute(
        """INSERT INTO tickets(title, description, priority, status, created_at)
        VALUES (?, ?, ?, 'open', ?)""",
        (title, description, priority, datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    ticket_id = cursor.lastrowid
    connection.close()
    return {"ok": True, "ticket_id": ticket_id, "status": "open", "priority": priority}


@tool
def system_health_check(service: str) -> dict:
    """Return deterministic service health for the lab environment."""
    return {"ok": True, "service": service, **HEALTH.get(service, {"status": "unknown"})}


@tool
def log_analyzer(log_text: str, limit: int = 20) -> dict:
    """Extract error and warning lines from support logs."""
    lines = log_text.splitlines()
    errors = [line for line in lines if re.search(r"\b(ERROR|FATAL|EXCEPTION)\b", line, re.I)]
    warnings = [line for line in lines if re.search(r"\bWARN(?:ING)?\b", line, re.I)]
    return {
        "ok": True,
        "errors": errors[:limit],
        "warnings": warnings[:limit],
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


@tool
def documentation_search(query: str, limit: int = 3) -> dict:
    """Search the local indexed product documentation."""
    results = _search(query, DOCS, limit)
    return {"ok": True, "results": results, "count": len(results)}


@tool
def package_lookup(package_name: str, version: str = "") -> dict:
    """Look up package versions in a deterministic local registry."""
    name = package_name.lower().replace("_", "-")
    versions = PACKAGES.get(name, [])
    return {
        "ok": True,
        "found": bool(versions),
        "package": name,
        "latest": versions[0] if versions else None,
        "versions": versions,
        "version_available": version in versions if version else None,
    }


@tool
def sql_query(query: str) -> dict:
    """Run one allowlisted read-only SELECT query against the tickets table."""
    lowered = query.strip().lower()
    functions = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", lowered))
    allowed_functions = {"avg", "count", "length", "lower", "max", "min", "sum", "upper"}
    if (
        not lowered.startswith("select ")
        or ";" in query
        or not re.search(r"\bfrom\s+tickets\b", lowered)
        or re.search(r"\b(insert|update|delete|drop|alter|create|pragma|attach)\b", lowered)
        or not functions <= allowed_functions
    ):
        return {"ok": False, "error": "Only safe SELECT queries on tickets are allowed"}
    _connect().close()
    connection = sqlite3.connect(f"file:{SUPPORT_DB_PATH.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(f"SELECT * FROM ({query}) LIMIT 100")]
    except sqlite3.Error as error:
        return {"ok": False, "error": str(error)}
    finally:
        connection.close()
    return {"ok": True, "rows": rows, "count": len(rows)}


@tool
def calculator(expression: str) -> dict:
    """Evaluate arithmetic without eval or function calls."""
    try:
        result = _calculate(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
        return {"ok": False, "error": str(error)}
    return {"ok": True, "result": result}


@tool
def file_search(query: str, path: str = ".", limit: int = 20) -> dict:
    """Search text files inside data/uploads only."""
    root = UPLOADS_ROOT.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        return {"ok": False, "error": "Path must remain inside data/uploads", "matches": []}
    files = [target] if target.is_file() else target.rglob("*") if target.exists() else []
    matches = []
    for file in files:
        if not file.is_file() or file.suffix.lower() not in {".txt", ".log", ".json", ".yaml", ".yml", ".toml"}:
            continue
        for number, line in enumerate(file.read_text(errors="ignore").splitlines(), 1):
            if query.lower() in line.lower():
                matches.append({"path": str(file.relative_to(root)), "line": number, "text": line})
                if len(matches) == limit:
                    return {"ok": True, "matches": matches, "count": len(matches)}
    return {"ok": True, "matches": matches, "count": len(matches)}


@tool
def web_search(query: str) -> dict:
    """Return a deterministic mock external-search result for the lab."""
    results = {
        "kubernetes": ("Kubernetes troubleshooting", "https://kubernetes.io/docs/tasks/debug/"),
        "postgresql": ("PostgreSQL documentation", "https://www.postgresql.org/docs/"),
        "python": ("Python packaging guide", "https://packaging.python.org/"),
    }
    match = next((value for key, value in results.items() if key in query.lower()), None)
    items = [{"title": match[0], "url": match[1]}] if match else []
    return {"ok": True, "results": items, "mock": True}


@tool
def escalate_to_human(reason: str, evidence: str) -> dict:
    """Create a high-priority escalation ticket with evidence."""
    ticket = ticket_create.invoke(
        {"title": f"ESCALATION: {reason}", "description": evidence, "priority": "high"}
    )
    return {"ok": ticket["ok"], "escalated": ticket["ok"], "reason": reason, "ticket": ticket}


@tool
def diagnostic_runbook(issue_type: str, service: str, symptom: str) -> dict:
    """Run health, log, and prior-ticket checks in sequence."""
    health = system_health_check.invoke({"service": service})
    steps = [
        {"step": "health_check", "result": health},
        {"step": "log_analysis", "result": log_analyzer.invoke({"log_text": symptom})},
        {"step": "prior_tickets", "result": ticket_search.invoke({"query": symptom})},
    ]
    status = "needs_human" if health["status"] == "unknown" else "diagnostics_complete"
    return {"ok": True, "issue_type": issue_type, "status": status, "steps": steps}


ALL_TOOLS = [
    knowledge_base_search,
    ticket_search,
    ticket_create,
    system_health_check,
    log_analyzer,
    documentation_search,
    package_lookup,
    sql_query,
    calculator,
    file_search,
    web_search,
    escalate_to_human,
    diagnostic_runbook,
]

TOOL_BY_NAME = {item.name: item for item in ALL_TOOLS}
