# Part 7/9 — Phase D — Domain Tool Layer

The system must expose all 13 tool contracts. To stay within the 5–7 hour timebox, at least 6 tools must be fully functional; the
remaining tools may use deterministic local/mock implementations, but every tool must have a clear schema and return structured
data.

| Tool | Purpose | Minimum implementation |
|---|---|---|
| knowledge_base_search | Retrieve internal KB passages | Local vector/TF-IDF or embeddings search |
| ticket_search | Find prior support cases | SQLite/PostgreSQL query |
| ticket_create | Create a new case | Insert structured record |
| system_health_check | Check service status | Mock/local HTTP health registry |
| log_analyzer | Find errors/patterns in logs | Regex + severity extraction |
| documentation_search | Search product docs | Local indexed docs |
| package_lookup | Version/package facts | Local registry or allowed web lookup |
| sql_query | Read support data | Read-only SQL with allowlist |
| calculator | Numeric calculation | Safe arithmetic |
| file_search | Search uploaded config/log files | Restricted local file search |
| web_search | External technical fallback | Provider wrapper / mock |
| escalate_to_human | Create escalation payload | Ticket + reason + evidence |
| diagnostic_runbook | Orchestrate known diagnostic steps | Calls multiple tools in sequence |

## Core tool implementations
```python
from typing import Any
from langchain_core.tools import tool
import re, sqlite3, math

@tool
def calculator(expression: str) -> dict:
    """Evaluate a restricted arithmetic expression."""
    allowed = set("0123456789.+-*/()% ")
    if not set(expression) <= allowed:
        return {"ok": False, "error": "Unsupported characters"}
    return {"ok": True, "result": eval(expression, {"__builtins__": {}}, {})}

@tool
def log_analyzer(log_text: str) -> dict:
    """Extract common ERROR/WARN patterns from support logs."""
    lines = log_text.splitlines()
    errors = [x for x in lines if re.search(r"\b(ERROR|FATAL|EXCEPTION)\b", x, re.I)]
    warnings = [x for x in lines if re.search(r"\bWARN(ING)?\b", x, re.I)]
    return {"errors": errors[:20], "warnings": warnings[:20], "error_count": len(errors)}

@tool
def ticket_search(query: str) -> dict:
    """Search previous support tickets in the local support DB."""
    con = sqlite3.connect("data/mock_support.db")
    rows = con.execute(
        "SELECT id, title, status, resolution FROM tickets WHERE title LIKE ? LIMIT 5",
        (f"%{query}%",),
    ).fetchall()
    con.close()
    return {"tickets": rows}

@tool
def ticket_create(title: str, description: str, priority: str = "medium") -> dict:
    """Create a support ticket."""
    con = sqlite3.connect("data/mock_support.db")
    cur = con.execute(
        "INSERT INTO tickets(title, description, priority, status) VALUES (?, ?, ?, 'open')",
        (title, description, priority),
    )
    con.commit(); ticket_id = cur.lastrowid; con.close()
    return {"ticket_id": ticket_id, "status": "open"}

@tool
def system_health_check(service: str) -> dict:
    """Return deterministic service health for the lab environment."""
    registry = {
        "api": {"status": "healthy", "latency_ms": 42},
        "database": {"status": "degraded", "connections_pct": 91},
        "gpu-worker": {"status": "healthy", "gpu_utilization": 74},
    }
    return registry.get(service, {"status": "unknown"})

@tool
def escalate_to_human(reason: str, evidence: str) -> dict:
    """Escalate a support case with evidence."""
    ticket = ticket_create.invoke({
        "title": f"ESCALATION: {reason}",
        "description": evidence,
        "priority": "high",
    })
    return {"escalated": True, "reason": reason, "ticket": ticket}
```

## Intentional ≤15% missing code
```python
# Tool contracts to implement with local/mock backends.
@tool
def knowledge_base_search(query: str) -> dict:
    """Return top trusted KB passages with scores and source IDs."""
    ...

@tool
def documentation_search(query: str) -> dict:
    """Search product/API documentation."""
    ...

@tool
def package_lookup(package_name: str, version: str = "") -> dict:
    """Look up package/version compatibility information."""
    ...

@tool
def sql_query(query: str) -> dict:
    """Execute an allowlisted read-only SQL query."""
    ...

@tool
def file_search(query: str, path: str = "data/uploads") -> dict:
    """Search approved uploaded logs/configs."""
    ...

@tool
def web_search(query: str) -> dict:
    """External technical fallback. Use an approved provider or a deterministic lab mock."""
    ...

# TODO(student 7): implement ONE of the six functions above fully.
# The other five may use deterministic mocks for this one-day challenge.

@tool
def diagnostic_runbook(issue_type: str, service: str, symptom: str) -> dict:
    """Run a deterministic diagnostic sequence before free-form reasoning."""
    steps = []
    health = system_health_check.invoke({"service": service})
    steps.append({"step": "health_check", "result": health})

    prior = ticket_search.invoke({"query": symptom})
    steps.append({"step": "prior_tickets", "result": prior})

    if health.get("status") == "unknown":
        return {"status": "needs_human", "steps": steps}

    return {"status": "diagnostics_complete", "steps": steps}
```
