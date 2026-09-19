"""LangGraph workflow for the multi-model support agent."""

import json
import os
from time import perf_counter
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from src.model_b.inference import run_qa_model
from src.model_c.inference import run_support_model
from src.routing.router_b import hybrid_router
from src.tools.domain import (
    diagnostic_runbook,
    escalate_to_human,
    knowledge_base_search,
)


class SupportState(TypedDict, total=False):
    user_message: str
    route: str
    intent: str
    confidence: float
    context: list[dict]
    tool_results: list[dict]
    answer: str
    escalate: bool
    trace_id: str
    latency_ms: float


def route_node(state: SupportState):
    decision = hybrid_router(state["user_message"])
    return {
        "route": decision["route"],
        "intent": decision.get("intent", ""),
        "confidence": decision["confidence"],
    }


def qa_node(state: SupportState):
    kb = knowledge_base_search.invoke({"query": state["user_message"]})
    context = kb["results"]
    answer = run_qa_model(state["user_message"], [item["text"] for item in context])
    return {"context": context, "answer": answer}


def _infer_service(message: str) -> str:
    lowered = message.lower()
    for service in ("database", "gpu-worker", "api"):
        if service.split("-")[0] in lowered:
            return service
    return "unknown"


def tool_node(state: SupportState):
    result = diagnostic_runbook.invoke(
        {
            "issue_type": state.get("intent", "unknown"),
            "service": _infer_service(state["user_message"]),
            "symptom": state["user_message"],
        }
    )
    return {"tool_results": [result]}


def support_node(state: SupportState):
    answer = run_support_model(
        state["user_message"],
        state.get("context", []),
        state.get("tool_results", []),
    )
    return {"answer": answer}


def escalation_node(state: SupportState):
    result = escalate_to_human.invoke(
        {
            "reason": "High-risk or unresolved technical incident",
            "evidence": json.dumps(state),
        }
    )
    ticket_id = result.get("ticket", {}).get("ticket_id")
    return {
        "escalate": True,
        "tool_results": [result],
        "answer": f"Escalated to human support. Ticket: {ticket_id}",
    }


def choose_after_router(state: SupportState) -> Literal["qa", "tools", "support", "escalate"]:
    return {
        "qa": "qa",
        "tools": "tools",
        "escalate": "escalate",
    }.get(state.get("route", ""), "support")


builder = StateGraph(SupportState)
builder.add_node("route", route_node)
builder.add_node("qa", qa_node)
builder.add_node("tools", tool_node)
builder.add_node("support", support_node)
builder.add_node("escalate", escalation_node)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", choose_after_router)
builder.add_edge("qa", END)
builder.add_edge("tools", "support")
builder.add_edge("support", END)
builder.add_edge("escalate", END)
graph = builder.compile()


def _langfuse_handler(trace_id: str):
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler(trace_context={"trace_id": trace_id.replace("-", "")})


def run_agent(user_message: str) -> SupportState:
    """Run the graph and optionally send its trace to Langfuse."""
    trace_id = str(uuid4())
    handler = _langfuse_handler(trace_id)
    config = {
        "run_name": "technical-support-agent",
        "metadata": {"project": "multi-agent-teq-support", "trace_id": trace_id},
    }
    if handler:
        config["callbacks"] = [handler]

    started = perf_counter()
    result = graph.invoke({"user_message": user_message, "trace_id": trace_id}, config=config)
    result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
    if handler:
        from langfuse import get_client

        get_client().flush()
    return result
