"""Router B: small instruction-model router and hybrid fallback."""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.paths import PROJECT_ROOT
from src.routing.router_a import baseline_router


load_dotenv(PROJECT_ROOT / ".env")

LLAMA_SERVER_URL = os.getenv(
    "LLAMA_SERVER_URL",
    "http://127.0.0.1:8080/v1/chat/completions",
)
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "local-qwen")
MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "160"))
THINKING_BUDGET_TOKENS = int(os.getenv("ROUTER_THINKING_BUDGET", "48"))


ROUTER_SYSTEM = """
You are a routing controller for a technical support system.
Return JSON only.
Allowed routes: qa, support_specialist, tools, escalate.
Use qa for questions answerable from trusted documentation.
Use tools for live/system/ticket/log/package information.
Use support_specialist for troubleshooting synthesis.
Use escalate for high-risk or unresolved production incidents.
"""

ROUTER_EXAMPLES = [
    {
        "user": "The production database may be corrupted and users are losing data.",
        "assistant": {"route": "escalate", "confidence": 1.0},
    },
    {
        "user": "Check the installed transformers version on the server.",
        "assistant": {"route": "tools", "confidence": 0.98},
    },
]


def build_router_prompt(user_message: str):
    messages = [{"role": "system", "content": ROUTER_SYSTEM}]

    for example in ROUTER_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(example["assistant"]),
            }
        )

    messages.append({"role": "user", "content": user_message})
    return messages


def validate_router_response(response: str):
    try:
        result = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("Router response must contain JSON only.") from error

    if not isinstance(result, dict):
        raise ValueError("Router response must be a JSON object.")
    if set(result) != {"route", "confidence"}:
        raise ValueError("Router response must contain route and confidence only.")
    if result["route"] not in {"qa", "support_specialist", "tools", "escalate"}:
        raise ValueError("Router returned an unsupported route.")
    if (
        isinstance(result["confidence"], bool)
        or not isinstance(result["confidence"], (int, float))
        or not 0 <= result["confidence"] <= 1
    ):
        raise ValueError("Router confidence must be a number from 0 to 1.")

    result["source"] = "llm"
    return result


def generate_with_llama_server(messages):
    payload = json.dumps(
        {
            "model": LLAMA_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "thinking_budget_tokens": THINKING_BUDGET_TOKENS,
            "chat_template_kwargs": {"enable_thinking": True},
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = Request(
        LLAMA_SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
    except URLError as error:
        raise ConnectionError(
            f"Cannot connect to llama-server at {LLAMA_SERVER_URL}."
        ) from error

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("llama-server returned an unexpected response.") from error


def llm_router(text: str, generate=generate_with_llama_server):
    response = generate(build_router_prompt(text))
    return validate_router_response(response)


def hybrid_router(text: str, generate=generate_with_llama_server):
    decision = baseline_router(text)
    if decision.get("source") == "rule":
        return decision
    if not decision["needs_fallback"]:
        return decision
    return llm_router(text, generate)
