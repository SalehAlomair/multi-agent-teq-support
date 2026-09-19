"""Use the local llama-server as the support specialist."""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen


LLAMA_SERVER_URL = os.getenv(
    "LLAMA_SERVER_URL",
    "http://127.0.0.1:8080/v1/chat/completions",
)
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "local-qwen")
MAX_TOKENS = int(os.getenv("SUPPORT_MAX_TOKENS", "256"))
THINKING_BUDGET = int(os.getenv("SUPPORT_THINKING_BUDGET", "64"))


def run_support_model(user_message: str, context: list[dict], tool_results: list[dict]) -> str:
    """Generate the final support answer through llama-server."""
    evidence = {"context": context, "tool_results": tool_results}
    payload = json.dumps(
        {
            "model": LLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise technical support specialist. "
                        "Use the supplied evidence and never invent tool results."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Request: {user_message}\nEvidence: {json.dumps(evidence)}",
                },
            ],
            "temperature": 0.1,
            "max_tokens": MAX_TOKENS,
            "thinking_budget_tokens": THINKING_BUDGET,
            "chat_template_kwargs": {"enable_thinking": THINKING_BUDGET > 0},
        }
    ).encode("utf-8")
    request = Request(
        LLAMA_SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read())
        return result["choices"][0]["message"]["content"].strip()
    except URLError as error:
        raise ConnectionError(f"Cannot connect to llama-server at {LLAMA_SERVER_URL}.") from error
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("llama-server returned an unexpected response.") from error
