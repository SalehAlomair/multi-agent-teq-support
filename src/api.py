"""OpenAI-compatible API for the complete support agent."""

import json
import os
import time
import uuid
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent import run_agent


MODEL_NAME = "tuwaiq-tech-support-agent"
API_KEY = os.getenv("OPENAI_API_KEY", "local-demo-key")
app = FastAPI(title="Tuwaiq Technical Support Agent")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    temperature: float = 0.2
    stream: bool = False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "saleh",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid API key")
    if request.model != MODEL_NAME:
        raise HTTPException(404, "Model not found")

    user_messages = [message.content for message in request.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(400, "No user message provided")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        async def events():
            result = run_agent(user_messages[-1])
            answer = result.get("answer", "Unable to produce an answer.")
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": answer},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            chunk["choices"][0]["delta"] = {}
            chunk["choices"][0]["finish_reason"] = "stop"
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    started = time.time()
    result = run_agent(user_messages[-1])
    answer = result.get("answer", "Unable to produce an answer.")

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "system_metadata": {
            "route": result.get("route"),
            "intent": result.get("intent"),
            "trace_id": result.get("trace_id"),
            "latency_seconds": round(time.time() - started, 4),
        },
    }
