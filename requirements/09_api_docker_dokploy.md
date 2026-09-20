# Part 9/9 — Phase F & G — OpenAI-Compatible FastAPI + Docker + Dokploy + Open WebUI

## Phase F — FastAPI
Open WebUI should see the entire multi-model system as one model. Internally, the API calls the LangGraph system and returns
an OpenAI-style chat completion response.

```python
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import time, uuid

app = FastAPI(title="Tuwaiq Technical Support Agent")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "tuwaiq-tech-support-agent"
    messages: List[ChatMessage]
    temperature: float = 0.2
    stream: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{
            "id": "tuwaiq-tech-support-agent",
            "object": "model",
            "owned_by": "student",
        }],
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, authorization: Optional[str] = Header(default=None)):
    if req.stream:
        raise HTTPException(400, "Streaming is not required for this challenge")

    user_messages = [m.content for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(400, "No user message provided")

    started = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    result = graph.invoke({
        "user_message": user_messages[-1],
        "trace_id": request_id,
    })
    answer = result.get("answer", "Unable to produce an answer.")

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "system_metadata": {
            "route": result.get("route"),
            "intent": result.get("intent"),
            "latency_seconds": round(time.time() - started, 4),
        },
    }
```

## Phase G — Docker, Dokploy, Open WebUI

### requirements.txt
```
fastapi
uvicorn[standard]
torch
transformers
datasets
peft
trl
accelerate
bitsandbytes; platform_system == "Linux"
langgraph
langchain-core
langfuse
sentence-transformers
scikit-learn
pandas
numpy
psycopg[binary]
```

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY data ./data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
services:
  support-agent:
    build: .
    container_name: support-agent
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: support
      POSTGRES_USER: support
      POSTGRES_PASSWORD: support
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - openwebui:/app/backend/data
    depends_on:
      - support-agent
    restart: unless-stopped

volumes:
  pgdata:
  openwebui:
```

### .env.example
```
LANGFUSE_PUBLIC_KEY=replace_me
LANGFUSE_SECRET_KEY=replace_me
LANGFUSE_HOST=https://cloud.langfuse.com
DATABASE_URL=postgresql://support:support@postgres:5432/support
OPENAI_API_KEY=local-demo-key
```

### Local verification
```bash
docker compose up --build

curl http://localhost:8000/health
curl http://localhost:8000/v1/models

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-demo-key" \
  -d '{
    "model": "tuwaiq-tech-support-agent",
    "messages": [{"role": "user", "content": "My API returns 503 after deployment."}]
  }'
```

### Dokploy deployment checklist
1. Push the repository to a Git provider or make it available to Dokploy.
2. Create a Docker Compose application in Dokploy.
3. Add environment variables and secrets; never commit private keys.
4. Expose the support-agent service on a domain with HTTPS.
5. Expose Open WebUI separately or keep it private behind authentication.
6. Run /health and /v1/models after deployment.
7. Point Open WebUI to the deployed OpenAI-compatible base URL ending in /v1.
8. Use the configured API key when Open WebUI requests authentication.
9. Send at least three test prompts and confirm Langfuse traces appear.

**Open WebUI target** — The UI should present one model: tuwaiq-tech-support-agent. Users should not need to know which
specialist model or tool handled the request.
