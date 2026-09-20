# Multi-Agent Technical Support

The repository is organized by component for code and by artifact type for
data, trained weights, and evaluation results.

```text
src/
├── model_a/     # intent classifier
├── model_b/     # extractive QA
├── model_c/     # LoRA support specialist
├── routing/     # rules, LLM router, hybrid policy, and router evaluation
└── tools/       # 13 structured domain tools

data/            # datasets grouped by model/component
models/          # trained artifacts grouped by model
results/         # evaluation outputs grouped by model/component
tests/           # tests grouped by component
notebooks/       # archived exploratory notebooks
```

Each Model C run keeps only two result files: `report.json` for automated
metrics, configuration, diagnostics, and training history; and `review.json`
for the standard and concise Golden Set reviews.

## Commands

Run the application:

```bash
uv run python main.py "Check the live status of the database"
```

The request now runs through the LangGraph workflow. To enable Langfuse traces,
set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env`.

Train each model:

```bash
uv run python -m src.model_a.train
uv run python -m src.model_b.train
uv run python -m src.model_c.train
```

Evaluate the routers while `llama-server` is running:

```bash
uv run python -m src.routing.evaluate
```

Run the automated tests:

```bash
uv run python -m unittest discover -s tests -v
```

Inspect the available tool contracts:

```bash
uv run python -c 'from src.tools import ALL_TOOLS; print([tool.name for tool in ALL_TOOLS])'
```

## API and Open WebUI

Run the OpenAI-compatible API locally:

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Its model ID is `tuwaiq-tech-support-agent`. Test it with:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-demo-key" \
  -d '{"model":"tuwaiq-tech-support-agent","messages":[{"role":"user","content":"According to the docs, how long do access tokens last?"}]}'
```

Run the API and Open WebUI together:

```bash
docker compose up --build
```

Open WebUI will be available at `http://localhost:3001`. For Dokploy, create a
Docker Compose application, add the variables from `.env.example`, mount the
`models` and `data` directories, expose port 8000 with HTTPS, and point Open
WebUI to the API base URL ending in `/v1`.

## Run on macOS

The final Model A, Model B, and Model C adapter artifacts are stored with Git
LFS. Training checkpoints and GGUF base models are intentionally excluded.

Install Git LFS before cloning, then prepare the project:

```bash
brew install git-lfs
git lfs install
git clone https://github.com/SalehAlomair/multi-agent-teq-support.git
cd multi-agent-teq-support
git lfs pull
cp .env.example .env
```

To reuse the Qwen llama-server running on the Windows PC, keep Tailscale and
the Windows startup script running and set this value in the Mac's `.env`:

```dotenv
LLAMA_SERVER_DOCKER_URL=http://100.79.62.113:8080/v1/chat/completions
```

Verify that the Mac can reach Qwen before starting the application:

```bash
curl http://100.79.62.113:8080/v1/models
```

Run the complete readiness check:

```bash
./scripts/check-mac-ready.sh
```

Then start the complete application:

```bash
docker compose up --build -d
docker compose ps
```

Open `http://localhost:3001`. The first account created in this fresh Open
WebUI volume is the administrator.
