# MacBook Handoff — Multi-Agent Technical Support

This document is the complete handoff for continuing this project on the MacBook.
Give this file to the next AI before it changes anything.

## 1. User preference

- Keep the implementation simple.
- Do not add abstractions or extra files unless they are necessary.
- Explain commands before asking the user to run unfamiliar ones.
- Never commit `.env`, passwords, Langfuse keys, API keys, or other secrets.

## 2. Project locations

- WSL project: `/home/saleh/Projects/multi-agent-teq-support`
- GitHub remote: `https://github.com/SalehAlomair/multi-agent-teq-support`
- Branch: `main`
- Windows helper script in the repository: `scripts/start-windows.ps1`
- Separate Windows copy should be: `C:\Scripts\start-support-agent.ps1`
- Do not overwrite the user's original `C:\Scripts\start-qwen.ps1`.

## 3. Git LFS model storage

The two incorrect unpushed commits that contained raw model binaries were
removed locally. Git LFS 3.4.1 was installed and the project was rebuilt from
`origin/main`. The staged Git objects for all three weights were verified as
small LFS pointers rather than the full binaries.

`git lfs ls-files` must list these three files:

```text
models/model_a/intent_classifier/model.safetensors
models/model_b/qa_model/model.safetensors
models/model_c/support_adapter_v2/adapter_model.safetensors
```

The final runtime artifacts total about 533 MB. Training checkpoints are
excluded. If GitHub reports an LFS quota problem, do not put the large binaries
back into normal Git; use a private Hugging Face repository or another artifact
store instead.

## 4. Git status

All portability changes, data, reports, final model artifacts, scripts, and
this handoff belong in the single `Prepare project for macOS` commit. Always run
`git status --short` before making later changes.

## 5. System architecture

```text
Browser
  -> Open WebUI on port 3001
  -> support-agent OpenAI-compatible API on port 8000
  -> LangGraph workflow
       -> Router A: rules + local fine-tuned Model A classifier
       -> Router B: remote Qwen through llama-server
       -> QA route: local Model B extractive QA
       -> Tools route: 13 local domain tools
       -> Support route: remote Qwen through llama-server
       -> Escalation route: local ticket creation
```

Important details:

- API model ID: `tuwaiq-tech-support-agent`
- Model A is loaded from `models/model_a/intent_classifier`.
- Model B is loaded lazily from `models/model_b/qa_model`.
- The tracked Model C adapter is in `models/model_c/support_adapter_v2`.
- The current production workflow does not load the Model C LoRA adapter.
- Router B and the final support specialist currently use the Windows
  `llama-server` running Qwen3.8-27B.
- The Model C adapter was trained for `Qwen/Qwen3-4B-Instruct-2507`; it cannot be
  attached directly to the Qwen3.8-27B GGUF because they are different bases.

## 6. Windows Qwen server

The large Qwen model is intentionally not stored in Git.

Current Windows/Tailscale details:

- Windows Tailscale IP: `100.79.62.113`
- Qwen endpoint: `http://100.79.62.113:8080/v1`
- Chat endpoint: `http://100.79.62.113:8080/v1/chat/completions`
- Open WebUI over Tailscale: `http://100.79.62.113:3001`

The WSL script outside the repository is `~/run-qwen38.sh`. It starts:

```text
llama-server
model: /mnt/e/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf
host: 0.0.0.0
port: 8080
GPU layers: 60
context: 16384
parallel slots: 1
```

The repository contains `scripts/start-windows.ps1`. The Windows copy is made
without touching the old script:

```powershell
Copy-Item "\\wsl$\Ubuntu\home\saleh\Projects\multi-agent-teq-support\scripts\start-windows.ps1" "C:\Scripts\start-support-agent.ps1" -Force
```

Run it from an Administrator PowerShell:

```powershell
C:\Scripts\start-support-agent.ps1
```

It performs these actions:

1. Starts/checks Tailscale.
2. Stops an old `llama-server` process.
3. Finds the current dynamic WSL IP.
4. Removes the old port `3000` proxy.
5. Proxies Windows ports `8080` and `3001` to WSL.
6. Runs `docker compose up -d` in the project.
7. Runs `~/run-qwen38.sh` in the foreground.
8. Checks Tailscale every five minutes in a background job.
9. If disconnected, starts the Tailscale service/GUI and runs `tailscale up`.

The PowerShell window must remain open while Qwen is in use.

Warnings about CORS, manually specified GPU layers, and unused `blk.64` tensors
have not prevented this GGUF from running. Wait until llama-server reports that
the model is loaded/listening before testing it.

## 7. Why Open WebUI uses port 3001

There was an older or stale Open WebUI instance/session on port `3000`. It
caused all of these symptoms:

- A user appeared logged in although the project database contained zero users.
- Signup appeared disabled.
- Open WebUI showed `No models available`.
- The browser returned a different `/api/config` than the project container.

The project was moved to port `3001` to avoid that collision. Do not change it
back to `3000` unless the old instance and browser state are deliberately
removed.

The correct local URL on Windows and Mac is:

```text
http://localhost:3001
```

The first user created in a fresh Open WebUI volume becomes the administrator.

## 8. Mac prerequisites

Install:

- Docker Desktop
- Tailscale
- Homebrew
- Git LFS
- Git

Commands:

```bash
brew install git-lfs
git lfs install
```

The Mac and Windows PC must be logged into the same Tailnet. The Windows PC must
remain powered on with `C:\Scripts\start-support-agent.ps1` running because the
Mac uses the Windows Qwen model remotely.

## 9. Clone and prepare on the Mac

After the repaired commit has been pushed:

```bash
git clone https://github.com/SalehAlomair/multi-agent-teq-support.git
cd multi-agent-teq-support
git lfs pull
cp .env.example .env
```

Confirm the LFS files were downloaded and are not small text pointer files:

```bash
git lfs ls-files
du -h models/model_a/intent_classifier/model.safetensors
du -h models/model_b/qa_model/model.safetensors
du -h models/model_c/support_adapter_v2/adapter_model.safetensors
```

Expected approximate sizes are 268 MB, 265 MB, and 12 MB.

After configuring `.env`, run the automated readiness check:

```bash
./scripts/check-mac-ready.sh
```

It verifies Git LFS, all three final artifacts, Docker Compose configuration,
and connectivity to the remote Qwen `/v1/models` endpoint. Do not start the app
until it prints `READY`.

## 10. Mac `.env`

Never commit `.env`. Start from `.env.example` and set real random values.

For the Mac using the Windows Qwen server, use:

```dotenv
LLAMA_SERVER_URL=http://100.79.62.113:8080/v1/chat/completions
LLAMA_SERVER_DOCKER_URL=http://100.79.62.113:8080/v1/chat/completions
LLAMA_MODEL=local-qwen

ROUTER_MAX_TOKENS=100
ROUTER_THINKING_BUDGET=0
SUPPORT_MAX_TOKENS=256
SUPPORT_THINKING_BUDGET=64

OPENAI_API_KEY=GENERATE_A_STRONG_RANDOM_VALUE
WEBUI_SECRET_KEY=GENERATE_A_DIFFERENT_STRONG_RANDOM_VALUE
OPEN_WEBUI_PORT=3001
```

Generate secrets locally, for example:

```bash
openssl rand -hex 32
openssl rand -hex 32
```

Langfuse is optional. If used, copy the user's real Langfuse values into `.env`
without showing or committing them.

## 11. Verify Windows connectivity from the Mac

Before starting Docker on the Mac:

```bash
tailscale status
curl http://100.79.62.113:8080/v1/models
```

If this fails:

1. Confirm the Windows PowerShell script is still running.
2. Confirm llama-server finished loading.
3. Confirm both computers are connected to Tailscale.
4. Check Windows Firewall access for TCP port `8080` on the Tailscale network.
5. Do not expose port `8080` through the public internet.

## 12. Start the application on the Mac

```bash
docker compose up --build -d
docker compose ps
```

Expected services:

- `support-agent`: healthy, port `8000`
- `open-webui`: healthy, port `3001`

Test the API:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Expected API model:

```text
tuwaiq-tech-support-agent
```

Open:

```text
http://localhost:3001
```

Create the first Open WebUI account. It should become `admin` automatically.

## 13. Test a full request

Replace `YOUR_OPENAI_API_KEY` with the value from the Mac `.env`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY" \
  -d '{"model":"tuwaiq-tech-support-agent","messages":[{"role":"user","content":"Check the live status of the database"}]}'
```

The expected answer should mention the deterministic local tool result: the
database is degraded and connection usage is 91%.

## 14. Useful Docker commands

```bash
docker compose ps
docker compose logs --tail=100 support-agent
docker compose logs --tail=100 open-webui
docker compose restart
docker compose down
```

Do not use `docker compose down -v` unless the user explicitly wants to delete
the Open WebUI database, accounts, and settings.

## 15. Troubleshooting

### Open WebUI says `No models available`

First verify the API itself:

```bash
curl http://localhost:8000/v1/models
```

Then verify from inside Open WebUI's Docker network:

```bash
docker exec multi-agent-teq-support-open-webui-1 \
  curl -sS http://support-agent:8000/v1/models
```

Open WebUI's connection must use:

```text
http://support-agent:8000/v1
```

The API key must match `OPENAI_API_KEY` in `.env`. The Compose configuration
already supplies both values automatically.

### Open WebUI shows an old account or signup is disabled

- Confirm the URL uses port `3001`, not `3000`.
- Use a private browser window.
- Clear site data/local storage for the old origin.
- Check `http://localhost:3001/api/config`.
- A fresh instance should show `"onboarding": true` before the first account is
  created.

### Qwen calls time out

```bash
curl http://100.79.62.113:8080/v1/models
docker compose logs --tail=100 support-agent
```

Router B and support responses cannot work without the remote llama-server.
Rule-only escalation and some local routes may still work.

### Model files are missing after cloning

```bash
git lfs install
git lfs pull
git lfs ls-files
```

Do not retrain immediately; first confirm that the Git LFS objects were pushed
successfully from WSL.

### Apple Silicon Docker build fails on PyTorch

The Docker image is intended to build natively. If the upstream PyTorch CPU
wheel does not support the Mac's Docker architecture, the next AI may add
`platform: linux/amd64` to the services as a fallback. This uses emulation and
will be slower, so do it only after seeing an architecture-specific build error.

## 16. Validation already completed

Before the latest port and handoff edits:

- `docker compose config --quiet` passed.
- All 23 Python unit tests passed.
- `support-agent` was healthy.
- Open WebUI 0.11.3 was healthy.
- `support-agent` returned `tuwaiq-tech-support-agent` from `/v1/models`.
- Open WebUI could reach `http://support-agent:8000/v1/models` from inside its
  container.
- A complete database-status request previously returned the local deterministic
  result and produced a Langfuse trace.

After any new changes, rerun:

```bash
docker compose config --quiet
UV_CACHE_DIR=/tmp/multi-agent-uv-cache uv run python -m unittest discover -s tests -v
```

## 17. Files that must stay out of Git

- `.env`
- `.venv/`
- full training checkpoints
- `*.gguf`
- `data/tools/*.db`
- passwords and API keys
- the 27B Qwen GGUF

Only the three final `.safetensors` artifacts should use Git LFS.

## 18. First actions for the next AI

1. Read this entire file.
2. Run `git status -sb` and `git log --oneline -3`.
3. Confirm Git LFS is installed and run `git lfs pull`.
4. Run `./scripts/check-mac-ready.sh` after configuring `.env`.
5. Test the Windows Qwen endpoint through Tailscale.
6. Start Docker and verify ports `8000` and `3001`.
7. Keep the solution simple and preserve unrelated user work.
