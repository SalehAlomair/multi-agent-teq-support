# Part 1/9 — Project Overview & System Design
(Tuwaiq Weekend Challenge: Multi-Model Agentic AI Technical Support System)

## 1. Project Mission
Build an AI Technical Support Agent that combines three specialist models, a routing layer, a tool-using LangGraph agent,
observability, and a deployable OpenAI-compatible API. The final system must decide when to classify, when to extract an answer
from trusted technical context, when to reason/generate with the instruction-tuned specialist, and when to call tools or escalate to
a human.

Individual · 5–7 hours · Open WebUI · OpenAI-compatible API · Langfuse · Docker · Dokploy · ≤15% student TODOs

### 1.1 What makes this a thinking challenge?
- Justify the intent taxonomy instead of accepting arbitrary labels.
- Compare a deterministic/classifier router with an LLM router, then define a hybrid fallback policy.
- Each fine-tuned model has different acceptance metrics; one global metric is not enough.
- A model that improves average score can still fail the project if it regresses on required Golden Set cases.
- The agent must decide when a tool is more reliable than a model answer.
- Deployment is accepted only when Open WebUI talks to the system through a valid OpenAI-compatible endpoint.

### 1.2 Learning outcomes
- Design a small multi-model AI system and document component boundaries.
- Fine-tune three models using three different task formulations.
- Evaluate before/after fine-tuning using task-appropriate metrics and real training logs.
- Create quality gates using baseline comparison, error analysis, Golden Set, and regression checks.
- Implement and compare deterministic/classifier routing against LLM routing.
- Build a LangGraph workflow that can invoke domain tools and specialist models.
- Expose the system through /v1/chat/completions and connect Open WebUI.
- Instrument the end-to-end workflow using Langfuse and deploy with Docker/Dokploy.

### 1.3 Recommended 5–7 hour timebox
| Block | Target time | Outcome |
|---|---|---|
| System design + data prep | 35–45 min | Architecture, labels, datasets, acceptance metrics |
| Model A: intent classifier | 45–55 min | Fine-tuned classifier + metrics |
| Model B: extractive QA | 45–55 min | QA model + EM/F1 |
| Model C: instruction specialist | 55–70 min | SFT + LoRA/QLoRA + eval |
| Router + tools + LangGraph | 65–80 min | Working multi-model agent |
| Langfuse + API + Docker | 45–60 min | Observable OpenAI-compatible service |
| Dokploy + Open WebUI + E2E QA | 35–50 min | Deployed demo + final report |

## 2. System Design Before Coding

### Architecture (text description)
Open WebUI → OpenAI-Compatible FastAPI (POST /v1/chat/completions) → LangGraph Orchestrator + Langfuse Trace →
Router A (Rules + Fine-tuned Classifier) & Router B (Small LLM Router) → Hybrid Routing Decision (confidence + fallback + policy) →
Model A (Intent Classifier), Model B (Technical Extractive QA), Model C (Support Specialist, SFT + LoRA/QLoRA).
Tool Layer: KB Search · Ticket Search/Create · Health Check · Log Analyzer · Docs Search · Package Lookup · SQL · Calculator ·
File Search · Web Search · Escalation · Diagnostic Runbook.

### 2.1 Functional requirements
- Classify incoming support requests into a stable intent taxonomy.
- Answer grounded documentation questions using extractive QA when an exact answer exists in trusted context.
- Handle multi-step troubleshooting and response synthesis using the instruction-tuned support specialist.
- Invoke technical tools when live/system information is required.
- Create or search tickets, inspect logs, query system health, and escalate unresolved issues.
- Expose a single OpenAI-compatible model surface to Open WebUI even though multiple internal models are used.

### 2.2 Non-functional requirements
| Concern | Minimum requirement |
|---|---|
| Latency | Record end-to-end latency and router latency |
| Traceability | Each request must carry a trace/request ID |
| Grounding | Documentation answers should prefer trusted context over free generation |
| Reliability | Required Golden Set cases must pass the project quality gate |
| Safety | Read-only SQL by default; explicit escalation for unsupported/high-risk cases |
| Deployment | One Dockerized service stack deployable via Dokploy |
| Observability | Langfuse trace for routing, tool calls, specialist selection, final response |

### 2.3 Required design decision record
Before training, create a short DESIGN_DECISIONS.md with:
- Intent classes and why they are separable.
- Classifier confidence threshold.
- When QA should win over generation.
- LoRA/QLoRA rank and target modules.
- Quality gate thresholds.
- Which tools are allowed automatically vs require escalation.
- Fallback order between rules/classifier and LLM router.

### 2.4 Suggested project structure
```
technical-support-agent/
├── data/
│   ├── intents.csv
│   ├── qa_train.json
│   ├── sft_train.jsonl
│   ├── golden_set.jsonl
│   └── mock_support.db
├── models/
│   ├── intent_classifier/
│   ├── qa_model/
│   └── support_adapter/
├── src/
│   ├── config.py
│   ├── routers.py
│   ├── specialists.py
│   ├── tools.py
│   ├── graph.py
│   ├── observability.py
│   └── api.py
├── tests/
│   ├── test_routing.py
│   ├── test_tools.py
│   └── test_golden_set.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
