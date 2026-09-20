# Part 10/10 — Test Scenarios, Deliverables, Rubric, TODOs, Checklist

## 10. End-to-End Test Scenarios
| Scenario | Expected route / behavior |
|---|---|
| "According to the deployment guide, which port must be exposed?" | KB retrieval → Model B QA → grounded answer |
| "My PostgreSQL pool is at 98% and requests time out." | Classifier → tools/runbook → Model C synthesis |
| "What is QLoRA?" | General/support specialist → Model C |
| "Production database may be corrupted after a failed migration." | Hard rule → escalate_to_human |
| "The API returns 503 after deployment; health says database degraded." | Tools → diagnostic runbook → Model C → possible ticket |
| Ambiguous request with classifier confidence below threshold | LLM router fallback |

### 10.1 Minimum automated tests
```python
def test_required_routes():
    cases = [
        ("Production database corruption suspected", "escalate"),
        ("According to the docs, what port is used?", "qa"),
    ]
    for text, expected in cases:
        assert baseline_router(text)["route"] == expected

def test_health_tool():
    result = system_health_check.invoke({"service": "api"})
    assert result["status"] in {"healthy", "degraded", "down", "unknown"}

# TODO(student 9): add one regression test from an error discovered during your own run.
```

## 11. Student Deliverables
- DESIGN_DECISIONS.md with architecture rationale and router policy.
- Three fine-tuned model artifacts or adapter references plus training/evaluation report.
- A comparison table: baseline vs fine-tuned for all three models.
- Router comparison: rules/classifier vs LLM router vs chosen hybrid.
- Golden Set and regression results.
- LangGraph agent with all 13 tool contracts and at least 6 functional implementations.
- Langfuse trace screenshot or trace URL/reference.
- FastAPI OpenAI-compatible endpoint.
- Dockerfile + docker-compose.yml + .env.example.
- Dokploy deployment URL and Open WebUI demo.
- README with exact run instructions and one architecture diagram.

## 12. Rubric — 100 Points
| Area | Points | What strong work looks like |
|---|---|---|
| System design | 12 | Clear boundaries, data flow, fallback policy, deployment architecture |
| Model A — classifier | 10 | Sound labels, baseline, macro metrics, confusion analysis |
| Model B — QA | 10 | Correct QA preprocessing, EM/F1, long-context handling |
| Model C — SFT/PEFT | 12 | Good instruction data, LoRA/QLoRA rationale, before/after evaluation |
| Evaluation & quality gates | 14 | Golden Set, regression checks, task metrics, real logs |
| Router experiment | 10 | Measured comparison and justified hybrid policy |
| Tools + LangGraph | 14 | Correct tool contracts, robust orchestration, escalation |
| Langfuse observability | 5 | Useful trace metadata and visible node/tool execution |
| API + Docker + Dokploy + Open WebUI | 9 | Working deployment and OpenAI-compatible integration |
| Code quality & documentation | 4 | Readable structure, env handling, reproducible instructions |

### 12.1 Automatic quality deductions
- −10: No baseline measurement before fine-tuning.
- −10: Deploying a model that fails a required Golden Set case without documenting the exception.
- −8: Open WebUI connects directly to a specialist model instead of the unified router/API.
- −8: Tools return unstructured free text only and cannot be traced.
- −5: Secrets/API keys committed to the repository.
- −5: No reproducible Docker run path.

### 12.2 Bonus — up to +10
- +3: MRR/Recall@K evaluation for KB retrieval with a small labeled retrieval set.
- +2: Confidence calibration or threshold sweep for Model A/router.
- +2: Streaming support for /v1/chat/completions.
- +2: Parallel tool calls or recovery path in LangGraph.
- +1: Resource/latency dashboard from Langfuse or exported traces.

## 13. Intentionally Missing Student Code (≤15%)
The guide deliberately leaves a small number of implementation decisions incomplete. These are the graded thinking points, not
missing infrastructure.

| TODO | Student responsibility |
|---|---|
| 1 | Build/clean intent dataset and stratified splits. |
| 2 | Create technical QA dataset. |
| 3 | Create SFT conversations with escalation/tool-result examples. |
| 4 | Add project-specific Golden Set cases. |
| 5 | Complete classifier confidence policy and intent→route mapping. |
| 6 | Add LLM-router few-shot examples and JSON validation. |
| 7 | Fully implement one additional domain tool; mocks allowed for remaining listed backends. |
| 8 | Implement measured hybrid LangGraph routing condition. |
| 9 | Add one regression test discovered from the student's own error analysis. |

Rule — Do not replace these TODOs with random boilerplate. Each TODO exists because it forces a design or evaluation
decision.

## 14. Final Submission Checklist
- [ ] Three distinct fine-tuning tasks completed.
- [ ] Baseline and post-FT metrics recorded for each model.
- [ ] Model quality gates documented.
- [ ] Router comparison completed.
- [ ] Golden Set + regression checks completed.
- [ ] LangGraph invokes specialists/tools correctly.
- [ ] Langfuse trace shows routing + tool/model calls.
- [ ] /v1/models and /v1/chat/completions work.
- [ ] Docker Compose runs from a clean environment.
- [ ] Dokploy deployment responds over HTTPS.
- [ ] Open WebUI connects to the unified agent endpoint.
- [ ] README contains architecture, setup, tests, and known limitations.

## Appendix A — Alignment with This Week's Notebooks
| Course concept | Applied in project |
|---|---|
| Instruction data + chat templates | Model C SFT dataset and inference formatting |
| LoRA / target modules | Model C PEFT training |
| QLoRA / NF4 / compute dtype | Optional CUDA path for Model C |
| Adapter save/reload | Deployable Model C artifact |
| Evaluation / quality gates | All models and end-to-end system |
| Golden Set / regression | Release gate |
| Accelerate / deployment mindset | Docker/Dokploy production path |
| OpenAI-compatible API | Open WebUI integration |

End of project brief. Build the smallest system that proves the architecture, measures the models honestly, and fails safely when
confidence is insufficient.
