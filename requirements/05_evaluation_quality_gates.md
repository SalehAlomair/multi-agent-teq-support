# Part 5/9 — Phase B — Evaluation and Quality Gates

No model is accepted because "training finished." Each model must beat a baseline and pass task-specific quality gates. Use real
trainer logs when possible; do not fabricate loss curves in the final submission.

| Component | Primary metrics | Required diagnostic |
|---|---|---|
| Model A — Intent | Accuracy, macro Precision/Recall/F1 | Confusion matrix + per-class recall |
| Model B — QA | Exact Match, token F1 | Long-context error cases |
| Model C — SFT/LoRA | Eval loss, perplexity, ROUGE where applicable | Golden Set + error categories |
| Retrieval tools | MRR, Recall@K | Failed-query inspection |
| Router | Routing accuracy, fallback rate, latency | Wrong-route analysis |
| End-to-end agent | Task success, latency, escalation rate | Regression / Golden Set |

## Evaluation log pattern
```python
import math
import pandas as pd

def perplexity_from_loss(loss):
    return math.exp(loss) if loss < 20 else float("inf")

# Always keep a baseline record BEFORE fine-tuning.
model_report = {
    "baseline": {},
    "fine_tuned": {},
    "quality_gate": {},
}

# Real Trainer logs:
history_df = pd.DataFrame(trainer_c.state.log_history)
train_logs = history_df[history_df["loss"].notna()].copy() if "loss" in history_df.columns else pd.DataFrame()
eval_logs = history_df[history_df["eval_loss"].notna()].copy() if "eval_loss" in history_df.columns else pd.DataFrame()

# Golden Set: critical behaviors, not just average metrics.
golden_set = [
    {
        "id": "G01",
        "category": "grounding",
        "prompt": "Answer only from the supplied KB. If missing, say you cannot verify.",
        "required": True,
    },
    {
        "id": "G02",
        "category": "escalation",
        "prompt": "Production database may be corrupted. What should you do?",
        "required": True,
    },
    {
        "id": "G03",
        "category": "instruction_following",
        "prompt": "Return exactly three troubleshooting steps.",
        "required": True,
    },
]

# TODO(student 4): add at least 7 more project-specific Golden Set cases
# derived from expected production failures and the errors you observed.
```

**Quality gate rule** — If a fine-tuned model improves an average score but fails a required Golden Set case that the baseline
passed, treat it as a regression and investigate before deployment.
