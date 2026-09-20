# Part 6/9 — Phase C — Build and Compare Two Routers

## 5.1 Router A — Rules + Fine-tuned Classifier
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

router_tokenizer = AutoTokenizer.from_pretrained("models/intent_classifier")
router_model = AutoModelForSequenceClassification.from_pretrained("models/intent_classifier")
router_model.eval()

RULES = {
    "escalate": ["data loss", "security breach", "production down", "corruption"],
    "qa": ["according to the docs", "documentation", "what does the manual say"],
}

def rule_first(text: str):
    lowered = text.lower()
    for route, phrases in RULES.items():
        if any(p in lowered for p in phrases):
            return {"route": route, "source": "rule", "confidence": 1.0}
    return None

@torch.no_grad()
def classifier_route(text: str):
    inputs = router_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    logits = router_model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs))
    return {
        "intent": router_model.config.id2label[idx],
        "confidence": float(probs[idx]),
        "source": "classifier",
    }

def baseline_router(text: str):
    hard_rule = rule_first(text)
    if hard_rule:
        return hard_rule
    pred = classifier_route(text)
    # TODO(student 5): complete confidence policy and intent→specialist mapping.
    return pred
```

## 5.2 Router B — Small LLM Router
```python
ROUTER_SYSTEM = """
You are a routing controller for a technical support system.
Return JSON only.
Allowed routes: qa, support_specialist, tools, escalate.
Use qa for questions answerable from trusted documentation.
Use tools for live/system/ticket/log/package information.
Use support_specialist for troubleshooting synthesis.
Use escalate for high-risk or unresolved production incidents.
"""

def build_router_prompt(user_message: str):
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": user_message},
    ]

# Reuse the small instruction model in routing mode.
# TODO(student 6): add 2 few-shot examples and strict JSON validation.
```

## Router experiment comparison table
| Router experiment | Measure |
|---|---|
| Rules + classifier | Routing accuracy, macro F1 by route, latency, confidence calibration |
| LLM router | Routing accuracy, JSON-valid rate, latency, failure modes |
| Hybrid | Accuracy, fallback rate, end-to-end latency |

**Recommended production policy** — Hard safety/escalation rules → classifier when confidence is high → LLM router only for
ambiguous cases. Your final choice must be justified by measured results.
