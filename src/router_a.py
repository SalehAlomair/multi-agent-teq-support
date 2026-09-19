"""Router A: rules and the fine-tuned intent classifier."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


router_tokenizer = AutoTokenizer.from_pretrained("models/intent_classifier")
router_model = AutoModelForSequenceClassification.from_pretrained(
    "models/intent_classifier"
)
router_model.eval()

RULES = {
    "escalate": ["data loss", "security breach", "production down", "corruption"],
    "qa": ["according to the docs", "documentation", "what does the manual say"],
}

CONFIDENCE_THRESHOLD = 0.80

INTENT_TO_ROUTE = {
    "authentication": "support_specialist",
    "network": "support_specialist",
    "deployment": "support_specialist",
    "database": "support_specialist",
    "gpu": "support_specialist",
    "api": "support_specialist",
    "package": "tools",
    "general": "qa",
}


def rule_first(text: str):
    lowered = text.lower()
    for route, phrases in RULES.items():
        if any(phrase in lowered for phrase in phrases):
            return {"route": route, "source": "rule", "confidence": 1.0}
    return None


@torch.no_grad()
def classifier_route(text: str):
    inputs = router_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    logits = router_model(**inputs).logits[0]
    probabilities = torch.softmax(logits, dim=-1)
    index = int(torch.argmax(probabilities))
    return {
        "intent": router_model.config.id2label[index],
        "confidence": float(probabilities[index]),
        "source": "classifier",
    }


def baseline_router(text: str):
    hard_rule = rule_first(text)
    if hard_rule:
        return hard_rule

    prediction = classifier_route(text)
    prediction["route"] = INTENT_TO_ROUTE.get(
        prediction["intent"],
        "support_specialist",
    )
    prediction["needs_fallback"] = (
        prediction["confidence"] < CONFIDENCE_THRESHOLD
    )
    return prediction
