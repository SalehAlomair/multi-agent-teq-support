"""Inference helpers for the trained Model A intent classifier."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.paths import MODELS_ROOT


MODEL_DIR = MODELS_ROOT / "model_a" / "intent_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


@torch.no_grad()
def predict_intent(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    logits = model(**inputs).logits[0]
    probabilities = torch.softmax(logits, dim=-1)
    index = int(torch.argmax(probabilities))
    return {
        "intent": model.config.id2label[index],
        "confidence": float(probabilities[index]),
        "source": "classifier",
    }

