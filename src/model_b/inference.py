"""Run the local extractive QA model."""

from functools import lru_cache

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.paths import MODELS_ROOT


MODEL_DIR = MODELS_ROOT / "model_b" / "qa_model"


@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def run_qa_model(question: str, contexts: list[str]) -> str:
    """Extract an answer from retrieved context."""
    if not contexts:
        return "I could not find this in the trusted documentation."

    tokenizer, model = _load_model()
    inputs = tokenizer(
        question,
        "\n".join(contexts),
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
    )
    with torch.no_grad():
        output = model(**inputs)

    start = int(output.start_logits.argmax())
    end = int(output.end_logits.argmax())
    if end < start:
        return "I could not extract a reliable answer from the documentation."
    answer = tokenizer.decode(inputs["input_ids"][0][start : end + 1], skip_special_tokens=True)
    return answer or "I could not extract a reliable answer from the documentation."

