# Part 2/9 — Phase A / Model A — Intent Classifier

Fine-tuning form: Sequence Classification. Purpose: classify the support request and provide a confidence score that can drive
routing.

| Suggested intent | Example |
|---|---|
| authentication | "I cannot sign in after resetting MFA." |
| network | "The API times out only from the office VPN." |
| deployment | "My Docker deployment exits after startup." |
| database | "PostgreSQL connections are exhausted." |
| gpu | "CUDA is available but training stays on CPU." |
| api | "The endpoint returns HTTP 422." |
| package | "Which version of transformers supports this?" |
| general | "Explain what LoRA is." |

Student design task — You may rename/add one class, but you must justify it using examples and avoid classes that overlap
heavily.

## Core training scaffold
```python
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_A = "distilbert-base-uncased"
LABELS = [
    "authentication", "network", "deployment", "database",
    "gpu", "api", "package", "general",
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
model_a = AutoModelForSequenceClassification.from_pretrained(
    MODEL_A,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label,
)

def tokenize_a(batch):
    return tokenizer_a(batch["text"], truncation=True, max_length=128)

def compute_cls_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }

# TODO(student 1): load/construct at least 12 examples per intent,
# then create stratified train/validation/test splits.

args_a = TrainingArguments(
    output_dir="models/intent_classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    report_to="none",
)

trainer_a = Trainer(
    model=model_a,
    args=args_a,
    train_dataset=train_a,
    eval_dataset=val_a,
    compute_metrics=compute_cls_metrics,
)
trainer_a.train()
trainer_a.evaluate(test_a)
trainer_a.save_model("models/intent_classifier")
tokenizer_a.save_pretrained("models/intent_classifier")
```

**Acceptance gate — Model A** — Recommended: macro F1 ≥ 0.80, no class recall < 0.60, and manual review of confusion
matrix. Adjust thresholds if your dataset is intentionally difficult.
