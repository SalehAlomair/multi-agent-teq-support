# Part 3/9 — Phase A / Model B — Technical Extractive QA

Fine-tuning form: Extractive Question Answering. Purpose: answer exact technical questions from trusted support documentation
instead of generating unsupported text.

## Long-context QA preprocessing
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_B = "distilbert-base-uncased"
tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
model_b = AutoModelForQuestionAnswering.from_pretrained(MODEL_B)

MAX_LENGTH = 384
DOC_STRIDE = 96

def prepare_qa_features(examples):
    tokenized = tokenizer_b(
        [q.strip() for q in examples["question"]],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offsets = tokenized.pop("offset_mapping")
    start_positions, end_positions = [], []

    for feature_index, feature_offsets in enumerate(offsets):
        input_ids = tokenized["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer_b.cls_token_id)
        sequence_ids = tokenized.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]

        answer_start = examples["answer_start"][sample_index]
        answer_end = answer_start + len(examples["answer_text"][sample_index])

        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        if (
            feature_offsets[context_start][0] > answer_start
            or feature_offsets[context_end][1] < answer_end
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        token_start = context_start
        while feature_offsets[token_start][1] <= answer_start:
            token_start += 1

        token_end = context_end
        while feature_offsets[token_end][0] >= answer_end:
            token_end -= 1

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

# TODO(student 2): create a small SQuAD-style technical-support dataset
# with trusted contexts from the supplied KB/docs and at least 30 QA pairs.

from transformers import TrainingArguments, Trainer

args_b = TrainingArguments(
    output_dir="models/qa_model",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)

trainer_b = Trainer(
    model=model_b,
    args=args_b,
    train_dataset=qa_train_features,
    eval_dataset=qa_val_features,
)
trainer_b.train()
trainer_b.save_model("models/qa_model")
tokenizer_b.save_pretrained("models/qa_model")
```

**Acceptance gate — Model B** — Evaluate with Exact Match and token-level F1 on held-out QA. Recommended gate: EM ≥ 0.65
and token F1 ≥ 0.80, plus manual inspection of long-context boundary cases.
