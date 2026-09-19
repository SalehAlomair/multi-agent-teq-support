"""Train and evaluate the Model B extractive question-answering model.

Converted from the MODEL B section of testing.ipynb.
"""

# MODEL B

import json
import re

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.paths import DATA_ROOT, MODELS_ROOT

DATA_PATH = DATA_ROOT / "model_b" / "qa.jsonl"
MODEL_DIR = MODELS_ROOT / "model_b" / "qa_model"
CHECKPOINT_DIR = MODELS_ROOT / "model_b" / "checkpoints"

MODEL_B = "distilbert-base-uncased-distilled-squad"
tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
model_b = AutoModelForQuestionAnswering.from_pretrained(MODEL_B)

MAX_LENGTH = 384
DOC_STRIDE = 96

raw_qa_data = [
    json.loads(line)
    for line in DATA_PATH.read_text().splitlines()
    if line.strip()
]

# Flatten into SQuAD-style records with computed answer_start (no manual counting)
qa_dataset = []
for entry in raw_qa_data:
    context = entry["context"]
    for question, answer_text in entry["qas"]:
        start = context.find(answer_text)
        if start == -1:
            raise ValueError(f"Answer text not found in context: {answer_text!r}")
        qa_dataset.append({
            "question": question,
            "context": context,
            "answer_text": answer_text,
            "answer_start": start,
        })

print(f"Total QA pairs: {len(qa_dataset)}")

train_b, temp_b = train_test_split(
    qa_dataset,
    test_size=0.3,
    random_state=42,
)

val_b, test_b = train_test_split(
    temp_b,
    test_size=0.5,
    random_state=42
)

print(len(train_b), len(val_b), len(test_b))

def prepare_qa_features(examples):
    tokenized = tokenizer_b(
        [q.strip() for q in examples['question']],
        examples['context'],
        truncation='only_second',
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )

    sample_mapping = tokenized.pop('overflow_to_sample_mapping')
    offsets = tokenized.pop('offset_mapping')
    start_positions, end_positions = [], []

    for feature_index, feature_offsets in enumerate(offsets):
        input_ids = tokenized['input_ids'][feature_index]
        cls_index = input_ids.index(tokenizer_b.cls_token_id)
        sequence_ids = tokenized.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]

        answer_start = examples['answer_start'][sample_index]
        answer_end = answer_start + len(examples['answer_text'][sample_index])

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

    tokenized['start_positions'] = start_positions
    tokenized['end_positions'] = end_positions
    return tokenized

train_b = Dataset.from_list(train_b)
val_b = Dataset.from_list(val_b)
test_b = Dataset.from_list(test_b)

qa_train_features = train_b.map(prepare_qa_features, batched=True, remove_columns=train_b.column_names)
qa_val_features = val_b.map(prepare_qa_features, batched=True, remove_columns=val_b.column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer_b)

args_b = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    learning_rate=2e-5,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    num_train_epochs=15,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    report_to='none',
    logging_steps=-1
)

trainer_b = Trainer(
    model=model_b,
    args=args_b,
    train_dataset=qa_train_features,
    eval_dataset=qa_val_features,
    data_collator=data_collator
)

trainer_b.train()
trainer_b.save_model(str(MODEL_DIR))
tokenizer_b.save_pretrained(str(MODEL_DIR))

test_b_tokenized = tokenizer_b(
    [ex["question"].strip() for ex in test_b],
    [ex["context"] for ex in test_b],
    truncation="only_second",
    max_length=MAX_LENGTH,
    stride=DOC_STRIDE,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)

print(len(test_b_tokenized["input_ids"]))

model_b.eval()

all_start_logits = []
all_end_logits = []

device = next(model_b.parameters()).device
print(device)  # confirm it's mps

with torch.no_grad():
    for i in range(len(test_b_tokenized["input_ids"])):
        input_ids = torch.tensor([test_b_tokenized["input_ids"][i]]).to(device)
        attention_mask = torch.tensor([test_b_tokenized["attention_mask"][i]]).to(device)
        outputs = model_b(input_ids=input_ids, attention_mask=attention_mask)
        all_start_logits.append(outputs.start_logits[0].cpu())
        all_end_logits.append(outputs.end_logits[0].cpu())

print(len(all_start_logits), all_start_logits[0].shape)

def get_predicted_answer(feature_index, start_logits, end_logits, offsets, context):
    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))

    if start_idx > end_idx or start_idx >= len(offsets) or end_idx >= len(offsets):
        return ""

    start_char = offsets[start_idx][0]
    end_char = offsets[end_idx][1]

    if start_char == 0 and end_char == 0:
        return ""  # points to [CLS]/padding, meaning "no answer found"

    return context[start_char:end_char]

sample_mapping = test_b_tokenized["overflow_to_sample_mapping"]
offsets = test_b_tokenized["offset_mapping"]

predicted_answers = {}

for feature_index in range(len(all_start_logits)):
    sample_index = sample_mapping[feature_index]
    context = test_b[sample_index]["context"]

    pred_text = get_predicted_answer(
        feature_index,
        all_start_logits[feature_index],
        all_end_logits[feature_index],
        offsets[feature_index],
        context,
    )

    # If a sample was split into multiple features, keep the first non-empty prediction
    if sample_index not in predicted_answers or predicted_answers[sample_index] == "":
        predicted_answers[sample_index] = pred_text

print(list(predicted_answers.items())[:5])

def normalize(s):
    return re.sub(r"[^\w\s]", "", s.lower()).strip()

em_count = 0
f1_total = 0

for i, example in enumerate(test_b):
    pred = predicted_answers.get(i, "")
    truth = example["answer_text"]

    pred_norm = normalize(pred)
    truth_norm = normalize(truth)

    # exact match
    if pred_norm == truth_norm:
        em_count += 1

    # token f1
    pred_words = pred_norm.split()
    truth_words = truth_norm.split()
    common = sum(1 for w in pred_words if w in truth_words)

    if len(pred_words) == 0 or len(truth_words) == 0:
        f1 = 0
    else:
        precision = common / len(pred_words)
        recall = common / len(truth_words)
        f1 = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

    f1_total += f1

em = em_count / len(test_b)
f1_avg = f1_total / len(test_b)

print(f"Exact Match: {em:.4f}")
print(f"Token F1: {f1_avg:.4f}")

for i in range(min(10, len(test_b))):
    pred = predicted_answers.get(i, "")
    truth = test_b[i]["answer_text"]
    print(f"[{i}] pred={pred!r} | truth={truth!r}")

print(trainer_b.state.best_model_checkpoint)
print(trainer_b.state.best_metric)
