"""Train and evaluate the Model A intent classifier.

Converted from the MODEL A section of testing.ipynb.
"""

import json

from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.paths import DATA_ROOT, MODELS_ROOT

DATA_PATH = DATA_ROOT / "model_a" / "intents.jsonl"
MODEL_DIR = MODELS_ROOT / "model_a" / "intent_classifier"

MODEL_A = 'distilbert-base-uncased'
LABELS = [
    'authentication',
    'network',
    'deployment',
    'database',
    'gpu',
    'api',
    'package',
    'general',
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
model_a = AutoModelForSequenceClassification.from_pretrained(
    MODEL_A,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label
    )

def tokenize_a(batch):
    return tokenizer_a(batch['text'], truncation=True, max_length=128)

def compute_cls_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision_macro': p,
        'recall_macro': r,
        'f1_macro': f1,
    }

data = [
    json.loads(line)
    for line in DATA_PATH.read_text().splitlines()
    if line.strip()
]

from datasets import Dataset

dataset = Dataset.from_list(data)
dataset = dataset.map(
    lambda x: {'label': label2id[x['label']]}
)

from collections import Counter
print(Counter(dataset['label']))

data_list = dataset.to_list()

texts = [example['text'] for example in data_list]
labels = [example['label'] for example in data_list]

from sklearn.model_selection import train_test_split

train_a, temp_a = train_test_split(
    data_list,
    test_size=0.3,
    stratify=[example['label'] for example in data_list],
    random_state=42
)

val_a, test_a = train_test_split(
temp_a,
test_size=0.5,
stratify=[example['label'] for example in temp_a],
random_state=42
)

from datasets import Dataset

train_a = Dataset.from_list(train_a)
val_a = Dataset.from_list(val_a)
test_a = Dataset.from_list(test_a)

train_a = train_a.map(tokenize_a, batched=True)
val_a = val_a.map(tokenize_a, batched=True)
test_a = test_a.map(tokenize_a, batched=True)

train_a = train_a.remove_columns(['text'])
val_a = val_a.remove_columns(['text'])
test_a = test_a.remove_columns(['text'])

print(train_a)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer_a)

args_a = TrainingArguments(
    output_dir=str(MODEL_DIR),
    learning_rate=2e-5,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    num_train_epochs=15,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    report_to='none',
    logging_steps=-1
)

trainer_a = Trainer(
    model=model_a,
    args=args_a,
    train_dataset=train_a,
    eval_dataset=val_a,
    compute_metrics=compute_cls_metrics,
    data_collator=data_collator
)

trainer_a.train()
trainer_a.evaluate(test_a)
trainer_a.save_model(str(MODEL_DIR))
tokenizer_a.save_pretrained(str(MODEL_DIR))
