"""Train and evaluate the Model C LoRA support assistant.

Converted from the Model C section of testing.ipynb. Cell boundaries are
retained so the manual Golden Set review workflow remains visible.
"""

# Model C

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_C = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer_c = AutoTokenizer.from_pretrained(MODEL_C)
if tokenizer_c.pad_token is None:
    tokenizer_c.pad_token = tokenizer_c.eos_token

# Refresh the cached check if bitsandbytes was installed after the kernel started.
is_bitsandbytes_available.cache_clear()
use_qlora = torch.cuda.is_available()

if use_qlora:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base_c = AutoModelForCausalLM.from_pretrained(
        MODEL_C,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_c = prepare_model_for_kbit_training(base_c)
else:
    base_c = AutoModelForCausalLM.from_pretrained(MODEL_C)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model_c = get_peft_model(base_c, lora_config)
model_c.print_trainable_parameters()

from model_c_eval import encode_conversation, completion_metrics, generate_answer

# Pre-tokenize and mask every prompt token; only assistant answers carry loss.
def format_for_sft(example):
    return encode_conversation(tokenizer_c, example['messages'])

import json
from pathlib import Path
from datasets import Dataset, DatasetDict

DATA_DIR = Path(__file__).resolve().parent / 'data'

# Curated lab conversations: troubleshooting, uncertainty, tool-result synthesis,
# and escalation. Splits are fixed and independent of Model B's QA dataset.
sft_records = [
    json.loads(line)
    for line in (DATA_DIR / 'support_sft.jsonl').read_text().splitlines()
    if line.strip()
]
sft_dataset = DatasetDict({
    split: Dataset.from_list([
        {'messages': record['messages']}
        for record in sft_records if record['split'] == split
    ])
    for split in ['train', 'validation', 'test']
})
sft_train = sft_dataset['train'].map(format_for_sft, remove_columns=['messages'])
sft_val = sft_dataset['validation'].map(format_for_sft, remove_columns=['messages'])
sft_test = sft_dataset['test'].map(format_for_sft, remove_columns=['messages'])
print(sft_dataset)

# Keep the held-out Golden Set separate from training and validation.
golden_set = [json.loads(line) for line in (DATA_DIR / 'golden_set.jsonl').read_text().splitlines() if line.strip()]
assert not {case['prompt'] for case in golden_set} & {
    r['messages'][1]['content'] for r in sft_records
}
assert all(any(label != -100 for label in row['labels']) for row in sft_train)

from datetime import datetime, timezone
from transformers import EarlyStoppingCallback, set_seed
import hashlib

# Run the Model C loading + LoRA cells first to create a fresh adapter.
# LoRA B weights start at zero; reject accidental continuation of a trained adapter.
if any(torch.count_nonzero(p).item() for name, p in model_c.named_parameters() if 'lora_B' in name):
    raise RuntimeError('Reload Model C and create a fresh LoRA adapter before this run.')
set_seed(42)
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
run_dir = Path('models') / ('support_adapter_' + run_id)
eval_dir = Path('results/model_c_evaluation') / run_id
eval_dir.mkdir(parents=True, exist_ok=False)
system_message = sft_records[0]['messages'][0]
manifest = {
    'model': MODEL_C, 'run_id': run_id, 'seed': 42,
    'quantized_4bit': use_qlora, 'loss_scope': 'assistant completion only',
    'max_new_tokens': 384, 'do_sample': False,
    'data_sha256': hashlib.sha256((DATA_DIR / 'support_sft.jsonl').read_bytes()).hexdigest(),
    'golden_sha256': hashlib.sha256((DATA_DIR / 'golden_set.jsonl').read_bytes()).hexdigest(),
    'note': 'Golden Set is a known regression suite, not an unseen final benchmark.',
}
(eval_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

# Record a real baseline before any optimization. Keep incremental outputs.
baseline_metrics = completion_metrics(model_c, tokenizer_c, sft_dataset['test'], baseline=True)
(eval_dir / 'baseline_metrics.json').write_text(json.dumps(baseline_metrics, indent=2))
baseline_answers = {}
for case in golden_set:
    baseline_answers[case['id']] = generate_answer(
        model_c, tokenizer_c, [system_message, {'role': 'user', 'content': case['prompt']}], baseline=True)
    (eval_dir / 'baseline_answers.json').write_text(json.dumps(baseline_answers, indent=2))
    print(case['id'], 'baseline recorded')

sft_args = SFTConfig(
    output_dir=str(run_dir), num_train_epochs=4,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=4, learning_rate=2e-5,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
    max_length=512, packing=False, report_to='none', seed=42, data_seed=42,
    bf16=bool(use_qlora and torch.cuda.is_bf16_supported()),
    fp16=bool(use_qlora and not torch.cuda.is_bf16_supported()),
    gradient_checkpointing=True, logging_steps=5,
)
trainer_c = SFTTrainer(
    model=model_c, args=sft_args, train_dataset=sft_train, eval_dataset=sft_val,
    processing_class=tokenizer_c,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
# Verify the actual trainer retains the assistant-only labels.
assert trainer_c.train_dataset[0]['labels'] == sft_train[0]['labels']
trainer_c.train()
trainer_c.save_model(str(run_dir / 'final'))
tokenizer_c.save_pretrained(str(run_dir / 'final'))
(eval_dir / 'training_history.json').write_text(json.dumps(trainer_c.state.log_history, indent=2))
print('Saved new run:', run_dir, eval_dir)

# Phase B — Model C evaluation and quality gates

if 'trainer_c' not in globals() or trainer_c.state.global_step == 0:
    raise RuntimeError('Run the new baseline and training cell first.')
eval_model_c = trainer_c.model
model_report = {
    'manifest': manifest,
    'baseline_method': 'Recorded before training with disabled adapter',
    'split': 'test', 'loss_scope': 'assistant completion only, token-weighted',
    'baseline': baseline_metrics,
    'fine_tuned': completion_metrics(eval_model_c, tokenizer_c, sft_dataset['test']),
}
model_report['quality_gate'] = {
    'loss_improved': model_report['fine_tuned']['loss'] < model_report['baseline']['loss'],
    'golden_set': 'pending human review', 'accepted': False,
}
(eval_dir / 'metrics.json').write_text(json.dumps(model_report, indent=2, allow_nan=False))
print(json.dumps(model_report, indent=2))

# Golden Set: generate both versions, then review
# These 10 cases are separate from the SFT data. Generation is deterministic. Judge each response against its rubric: keyword matches alone cannot establish groundedness or safe escalation. Review files retain both responses and start with null verdicts; no case passes automatically. Generation may take several minutes on CPU.

review_path = eval_dir / ('golden_review_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
reviews = []
for case in golden_set:
    baseline = baseline_answers[case['id']]
    tuned = generate_answer(eval_model_c, tokenizer_c,
        [system_message, {'role': 'user', 'content': case['prompt']}])
    reviews.append({**case,
        'baseline_response': baseline['text'], 'fine_tuned_response': tuned['text'],
        'baseline_generation': {k: v for k, v in baseline.items() if k != 'text'},
        'fine_tuned_generation': {k: v for k, v in tuned.items() if k != 'text'},
        'baseline_pass': None, 'fine_tuned_pass': None,
        'error_category': '', 'review_notes': '',
    })
    review_path.write_text(json.dumps(reviews, indent=2))
    print(case['id'], 'truncated:', baseline['truncated'], tuned['truncated'])
print('Review file:', review_path)

# Apply the gate after review
# In the generated review JSON, set baseline_pass and fine_tuned_pass to true or false and explain the verdict in review_notes. For failures, record an error_category such as hallucination, missed escalation, missing clarification, or instruction following. Then run the cell below. Acceptance requires lower test loss, all required fine-tuned cases passing, and completed reviews. This is Model C's gate only, not a system deployment gate.

reviews = json.loads(review_path.read_text())
expected = {case['id'] for case in golden_set}
if len(reviews) != len(expected) or {row['id'] for row in reviews} != expected:
    raise ValueError('Review must contain each Golden Set case exactly once.')
required_ids = {case['id'] for case in golden_set if case['required']}
complete = all(
    type(row['baseline_pass']) is bool and type(row['fine_tuned_pass']) is bool
    and bool(row['review_notes'].strip()) for row in reviews
)
regressions = [row['id'] for row in reviews
               if row['id'] in required_ids and row['baseline_pass'] is True
               and row['fine_tuned_pass'] is False]
truncated_cases = [row['id'] for row in reviews if any(
    row[key]['truncated'] for key in ['baseline_generation', 'fine_tuned_generation'])]
required_pass = complete and all(row['fine_tuned_pass'] is True
                               for row in reviews if row['id'] in required_ids)
model_report['quality_gate'].update({
    'golden_set': 'reviewed' if complete else 'pending human review',
    'required_cases_pass': required_pass,
    'regressions': regressions,
    'truncated_cases': truncated_cases,
    'review_file': str(review_path),
    'accepted': bool(model_report['quality_gate']['loss_improved'] and required_pass and not regressions and not truncated_cases),
})
(eval_dir / 'metrics.json').write_text(json.dumps(model_report, indent=2, allow_nan=False))
print(json.dumps(model_report['quality_gate'], indent=2))

# Diagnose the current adapter and compare concise responses
# Run the next cell directly while the trained model remains in memory. It does not train or replace previous results. New training examples only take effect after rerunning data preparation and a fresh training run.

# Run only this cell on the current trained model; no retraining required.
import importlib
import model_c_eval
importlib.reload(model_c_eval)
from model_c_eval import adapter_diagnostics, generate_answer

if 'trainer_c' not in globals() or trainer_c.state.global_step == 0:
    raise RuntimeError('This cell needs the trained trainer_c from the current run.')
current_model = trainer_c.model
probe_messages = [sft_records[0]['messages'][0], {
    'role': 'user', 'content': 'A background task is delayed. What information should I collect first?'}]
diagnostics = adapter_diagnostics(current_model, tokenizer_c, probe_messages)
print(json.dumps(diagnostics, indent=2))
comparison_dir = eval_dir / ('concise_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
comparison_dir.mkdir(parents=True, exist_ok=False)
(comparison_dir / 'adapter_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
if not diagnostics['effect_detected']:
    raise RuntimeError('No adapter effect on this probe. Investigate before retraining or scoring.')

# Same instruction and token budget for both models. This is a new prompting
# experiment, not directly comparable to the earlier prompting configuration.
concise_system = dict(sft_records[0]['messages'][0])
concise_system['content'] += (
    ' Answer directly in at most 100 words. Put the decision and essential next '
    'action first. Avoid preambles, repeated explanations, and speculative advice.'
)
(comparison_dir / 'generation_config.json').write_text(json.dumps({
    'system_message': concise_system, 'max_new_tokens': 384, 'do_sample': False,
    'model': MODEL_C, 'adapter_run': str(run_dir),
    'baseline_method': 'Current base with adapter disabled; prompting experiment after training',
}, indent=2))
concise_reviews = []
for case in golden_set:
    messages = [concise_system, {'role': 'user', 'content': case['prompt']}]
    base = generate_answer(current_model, tokenizer_c, messages, baseline=True)
    tuned = generate_answer(current_model, tokenizer_c, messages)
    concise_reviews.append({**case,
        'baseline_response': base['text'], 'fine_tuned_response': tuned['text'],
        'baseline_generation': {k:v for k,v in base.items() if k != 'text'},
        'fine_tuned_generation': {k:v for k,v in tuned.items() if k != 'text'},
        'baseline_pass': None, 'fine_tuned_pass': None, 'error_category': '', 'review_notes': '',
    })
    (comparison_dir / 'review.json').write_text(json.dumps(concise_reviews, indent=2))
    print(case['id'], base['finish_reason'], tuned['finish_reason'])
print('Review the new experiment separately:', comparison_dir / 'review.json')
