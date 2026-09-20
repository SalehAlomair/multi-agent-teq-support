# Part 4/9 — Phase A / Model C — Instruction-Tuned Support Specialist

Fine-tuning form: Supervised Instruction Fine-Tuning with PEFT. Use LoRA by default; use QLoRA when CUDA and bitsandbytes
are available. Purpose: troubleshooting explanation, synthesis of tool results, and final technical response generation.

## SFT + LoRA/QLoRA scaffold
```python
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_C = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer_c = AutoTokenizer.from_pretrained(MODEL_C)
if tokenizer_c.pad_token is None:
    tokenizer_c.pad_token = tokenizer_c.eos_token

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

# TODO(student 3): build 40–80 high-quality technical-support conversations
# in messages format. Include successful troubleshooting, uncertainty,
# tool-result synthesis, and human-escalation examples.

def format_for_sft(example):
    return {
        "text": tokenizer_c.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

sft_train = sft_dataset["train"].map(format_for_sft)
sft_val = sft_dataset["validation"].map(format_for_sft)

sft_args = SFTConfig(
    output_dir="models/support_adapter",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataset_text_field="text",
    max_length=512,
    packing=False,
    report_to="none",
)

trainer_c = SFTTrainer(
    model=model_c,
    args=sft_args,
    train_dataset=sft_train,
    eval_dataset=sft_val,
    processing_class=tokenizer_c,
)
trainer_c.train()
model_c.save_pretrained("models/support_adapter")
tokenizer_c.save_pretrained("models/support_adapter")
```

**Acceptance gate — Model C** — Compare baseline vs fine-tuned loss/perplexity. Add ROUGE where a reference answer exists.
Required Golden Set behaviors must pass 100% for groundedness/escalation cases.
