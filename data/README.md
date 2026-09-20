# Model training data

Training and evaluation records are grouped by component:

- `model_a/intents.jsonl`: 160 intent-classification examples across eight labels.
- `model_b/qa.jsonl`: eight support contexts, each with ten extractive QA pairs.
- `model_c/support_sft.jsonl`: Model C supervised fine-tuning conversations.
- `model_c/golden_set.jsonl`: Model C regression and human-review cases.
- `routing/router_eval.jsonl`: balanced labeled cases for comparing both routers and the
  hybrid routing policy.
- `tools/support.db`: local SQLite ticket store, created and seeded automatically.
- `uploads/`: the only directory the restricted `file_search` tool may inspect.

Each JSONL file contains one JSON object per line. The training scripts resolve
these paths relative to the project directory, so they can be launched from a
different working directory.

## Model C instruction data

`model_c/support_sft.jsonl` originally contained 64 manually authored, synthetic lab conversations:
16 each for troubleshooting, uncertainty, tool-result synthesis, and escalation.
Tool results included in user messages are simulated evidence, not live checks.
The assistant responses distinguish observations from hypotheses and do not claim
unperformed actions.

Each record has an ID, category, fixed split, and system/user/assistant messages.
Each category contributes 12 training, 2 validation, and 2 test examples (48/8/8
overall). The notebook loads these splits independently of Model B's QA data and
formats them with Model C's chat template. Use validation for model selection and
reserve test examples for final evaluation.

This small authored dataset supports the project exercise; it does not replace
baseline measurements, a separate Golden Set, or evaluation on real support cases.

## Focused revision

Added 12 training-only parallel scenarios after regression review: 76 records in
all (60 train, 8 validation, 8 test). Original validation/test records are unchanged.
New cases cover incident escalation, uncertain write results, conflicting documents,
resource-pressure diagnosis, notebook environments, and missing variables. They do
not copy Golden Set prompts. The Golden Set is now a known regression suite; final
claims need a separate unseen evaluation. These records take effect only on a new
training run. The notebook's final diagnostic cell uses the current adapter without
retraining and stores a separate, symmetric concise-prompt comparison.
