# Router evaluation

Cases: 32
LLM model: local-qwen

| Router | Accuracy | Macro F1 | Avg latency (ms) |
|---|---:|---:|---:|
| router_a | 0.562 | 0.559 | 8.66 |
| router_b | 1.000 | 1.000 | 5787.14 |
| hybrid | 1.000 | 1.000 | 4359.45 |

Router B JSON-valid rate: 1.000
Hybrid fallback rate: 0.750
Router A ECE: 0.097

Recommended policy: **router_b**. router_b achieved the best measured accuracy and Macro F1 combination. Hard escalation rules remain first in the hybrid policy.
