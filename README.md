# Qwen3.5-9B abliteration — optimization checkpoints

Optuna/Heretic study state for a controlled abliteration of `Qwen/Qwen3.5-9B`
(commit c202236235762e1c871ad0ccb60c8ee5ba337b9a).

## Contents
- `checkpoints/*.jsonl` — deduplicated study-state snapshots (14 unique of 119).
- Each file is newline-delimited JSON; `op_code` 6 records hold per-trial
  `[keyword_refusal_rate, kl_divergence]` values.

## Why the model was rebuilt instead of using the Heretic export
Heretic 2.1.0 exports by PEFT `merge_and_unload()`, which produced degenerate
weights for this hybrid linear-attention architecture (`Qwen3_5ForConditionalGeneration`):
the exported model emitted a repeated single token for every prompt.
Direct rank-1 orthogonalisation of the attention output projections, with a
strength sweep gated on KL divergence and capability probes, preserved the model.

## Measured tradeoff (Qwen3.5-9B, attention out-proj ablation, layers 8-31)

| scale | KL vs base | refusal rate | capability |
|---|---|---|---|
| 0.75 | 0.033 | 100% | 1.000 |
| 1.25 | 0.095 | 100% | 1.000 |
| 1.75 | 0.196 | 62.5% | 1.000 |
| 2.25 | 0.358 | 25% | 1.000 |
| 3.00 | 0.695 | 0% | 1.000 |

Capability stayed 1.000 at every strength, i.e. no collapse at any tested point.
