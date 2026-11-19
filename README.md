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

## v2 — shipped model: `qwen35-9b-abliterated-v2`

Full run recorded in `v2/` (scripts, eval JSON, metrics).

**Method:** orthogonal projection abliteration applied as a *direct weight edit*
(`W = W_base − s·outer(d, v)`), restored from pristine base weights for every
configuration. No LoRA and no PEFT merge anywhere, so the failure mode above cannot occur.

- Layers: **0–31** (all), not 8–31
- Modules: **attn out-proj AND `mlp.down_proj`**
- Directions: difference of means, last token, 170 harmful / 160 harmless prompts
- Scales: **attention 0.75, MLP 2.25**

The module set and layer range follow the published settings of
`lukey03/Qwen3.5-9B-abliterated` (`abliteration_meta.json`, archived in
`v2/reference/`). Only that 1 KB metadata file was read — no weights or GGUF were
downloaded from that repo, and nothing from it enters the model.

### Why `mlp.down_proj` matters (controlled test, `v2/evals/`)

Same harness, pristine weights restored per config, KL measured on **held-out** benign prompts:

| attn | mlp | KL | refusal | cap |
|---|---|---|---|---|
| 0.00 | 0.00 | 0.0000 | 8/8 | 1.00 |
| 0.75 | 0.00 | 0.1732 | 8/8 | 1.00 |
| 0.75 | 0.75 | 0.3291 | 5/8 | 1.00 |
| 1.00 | 0.00 | 0.3763 | 8/8 | 0.90 |
| 1.00 | 1.00 | 0.7125 | 1/8 | 1.00 |
| 0.75 | 2.25 | 0.5640 | **0/16** | 28/30 |

Attention-only ablation stalls at 8/8 refusals while already costing capability;
MLP strength is what actually removes refusals, at lower cost. Attention strength is
the expensive, damaging axis.

### Final measurements

| | capability | refusals | KL | degeneration |
|---|---|---|---|---|
| base model (control) | 29/30 | 16/16 | — | — |
| fp model (bf16, pre-quant) | 28/30 | 0/16 | 0.564 | — |
| **shipped GGUF Q5_K_M (CUDA)** | **29/30** | **0/16** | — | **0/20** |
| **shipped GGUF Q5_K_M (local Metal)** | **29/30** | **0/16** | — | **0/20** |

Artifact: `qwen35-9b-abliterated-v2-Q5_K_M.gguf`, 6,494,705,568 bytes,
SHA-256 `7d5d7c03609f7185dea363facb96fe73962ca3c1ad4092bf525935072c1e1e75`
(verified instance → R2 → local). Hosted at `r2:ob-models/qwen35-9b-abliterated-v2/`
(6.049 GiB total, v1 purged). Loads in llama.cpp b12549 and in 2.79.0; the current
converter maps `Qwen3_5ForCausalLM → qwen` natively, so the earlier `qwen2` patch is
no longer needed.

### Corrections to the earlier table above

The first table's "capability 1.000" came from a 10-question smoke test, and its
refusal figures came from a substring detector over 8 prompts that flags words like
"illegal" anywhere in the output. Both overstate certainty. The 30-prompt suite with a
base-model control is the better guide: at matched settings the abliterated model
matches base (29/30), but a stricter reading of the 8-prompt numbers is that refusal
*granularity* was too coarse to resolve the knee of the curve.

## v3 — MTP attempt and the container fix

Adding the base model's multi-token-prediction block turned out to be a dead end worth
documenting, because it explains both the earlier LM Studio failure and why the final
artifact is text-only.

**What the base model actually contains:** `Qwen3_5ForConditionalGeneration` ships **775
tensors**, including **15 `mtp.*` tensors** (`mtp.fc`, `mtp.layers.0.*`, `mtp.norm`, ...).
The abliteration pipeline loads `AutoModelForCausalLM`, which does not model the MTP block,
so `save_pretrained()` silently dropped those 15 tensors while `config.json` still reported
`mtp_num_hidden_layers = 1`. The converter then emitted `block_count = 33`,
`nextn_predict_layers = 1` and a 33-entry `recurrent_layers` array, with **no** MTP tensors
behind them.

### Why LM Studio could not load v2

```
src/models/qwen35.cpp:523: GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj")
  graph_mtp -> build_arch_graph -> graph_reserve -> llama_init_from_model
  -> common_params_fit_impl -> common_fit_apply
```

LM Studio always runs `common_fit_apply`, which reserves the MTP graph and aborted on the
missing tensor. Plain `llama-server -c 8192` never reserves that graph, which is why the file
appeared to work for us and not for LM Studio. Metadata-only patches cannot fix this: shortening
`recurrent_layers` to 32 bytes-to-32-entries requires rewriting the whole file, and every partial
patch just moves the error (`blk.32.attn_norm.weight not found` →
`recurrent_layers has wrong array length; expected 32, got 33`).

### What happened when MTP was actually added

Carrying the 15 tensors over (`mtp.fc` → `blk.32.nextn.eh_proj`, etc.) produced a GGUF with
**394 tensors, block_count 33, nextn_predict_layers 1** and all four nextn tensors present.
It then **loaded successfully** — both in LM Studio and in llama-server — but **failed at
inference** with `Compute error` / `ggml_backend_sched_graph_compute_async failed` after roughly
five prompts. So the MTP graph is not merely under-specified in our export; it does not run.

### Shipped: text-only with consistent metadata

Converting the same abliterated weights with `--no-nextn` lets the converter write metadata that
matches the tensors:

| | v2 (shipped, broken metadata) | **v3-text (shipped)** |
|---|---|---|
| tensors | 379 | 379 |
| `block_count` | 33 | **32** |
| `nextn_predict_layers` | 1 (no tensors) | **absent** |
| `recurrent_layers` entries | 33 | **32** |
| loads in LM Studio | no (assert) | **yes** |
| runs | yes (llama-server) | **yes (both)** |

Artifact: `qwen35-9b-abliterated-v3-text-Q5_K_M.gguf`, 6,494,705,536 bytes,
SHA-256 `d87cd45617c75c48db9b661a22f6b8426637fcb95057e3439fc85e650b76610e`,
hosted at `r2:ob-models/qwen35-9b-abliterated-v3-text/` (single object, 6.049 GiB).
Degeneration suite on this exact hash: **0/20**.

Superseded artifacts removed after validation (recorded here for traceability):
`qwen35-9b-abliterated-v2-Q5_K_M.gguf` (sha256 `7d5d7c03609f7185dea363facb96fe73962ca3c1ad4092bf525935072c1e1e75`)
and the MTP build `qwen35-9b-abliterated-v3-mtp-Q5_K_M.gguf`
(sha256 `7709dbbed81655bb4ae47740530a7f02ff3c65595588aa36dccc454160c15012`,
6,669,280,064 bytes — loaded but failed inference with `Compute error`).

### Caveats

- 30 capability prompts, 16 refusal prompts, 20 degeneration prompts: smoke tests, not benchmarks.
- Greedy decoding, single turn.
- "0/16 refusals" means no refusal on those 16 prompts — not proof of zero refusal generally.
- No safety evaluation was performed; this is a capability-retention measurement.
