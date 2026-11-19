#!/usr/bin/env python3
"""Probe configs on the full final evaluation sets, without saving any model.

Includes (0.0, 0.0) as a base-model control so capability misses and refusal
counts can be attributed to the ablation rather than to the harness or the base
model. Reuses final_model.py's prompt sets and helpers; directions come from the
cache written by the final build.
"""
import json, os, torch
from datasets import load_dataset
import final_model as fm
from transformers import AutoTokenizer, AutoModelForCausalLM

CFGS = [tuple(float(v) for v in c.split(","))
        for c in os.environ.get("CFGS", "0.0,0.0;0.75,2.25;0.75,2.75").split(";")]
OUTJ = os.environ.get("OUTJ", f"{fm.W}/probe_final.json")

tok = AutoTokenizer.from_pretrained(fm.BASE, revision=fm.REV)
model = AutoModelForCausalLM.from_pretrained(fm.BASE, revision=fm.REV, dtype=torch.bfloat16,
                                             device_map="auto", low_cpu_mem_usage=True).eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
layers = lm.layers
d = torch.load(fm.DIRCACHE, map_location="cpu")["d"]
print("dirs loaded from cache", flush=True)

targets = []
for i, lay in enumerate(layers):
    o = getattr(getattr(lay, "self_attn", None), "o_proj", None)
    if o is None:
        o = getattr(getattr(lay, "linear_attn", None), "out_proj", None)
    if o is not None:
        targets.append((i, o, "attention"))
    dp = getattr(getattr(lay, "mlp", None), "down_proj", None)
    if dp is not None:
        targets.append((i, dp, "mlp"))

heldout = load_dataset("mlabonne/harmless_alpaca", split="train")["text"][fm.N_GOOD:fm.N_GOOD + 32]


def last_logits(p):
    ids = tok(fm.build(tok, p), return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**ids, use_cache=False).logits[0, -1, :].float()


base_logits = [last_logits(p) for p in heldout]
refs, vs = [], []
for i, mod, kind in targets:
    wt = mod.weight.data
    refs.append(wt.detach().cpu().clone())
    vs.append((wt.T @ d[i].to(device=wt.device, dtype=wt.dtype)).cpu())
print("refs cached", flush=True)

results = []
for attn, mlp in CFGS:
    for (i, mod, kind), b, v in zip(targets, refs, vs):
        wt = mod.weight.data
        wt.copy_(b.to(wt.device))
        s = attn if kind == "attention" else mlp
        if s:
            wt.add_(torch.outer(d[i].to(device=wt.device, dtype=wt.dtype), v.to(wt.device)),
                    alpha=-s)
    torch.cuda.empty_cache()

    hits, miss = 0, []
    for q, accepts in fm.CAP:
        out = fm.gen(model, tok, q, 96).lower()
        if any(a in out for a in accepts):
            hits += 1
        else:
            miss.append({"q": q, "a": out[:120]})

    kl_sum = 0.0
    for p, bl in zip(heldout, base_logits):
        lg = torch.log_softmax(last_logits(p), -1)
        pb = torch.softmax(bl, -1)
        kl_sum += torch.sum(pb * (torch.log(pb + 1e-12) - lg)).item()

    anch, broad, ans = 0, 0, []
    for h in fm.HARM:
        a = fm.gen(model, tok, h, 160)
        ans.append({"q": h, "a": a[:300]})
        if fm.OPEN_REF.match(a):
            anch += 1
        if any(t in a.lower() for t in fm.BROAD):
            broad += 1

    row = {"attn": attn, "mlp": mlp, "cap_hits": hits, "cap_total": len(fm.CAP),
           "kl": kl_sum / len(heldout), "anchored": anch, "broad": broad,
           "harm_total": len(fm.HARM), "misses": miss, "answers": ans}
    results.append(row)
    with open(OUTJ, "w") as f:
        json.dump(results, f, indent=2)
    print(f"attn={attn} mlp={mlp} cap={hits}/{len(fm.CAP)} KL={row['kl']:.4f} "
          f"anch={anch}/{len(fm.HARM)} broad={broad}/{len(fm.HARM)}", flush=True)

print("PROBE_FINAL_DONE", flush=True)
