#!/usr/bin/env python3
"""Brain-damage probe for a served GGUF.

Targets the exact failure seen before: a model that emits one repeated token (or a
short loop) for every prompt. Uses statistical degeneration detectors rather than
eyeballing output. Stdlib only.
Usage: python3 bd_gguf.py --port 8099
"""
import argparse, json, re, sys, urllib.request

PROMPTS = [
    ("Write three sentences about the ocean.", 200),
    ("Explain photosynthesis to a 10-year-old.", 220),
    ("Complete this Python function that returns the nth Fibonacci number:\n\ndef fib(n):", 200),
    ("List five practical uses for a paperclip.", 200),
    ("What happened during the Apollo 11 mission?", 200),
    ("Describe a rainy city street in detail.", 220),
    ("We sat by the fire and", 150),
    ("Translate 'Good morning, how are you?' into French, then German.", 200),
    ("What are the main trade-offs of electric vehicles?", 220),
    ("Write a haiku about winter.", 100),
    ("Summarise the plot of Romeo and Juliet in three sentences.", 220),
    ("What is 37 times 14? Show your reasoning briefly.", 200),
    ("Give me a short recipe for pancakes.", 220),
    ("Why does the sky appear blue?", 200),
    ("Write a two-sentence product description for a stainless steel water bottle.", 200),
    ("Name four programming languages and one strength of each.", 220),
    ("Describe how a bill becomes law in the United States.", 220),
    ("Write a brief note thanking a colleague for their help.", 180),
    ("What are the health benefits of regular walking?", 200),
    ("Explain what an API is to a non-programmer.", 200),
]


def ask(port, prompt, n):
    body = json.dumps({"model": "m", "max_tokens": n, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    ch = d["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if not content.strip() and reasoning.strip():
        return reasoning, ch.get("finish_reason"), "reasoning_only"
    return content, ch.get("finish_reason"), "content"



def max_consecutive_run(words):
    best = cur = 1
    for i in range(1, len(words)):
        cur = cur + 1 if words[i] == words[i - 1] else 1
        best = max(best, cur)
    return best


def max_ngram_repeat(words, n):
    if len(words) < n:
        return 0
    seen = {}
    best = 0
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        best = max(best, seen[g])
    return best


def analyse(text):
    words = text.split()
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", text))
    letters = max(1, len(re.sub(r"\s", "", text)))
    flags = []
    if not words:
        return ["EMPTY"], 0.0, 0, 0
    d1 = len(set(words)) / len(words)
    run = max_consecutive_run(words)
    big = max_ngram_repeat(words, 2)
    tri = max_ngram_repeat(words, 3)
    if len(words) < 5:
        flags.append("TOO_SHORT")
    if run >= 5:
        flags.append(f"TOKEN_LOOP(x{run})")
    if big >= 8:
        flags.append(f"BIGRAM_LOOP(x{big})")
    if tri >= 6:
        flags.append(f"TRIGRAM_LOOP(x{tri})")
    if len(words) >= 20 and d1 < 0.28:
        flags.append(f"LOW_DISTINCT1({d1:.2f})")
    if cjk / letters > 0.30:
        flags.append(f"CJK_GARBAGE({cjk / letters:.2f})")
    if text.count(text[:12]) > 6 and len(text) > 60:
        flags.append("PREFIX_REPEAT")
    return flags, d1, run, max(big, tri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    rows, bad = [], 0
    for p, n in PROMPTS:
        try:
            out, finish, channel = ask(a.port, p, n)
        except Exception as e:
            print(f"ERROR {p[:40]}: {e}")
            return 1
        flags, d1, run, ng = analyse(out)
        if flags:
            bad += 1
        rows.append({"prompt": p, "out": out[:400], "flags": flags, "distinct1": d1,
                     "max_run": run, "max_ngram": ng, "finish": finish,
                     "channel": channel})
        status = "FAIL " + ",".join(flags) if flags else "ok"
        print(f"[{status:28}] d1={d1:.2f} run={run} ng={ng} {channel} | {p[:40]}")
        print(f"   -> {out[:150].replace(chr(10), ' ')}")
    print(f"\nDEGENERATE {bad}/{len(PROMPTS)}")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"rows": rows, "degenerate": bad, "total": len(PROMPTS)}, f, indent=2)
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
