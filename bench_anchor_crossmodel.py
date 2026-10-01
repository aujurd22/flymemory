"""P-ANCH-1 cross-model replication: deepseek-reasoner (R1 reasoning mode).

Same 48 turns, same two conditions, same criteria as the original
P-ANCH-1 (bench_anchor_pairing.py) -- the ACTOR swapped from
deepseek-chat to deepseek-reasoner. This is a WITHIN-family,
cross-reasoning-mode replication (a true cross-vendor run needs an ARK
key for doubao, which is not available in this session; recorded as the
remaining gap).

Pre-registered replication bands (weaker than the original: a different
model may shift absolute accuracy):
  REPLICATED  : acc(A) >= 40/48 AND discordant U-wrong/A-right >= 3x
                U-right/A-wrong (same direction, strong margin)
  PARTIAL     : acc(A) >= 34/48 with direction preserved
  NOT-REPLICATED : acc(A) < 34/48 or direction flips
  max_tokens raised to 2000 (reasoning tokens; 100 would truncate).

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_anchor_crossmodel.py
"""
import json
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ACTOR_MODEL = "deepseek-reasoner"
BASE_URL = "https://api.deepseek.com"

# reuse the exact trial set and prompts from the original experiment
from bench_anchor_pairing import (  # noqa: E402
    SYSTEM_U, SYSTEM_A, load_set, parse_bool)


def ds_chat(client, system, user):
    r = client.chat.completions.create(
        model=ACTOR_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=2000)
    # reasoner puts reasoning in reasoning_content; final text in content
    return r.choices[0].message.content or ""


def main():
    from bench_anchor_pairing import main as _unused  # noqa: F401
    items = load_set()
    from bench_memory_judgment import ds_client  # noqa: E402
    client = ds_client()
    traces = []
    pair = Counter()
    acc = {"U": 0, "A": 0}
    cls_err = {"U": Counter(), "A": Counter()}

    for i, (cid, entry, turn, anchor, cls) in enumerate(items):
        u_raw = ds_chat(client, SYSTEM_U, f"User message:\n\"{turn}\"")
        asserts = parse_bool(u_raw, "asserts_fact")
        u_verdict = "YES" if asserts else ("NO" if asserts is not None
                                           else "PARSE_FAIL")
        a_raw = ds_chat(client, SYSTEM_A,
                        f"Stored memory entry:\n\"{entry}\"\n\n"
                        f"User's new message:\n\"{turn}\"")
        inv = parse_bool(a_raw, "invalidates")
        a_verdict = "YES" if inv else ("NO" if inv is not None
                                       else "PARSE_FAIL")
        u_ok, a_ok = u_verdict == anchor, a_verdict == anchor
        acc["U"] += u_ok
        acc["A"] += a_ok
        pair[(u_ok, a_ok)] += 1
        if not u_ok:
            cls_err["U"][cls] += 1
        if not a_ok:
            cls_err["A"][cls] += 1
        traces.append({"case_id": cid, "cls": cls, "anchor": anchor,
                       "u": u_verdict, "a": a_verdict})
        print(f"[{i+1}/48] {cid:14s} ({cls:9s}) gold={anchor:3s} "
              f"U={u_verdict:3s}{'✗' if not u_ok else ' '} "
              f"A={a_verdict:3s}{'✗' if not a_ok else ' '}", flush=True)

    n = len(items)
    b = pair[(True, False)]
    c = pair[(False, True)]
    resist_a = sum(1 for t in traces if t["cls"] == "resist" and
                   t["a"] == "NO")
    print(f"\n=== P-ANCH-1 cross-model (deepseek-reasoner) | n={n} ===")
    print(f"acc(U) {acc['U']}/{n} | acc(A) {acc['A']}/{n} | "
          f"discordant U-wrong/A-right={c} U-right/A-wrong={b}")
    print(f"resist-class A repair: {resist_a}/12")

    if acc["A"] >= 40 and c >= 3 * max(b, 1):
        verdict = "REPLICATED"
    elif acc["A"] >= 34:
        verdict = "PARTIAL (direction preserved)"
    else:
        verdict = "NOT-REPLICATED"
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"anchor_crossmodel_{ts}.json")
    json.dump({"experiment": "P-ANCH-1-crossmodel", "model": ACTOR_MODEL,
               "acc_U": acc["U"], "acc_A": acc["A"],
               "discordant": {"b": b, "c": c}, "resist_a_ok": resist_a,
               "verdict": verdict, "traces": traces},
              open(out, "w"), indent=1)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
