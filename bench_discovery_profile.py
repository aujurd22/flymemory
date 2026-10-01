"""P-DISCOVERY: Insight-score profile of consolidation entries
(descriptive; cross-judge practice for intuition-mechanism P155).

Scores each consolidated entry on the Insight Arena's four axes
(C/N/V/T, P152) with the INDEPENDENT judge (doubao via ARK -- the
cross-judge discipline P155 validated), reports the distribution and
the cross-axis correlations. Descriptive by design: no hypothesis test,
no adoption decision -- this calibrates whether a Discovery-Score-style
filter could rank consolidation entries for downstream value.

Run:  py -3.13 run_discovery_ark.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

CACHE = os.path.join(_HERE, "reports", "consolidated_entries.json")
N_SCORE = 30

SCORE_SYSTEM = """You are scoring a consolidated memory entry for an AI agent's
long-term memory, on four axes (0, 1, or 2 each):
  C compression: does it merge many underlying facts into a compact
    statement? (2 = clearly synthesizes several facts)
  N novelty: does it state something non-obvious, beyond any single
    source sentence? (2 = a genuine synthesis)
  V value: would this entry likely matter for future questions?
    (2 = high future utility)
  T transferability: does the insight apply beyond the exact original
    context? (2 = generalizes)
Be conservative; most everyday consolidation entries are 0-1 on most
axes. Reply with ONLY:
{"C": 0, "N": 0, "V": 0, "T": 0}"""


def main():
    import llm_client  # ARK doubao -- independent judge (P155 discipline)
    cons = json.load(open(CACHE, encoding="utf-8"))
    entries = []
    for c in cons:
        for e in c["entries"]:
            entries.append({"sid": c["sid"], "text": e})
    print(f"consolidation entries available: {len(entries)}")
    step = max(1, len(entries) // N_SCORE)
    sample = entries[::step][:N_SCORE]

    scored = []
    for i, e in enumerate(sample):
        raw = llm_client.ask(
            SCORE_SYSTEM + "\n\nENTRY:\n" + e["text"][:600],
            temperature=0.0, max_tokens=200)
        try:
            sc = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except Exception:
            sc = {}
        axes = {k: int(sc.get(k, -1)) for k in "CNVT"}
        axes["len"] = len(e["text"])
        axes["sid"] = e["sid"]
        scored.append(axes)
        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{len(sample)}", flush=True)

    arr = {k: np.asarray([s[k] for s in scored], dtype=float)
           for k in "CNVT"}
    print("\naxis distributions (0-2):")
    for k in "CNVT":
        v = arr[k]
        print(f"  {k}: mean {v.mean():.2f} | 0:{int((v==0).sum())} "
              f"1:{int((v==1).sum())} 2:{int((v==2).sum())} "
              f"(invalid {int((v<0).sum())})")
    total = sum(arr.values())
    print(f"total score: mean {total.mean():.2f}/{total.max():.0f} max | "
          f"entries with total>=5: {int((total>=5).sum())}/{len(scored)}")
    print("\ncross-axis Spearman (valid entries only):")
    from scipy.stats import spearmanr
    for a in "CNV":
        for b in "NVT":
            if a < b:
                m = (arr[a] >= 0) & (arr[b] >= 0)
                if m.sum() > 5:
                    rho, p = spearmanr(arr[a][m], arr[b][m])
                    print(f"  {a}-{b}: rho={rho:.3f} (p={p:.3f}, n={int(m.sum())})")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"discovery_profile_{ts}.json")
    json.dump({"experiment": "P-DISCOVERY", "n_scored": len(scored),
               "scored": scored, "summary": {
                   k: {"mean": float(arr[k].mean())} for k in "CNVT"}},
              open(out, "w"), indent=1)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
