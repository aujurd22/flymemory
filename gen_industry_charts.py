"""Industry memory-tool comparison charts -- every bar carries its source
and protocol. Groups are visually separated because cross-protocol
comparison is INVALID (the program's own restraint policy):
  Panel 1: LOCOMO (Mem0 paper, LLM-as-Judge, J%) -- 8 tools, one protocol
  Panel 2: LongMemEval per-ability (Zep blog, gpt-4o) -- Zep vs
           full-context only (no other vendor published)
  Panel 3: our internal numbers (own protocol, own judge) -- plotted
           separately and explicitly NOT comparable to panels 1-2.

Run:  py -3.13 gen_industry_charts.py  -> docs/img/industry_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "img")
os.makedirs(OUT, exist_ok=True)

C_MEM0, C_ZEP, C_GRAY, C_FM, C_OURS = ("#3f7fd9", "#d97b3f", "#9aa0a6",
                                       "#3f9d55", "#8e5bd9")

# ---------------- Panel 1: LOCOMO overall J (Mem0 paper Table 2) ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
tools = ["Full-context\n(~26K tok)", "Mem0-graph", "Mem0", "Zep",
         "Best RAG", "LangMem", "OpenAI\nmemory", "A-Mem"]
j = [72.90, 68.44, 66.88, 65.99, 60.97, 58.10, 52.90, 48.38]
colors = [C_GRAY] + [C_MEM0] * 2 + [C_ZEP] + [C_GRAY] * 4
bars = a1.bar(tools, j, color=colors, width=0.62)
for b, v in zip(bars, j):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
            ha="center", fontsize=9)
a1.set_ylabel("LLM-as-Judge score (J, %)")
a1.set_ylim(0, 82)
a1.set_title("LOCOMO — one protocol, one judge\n(Mem0 paper Table 2, 10-run mean)")
a1.tick_params(axis="x", labelsize=8)

# ---------------- Panel 2: LongMemEval per-ability (Zep blog, gpt-4o) -----
cats = ["ss-pref", "ss-user", "temporal", "multi-session",
        "knowledge-update", "ss-assistant"]
full_ctx = [20.0, 81.4, 45.1, 44.3, 78.2, 94.6]
zep = [56.7, 92.9, 62.4, 57.9, 83.3, 80.4]
x = np.arange(len(cats))
w = 0.36
b1 = a2.bar(x - w / 2, full_ctx, w, label="full-context (115K tok)",
            color=C_GRAY)
b2 = a2.bar(x + w / 2, zep, w, label="Zep (1.6K tok)", color=C_ZEP)
for b, v in zip(b1, full_ctx):
    a2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
            ha="center", fontsize=8)
for b, v in zip(b2, zep):
    a2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}",
            ha="center", fontsize=8)
a2.set_xticks(x, cats, fontsize=8)
a2.set_ylabel("accuracy, %")
a2.set_ylim(0, 105)
a2.set_title("LongMemEval per ability (Zep blog, gpt-4o)\n"
             "-- no other vendor published on this protocol", fontsize=10)
a2.legend(frameon=False, fontsize=8)

fig.suptitle("Published memory-tool numbers — panels are NOT comparable "
             "across protocols", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "industry_locomo_zep.png"))
plt.close(fig)

# ---------------- Panel 3: our internal numbers (own protocol) ------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

# LongMemEval-S oracle retrieval ladder (own protocol)
ladder = ["dense", "full\nscoring", "BM25", "RRF", "RRF+CE", "oracle@10"]
lv = [27.4, 29.0, 33.6, 40.8, 45.8, 55.4]
bars = a1.bar(ladder, lv, color=[C_GRAY] * 3 + [C_FM] * 2 + [C_GRAY],
              width=0.6)
for b, v in zip(bars, lv):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
            ha="center", fontsize=9)
a1.set_ylabel("evidence-hit@3, % of 500")
a1.set_ylim(0, 62)
a1.set_title("FlyMemory internal: LongMemEval-S retrieval ladder\n"
             "(199K-turn store, own protocol — NOT comparable to LOCOMO J)",
             fontsize=10)

# stale/current resolution (the axis others don't measure)
conds = ["old-prefix\nquery (A1)", "+ cluster layer\n(P-CLUSTER)",
         "+ supersede marks\n(state machine)", "+ anchor\n(L6, paired)"]
cv = [92.0, None, 0.0, 2.1]          # % wrong on current-state questions
cv[1] = 92.0                          # P-CLUSTER: no improvement (NULL)
bars = a2.bar(conds, [92, 92, 0, 2.1],
              color=[C_GRAY, C_GRAY, C_GREEN := "#3f9d55", C_GREEN],
              width=0.55)
labels = ["92%", "92%\n(NULL)", "0%", "2.1%"]
for b, l in zip(bars, labels):
    a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, l,
            ha="center", fontsize=9)
a2.set_ylabel("% WRONG on current-state questions")
a2.set_ylim(0, 108)
a2.set_title("The axis published benchmarks don't measure:\n"
             "stale-current resolution (FlyMemory, own runs)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "industry_flymemory_internal.png"))
plt.close(fig)

print("charts written:",
      sorted(f for f in os.listdir(OUT) if f.startswith("industry_")))
