"""Generate the README charts from recorded benchmark numbers.
Deterministic; every number is traceable to a reports/*.json artifact
(the source is cited in each chart's caption inside README.md).

Run:  py -3.13 gen_readme_charts.py   -> docs/img/*.png
"""
import os

import numpy as np
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "img")
os.makedirs(OUT, exist_ok=True)

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = ("#3572b0", "#e8873a",
                                            "#3f9d55", "#c94f4f", "#9aa0a6")

# ---- chart 1: LongMemEval-S retrieval ladder (bench_rerank_full.py) ----
fig, ax = plt.subplots(figsize=(7.2, 3.4))
methods = ["dense", "dense+lex\n(full)", "BM25", "RRF\n(dense+lex)",
           "RRF +\ncross-encoder", "oracle@10\n(ceiling)"]
vals = [27.4, 29.0, 33.6, 40.8, 45.8, 55.4]
colors = [C_GRAY, C_GRAY, C_GRAY, C_BLUE, C_BLUE, C_GREEN]
bars = ax.bar(methods, vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
            ha="center", fontsize=10)
ax.set_ylabel("answer-level hit, % of 500")
ax.set_ylim(0, 62)
ax.set_title("LongMemEval-S retrieval ladder (500 questions)")
fig.savefig(os.path.join(OUT, "retrieval_ladder.png"))
plt.close(fig)

# ---- chart 2: consolidation overlay + anchoring (two panels) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
bars = a1.bar(["turn-only", "replace-style\n(abstract ON)",
               "overlay\n(abstract ON)"], [32.0, 22.0, 42.8],
              color=[C_GRAY, C_RED, C_GREEN], width=0.6)
for b, v in zip(bars, [32.0, 22.0, 42.8]):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}",
            ha="center", fontsize=10)
a1.set_ylabel("strict accuracy, % of 500")
a1.set_ylim(0, 50)
a1.set_title("L1 overlay law (LongMemEval, n=500)")
a1.annotate("abstraction must\noverlay, never replace",
            xy=(1, 22), xytext=(1.35, 34), fontsize=9, color=C_RED,
            arrowprops=dict(arrowstyle="->", color=C_RED))

conds = ["unanchored\n(classify + rule)", "anchored\n(entry vs message)"]
vals2 = [70.8, 97.9]
bars2 = a2.bar(conds, vals2, color=[C_GRAY, C_GREEN], width=0.45)
for b, v in zip(bars2, vals2):
    a2.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center",
            fontsize=10)
a2.set_ylabel("state-change decision acc, % of 48")
a2.set_ylim(0, 108)
a2.set_title("L6 anchoring law (P-ANCH-1, paired)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "laws.png"))
plt.close(fig)

# ---- chart 3: memory judgment v1.5 (168 cases) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
metrics = ["supersede\nprecision", "supersede\nrecall", "forget\nprecision",
           "forget\nrecall"]
mvals = [98.9, 100.0, 100.0, 100.0]
bars = a1.bar(metrics, mvals, color=C_BLUE, width=0.58)
for b, v in zip(bars, mvals):
    a1.text(b.get_x() + b.get_width()/2, v + 1.0, f"{v:.1f}", ha="center",
            fontsize=10)
a1.set_ylabel("% (execution-aware, n=89/8)")
a1.set_ylim(0, 112)
a1.set_title("Memory Judgment v1.5, deepseek-chat (168 cases)")

cats = ["classic\nno-ops\n(36)", "balanced\nSTALE (12)", "balanced\nRESIST (12)"]
cvals = [36, 12, 9]
ctot = [36, 12, 12]
bars2 = a2.bar(cats, cvals, color=[C_GREEN, C_GREEN, C_ORANGE], width=0.55)
for b, v, t in zip(bars2, cvals, ctot):
    a2.text(b.get_x() + b.get_width()/2, v + 0.35, f"{v}/{t}", ha="center",
            fontsize=10)
a2.set_ylabel("correctly silent / held")
a2.set_ylim(0, 42)
a2.set_title("adversarial holds (response-bias block)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "judgment.png"))
plt.close(fig)

# ---- chart 4: eviction benefit-cost mirror (P-CLEANUP pre-study) ----
fig, ax = plt.subplots(figsize=(7.6, 3.6))
pols = ["LRU\n(current)", "core-first", "random", "fringe-first"]
rec = [6.3, 27.1, 17.7, 0.0]
keep = [91.0, 70.0, 87.2, 91.7]
x = np.arange(len(pols))
w = 0.38
b1 = ax.bar(x - w/2, rec, w, label="evicted-recoverability %",
            color=C_ORANGE)
b2 = ax.bar(x + w/2, keep, w, label="holdout top-5 keep %",
            color=C_BLUE)
for b, v in zip(b1, rec):
    ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha="center",
            fontsize=9)
for b, v in zip(b2, keep):
    ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha="center",
            fontsize=9)
ax.set_xticks(x, pols)
ax.set_ylabel("%")
ax.set_ylim(0, 104)
ax.set_title("The benefit-cost mirror: what eviction frees, it perturbs")
ax.legend(frameon=False, fontsize=9)
fig.savefig(os.path.join(OUT, "eviction_mirror.png"))
plt.close(fig)

print("charts written to docs/img/:",
      sorted(os.listdir(OUT)))
