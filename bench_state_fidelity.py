"""State-fidelity audit: what does the engine DO with a real state update?

Naive-RAG's core failure (Phase 2) is stale facts outranking new ones, but the
deeper engine question is: when a user states an updated fact ("my number is
now 139..."), does the store end up holding the NEW state?  The dedup ladder
has three outcomes for a (old, new) pair:

  sim > dup_t (0.92/0.95)  -> strengthen : access refreshed, text NOT updated
                              => NEW STATE SILENTLY DROPPED
  sim > merge_t (0.75/0.85)-> merge      : text rewritten to the longer side
                              => in-place update (history lost, state correct)
  else                     -> both kept  => stale risk at retrieval

This audit runs N realistic state-update pairs through exactly that ladder
and reports which outcome each got, plus whether the store's final text
actually contains the new state.

Run: python bench_state_fidelity.py
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flymemory"))

from flymemory.v3 import SmartMemory, _embed  # noqa: E402
import numpy as np  # noqa: E402

# (old, new, new-state keyword, stale keyword)
PAIRS = [
    ("The user's phone number is 138-0000-1111.",
     "The user's phone number is 139-9999-8888.", "139-9999-8888", "138-0000-1111"),
    ("The user's production server is 192.168.1.50.",
     "The user's production server is 192.168.1.99.", "192.168.1.99", "192.168.1.50"),
    ("The user lives at Oak Street 5.",
     "The user lives at Oak Street 8.", "Oak Street 8", "Oak Street 5"),
    ("The user's team is called Team Falcon.",
     "The user's team is called Team Kestrel.", "Team Kestrel", "Team Falcon"),
    ("The user drives a gasoline hatchback.",
     "The user drives an electric car.", "electric", "gasoline"),
    ("The user studies Japanese with a paper textbook.",
     "The user studies Japanese with a spaced-repetition app.", "spaced-repetition", "paper textbook"),
    ("The user's warehouse uses the XinDa label printer.",
     "The user's warehouse uses the HTW-109 label printer.", "HTW-109", "XinDa"),
    ("The user writes most backend code in Python.",
     "The user writes most backend code in Rust.", "Rust", "Python"),
    ("The user hosts their blog on a rented VPS.",
     "The user hosts their blog on a static hosting platform.", "static hosting", "VPS"),
    ("The user reports to Manager Zhang.",
     "The user reports to Manager Liu.", "Manager Liu", "Manager Zhang"),
    ("The user prefers dark mode in every app.",
     "The user prefers light mode in every app.", "light mode", "dark mode"),
    ("The user's primary editor is PyCharm.",
     "The user's primary editor is Neovim.", "Neovim", "PyCharm"),
    ("The user lives in Shenzhen.",
     "The user lives in Hangzhou.", "Hangzhou", "Shenzhen"),
    ("The user subscribes to the Pro tier.",
     "The user is on the free tier.", "free tier", "Pro tier"),
    ("The user's laptop runs Windows 11.",
     "The user's laptop runs Fedora Linux.", "Fedora", "Windows 11"),
    ("The user's meeting is scheduled for 15:00.",
     "The user's meeting is scheduled for 16:00.", "16:00", "15:00"),
    ("The user works from the downtown office.",
     "The user works from the riverside office.", "riverside", "downtown"),
    ("The user's admin password hint is 'old one'.",
     "The user's admin password hint is 'blue fish'.", "blue fish", "old one"),
    ("The user deploys on AWS.",
     "The user deploys on Hetzner.", "Hetzner", "AWS"),
    ("The user's cat is called Mochi.",
     "The user's cat is called Daifuku.", "Daifuku", "Mochi"),
]


def main():
    outcomes = {"strengthen_dropped": [], "merged_inplace": [], "kept_separate": [],
                "rejected": []}
    for old, new, new_kw, stale_kw in PAIRS:
        mem = SmartMemory(n_bits=4096)
        r1 = mem.remember(old, source="import", force_new=True)
        r2 = mem.remember(new, source="model")
        e1, e2 = _embed(old), _embed(new)
        sim = float(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2))
        final_texts = [m.text for m in mem.memories]
        has_new = any(new_kw in t for t in final_texts)
        act = r2["action"]
        if not has_new:
            # strengthen (text untouched) or merge-without-rewrite (new text
            # was SHORTER, so nothing was written): the new state is gone
            cat = "new_state_dropped"
        elif act == "new":
            cat = "kept_separate"
        elif act == "merged":
            cat = "merged_inplace"
        else:
            cat = f"other_{act}"
        outcomes.setdefault(cat, []).append((old, new, round(sim, 3)))

    n = len(PAIRS)
    print(f"pairs: {n}")
    for k, v in outcomes.items():
        print(f"\n{k}: {len(v)}/{n}")
        for old, new, sim in v:
            print(f"  sim={sim}  {old[:40]!r} -> {new[:40]!r}")

    print("\nverdict per outcome:")
    print("  merged_inplace  = state correct, HISTORY LOST (no lineage)")
    print("  kept_separate   = history kept, STALE RISK at retrieval")
    print("  strengthen_dropped = NEW STATE LOST (engine bug territory)")


if __name__ == "__main__":
    main()
