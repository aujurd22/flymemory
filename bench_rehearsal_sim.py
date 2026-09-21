"""Rehearsal-policy simulation: does the wide rehearsal rule (refresh every
entry with sim > 0.5, the shipped behavior) create an immortal popular-memory
bias, compared to top-k-only rehearsal?

Simulation: a library grows over time (new entries), a Zipf-flavored query
stream repeatedly touches some topics and abandons others. Three policies:
  A: top-k-only rehearsal (k=3)
  B: sim>0.5 wide rehearsal (shipped behavior)
  C: sim>0.5 rehearsal weighted by sim (refresh amount proportional)
Metrics at the end: access-count Gini, fraction of entries with retention
pinned at 1.0, retention of ABANDONED topics (can they be forgotten at all?).

Run: python bench_rehearsal_sim.py [--steps 6000]
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import SmartMemory, _embed  # noqa: E402

DAY = 86400.0


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def run_policy(policy, steps, tau_days, seed, new_every=40):
    """One simulated agent lifetime. Returns per-entry access counts and the
    identity of abandoned vs hot entries for final analysis."""
    rng = np.random.default_rng(seed)
    topics = [f"主题{i}：{kw}" for i, kw in enumerate(
        ["果蝇视觉", "打印机库存", "跨境电商", "GPU训练", "MCP运维",
         "数据库备份", "界面设计", "音频降噪", "法律合规", "风控模型",
         "农业灌溉", "航天材料", "烹饪方法", "登山装备", "古典音乐"])]
    # Zipf popularity over topics; the last 5 topics are ABANDONED early
    pop = 1.0 / (np.arange(1, len(topics) + 1) ** 1.2)
    pop[-5:] *= 0.05

    texts = []
    for i, t in enumerate(topics):
        for j in range(12):
            texts.append(f"{t}的记忆条目{j}号：相关细节与数值。")
    embs = [_embed(t) for t in texts]
    E = np.stack(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    now = time.time()
    access = np.zeros(len(texts), dtype=np.int64)
    created = now - np.linspace(60 * DAY, 0, len(texts))  # spread over 60 days
    last = created.copy()

    topic_of = np.array([i // 12 for i in range(len(texts))])

    for step in range(steps):
        # pick a query topic by popularity, then the 3 nearest entries
        ti = rng.choice(len(topics), p=pop / pop.sum())
        qe = embs[(ti * 12 + int(rng.integers(0, 12))) % len(texts)]
        sims = E @ qe
        order = np.argsort(-sims)
        if policy == "A":
            for idx in order[:3]:
                access[idx] += 1
                last[idx] = now - (steps - step) / steps * 60 * DAY
        elif policy in ("B", "C"):
            hot = order[sims[order] > 0.5]
            for idx in hot:
                access[idx] += 1
                last[idx] = now - (steps - step) / steps * 60 * DAY

    return access, last, topic_of, len(topics)


def retention(last, access, tau, now):
    dt = now - last
    base = (1.0 + dt / tau) ** -0.5
    rehearse = np.minimum(1.0 + 0.5 * np.log2(1 + access), 1.0)
    return np.minimum(base * rehearse, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()

    tau = 30 * DAY
    now = time.time()
    print(f"steps={args.steps}, tau=30d; last 5 topics are ABANDONED "
          f"(popularity x0.05)\n")
    print(f"{'policy':8s} {'accessGini':>10s} {'pinned@1.0':>10s} "
          f"{'abandoned-retention':>20s} {'hot-retention':>14s}")
    for policy in ("A", "B", "C"):
        access, last, topic_of, n_topics = run_policy(policy, args.steps, 30, seed=3)
        ret = retention(last, access, tau, now)
        abandoned = topic_of >= n_topics - 5
        hot = ~abandoned
        g = gini(access)
        pinned = float((ret >= 0.999).mean())
        ab_ret = float(ret[abandoned].mean()) if abandoned.any() else float("nan")
        hot_ret = float(ret[hot].mean()) if hot.any() else float("nan")
        print(f"{policy:8s} {g:10.3f} {pinned:10.3f} {ab_ret:20.3f} {hot_ret:14.3f}")

    print("\nA: top-k only; B: wide sim>0.5 (shipped); C: same as B in count, "
          "marked for future sim-weighted refresh.")
    print("watch: B/C 'abandoned-retention' -- if abandoned memories stay near "
          "1.0 while A lets them decay, the wide rule creates immortal chatter.")


if __name__ == "__main__":
    main()
