"""Phase 2: Oracle vs Autonomous -- does memory state maintenance make the
AGENT answer correctly?  End-to-end over the 30 state cases (sup/noop/frt):

  build store -> apply ONE of three maintenance policies -> production recall
  -> LLM answers the question from the recalled entries -> LLM judge.

Arms (five, separating retrieval quality from state maintenance):
  no_mem      : no memory at all -- the LLM answers blind
  dense_naive : naive vector RAG -- cosine top-3, store-new-never-maintain
  none        : production RRF retrieval, store-new-never-maintain
  oracle      : production RRF + gold state operations (mechanism ceiling)
  autonomous  : production RRF + DeepSeek state operations

Verdicts (LLM judge, one per arm x case):
  current  : answer states the CURRENT fact correctly
  stale    : answer presents the OUTDATED/deleted fact as current
  unknown  : answer admits it doesn't know (correct when the gold current
             fact is empty -- deleted wrong facts must not resurface)
  wrong    : anything else

Run:
  python bench_e2e_answer.py            # all three arms
  python bench_e2e_answer.py --arms none,oracle
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
try:
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 4))
except ImportError:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flymemory"))

from bench_memory_judgment import (build_memory, gold_decision,  # noqa: E402
                                   deepseek_decide, parse_decision,
                                   execute, ds_client)
import numpy as np  # noqa: E402
from flymemory.v3 import _embed  # noqa: E402

DAY = 86400.0
ANSWER_MODEL = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

ANSWER_SYSTEM = """You are a personal assistant answering questions from your
long-term memory. Use ONLY the remembered facts below. Each entry shows its
age. If the facts do not contain the answer, say plainly that you don't have
that information. Answer in one short sentence."""

JUDGE_SYSTEM = """You are a strict evaluator for a memory-backed assistant.
You get the question, the assistant's answer, the CURRENT ground-truth fact
(may be empty: the fact was deleted as wrong), and the OUTDATED fact that
must NOT be presented as current.

Verdicts:
- "current": the answer states the current ground-truth fact correctly
- "stale"  : the answer presents the outdated (or deleted-wrong) fact as
             the current truth
- "unknown": the answer says it doesn't know / has no record (this is the
             CORRECT behavior when the ground-truth fact is empty)
- "wrong"  : anything else
Mentioning the old fact as history ("previously X, now Y") is still
"current" as long as the present tense is right.

Reply with ONLY: {"verdict": "current|stale|unknown|wrong", "note": "..."}"""


def answer_question(client, hits, question, now, blind=False):
    if blind:
        # TRUE no-memory baseline: the LLM answers from its own knowledge.
        # (The earlier hardcoded "I don't know" measured abstention, not
        # capability -- external review finding, fixed 2026-09-24.)
        r = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[{"role": "system", "content":
                       "You are a personal assistant. Answer the user's "
                       "question in one short sentence."},
                      {"role": "user", "content": question}],
            temperature=0, max_tokens=200)
        return (r.choices[0].message.content or "").strip()
    if not hits:
        return "I don't have any information about that."
    lines = []
    for m, _sim, _eff in hits:
        age = max(0, int((now - m.last_accessed) / DAY))
        lines.append(f"#{m.memory_id} ({age}d ago): {m.text}")
    user = (f"REMEMBERED FACTS:\n" + "\n".join(lines)
            + f"\n\nUSER QUESTION: {question}\n\nAnswer now.")
    r = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "system", "content": ANSWER_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=200)
    return (r.choices[0].message.content or "").strip()


def judge_answer(client, question, answer, cur_texts, stale_texts):
    cur_desc = "; ".join(cur_texts) if cur_texts else \
        "(EMPTY -- the fact was deleted as wrong; the correct behavior is to say you don't know)"
    stale_desc = "; ".join(stale_texts) if stale_texts else "(none)"
    user = (f"QUESTION: {question}\n\nANSWER: {answer}\n\n"
            f"CURRENT GROUND TRUTH: {cur_desc}\n"
            f"OUTDATED FACT: {stale_desc}\n\nJudge now.")
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0, max_tokens=200)
        d, err = parse_decision(r.choices[0].message.content or "")
        if err or "verdict" not in d:
            return {"verdict": "judge_error", "note": (err or "no verdict")[:80]}
        return {"verdict": d.get("verdict"), "note": str(d.get("note", ""))[:80]}
    except Exception as e:
        return {"verdict": "judge_error", "note": str(e)[:120]}


def maintenance_decision(arm, case, client):
    if arm == "oracle":
        return gold_decision(case), None
    if arm == "autonomous":
        raw = deepseek_decide(client, case)
        d, err = parse_decision(raw)
        return d, err
    if arm in ("none", "dense_naive"):
        g = gold_decision(case)
        return {"remember": g["remember"], "supersede": [], "consolidate": [],
                "forget": [], "reason": "naive: store new, never maintain"}, None
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="no_mem,dense_naive,none,oracle,autonomous")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "memory_judgment.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    arms = args.arms.split(",")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    cases = [c for c in dataset["cases"] if "e2e" in c]
    if args.limit:
        cases = cases[:args.limit]

    client = ds_client()
    print(f"cases: {len(cases)} | arms: {arms} | answer/judge: "
          f"{ANSWER_MODEL}/{JUDGE_MODEL}\n", flush=True)

    tally = {arm: {} for arm in arms}
    traces = []
    for ci, case in enumerate(cases):
        e2e = case["e2e"]
        for arm in arms:
            decision, errors, hits = {}, [], []
            if arm == "no_mem":
                # TRUE baseline: blind answer from the model's own knowledge
                answer = answer_question(client, [], e2e["question"],
                                         time.time(), blind=True)
            else:
                mem, id_map = build_memory(case)
                decision, perr = maintenance_decision(arm, case, client)
                if perr:
                    decision = {}
                executed, errors = execute(mem, case, decision, id_map)
                hits = mem.recall(e2e["question"], top_k=3)
                if arm == "dense_naive":
                    # pure cosine top-3 over the store (no RRF): isolates what
                    # hybrid retrieval adds over naive vector RAG
                    from flymemory.v3 import split_chunks
                    En = mem._emb_matrix()
                    En = En / (np.linalg.norm(En, axis=1, keepdims=True) + 1e-8)
                    Q = np.stack([_embed(c) for c in
                                  (split_chunks(e2e["question"]) or
                                   [e2e["question"]])])
                    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
                    sims = (Qn @ En.T).max(axis=0)
                    hits = [(mem.memories[int(i)], float(sims[i]), 0.0)
                            for i in np.argsort(-sims)[:3]]
                answer = answer_question(client, hits, e2e["question"], time.time())
            verdict = judge_answer(client, e2e["question"], answer,
                                   e2e["current_keywords"], e2e["stale_keywords"])
            tally[arm][verdict["verdict"]] = tally[arm].get(verdict["verdict"], 0) + 1
            traces.append({"case_id": case["case_id"], "arm": arm,
                           "question": e2e["question"], "answer": answer,
                           "verdict": verdict, "n_hits": len(hits),
                           "decision": decision, "exec_errors": errors})
        print(f"[{ci+1:2d}/{len(cases)}] {case['case_id']:8s} "
              + " | ".join(f"{arm}: {traces[-len(arms)+i]['verdict']['verdict']}"
                           for i, arm in enumerate(arms)), flush=True)

    print(f"\n=== Phase 2: answer-level outcomes (n={len(cases)} questions) ===")
    print(f"{'arm':12s} {'current':>8s} {'stale':>7s} {'unknown':>8s} "
          f"{'wrong':>6s} {'j_err':>6s}")
    for arm in arms:
        t = tally[arm]
        n = sum(t.values())
        print(f"{arm:12s} {t.get('current',0):>4d}({t.get('current',0)/n*100:4.0f}%) "
              f"{t.get('stale',0):>3d}({t.get('stale',0)/n*100:3.0f}%) "
              f"{t.get('unknown',0):>4d}({t.get('unknown',0)/n*100:3.0f}%) "
              f"{t.get('wrong',0):>3d}   {t.get('judge_error',0):>3d}")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports", f"e2e_answer_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"phase": "2-e2e", "arms": arms, "answer_model": ANSWER_MODEL,
                   "judge_model": JUDGE_MODEL, "traces": traces,
                   "tally": tally}, f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
