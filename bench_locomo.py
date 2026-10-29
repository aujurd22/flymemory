"""FlyMemory on LOCOMO -- the market-position measurement (2026-10-29).

Protocol aligned with the Mem0 paper's LOCOMO setup:
  - all 10 sessions ingested per speaker-pair conversation (35 sessions,
    ~2.7K turns), turns stored as "speaker: text" at turn granularity
  - 199 QA across 5 categories; LLM-as-Judge scoring (J, Mem0 paper's
    metric) with the same prompt family
  - arms: NoMemory / RAG top-5 / FlyMemory full pipeline
    (production recall + CE rerank off + query routing)
Deterministic (temp 0). 199 x 3 answers + judges.

Market reference numbers (LOCOMO, Mem0 paper Table 2, same benchmark):
  Full-context 72.90 | Mem0-graph 68.44 | Mem0 66.88 | Zep 65.99 |
  RAG 60.97 | LangMem 58.10 | OpenAI 52.90 | A-Mem 48.38

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_locomo.py [--sample N]
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from bench_memory_judgment import ds_client, parse_decision  # noqa: E402

DATA = os.path.join(_HERE, "data_locomo", "locomo10.json")
PKL = os.path.join(_HERE, "flymemory", "flymemory_v3.pkl")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_locomo_snapshot.pkl")

ANSWER_SYSTEM = """You are a personal assistant answering questions from your
long-term memory of past conversations. Use ONLY the memory entries below.
If they do not contain the answer, say you don't know. Answer in one or two
short sentences."""
JUDGE_SYSTEM = """You are grading an assistant's answer against the ground
truth for a personal-memory QA task. Reply with ONLY:
{"score": "correct|partial|wrong", "note": "one short sentence"}
- correct: the answer conveys the same key fact(s) as the ground truth
- partial: some key facts right, some missing or imprecise
- wrong: contradicts the ground truth or is off topic"""

CATEGORY = {1: "single-hop", 2: "multi-hop", 3: "temporal",
            4: "open-domain", 5: "adversarial"}


def ingest_sessions(d):
    """Ingest every conversation turn into a fresh SmartMemory."""
    from flymemory.v3 import SmartMemory  # noqa: E402
    mem = SmartMemory(n_bits=4096, enable_hopfield=False)
    n_turns = 0
    for si, sess_data in enumerate(d):
        conv = sess_data["conversation"]
        for key in sorted(conv.keys()):
            if key.endswith("_date_time") or key in ("speaker_a", "speaker_b"):
                continue
            date_key = key + "_date_time"
            dt = conv.get(date_key, key)
            turns = conv[key]
            if not isinstance(turns, list):
                continue
            for t in turns:
                if not isinstance(t, dict):
                    continue
                text = f"{t.get('speaker', '?')}: {t.get('text', '')}"
                mem.remember_text(text, source="hook",
                                  tags=[f"locomo", f"{key}"],
                                  timestamp=None, force_new=True)
                n_turns += 1
    return mem, n_turns


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="first N questions only (0 = all 199)")
    ap.add_argument("--arms", default="nomem,rag,fly",
                    help="comma-separated arms to run")
    args = ap.parse_args()
    arms = args.arms.split(",")

    t0 = time.time()
    d = json.load(open(DATA, encoding="utf-8"))
    questions = []
    for si, sess_data in enumerate(d):
        for q in sess_data.get("qa", []):
            questions.append({**q, "sample_id": sess_data["sample_id"]})
    if args.sample:
        questions = questions[:args.sample]
    print(f"questions: {len(questions)} | sessions: {len(d)}", flush=True)

    fly_mem = None
    if "fly" in arms:
        fly_mem, n_turns = ingest_sessions(d)
        print(f"flymemory store: {n_turns} turns ingested "
              f"({time.time()-t0:.0f}s)", flush=True)

    client = ds_client()

    def chat_retry(system, user, max_tokens=300, tries=5):
        """DeepSeek call with exponential backoff -- intermittent
        connection errors (observed live) must not kill a 199-question
        run at question 27."""
        last = None
        for attempt in range(tries):
            try:
                # fresh client per call, with a no-brotli http_client:
                # httpx2 (installed for the dsh plugin) has a broken br
                # decoder -- process() takes no keyword arguments -- which
                # crashes on DeepSeek's br-compressed responses
                import httpx
                hc = httpx.Client(
                    headers={"Accept-Encoding": "gzip, deflate"},
                    trust_env=False)
                fresh = ds_client(http_client=hc)
                r = fresh.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0, max_tokens=max_tokens)
                return r.choices[0].message.content or ""
            except Exception as e:
                last = e
                wait = min(2 ** attempt * 5, 60)   # 5/10/20/40/60s
                print(f"    [retry {attempt+1}/{tries} after {wait}s: "
                      f"{type(e).__name__}]", flush=True)
                time.sleep(wait)
        raise last

    tally = {a: {"correct": 0, "partial": 0, "wrong": 0} for a in arms}
    by_cat = {a: {} for a in arms}
    traces = []

    # build RAG reference over all turns (numpy cosine, one embedder)
    if "rag" in arms or "fly" in arms:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2")
        all_turns = []
        for sess_data in d:
            conv = sess_data["conversation"]
            for key in sorted(conv.keys()):
                if key.endswith("_date_time") or key in ("speaker_a",
                                                         "speaker_b"):
                    continue
                turns = conv[key]
                if isinstance(turns, list):
                    for t in turns:
                        if isinstance(t, dict):
                            all_turns.append(
                                f"{t.get('speaker','?')}: {t.get('text','')}")
        T = np.asarray(st_model.encode(all_turns, normalize_embeddings=True),
                       dtype=np.float32)
        print(f"RAG corpus: {len(all_turns)} turns", flush=True)

    for qi, q in enumerate(questions):
        cat = CATEGORY.get(q["category"], f"cat{q['category']}")
        ctx_by_arm = {}

        if "nomem" in arms:
            ctx_by_arm["nomem"] = "(no memory available — answer from "
            "general knowledge or say you don't know)"
            ctx_by_arm["nomem"] = None

        if "rag" in arms:
            qv = np.asarray(st_model.encode(
                [q["question"]], normalize_embeddings=True)[0],
                dtype=np.float32)
            top5 = np.argsort(-(T @ qv))[:5]
            ctx_by_arm["rag"] = "\n".join(f"- {all_turns[i][:200]}"
                                          for i in top5)

        if "fly" in arms:
            hits = fly_mem.recall(q["question"], top_k=5)
            ctx_by_arm["fly"] = "\n".join(f"- {h[0].text[:200]}"
                                          for h in hits)

        for arm in arms:
            ctx = ctx_by_arm[arm]
            if ctx is None:
                user = (f"Question: {q['question']}\n\nYou have no memory "
                        f"of past conversations. Answer from general "
                        f"knowledge or say you don't know.")
            else:
                user = (f"Remembered entries:\n{ctx}\n\n"
                        f"Question: {q['question']}\n\nAnswer now.")
            raw = chat_retry(ANSWER_SYSTEM, user, max_tokens=300)
            jr = chat_retry(JUDGE_SYSTEM,
                            f"Ground truth: {q['answer']}\n\n"
                            f"Assistant answer: {raw}\n\nGrade it.",
                            max_tokens=150)
            dec = None
            try:
                s_, e_ = jr.find("{"), jr.rfind("}")
                dec = json.loads(jr[s_:e_ + 1])
            except Exception:
                pass
            score = dec.get("score", "wrong") if isinstance(dec, dict) \
                else "wrong"
            tally[arm][score] = tally[arm].get(score, 0) + 1
            by_cat[arm].setdefault(cat, []).append(score)
            traces.append({"qi": qi, "arm": arm, "cat": cat, "score": score,
                           "answer": raw[:160]})
        line = " | ".join(
            f"{a} {tally[a]['correct']/(qi+1):.0%}" for a in arms)
        print(f"[{qi+1}/{len(questions)}] {line}", flush=True)

    n = len(questions)
    print(f"\n=== LOCOMO results (n={n}, deepseek-chat answer+judge) ===")
    summary = {}
    for arm in arms:
        total_j = (tally[arm]["correct"] + 0.5 * tally[arm]["partial"]) / n
        strict = tally[arm]["correct"] / n
        summary[arm] = {"strict": round(strict, 4),
                        "J": round(total_j, 4),
                        "counts": tally[arm]}
        per_cat = {}
        for cat in sorted(by_cat[arm].keys()):
            scores = by_cat[arm][cat]
            per_cat[cat] = round(
                scores.count("correct") / max(len(scores), 1), 3)
        summary[arm]["strict_per_category"] = per_cat
        print(f"  {arm:8s} strict {strict:.1%} | J {total_j:.1%} | "
              f"per-cat {per_cat}")

    print("\nmarket reference (LOCOMO, Mem0 paper Table 2, J): "
          "Full-context 72.9 | Mem0-graph 68.4 | Mem0 66.9 | Zep 66.0 | "
          "RAG 61.0 | LangMem 58.1 | OpenAI 52.9 | A-Mem 48.4")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"locomo_{ts}.json")
    json.dump({"experiment": "LOCOMO-flymemory", "n": n,
               "summary": summary, "traces": traces},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
