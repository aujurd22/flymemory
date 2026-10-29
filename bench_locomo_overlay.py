"""P-LOCO-OVERLAY: overlay arm on the LOCOMO 30-question sample
(registered, research/RESEARCH.md P-2026-10-29-LOCO-OVERLAY -- verdict
PENDING there; criteria locked before run).

Arms: FLY (turn-only, replicates the 10% baseline) / FLY+OVERLAY
(per-session deepseek summaries stored on top of turns) / RAG (control).
Single-hop category (0/10 baseline) is the targeted fix.

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_locomo_overlay.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

DATA = os.path.join(_HERE, "data_locomo", "locomo10.json")
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_locomo_snapshot.pkl")
OVERLAY_CACHE = os.path.join(_HERE, "reports", "locomo_overlay_cache.json")
N_QUESTIONS = 30

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
SUMMARY_SYSTEM = """Condense this conversation session into durable facts:
names, relationships, numbers, dates, plans, preferences, outcomes.
ONE compact paragraph. Include the session date. Do not invent."""
CATEGORY = {1: "single-hop", 2: "multi-hop", 3: "temporal",
            4: "open-domain", 5: "adversarial"}


def main():
    t0 = time.time()
    import shutil
    shutil.copy(os.path.join(_HERE, "flymemory", "flymemory_v3.pkl"), SNAP)
    d = json.load(open(DATA, encoding="utf-8"))

    # ---- generate session summaries (cached) ----
    if os.path.exists(OVERLAY_CACHE):
        summaries = json.load(open(OVERLAY_CACHE, encoding="utf-8"))
        print(f"overlay cache loaded: {len(summaries)} sessions", flush=True)
    else:
        summaries = {}
    from flymemory.v3 import SmartMemory  # noqa: E402

    client = None
    missing = []
    for si, sess_data in enumerate(d):
        sid = sess_data["sample_id"]
        if sid in summaries:
            continue
        missing.append((si, sid, sess_data))
    if missing:
        from bench_memory_judgment import ds_client
        client = ds_client()
        for k, (si, sid, sess_data) in enumerate(missing):
            conv = sess_data["conversation"]
            chunks = []
            for key in sorted(conv.keys()):
                if key.endswith("_date_time") or key in ("speaker_a",
                                                         "speaker_b"):
                    continue
                dt = conv.get(key + "_date_time", "")
                turns = conv[key]
                if not isinstance(turns, list):
                    continue
                body = " ".join(f"{t.get('speaker','?')}: {t.get('text','')}"
                                for t in turns if isinstance(t, dict))
                chunks.append(f"[{dt}] {body[:1500]}")
            convo = "\n".join(chunks)[:8000]
            raw = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": SUMMARY_SYSTEM},
                          {"role": "user", "content": convo}],
                temperature=0, max_tokens=600).choices[0].message.content
            summaries[sid] = raw or ""
            if (k + 1) % 5 == 0:
                print(f"  summarized {k+1}/{len(missing)}", flush=True)
        json.dump(summaries, open(OVERLAY_CACHE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"generated {len(missing)} session summaries", flush=True)

    # ---- build both stores ----
    # FLY: turn-only (replicates baseline)
    fly_mem = SmartMemory(n_bits=4096, enable_hopfield=False)
    n_turns = 0
    for si, sess_data in enumerate(d):
        conv = sess_data["conversation"]
        for key in sorted(conv.keys()):
            if key.endswith("_date_time") or key in ("speaker_a", "speaker_b"):
                continue
            turns = conv[key]
            if not isinstance(turns, list):
                continue
            for t in turns:
                if isinstance(t, dict):
                    fly_mem.remember_text(
                        f"{t.get('speaker','?')}: {t.get('text','')}",
                        source="hook", force_new=True)
                    n_turns += 1
    # FLY+OVERLAY: turns + one summary entry per session
    ov_mem = SmartMemory(n_bits=4096, enable_hopfield=False)
    for m in fly_mem.memories:
        ov_mem.remember_text(m.text, source="hook",
                             timestamp=m.timestamp, force_new=True)
    for sid, text in summaries.items():
        ov_mem.remember_text(f"[Session summary {sid}] {text}",
                             source="model", force_new=True)
    print(f"fly store: {n_turns} turns | overlay store: +{len(summaries)} "
          f"summaries ({time.time()-t0:.0f}s)", flush=True)

    # ---- questions (same 30 as the baseline sample: first 30 in file order) --
    questions = []
    for si, sess_data in enumerate(d):
        for q in sess_data.get("qa", []):
            questions.append({**q, "sample_id": sess_data["sample_id"]})
        if len(questions) >= N_QUESTIONS:
            questions = questions[:N_QUESTIONS]
            break

    from bench_memory_judgment import ds_client  # noqa: E402
    if client is None:
        client = ds_client()

    arms = ["RAG", "FLY", "FLY+OVERLAY"]
    tally = {a: {"correct": 0, "partial": 0, "wrong": 0} for a in arms}
    cat_correct = {a: {} for a in arms}
    traces = []

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
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
    T = np.asarray(st.encode(all_turns, normalize_embeddings=True),
                   dtype=np.float32)

    def judge(answer, gold):
        raw = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user",
                       "content": f"Ground truth: {gold}\n\n"
                                  f"Assistant answer: {answer}\n\nGrade it."}],
            temperature=0, max_tokens=150).choices[0].message.content or ""
        s_, e_ = raw.find("{"), raw.rfind("}")
        try:
            return json.loads(raw[s_:e_ + 1]).get("score", "wrong")
        except Exception:
            return "wrong"

    for qi, q in enumerate(questions):
        qv = np.asarray(st.encode([q["question"]],
                                  normalize_embeddings=True)[0],
                        dtype=np.float32)
        ctx_by = {}
        top5 = np.argsort(-(T @ qv))[:5]
        ctx_by["RAG"] = "\n".join(f"- {all_turns[i][:200]}" for i in top5)
        fly_hits = fly_mem.recall(q["question"], top_k=5)
        ctx_by["FLY"] = "\n".join(f"- {h[0].text[:200]}" for h in fly_hits)
        ov_hits = ov_mem.recall(q["question"], top_k=5)
        ctx_by["FLY+OVERLAY"] = "\n".join(f"- {h[0].text[:200]}"
                                          for h in ov_hits)

        for arm in arms:
            user = (f"Remembered entries:\n{ctx_by[arm]}\n\n"
                    f"Question: {q['question']}\n\nAnswer now.")
            raw = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": ANSWER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0, max_tokens=300).choices[0].message.content
            score = judge(raw or "", q["answer"])
            cat = CATEGORY.get(q["category"], "other")
            tally[arm][score] = tally[arm].get(score, 0) + 1
            cat_correct[arm].setdefault(cat, []).append(
                score == "correct")
            traces.append({"qi": qi, "arm": arm, "cat": cat,
                           "score": score, "answer": (raw or "")[:150]})
        line = " | ".join(f"{a} {tally[a]['correct']/(qi+1):.0%}"
                          for a in arms)
        print(f"[{qi+1}/{len(questions)}] {line}", flush=True)

    n = len(questions)
    print(f"\n=== P-LOCO-OVERLAY (n={n}) ===")
    summary = {}
    for arm in arms:
        strict = tally[arm]["correct"] / n
        per = {c: sum(v) / len(v) for c, v in cat_correct[arm].items()}
        summary[arm] = {"strict": round(strict, 4),
                        "per_category": {c: round(v, 3) for c, v
                                         in per.items()}}
        print(f"  {arm:12s} strict {strict:.1%} | per-cat "
              f"{ {c: f'{v:.0%}' for c, v in per.items()} }")

    sh_base = cat_correct["FLY+OVERLAY"].get("single-hop", [])
    sh_fly = cat_correct["FLY"].get("single-hop", [])
    print(f"\nsingle-hop: FLY {sum(sh_fly)}/{len(sh_fly)} -> "
          f"FLY+OVERLAY {sum(sh_base)}/{len(sh_base)}")
    print("VERDICT rule: SUPPORTED iff overlay >= FLY + 5pp total AND "
          "single-hop improves; partial/no-gain otherwise (judged on the "
          "numbers above).")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"loco_overlay_{ts}.json")
    json.dump({"experiment": "P-LOCO-OVERLAY", "n": n,
               "summary": summary, "traces": traces},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
