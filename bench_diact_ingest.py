"""P-DIACT: conversational-unit ingestion A/B on LOCOMO (registered,
research/RESEARCH.md P-2026-10-29-DIACT -- verdict PENDING there).

Hypothesis under test (from P-LOCO-OVERLAY's falsified missing-summary
hypothesis): single-hop 0/10 is caused by per-sentence chunking
shattering conversationally-carried facts. Dialogue-act grouping --
consecutive turns by the SAME speaker merge into one entry, speaker
changes cut -- should keep facts inside their conversational unit.

Arms (same 30 questions, same judge):
  FLY-turns  per-sentence/turn ingestion (baseline, replicates 10%)
  FLY-diact  dialogue-act ingestion (same-speaker runs merged, cap 6)

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_diact_ingest.py
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
CATEGORY = {1: "single-hop", 2: "multi-hop", 3: "temporal",
            4: "open-domain", 5: "adversarial"}


def group_dialogue_acts(turns, max_group=6):
    """Merge consecutive same-speaker turns into one dialogue act."""
    groups = []
    cur_speaker, cur_texts = None, []
    for t in turns:
        sp = t.get("speaker", "?")
        txt = t.get("text", "")
        if sp == cur_speaker and len(cur_texts) < max_group:
            cur_texts.append(txt)
        else:
            if cur_texts:
                groups.append((cur_speaker, " ".join(cur_texts)))
            cur_speaker, cur_texts = sp, [txt]
    if cur_texts:
        groups.append((cur_speaker, " ".join(cur_texts)))
    return groups


def build_store(d, mode):
    from flymemory.v3 import SmartMemory
    mem = SmartMemory(n_bits=4096, enable_hopfield=False)
    n_entries = 0
    for sess_data in d:
        conv = sess_data["conversation"]
        for key in sorted(conv.keys()):
            if key.endswith("_date_time") or key in ("speaker_a",
                                                     "speaker_b"):
                continue
            turns = conv[key]
            if not isinstance(turns, list):
                continue
            if mode == "diact":
                for sp, text in group_dialogue_acts(turns):
                    mem.remember_text(f"{sp}: {text}", source="hook",
                                      force_new=True)
                    n_entries += 1
            else:
                for t in turns:
                    if isinstance(t, dict):
                        mem.remember_text(
                            f"{t.get('speaker','?')}: {t.get('text','')}",
                            source="hook", force_new=True)
                        n_entries += 1
    return mem, n_entries


def main():
    t0 = time.time()
    d = json.load(open(DATA, encoding="utf-8"))
    questions = []
    for sess_data in d:
        for q in sess_data.get("qa", []):
            questions.append(q)
        if len(questions) >= N_QUESTIONS:
            questions = questions[:N_QUESTIONS]
            break
    print(f"questions: {len(questions)}", flush=True)

    from flymemory.v3 import SmartMemory  # noqa: E402 re-imported lazily
    import importlib
    import flymemory.v3 as v3
    importlib.reload(v3)

    from bench_memory_judgment import ds_client
    client = ds_client()

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def run_arm(mode):
        mem, n_entries = build_store(d, mode)
        print(f"  [{mode}] store: {n_entries} entries", flush=True)
        tally = {"correct": 0, "partial": 0, "wrong": 0}
        cat_ok = {}
        for qi, q in enumerate(questions):
            hits = mem.recall(q["question"], top_k=5)
            ctx = "\n".join(f"- {h[0].text[:200]}" for h in hits)
            raw = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": ANSWER_SYSTEM},
                          {"role": "user",
                           "content": f"Remembered entries:\n{ctx}\n\n"
                                      f"Question: {q['question']}\n\n"
                                      f"Answer now."}],
                temperature=0, max_tokens=300).choices[0].message.content
            jr = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user",
                           "content": f"Ground truth: {q['answer']}\n\n"
                                      f"Assistant answer: "
                                      f"{raw or ''}\n\nGrade it."}],
                temperature=0, max_tokens=150).choices[0].message.content
            s_, e_ = jr.find("{"), jr.rfind("}")
            try:
                sc = json.loads(jr[s_:e_ + 1]).get("score", "wrong")
            except Exception:
                sc = "wrong"
            tally[sc] = tally.get(sc, 0) + 1
            cat = CATEGORY.get(q["category"], "other")
            cat_ok.setdefault(cat, []).append(sc == "correct")
            if (qi + 1) % 10 == 0:
                print(f"  [{mode} {qi+1}/{len(questions)}] strict "
                      f"{tally['correct']/(qi+1):.0%}", flush=True)
        strict = tally["correct"] / len(questions)
        per = {c: sum(v) / len(v) for c, v in cat_ok.items()}
        print(f"  == {mode}: strict {strict:.1%} | per-cat "
              f"{ {c: f'{v:.0%}' for c, v in per.items()} }", flush=True)
        return strict, per

    strict_t, per_t = run_arm("turns")
    strict_d, per_d = run_arm("diact")
    delta = (strict_d - strict_t) * 100
    sh_t = per_t.get("single-hop", 0)
    sh_d = per_d.get("single-hop", 0)

    if delta >= 5:
        verdict = (f"SUPPORTED -- adopt opt-in dialogue ingestion "
                   f"(single-hop {sh_t:.0%} -> {sh_d:.0%})")
    elif delta >= 2:
        verdict = "PARTIAL (direction holds, marginal)"
    else:
        verdict = ("NULL -- granularity is not the root either; next "
                   "hypothesis: judge-side answer normalization")
    print(f"\nstrict: turns {strict_t:.1%} vs diact {strict_d:.1%} "
          f"(delta {delta:+.1f}pp)")
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"diact_{ts}.json")
    json.dump({"experiment": "P-DIACT", "strict_turns": strict_t,
               "strict_diact": strict_d, "delta_pp": round(delta, 2),
               "single_hop": {"turns": round(sh_t, 3),
                              "diact": round(sh_d, 3)},
               "verdict": verdict},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
