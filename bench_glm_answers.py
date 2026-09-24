"""Grade the GLM answer arm (50 questions, seed 7) with the same DeepSeek judge.

Reads reports/glm_answers_50.json (idx + answer), grades each against the
dataset gold with the same JUDGE_SYSTEM protocol as bench_lme_e2e.py, and
reports the comparison: DeepSeek answer arm vs GLM answer arm.

Run:
  DEEPSEEK_API_KEY=sk-... python bench_glm_answers.py
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_memory_judgment import ds_client, parse_decision  # noqa: E402
from bench_lme_e2e import JUDGE_SYSTEM, MODEL  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
answers = json.load(open(os.path.join(here, "reports", "glm_answers_50.json"),
                         encoding="utf-8"))
with open(os.path.join(here, "data_longmemeval", "longmemeval_oracle.json"),
          encoding="utf-8") as f:
    data = json.load(f)

client = ds_client()
tally = {"correct": 0, "partial": 0, "wrong": 0, "judge_error": 0}
traces = []
t0 = time.time()
for a in answers:
    q = data[a["idx"]]
    juser = (f"QUESTION: {q['question']}\n\nGROUND TRUTH: "
             f"{q.get('answer', '')}\n\nASSISTANT ANSWER: {a['answer']}\n\nGrade now.")
    verdict = "judge_error"
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": juser}],
            temperature=0, max_tokens=200)
        d, err = parse_decision(r.choices[0].message.content or "")
        if not err and d.get("score"):
            verdict = d["score"]
    except Exception:
        pass
    tally[verdict] = tally.get(verdict, 0) + 1
    traces.append({"idx": a["idx"], "question": q["question"],
                   "answer": a["answer"], "gold": q.get("answer", ""),
                   "verdict": verdict})
    if len(traces) % 10 == 0:
        print(f"  {len(traces)}/50 ({time.time()-t0:.0f}s) {tally}", flush=True)

n = len(answers)
n_scored = sum(v for k, v in tally.items() if k != "judge_error")
print(f"\n=== GLM answer arm (n={n}) ===")
for k in ("correct", "partial", "wrong", "judge_error"):
    print(f"  {k:12s} {tally.get(k,0):3d}")
if n_scored:
    acc = (tally.get("correct", 0) + 0.5 * tally.get("partial", 0)) / n_scored
    print(f"  score = {acc:.3f}  (strict = {tally.get('correct',0)/n_scored:.3f})")
    print("  DeepSeek answer arm (same 50, seed 7): 0.320 strict / 0.350 weighted")

out = os.path.join(here, "reports", f"glm_graded_{int(time.time())}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"answer_model": "GLM (ZCode session)", "judge_model": MODEL,
               "tally": tally, "traces": traces}, f, ensure_ascii=False, indent=1)
print(f"trace: {out}")
