"""Full hallucination audit of the 940 session-consolidated entries.

Every consolidated entry that entered the overlay store (bench_granularity.py)
is judged against its source conversation by an LLM judge (same protocol as
the 30-session pilot, which found 0/87 hallucinated). This full audit answers:
is it safe to ingest LLM-consolidated entries into the memory library at all,
or do hallucinations accumulate?

Output: hallucination rate per entry + per session, full trace in reports/.

Run:
  DEEPSEEK_API_KEY=sk-... python bench_consolidation_audit.py [--sample N]
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_memory_judgment import ds_client, parse_decision  # noqa: E402

JUDGE_SYSTEM = """You are a strict factuality judge. Given the full CONVERSATION and
distilled MEMORY ENTRIES extracted from it, decide for each entry whether
every claim in it is supported by the conversation. Reply with ONLY:
{"hallucinated_entries": [indices of entries containing unsupported claims],
 "notes": "short"}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="judge only N random sessions (0 = all 940)")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cons = json.load(open(os.path.join(here, "reports",
                                       "consolidated_entries.json"),
                          encoding="utf-8"))
    with open(os.path.join(here, "data_longmemeval", "longmemeval_oracle.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    sessions = {}
    for q in data:
        for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            if sid not in sessions:
                sessions[sid] = [f"{t.get('role', 'user')}: {t.get('content', '')}"
                                 for t in sess]
    if args.sample:
        cons = random.Random(args.seed).sample(cons, args.sample)

    client = ds_client()
    n_entries = 0
    bad_sessions = []
    t0 = time.time()
    for si, c in enumerate(cons):
        convo = "\n".join(sessions[c["sid"]])[:6000]
        entries = "\n".join(f"[{i}] {e}" for i, e in enumerate(c["entries"]))
        user = f"CONVERSATION:\n{convo}\n\nDISTILLED ENTRIES:\n{entries}\n\nJudge now."
        try:
            r = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0, max_tokens=300)
            d, err = parse_decision(r.choices[0].message.content or "")
            hall = d.get("hallucinated_entries", []) if not err else "judge_error"
        except Exception as e:
            hall, err = "api_error", str(e)[:100]
        n_entries += len(c["entries"])
        if hall:
            bad_sessions.append({"sid": c["sid"], "hallucinated": hall,
                                 "error": err if err else None,
                                 "entries": c["entries"]})
        if (si + 1) % 100 == 0:
            print(f"  {si+1}/{len(cons)} ({time.time()-t0:.0f}s) "
                  f"bad sessions: {len(bad_sessions)}", flush=True)

    n = len(cons)
    print(f"\n=== Consolidation hallucination audit (sessions={n}, "
          f"entries={n_entries}) ===")
    print(f"  sessions with hallucinated entries: {len(bad_sessions)}")
    print(f"  hallucination-free rate (session level): "
          f"{(n-len(bad_sessions))/n*100:.1f}%")

    os.makedirs(os.path.join(here, "reports"), exist_ok=True)
    out = os.path.join(here, "reports",
                       f"consolidation_audit_{n}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sessions": n, "entries": n_entries,
                   "bad_sessions": len(bad_sessions), "detail": bad_sessions},
                  f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
