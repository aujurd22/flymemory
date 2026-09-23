"""Memory Judgment Benchmark v1 -- Phase 1 of the evaluation roadmap.

Question: when an LLM sees FlyMemory's current state plus a new user message,
does it issue the RIGHT state-machine operations (remember / supersede /
consolidate / forget)?  This must be measured BEFORE any end-to-end answer
benchmark, otherwise gains cannot be attributed to the architecture vs the
oracle judgments it silently receives.

Offline JSON protocol (no MCP tool-calling in Phase 1): the actor sees the
memory snapshot + the turn and emits strict JSON; the harness executes.
Mechanical validity (id existence, active supersede target, cycles) is
checked in code -- never by an LLM. Semantic validity (unsupported inference
in consolidation conclusions) goes to an LLM judge, reported separately.

The engine (flymemory/v3.py) is FROZEN while this benchmark exists; do not
change engine behavior to make numbers look better (evaluation leakage).

Arms:
  oracle   : replays the gold annotations -- harness sanity check, must be ~perfect
  deepseek : DeepSeek chat as the autonomous memory actor (DEEPSEEK_API_KEY env)

Metrics:
  supersede      precision / recall / F1 over gold old_id sets
                 (noop cases contribute negatives; a noop case with any
                  mutation counts toward unnecessary-mutation rate)
  forget         precision / recall / F1 + n_positive (report with n -- small)
  consolidation  evidence-selection precision / recall (mechanical)
                 + unsupported-inference rate (LLM judge, calibration pending)
  overall        unnecessary mutation rate = noop cases with any mutation
                 / total noop cases

Every case's raw actor output, parsed decision and gold are persisted to
reports/ so future models can be re-scored without re-annotating.

Run:
  python bench_memory_judgment.py --actor oracle
  DEEPSEEK_API_KEY=sk-... python bench_memory_judgment.py --actor deepseek
  python bench_memory_judgment.py --actor deepseek --judge   # + LLM semantic judge
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
try:
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 4))
except ImportError:
    pass
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import SmartMemory, _embed  # noqa: E402

DAY = 86400.0
BENCHMARK_VERSION = "1.0"
ACTOR_MODEL = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

ACTOR_SYSTEM = """You are the memory-policy module of a personal AI assistant.
Given the assistant's current long-term memory entries and the user's new
message, decide which memory operations to perform.

Rules:
- remember: genuinely NEW facts worth keeping long-term.
- supersede: an existing entry is OUTDATED because the message updates that
  exact fact. Mentioning an old fact is NOT an update. point remember_index
  at the remember array entry that holds the new state.
- forget: an entry is factually WRONG (a mistake). Merely old-but-true facts
  are superseded, NEVER forgotten.
- consolidate: several entries are fragments of one topic and a higher-level
  conclusion would serve future recall better. The conclusion must contain
  ONLY information present in the selected entries. When the user asks for a
  summary/overview of some area, that is the moment to consolidate that
  area's fragments -- the summary should then come from the consolidated
  entry, not from re-reading fragments every time.
- If nothing qualifies, return empty arrays. Most turns should NOT mutate
  long-term memory.

Respond with ONLY this JSON (no markdown fence, no extra text):
{"remember": ["..."], "supersede": [{"old_id": 1, "remember_index": 0}],
 "consolidate": [{"memory_ids": [1, 2], "conclusion": "..."}],
 "forget": [{"memory_id": 5}], "reason": "one short sentence"}"""

JUDGE_SYSTEM = """You are a strict factuality judge. Given EVIDENCE entries
and a CONCLUSION drawn from them, decide whether the conclusion contains
information NOT supported by the evidence. Reply with ONLY:
{"supported": true/false, "unsupported_claims": ["..."]}"""


def ds_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ds_chat(client, system, user, max_tokens=512):
    r = client.chat.completions.create(
        model=ACTOR_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=max_tokens)
    return r.choices[0].message.content or ""


def parse_decision(raw):
    """Extract the decision JSON from raw actor output. Returns (dict, error)."""
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1]), None
        except json.JSONDecodeError as e:
            return {}, f"json parse error: {e}"
    return {}, "no JSON object in output"


def build_memory(case):
    """Fresh store per case, entries backdated so decay is realistic.

    Returns (mem, id_map): dataset ids are 1-based, store ids are 0-based --
    the harness translates in both directions so scores stay in dataset-id
    space (the space the actor sees and the data files use)."""
    mem = SmartMemory(n_bits=4096)
    now = time.time()
    id_map = {}
    for m in case["memories"]:
        r = mem.remember_text(m["text"], source="import",
                              timestamp=now - m.get("days_ago", 30) * DAY,
                              force_new=True)
        id_map[m["id"]] = r["memory_id"]
    return mem, id_map


def gold_decision(case):
    g = case["gold"]
    cons = []
    for c in g.get("consolidate", []):
        # the oracle must produce real conclusion text -- conclusion_points are
        # the annotation format, an empty conclusion would be junk-rejected
        cons.append({"memory_ids": c["memory_ids"],
                     "conclusion": ". ".join(c["conclusion_points"]) + "."})
    return {"remember": list(g.get("remember", [])),
            "supersede": list(g.get("supersede", [])),
            "consolidate": cons,
            "forget": list(g.get("forget", [])),
            "reason": "gold"}


def oracle_decide(case):
    return json.dumps(gold_decision(case))


def deepseek_decide(client, case):
    lines = "\n".join(f"#{m['id']}: {m['text']}" for m in case["memories"])
    user = (f"Current memory:\n{lines}\n\nNew user message:\n"
            f"\"{case['turn']}\"\n\nDecide the memory operations now.")
    return ds_chat(client, ACTOR_SYSTEM, user)


def execute(mem, case, decision, id_map):
    """Apply the decision to the store. Actor ids are dataset ids; the harness
    translates via id_map and reports errors back in dataset ids.
    Returns (executed, mechanical_errors)."""
    executed = {"remember_ids": [], "supersede": [], "consolidate": [],
                "forget": []}
    errors = []
    new_ids = []
    for text in decision.get("remember", []):
        r = mem.remember_text(str(text), source="model", force_new=True)
        if r.get("stored"):
            new_ids.append(r["memory_id"])
            executed["remember_ids"].append(r["memory_id"])
        else:
            errors.append(f"remember rejected: {str(text)[:40]}")
            new_ids.append(None)
    for s in decision.get("supersede", []):
        old_id = int(s.get("old_id", -1))
        idx = s.get("remember_index")
        new_id = new_ids[idx] if isinstance(idx, int) and idx < len(new_ids) else None
        if new_id is None:
            errors.append(f"supersede {old_id}: remember_index {idx} unusable")
            executed["supersede"].append({"old_id": old_id, "ok": False,
                                          "error": "bad remember_index"})
            continue
        actual_old = id_map.get(old_id)
        if actual_old is None:
            errors.append(f"supersede {old_id}: unknown dataset id")
            executed["supersede"].append({"old_id": old_id, "ok": False,
                                          "error": "unknown id"})
            continue
        ok, reason = mem.supersede(actual_old, new_id)
        executed["supersede"].append({"old_id": old_id, "new_id": new_id,
                                      "ok": ok, "error": None if ok else reason})
        if not ok:
            errors.append(f"supersede {old_id}->{new_id}: {reason}")
    for c in decision.get("consolidate", []):
        ds_ids = [int(i) for i in c.get("memory_ids", [])]
        conclusion = str(c.get("conclusion", ""))
        actual_ids = [id_map[i] for i in ds_ids if i in id_map]
        by_id = {m.memory_id for m in mem.memories}
        missing = [i for i in actual_ids if i not in by_id]
        if len(set(actual_ids)) < 2:
            errors.append(f"consolidate: fewer than 2 usable ids in {ds_ids}")
            executed["consolidate"].append({"ids": ds_ids, "ok": False,
                                            "error": "needs >= 2"})
            continue
        if missing:
            errors.append(f"consolidate missing ids: {missing}")
            executed["consolidate"].append({"ids": ds_ids, "ok": False,
                                            "error": f"missing {missing}"})
            continue
        r = mem.remember_text(conclusion, tags=["consolidation"],
                              source="model", force_new=True)
        if not r.get("stored"):
            errors.append("consolidate conclusion not stored")
            executed["consolidate"].append({"ids": ds_ids, "ok": False,
                                            "error": "not stored"})
            continue
        evidence = sorted(set(actual_ids))
        for cid in r.get("memory_ids", []):
            entry = next((m for m in mem.memories if m.memory_id == cid), None)
            if entry is not None:
                entry.evidence_ids = evidence
        executed["consolidate"].append({"ids": ds_ids, "conclusion": conclusion,
                                        "node_ids": r.get("memory_ids", []),
                                        "ok": True, "error": None})
    for f in decision.get("forget", []):
        ds_id = int(f.get("memory_id", -1))
        actual = id_map.get(ds_id)
        if actual is None:
            errors.append(f"forget {ds_id}: unknown dataset id")
            executed["forget"].append({"memory_id": ds_id, "ok": False,
                                       "error": "unknown id"})
            continue
        text = mem.forget(actual)
        executed["forget"].append({"memory_id": ds_id,
                                   "ok": text is not None,
                                   "error": None if text is not None else "id missing"})
        if text is None:
            errors.append(f"forget {ds_id}: id missing")
    return executed, errors


def prf(pred_ids, gold_ids):
    tp = len(pred_ids & gold_ids)
    p = tp / len(pred_ids) if pred_ids else None
    r = tp / len(gold_ids) if gold_ids else None
    f1 = (2 * p * r / (p + r)) if (p is not None and r is not None
                                   and p + r > 0) else None
    return p, r, f1


def judge_conclusion(client, evidence_texts, conclusion):
    user = ("EVIDENCE:\n" + "\n".join(f"- {t}" for t in evidence_texts)
            + f"\n\nCONCLUSION:\n{conclusion}\n\nJudge now.")
    try:
        raw = ds_chat(client, JUDGE_SYSTEM, user)
        d, err = parse_decision(raw)
        if err:
            return {"judge_error": err}
        return {"supported": bool(d.get("supported")),
                "unsupported_claims": list(d.get("unsupported_claims", []))}
    except Exception as e:  # API failures must not kill the run
        return {"judge_error": str(e)[:120]}


def cos_text(a, b):
    ea, eb = _embed(a), _embed(b)
    return float(ea @ eb / (np.linalg.norm(ea) + 1e-8) / (np.linalg.norm(eb) + 1e-8))


def score_case(case, decision, executed, errors, client, use_judge):
    gold = case["gold"]
    ctype = case["type"]
    sc = {"type": ctype}

    pred_sup = {int(s.get("old_id", -1)) for s in decision.get("supersede", [])}
    gold_sup = {int(s["old_id"]) for s in gold.get("supersede", [])}
    sc["supersede"] = prf(pred_sup, gold_sup)

    pred_fgt = {int(f.get("memory_id", -1)) for f in decision.get("forget", [])}
    gold_fgt = {int(f["memory_id"]) for f in gold.get("forget", [])}
    sc["forget"] = prf(pred_fgt, gold_fgt)
    sc["forget_n_positive"] = len(gold_fgt)

    pred_cons = [sorted(int(i) for i in c.get("memory_ids", []))
                 for c in decision.get("consolidate", [])]
    gold_cons = [sorted(int(i) for i in c.get("memory_ids", []))
                 for c in gold.get("consolidate", [])]
    if gold_cons:
        gold_set = {tuple(x) for x in gold_cons}
        pred_set = {tuple(x) for x in pred_cons} or {tuple()}
        tp = len(pred_set & gold_set)
        sc["consolidation"] = {
            "evidence_precision": tp / len(pred_set) if pred_set else None,
            "evidence_recall": tp / len(gold_set) if gold_set else None,
            "n_gold": len(gold_cons)}
        if use_judge and client and pred_cons:
            ev = next((m["text"] for m in case["memories"]
                       if sorted([m["id"]]) == pred_cons[0]), None)
            ev_texts = [m["text"] for m in case["memories"]
                        if m["id"] in pred_cons[0]]
            sc["consolidation"]["judge"] = judge_conclusion(
                client, ev_texts, decision.get("consolidate", [{}])[0]
                .get("conclusion", ""))
    elif pred_cons:
        sc["consolidation"] = {"evidence_precision": 0.0, "spurious": pred_cons}

    mutated = bool(decision.get("remember") or decision.get("supersede")
                   or decision.get("consolidate") or decision.get("forget"))
    sc["mutated"] = mutated
    gold_mut = bool(gold.get("remember") or gold.get("supersede")
                    or gold.get("consolidate") or gold.get("forget"))
    sc["unnecessary_mutation"] = mutated and not gold_mut
    sc["missed_mutation"] = gold_mut and not mutated

    # faithfulness of remembered text vs gold (report-level, not pass/fail)
    sims = []
    for pt in decision.get("remember", []):
        best = max((cos_text(str(pt), gt) for gt in gold.get("remember", [])),
                   default=None)
        if best is not None:
            sims.append(round(best, 3))
    sc["remember_sim_vs_gold"] = sims

    sc["mechanical_errors"] = errors
    sc["executed"] = executed
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", choices=["oracle", "deepseek"], default="oracle")
    ap.add_argument("--data", default=os.path.join(_HERE, "data", "memory_judgment.json"))
    ap.add_argument("--judge", action="store_true",
                    help="LLM semantic judge for consolidation conclusions")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    cases = dataset["cases"]
    if args.limit:
        cases = cases[:args.limit]

    client = None
    if args.actor == "deepseek" or args.judge:
        client = ds_client()

    prompt_hash = hashlib.sha1(ACTOR_SYSTEM.encode()).hexdigest()[:10]
    print(f"cases: {len(cases)} | actor: {args.actor} "
          f"(model {ACTOR_MODEL if args.actor == 'deepseek' else 'gold'}, "
          f"prompt {prompt_hash}) | judge: {args.judge}\n", flush=True)

    traces, agg = [], {"supersede_p": [], "supersede_r": [], "forget_p": [],
                       "forget_r": [], "noop_mutated": 0, "noop_total": 0,
                       "mech_error_cases": 0}
    for ci, case in enumerate(cases):
        mem, id_map = build_memory(case)
        raw = (oracle_decide(case) if args.actor == "oracle"
               else deepseek_decide(client, case))
        decision, perr = parse_decision(raw)
        if perr:
            decision = {}
        executed, errors = execute(mem, case, decision, id_map)
        sc = score_case(case, decision, executed, errors, client,
                        args.judge and args.actor == "deepseek")
        sc.update({"case_id": case["case_id"], "parse_error": perr,
                   "raw_output": raw, "decision": decision})
        traces.append(sc)
        for k, dst in (("supersede", "supersede_p"), (None, None)):
            pass
        if sc["supersede"][0] is not None:
            agg["supersede_p"].append(sc["supersede"][0])
        if sc["supersede"][1] is not None:
            agg["supersede_r"].append(sc["supersede"][1])
        if sc["forget"][0] is not None:
            agg["forget_p"].append(sc["forget"][0])
        if sc["forget"][1] is not None:
            agg["forget_r"].append(sc["forget"][1])
        if case["type"] == "noop":
            agg["noop_total"] += 1
            agg["noop_mutated"] += int(sc["unnecessary_mutation"])
        if errors or perr:
            agg["mech_error_cases"] += 1
        flag = "MUT!" if sc["unnecessary_mutation"] else (
            "ERR!" if (errors or perr) else "ok  ")
        print(f"[{ci+1:2d}/{len(cases)}] {flag} {case['case_id']:10s} "
              f"sup P/R={sc['supersede'][0]}/{sc['supersede'][1]} "
              f"fgt P/R={sc['forget'][0]}/{sc['forget'][1]} "
              f"mut={sc['mutated']}", flush=True)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    print(f"\n=== Memory Judgment v{BENCHMARK_VERSION} | actor={args.actor} ===")
    print(f"  supersede precision        {mean(agg['supersede_p'])}  (n={len(agg['supersede_p'])})")
    print(f"  supersede recall           {mean(agg['supersede_r'])}  (n={len(agg['supersede_r'])})")
    print(f"  forget precision           {mean(agg['forget_p'])}  (n={len(agg['forget_p'])})")
    print(f"  forget recall              {mean(agg['forget_r'])}  (n={len(agg['forget_r'])})")
    print(f"  unnecessary mutation rate  {agg['noop_mutated']}/{agg['noop_total']}")
    print(f"  cases w/ mechanical errors {agg['mech_error_cases']}")

    os.makedirs(os.path.join(_HERE, "reports"), exist_ok=True)
    out = os.path.join(_HERE, "reports",
                       f"memory_judgment_{args.actor}_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmark_version": BENCHMARK_VERSION,
                   "actor": args.actor, "actor_model": ACTOR_MODEL,
                   "temperature": 0, "prompt_hash": prompt_hash,
                   "aggregate": {k: mean(v) if isinstance(v, list) else v
                                 for k, v in agg.items()},
                   "traces": traces}, f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
