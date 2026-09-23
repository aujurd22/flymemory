"""Memory Judgment Benchmark -- Phase 1.5: real tool-calling.

Same 38 cases as bench_memory_judgment.py, but the actor no longer emits an
offline JSON decision: it drives the REAL tool surface (remember / supersede /
consolidate / forget, with production semantics including dedup on remember)
through a function-calling loop, and must handle the id flow itself (remember
first, take the returned id, then supersede old -> new).

Purpose: measure the protocol->tooling transfer loss. Phase 1 (offline JSON)
and Phase 1.5 (tools) share the dataset, the model and the scorer; the delta
is attributable to the interface, not the judgments.

Snapshot ids are the REAL store ids (the builder aligns them), so the model
sees exactly the ids the tools accept -- no translation layer.

Run:
  DEEPSEEK_API_KEY=sk-... python bench_memory_judgment_tools.py
"""
import argparse
import hashlib
import json
import os
import sys
import time

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

from flymemory.v3 import SmartMemory  # noqa: E402
from bench_memory_judgment import (score_case, ds_client,  # noqa: E402
                                   BENCHMARK_VERSION, DAY)

ACTOR_MODEL = "deepseek-chat"

TOOLS_SYSTEM = """You are the memory-policy module of a personal AI assistant.
You operate its long-term memory through real tools. Given the current memory
entries and the user's new message, call tools as needed.

Rules:
- flymemory_remember: genuinely NEW facts worth keeping long-term.
- flymemory_supersede: an existing entry is OUTDATED because the message
  updates that exact fact. Mentioning an old fact is NOT an update. A
  supersede needs a NEW entry: call flymemory_remember first, take the id it
  returns, then supersede(old_id, new_id).
- flymemory_forget: an entry is factually WRONG (a mistake). Merely
  old-but-true facts are superseded, NEVER forgotten.
- flymemory_consolidate: several entries are fragments of one topic and a
  higher-level conclusion would serve future recall better. The conclusion
  must contain ONLY information present in the selected entries. When the
  user asks for a summary/overview of some area, that is the moment to
  consolidate that area's fragments.
- If nothing qualifies, call NO tool and just reply "done".

Most turns should NOT mutate long-term memory."""

TOOLS = [
    {"type": "function", "function": {
        "name": "flymemory_remember",
        "description": "Store a new long-term fact. Returns the new entry id.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The fact to store"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "flymemory_supersede",
        "description": "Mark an outdated entry as superseded by a newer one "
                       "(the new entry must already exist -- remember it first).",
        "parameters": {"type": "object", "properties": {
            "old_id": {"type": "integer"}, "new_id": {"type": "integer"}},
            "required": ["old_id", "new_id"]}}},
    {"type": "function", "function": {
        "name": "flymemory_consolidate",
        "description": "Consolidate >= 2 fragment entries into one higher-order "
                       "conclusion entry. Conclusion must only use info from the "
                       "selected entries.",
        "parameters": {"type": "object", "properties": {
            "memory_ids": {"type": "array", "items": {"type": "integer"}},
            "conclusion": {"type": "string"}},
            "required": ["memory_ids", "conclusion"]}}},
    {"type": "function", "function": {
        "name": "flymemory_forget",
        "description": "Hard-delete one factually WRONG entry (never a merely "
                       "old-but-true one -- supersede those).",
        "parameters": {"type": "object", "properties": {
            "memory_id": {"type": "integer"}},
            "required": ["memory_id"]}}},
]


def build_tools_memory(case):
    """Store ids aligned with dataset ids (start _next_id at 1): the model then
    sees exactly the ids the tools accept. Returns (mem, id_map, inv_map)."""
    mem = SmartMemory(n_bits=4096)
    mem._next_id = 1
    now = time.time()
    id_map, inv_map = {}, {}
    for m in case["memories"]:
        r = mem.remember_text(m["text"], source="import",
                              timestamp=now - m.get("days_ago", 30) * DAY,
                              force_new=True)
        id_map[m["id"]] = r["memory_id"]
        inv_map[r["memory_id"]] = m["id"]
    return mem, id_map, inv_map


def consolidate_impl(mem, memory_ids, conclusion):
    """Production consolidate semantics (mcp_v3 parity): force-new node, every
    conclusion chunk carries the full evidence list."""
    ids = [int(i) for i in memory_ids]
    if len(set(ids)) < 2:
        return "rejected: consolidation needs at least 2 distinct memories"
    by_id = {m.memory_id for m in mem.memories}
    missing = [i for i in ids if i not in by_id]
    if missing:
        return f"rejected: missing memory ids {missing}"
    r = mem.remember_text(conclusion, tags=["consolidation"],
                          source="model", force_new=True)
    if not r.get("stored"):
        return "rejected: conclusion not stored"
    evidence = sorted(set(ids))
    for cid in r.get("memory_ids", []):
        entry = next((m for m in mem.memories if m.memory_id == cid), None)
        if entry is not None:
            entry.evidence_ids = evidence
    return (f"consolidated -> #{r['memory_id']}, evidence {evidence}")


def apply_tool(mem, name, args):
    """Execute one tool call with production semantics; returns the tool
    result text fed back to the model."""
    try:
        if name == "flymemory_remember":
            r = mem.remember_text(str(args.get("text", "")), source="model")
            if not r.get("stored"):
                return f"rejected: {r.get('reason') or 'not novel enough'}"
            return f"stored as #{r['memory_id']}"
        if name == "flymemory_supersede":
            ok, reason = mem.supersede(int(args.get("old_id", -1)),
                                       int(args.get("new_id", -1)))
            return "ok" if ok else f"rejected: {reason}"
        if name == "flymemory_consolidate":
            return consolidate_impl(mem, args.get("memory_ids", []),
                                    str(args.get("conclusion", "")))
        if name == "flymemory_forget":
            text = mem.forget(int(args.get("memory_id", -1)))
            return "deleted" if text is not None else "rejected: id missing"
        return f"unknown tool {name}"
    except (TypeError, ValueError) as e:
        return f"rejected: bad arguments ({e})"


def run_tools_case(client, case, max_rounds=8):
    """One case through the tool loop. Returns (decision_like, errors, raw_log)."""
    mem, id_map, inv_map = build_tools_memory(case)
    lines = "\n".join(f"#{m['id']}: {m['text']}" for m in case["memories"])
    messages = [
        {"role": "system", "content": TOOLS_SYSTEM},
        {"role": "user", "content":
            f"Current memory:\n{lines}\n\nNew user message:\n\"{case['turn']}\"\n\n"
            f"Operate memory as needed, or call nothing and reply done."},
    ]
    d_like = {"remember": [], "supersede": [], "consolidate": [], "forget": []}
    errors, raw_log = [], []
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=ACTOR_MODEL, messages=messages, tools=TOOLS,
            temperature=0, max_tokens=512)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            raw_log.append(msg.content or "")
            break
        messages.append({"role": "assistant",
                         "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                errors.append(f"{name}: bad JSON arguments")
            raw_log.append(f"{name}({tc.function.arguments})")
            if name == "flymemory_remember":
                d_like["remember"].append(str(args.get("text", "")))
            elif name == "flymemory_supersede":
                try:
                    old_ds = inv_map[int(args.get("old_id", -1))]
                    d_like["supersede"].append({"old_id": old_ds})
                except (KeyError, ValueError):
                    errors.append(f"supersede unknown old_id {args.get('old_id')}")
            elif name == "flymemory_consolidate":
                ds_ids = sorted(inv_map.get(int(i), int(i))
                                for i in args.get("memory_ids", []))
                d_like["consolidate"].append({"memory_ids": ds_ids,
                                              "conclusion": str(args.get("conclusion", ""))})
            elif name == "flymemory_forget":
                try:
                    d_like["forget"].append({"memory_id": inv_map[int(args.get("memory_id", -1))]})
                except (KeyError, ValueError):
                    errors.append(f"forget unknown id {args.get('memory_id')}")
            result = apply_tool(mem, name, args)
            if result.startswith("rejected"):
                errors.append(f"{name}: {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result})
    return d_like, errors, raw_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "data", "memory_judgment.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    cases = dataset["cases"]
    if args.limit:
        cases = cases[:args.limit]

    client = ds_client()
    prompt_hash = hashlib.sha1(TOOLS_SYSTEM.encode()).hexdigest()[:10]
    print(f"cases: {len(cases)} | protocol: REAL TOOLS | model {ACTOR_MODEL} "
          f"(prompt {prompt_hash})\n", flush=True)

    traces, agg = [], {"supersede_p": [], "supersede_r": [], "forget_p": [],
                       "forget_r": [], "noop_mutated": 0, "noop_total": 0,
                       "mech_error_cases": 0}
    for ci, case in enumerate(cases):
        d_like, errors, raw_log = run_tools_case(client, case)
        executed = {}  # tools mode executes inline; scoring is decision-level
        sc = score_case(case, d_like, executed, errors, None, False)
        sc.update({"case_id": case["case_id"], "raw_output": raw_log,
                   "decision": d_like})
        traces.append(sc)
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
        if errors:
            agg["mech_error_cases"] += 1
        flag = "MUT!" if sc["unnecessary_mutation"] else ("ERR!" if errors else "ok  ")
        print(f"[{ci+1:2d}/{len(cases)}] {flag} {case['case_id']:10s} "
              f"sup P/R={sc['supersede'][0]}/{sc['supersede'][1]} "
              f"fgt P/R={sc['forget'][0]}/{sc['forget'][1]} "
              f"mut={sc['mutated']}", flush=True)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    print(f"\n=== Memory Judgment v{BENCHMARK_VERSION} Phase 1.5 (real tools) "
          f"| actor={ACTOR_MODEL} ===")
    print(f"  supersede precision        {mean(agg['supersede_p'])}  (n={len(agg['supersede_p'])})")
    print(f"  supersede recall           {mean(agg['supersede_r'])}  (n={len(agg['supersede_r'])})")
    print(f"  forget precision           {mean(agg['forget_p'])}  (n={len(agg['forget_p'])})")
    print(f"  forget recall              {mean(agg['forget_r'])}  (n={len(agg['forget_r'])})")
    print(f"  unnecessary mutation rate  {agg['noop_mutated']}/{agg['noop_total']}")
    print(f"  cases w/ mechanical errors {agg['mech_error_cases']}")

    os.makedirs(os.path.join(_HERE, "reports"), exist_ok=True)
    out = os.path.join(_HERE, "reports",
                       f"memory_judgment_tools_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmark_version": BENCHMARK_VERSION, "phase": "1.5-tools",
                   "actor_model": ACTOR_MODEL, "temperature": 0,
                   "prompt_hash": prompt_hash,
                   "aggregate": {k: mean(v) if isinstance(v, list) else v
                                 for k, v in agg.items()},
                   "traces": traces}, f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
