"""FlyMemory "dreaming": idle-time consolidation pass (v4-rfc §9).

Letta's sleep-time compute insight (arXiv 2504.13171): agents can think over
their memory during downtime, so test-time queries get pre-digested context.
FlyMemory mapping: during idle hours, re-run overlay consolidation over the
RECENT-channel window (last ~90 min of conversation) and store the distilled
entries as overlay -- engine untouched, pure post-processing.

Storage goes through the RUNNING SERVICE (HTTP jsonrpc flymemory_remember),
never by writing the pkl directly: the service holds the authoritative
in-memory copy and would overwrite any direct pkl writes on its next save.

Safety gates before anything is written:
  1. Faithfulness audit: every distilled entry is judged against the source
     turns by the LLM (bench_consolidation_audit protocol); failures dropped.
  2. Entries enter via flymemory_remember (overlay semantics, credential
     filter, dedup ladder all apply server-side).

Run (manual or via scheduled task):
  DEEPSEEK_API_KEY=sk-... python dream.py [--window 90] [--min-turns 6]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "flymemory"))

from bench_memory_judgment import ds_client, parse_decision  # noqa: E402
from bench_lme_e2e import ANSWER_SYSTEM, JUDGE_SYSTEM, MODEL  # noqa: E402

MCP_URL = "http://127.0.0.1:8765/mcp"

DISTILL_SYSTEM = """You are the memory-consolidation module of a personal AI
assistant. Given a recent conversation window, distill the DURABLE facts and
decisions worth keeping long-term: preferences, numbers, names, dates,
plans, outcomes, corrections. Write 1-6 short self-contained entries. Do NOT
invent anything not present in the conversation. Reply with ONLY a JSON
array of strings."""

AUDIT_SYSTEM = """You are a strict factuality judge. Given SOURCE TURNS and a
distilled MEMORY ENTRY, decide whether every claim in the entry is supported
by the source turns. Reply with ONLY:
{"supported": true/false, "unsupported_claims": ["..."]}"""


def ds_chat(client, system, user, max_tokens=900):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=max_tokens)
    return r.choices[0].message.content or ""


def mcp_call(name, arguments):
    """Call a tool on the running flymemory service over streamable-http."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": arguments}})
    req = urllib.request.Request(
        MCP_URL, data=body.encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read().decode("utf-8", "replace")
    for line in payload.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            break
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        return "mcp_error: unparseable response"
    result = d.get("result", {})
    content = result.get("content", [])
    return content[0].get("text", "") if content else json.dumps(result)[:200]


def parse_array(raw):
    """Extract a JSON string array from raw output, with truncation repair.

    Returns (list, error) -- error None on success.
    Progressive repair for truncated output: try the raw slice, then
    progressively shorter prefixes that end at a complete string element,
    each closed with ]."""
    start = raw.find("[")
    if start < 0:
        return None, "no JSON array in output"
    body = raw[start:]
    candidates = [body]
    q = body.rfind('"')
    while q > 0:
        candidates.append(body[:q + 1] + "]")
        q = body.rfind('"', 0, q)
    last_err = None
    for cand in candidates:
        try:
            arr = json.loads(cand)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()], None
        except json.JSONDecodeError as e:
            last_err = f"@{e.pos}: {e.msg}"
    return None, last_err or "unparseable"


def distill(client, turns):
    convo = "\n".join(turns)[:8000]
    raw = ds_chat(client, DISTILL_SYSTEM, convo, max_tokens=900)
    entries, err = parse_array(raw)
    return entries or [], err


def audit_entry(client, source_turns, entry):
    src = "\n".join(source_turns)[:6000]
    d, err = parse_decision(ds_chat(
        client, AUDIT_SYSTEM,
        f"SOURCE TURNS:\n{src}\n\nENTRY:\n{entry}\n\nJudge now.",
        max_tokens=300))
    if err:
        return False, f"audit error: {err}"
    return bool(d.get("supported")), "; ".join(d.get("unsupported_claims", [])[:2])


def main():
    from flymemory.v3 import load  # read-only pkl access for the window

    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=90,
                    help="RECENT window in minutes (default 90)")
    ap.add_argument("--min-turns", type=int, default=6,
                    help="skip if fewer than this many turns in window")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be stored, write nothing")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    # the RUNNING service owns flymemory/flymemory_v3.pkl; we only READ it
    # here (window snapshot) -- all writes go through MCP flymemory_remember
    pkl = os.path.join(here, "flymemory", "flymemory_v3.pkl")
    if not os.path.exists(pkl):
        sys.exit(f"memory store not found: {pkl}")
    mem = load(pkl, enable_hopfield=False)
    now = time.time()
    window_start = now - args.window * 60

    recent = [m for m in mem.memories
              if m.timestamp >= window_start and m.superseded_by is None]
    if len(recent) < args.min_turns:
        print(f"dream: only {len(recent)} turns in the last {args.window}min "
              f"(min {args.min_turns}) -- skipping", flush=True)
        return
    recent.sort(key=lambda m: m.timestamp)
    turns = [f"{m.source}: {m.text}" for m in recent]
    span_new = (now - recent[0].timestamp) / 60
    print(f"dream: window has {len(recent)} turns ({span_new:.0f} min ago .. now)",
          flush=True)

    client = ds_client()
    entries, derr = distill(client, turns)
    if derr:
        print(f"  [distill error] {derr}", flush=True)
    if not entries:
        print("dream: distillation produced nothing -- skipping", flush=True)
        return

    stored = []
    for entry in entries:
        supported, note = audit_entry(client, turns, entry)
        print(f"  [{'KEEP' if supported else 'DROP'}] {entry[:80]}"
              + (f"  ({note})" if not supported else ""), flush=True)
        if not supported:
            continue
        if args.dry_run:
            stored.append(entry)
            continue
        result = mcp_call("flymemory_remember", {"text": entry})
        print(f"    -> {result[:100]}", flush=True)
        stored.append(entry)
    mode = "(dry-run, not stored)" if args.dry_run else "stored via MCP"
    print(f"dream: {len(stored)} entries {mode}", flush=True)


if __name__ == "__main__":
    main()
