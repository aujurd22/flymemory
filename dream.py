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
import re
import pathlib
import sys
import time
import urllib.request

# silent under pythonw (scheduled task): route prints to a log file.
# Text mode + line buffering -- the pythonw/buffering lesson from
# server.log (buffering=0 binary streams crash sys.stderr writes).
if sys.stdout is None or sys.stderr is None:
    _log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dream.log"), "a", buffering=1,
                encoding="utf-8")
    sys.stdout = sys.stderr = _log

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "flymemory"))

# scheduled-task context has no shell env: fall back to the verified
# key source (same file the run_*_with_key.py helpers use). Key stays
# out of the repo and out of logs.  NOTE: the sys.path inserts above
# must precede this block -- local_config lives next to dream.py.
if not os.environ.get("DEEPSEEK_API_KEY"):
    # key file path is machine-local (flymemory/local_config.py is
    # git-ignored); set DEEPSEEK_KEY_FILE or create local_config.py with
    # DEEPSEEK_KEY_FILE=<path> on your own machine.
    try:
        import re as _re
        try:
            from local_config import DEEPSEEK_KEY_FILE  # type: ignore
        except ImportError:
            DEEPSEEK_KEY_FILE = os.environ.get("DEEPSEEK_KEY_FILE", "")
        if DEEPSEEK_KEY_FILE and os.path.exists(DEEPSEEK_KEY_FILE):
            _src = open(DEEPSEEK_KEY_FILE, encoding="utf-8").read()
            _m = _re.search(r'api_key="([^"]+)"', _src)
            if _m:
                os.environ["DEEPSEEK_API_KEY"] = _m.group(1)
    except Exception:
        pass
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

MERGE_SYSTEM = """You decide whether two memory entries state the SAME fact.

Reply with ONLY this JSON:
{"same": true/false, "merged": "...", "why": "..."}

Rules:
- same=true ONLY when both describe the same subject and agree -- a rewording,
  a duplicate, or one strictly containing the other.
- If they disagree (different values, different dates, one replaces the other),
  set same=false. Do not reconcile a contradiction into one sentence.
- When same=true, "merged" is ONE self-contained sentence in the source language
  that keeps EVERY concrete detail from both (numbers, names, dates, paths).
  Never add anything that is not in the two entries.
- When same=false, "merged" is an empty string.
"""

CLUSTER_SYSTEM = """You are given several memory entries. Decide whether they ALL state
the same fact.

Reply with ONLY this JSON:
{"same": true/false, "merged": "...", "keep": [ids that differ], "why": "..."}

Rules:
- same=true ONLY if every entry describes the same subject and they agree.
- If one entry carries a different value, date, path or decision, set same=false and
  list that entry's id in "keep". Do not reconcile a disagreement into one sentence.
- When same=true, "merged" is ONE self-contained sentence in the source language that
  keeps EVERY concrete detail from every entry (numbers, names, dates, paths). Never
  add anything that is not in the given entries.
"""

AUDIT_SYSTEM = """You are a strict factuality judge. Given SOURCE TURNS and a
distilled MEMORY ENTRY, decide whether every claim in the entry is supported
by the source turns. Reply with ONLY:
{"supported": true/false, "unsupported_claims": ["..."]}"""


DREAM_BASE = os.environ.get("FLYMEMORY_DREAM_BASE_URL",
                            "https://developer.amd.com.cn/radeon/api/v1").rstrip("/")
DREAM_MODEL = os.environ.get("FLYMEMORY_DREAM_MODEL", "Qwen3.8-Flash-Next")
# Measured on that endpoint 2026-10-31: Qwen3.8-Flash-Next returns clean JSON in
# ~13s, GLM-5.3-Flash ~27s, Qwen3.8-27B intermittently empty, MiMo 61s,
# DeepSeek-V4.1-Flash 429 under load.
DREAM_FALLBACKS = ("GLM-5.3-Flash", "DeepSeek-V4-Flash")
_DREAM_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def _dream_key():
    return (os.environ.get("FLYMEMORY_DREAM_API_KEY")
            or os.environ.get("AMDRADEON_API_KEY") or "")


def _free_endpoint_chat(system, user, max_tokens):
    """One call to the free OpenAI-compatible endpoint; "" when unavailable."""
    key = _dream_key()
    if not key:
        return ""
    chain = [DREAM_MODEL] + [m for m in DREAM_FALLBACKS if m != DREAM_MODEL]
    for model_name in chain:
        body = json.dumps({"model": model_name,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}],
                           "temperature": 0, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(
            DREAM_BASE + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "Authorization": "Bearer " + key, "User-Agent": _DREAM_UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode())
            text = payload["choices"][0]["message"].get("content") or ""
            if text.strip():
                return text
        except Exception:
            continue
    return ""


def ds_chat(client, system, user, max_tokens=900):
    """Free endpoint when configured, otherwise the original DeepSeek client."""
    text = _free_endpoint_chat(system, user, max_tokens)
    if text:
        return text
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


def merge_pass(url=None, limit=8, min_sim=0.80, apply=False, chat=None, pool=2000):
    """Ask the model which near-duplicate pairs are actually one fact.

    Mechanical organize can only retire exact copies and stale state keys. ``merge``
    covers the rest: the model reads both entries and either produces one sentence
    that keeps every detail, or says they differ (in which case nothing is touched).
    """
    call = chat or llm_chat
    # Rotate a persisted cursor so successive runs walk the WHOLE library: one
    # pass over 20k entries is an O(n^2) scan, a 2k window per run is cheap.
    cursor_path = pathlib.Path(os.environ.get("FLYMEMORY_MERGE_CURSOR",
                                              os.path.join(os.environ.get("TEMP", "."),
                                                           "flymemory-merge-cursor.json")))
    cursor = 0
    try:
        cursor = int(json.loads(cursor_path.read_text(encoding="utf-8")).get("offset", 0))
    except Exception:                        # noqa: BLE001 - first run
        cursor = 0
    raw = mcp_call(url, "flymemory_conflict_pairs",
                   {"min_sim": min_sim, "limit": max(1, int(limit)),
                    "pool_limit": max(100, int(pool)), "offset": cursor})
    if pool:
        try:
            cursor_path.parent.mkdir(parents=True, exist_ok=True)
            cursor_path.write_text(json.dumps({"offset": cursor + int(pool)}), encoding="utf-8")
        except Exception:                    # noqa: BLE001 - cursor is best effort
            pass
    try:
        pairs = json.loads(raw) if raw.strip().startswith("[") else []
    except json.JSONDecodeError:
        pairs = []
    if not pairs and raw.strip() and not raw.strip().startswith("["):
        print("merge: candidate query failed -> " + raw.strip()[:160], flush=True)
    if not pairs:
        print("merge: no candidate pairs", flush=True)
        return 0

    merged_count = 0
    for pair in pairs:
        prompt = "A:\n" + str(pair.get("a_text", "")) + "\n\nB:\n" + str(pair.get("b_text", ""))
        decision, error = parse_decision(call(MERGE_SYSTEM, prompt, max_tokens=400))
        if error:
            print(f"  [skip] #{pair['a_id']}~#{pair['b_id']} ({error})", flush=True)
            continue
        if not decision.get("same"):
            print(f"  [DIFFERENT] #{pair['a_id']} ~ #{pair['b_id']} "
                  f"sim={pair['sim']}: {decision.get('why', '')[:90]}", flush=True)
            continue
        merged_text = str(decision.get("merged") or "").strip()
        if not merged_text:
            print(f"  [skip] #{pair['a_id']}~#{pair['b_id']}: same but no merged text",
                  flush=True)
            continue
        print(f"  [MERGE] #{pair['a_id']} + #{pair['b_id']} -> {merged_text[:90]}", flush=True)
        merged_count += 1
        if not apply:
            continue
        written = mcp_call(url, "flymemory_remember",
                           {"text": merged_text, "tags": "dream,merged", "source": "model"})
        match = re.search(r"#(\d+)", str(written))
        if not match:
            print(f"    -> write failed: {str(written)[:80]}", flush=True)
            continue
        new_id = int(match.group(1))
        retired = 0
        for old_id in (pair["a_id"], pair["b_id"]):
            if int(old_id) == new_id:
                # The merged text deduped into one of the originals: that entry IS
                # the merge target now, so superseding it into itself is pointless.
                continue
            mcp_call(url, "flymemory_supersede",
                     {"old_memory_id": int(old_id), "new_memory_id": new_id})
            retired += 1
        where = "in place" if retired < 2 else "as a new entry"
        print(f"    -> merge landed {where} at #{new_id}, {retired} original(s) retired",
              flush=True)

    mode = "applied" if apply else "dry-run, nothing written"
    print(f"merge: {merged_count} pair(s) merged ({mode})", flush=True)
    return 0



def cluster_pass(url=None, limit=5, min_sim=0.82, apply=False, pool=4000):
    """Fold a whole group of synonyms into one entry with a single model call."""
    raw = mcp_call(url, "flymemory_clusters",
                   {"min_sim": min_sim, "limit": max(1, int(limit)),
                    "min_size": 3, "max_size": 8, "pool_limit": max(1, int(pool))})
    try:
        clusters = json.loads(raw) if raw.strip().startswith("[") else []
    except json.JSONDecodeError:
        clusters = []
    if not clusters and raw.strip() and not raw.strip().startswith("["):
        print("cluster: query failed -> " + raw.strip()[:160], flush=True)
    if not clusters:
        print("cluster: no groups of 3+ found", flush=True)
        return 0

    merged_count = 0
    for group in clusters:
        ids = group.get("ids") or []
        texts = group.get("texts") or []
        if len(ids) < 3:
            continue
        numbered = "\n".join("[%s] %s" % (i, t) for i, t in zip(ids, texts))
        answer = llm_chat(CLUSTER_SYSTEM, "ENTRIES:\n" + numbered, max_tokens=600)
        if not (answer or "").strip():
            print("  [skip] cluster %s (no model reply)" % (ids,), flush=True)
            continue
        parsed, parse_error = parse_decision(answer)
        if parse_error:
            print("  [skip] cluster %s (%s)" % (ids, parse_error), flush=True)
            continue
        if not parsed.get("same"):
            keep = ", ".join("#%s" % i for i in parsed.get("keep", [])[:4])
            print("  [DIFFERENT] %d entries, kept: %s -- %s"
                  % (len(ids), keep or "n/a", str(parsed.get("why", ""))[:80]), flush=True)
            continue
        merged_text = str(parsed.get("merged") or "").strip()
        if not merged_text:
            print("  [skip] cluster %s: same but no merged text" % (ids,), flush=True)
            continue
        print("  [CLUSTER %d] %s -> %s" % (len(ids),
              ", ".join("#%s" % i for i in ids), merged_text[:80]), flush=True)
        merged_count += 1
        if not apply:
            continue
        written = mcp_call(url, "flymemory_remember",
                           {"text": merged_text, "tags": "dream,merged,cluster",
                            "source": "model"})
        match = re.search(r"#(\d+)", str(written))
        if not match:
            print("    -> write failed: %s" % str(written)[:80], flush=True)
            continue
        new_id = int(match.group(1))
        retired = 0
        for old_id in ids:
            if int(old_id) == new_id:
                continue
            mcp_call(url, "flymemory_supersede",
                     {"old_memory_id": int(old_id), "new_memory_id": new_id})
            retired += 1
        print("    -> #%d consolidated, %d original(s) retired" % (new_id, retired),
              flush=True)

    mode = "applied" if apply else "dry-run, nothing written"
    print("cluster: %d group(s) consolidated (%s)" % (merged_count, mode), flush=True)
    return 0


def main():
    from flymemory.v3 import load  # read-only pkl access for the window

    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=90,
                    help="RECENT window in minutes (default 90)")
    ap.add_argument("--min-turns", type=int, default=6,
                    help="skip if fewer than this many turns in window")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be stored, write nothing")
    ap.add_argument("--shadow", action="store_true",
                    help="write proposals to a JSON file instead of the store")
    ap.add_argument("--promote", default="",
                    help="write the proposals saved by an earlier --shadow run")
    ap.add_argument("--shadow-dir", default="",
                    help="where --shadow writes (default: next to this file)")
    ap.add_argument("--compartment", default="",
                    help="dream only entries tagged comp:<name> (default: "
                         "ungrouped entries without any comp: tag). Use "
                         "'all' to dream everything together")
    args = ap.parse_args()

    if args.cluster:
        return cluster_pass(None, args.cluster_limit,
                            args.cluster_min_sim, args.cluster_apply,
                            pool=args.cluster_pool)

    if args.merge:
        return merge_pass(None, args.merge_limit, args.merge_min_sim,
                          args.merge_apply, pool=args.merge_pool)

    if args.organize or args.organize_apply:
        print(mcp_call("flymemory_organize",
                       {"apply": bool(args.organize_apply)}), flush=True)
        print(flush=True)

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
    comp_tag = f"comp:{args.compartment}" if args.compartment and args.compartment != "all" else None
    if comp_tag:
        # compartment-scoped dreaming: exclude entries from OTHER
        # compartments (cross-domain contamination fix, 2026-09-26);
        # ungrouped entries (no comp: tag) stay included
        recent = [m for m in recent
                  if not any(t.startswith("comp:") and t != comp_tag
                             for t in m.tags)]
    if len(recent) < args.min_turns:
        print(f"dream: only {len(recent)} turns in the last {args.window}min "
              f"(min {args.min_turns}) -- skipping", flush=True)
        return
    recent.sort(key=lambda m: m.timestamp)
    turns = [f"{m.source}: {m.text}" for m in recent]
    span_new = (now - recent[0].timestamp) / 60
    print(f"dream: window has {len(recent)} turns ({span_new:.0f} min ago .. now)"
          + (f" [compartment={args.compartment}]" if args.compartment else ""),
          flush=True)

    client = ds_client()
    entries, derr = distill(client, turns)
    if derr:
        print(f"  [distill error] {derr}", flush=True)
    if not entries:
        print("dream: distillation produced nothing -- skipping", flush=True)
        return

    if args.promote:
        return promote(args)

    stored = []
    proposals = []
    for entry in entries:
        supported, note = audit_entry(client, turns, entry)
        print(f"  [{'KEEP' if supported else 'DROP'}] {entry[:80]}"
              + (f"  ({note})" if not supported else ""), flush=True)
        if not supported:
            continue
        if args.dry_run:
            stored.append(entry)
            continue
        if args.shadow:
            proposals.append(entry)
            stored.append(entry)
            continue
        write_args = {"text": entry}
        if args.compartment and args.compartment != "all":
            write_args["compartment"] = args.compartment
        result = mcp_call("flymemory_remember", write_args)
        print(f"    -> {result[:100]}", flush=True)
        stored.append(entry)
    if args.shadow and proposals:
        directory = pathlib.Path(args.shadow_dir) if args.shadow_dir else pathlib.Path(
            os.path.dirname(os.path.abspath(__file__)))
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ("dream-shadow-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
        target.write_text(json.dumps({
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "window_minutes": args.window,
            "compartment": args.compartment,
            "proposals": proposals,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"dream: {len(proposals)} proposals written to {target}", flush=True)
        print("       review, then: python dream.py --promote " + str(target), flush=True)
        return

    mode = "(dry-run, not stored)" if args.dry_run else "stored via MCP"
    print(f"dream: {len(stored)} entries {mode}", flush=True)


def promote(args):
    """Write the proposals a --shadow run left on disk."""
    path = pathlib.Path(args.promote)
    if not path.exists():
        print(f"dream: proposal file not found: {path}", flush=True)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    written = 0
    for entry in data.get("proposals", []):
        write_args = {"text": entry}
        compartment = data.get("compartment") or args.compartment
        if compartment and compartment != "all":
            write_args["compartment"] = compartment
        result = mcp_call("flymemory_remember", write_args)
        written += 1
        print(f"  -> {result[:100]}", flush=True)
    print(f"dream: promoted {written} entries from {path.name}", flush=True)


if __name__ == "__main__":
    main()
