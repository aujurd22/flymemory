"""FlyMemory CLI (v3) — human-facing inspection and maintenance.

Borrowed from scope-recall-hermes (2026-10-24): a memory system needs a
human door, not only an MCP surface. Read operations run offline on a
snapshot copy of the pkl (safe while the service runs); write operations
(correct/forget) go through the RUNNING SERVICE at 127.0.0.1:8765 —
direct pkl writes are forbidden while the service holds the authoritative
copy (the 改库先停服 iron rule).

Commands:
  stats                      store overview (counts, model, lineage stats)
  search <query> [--top N] [--comp X]     recall with stamps
  history <id>               supersede lineage around an entry (who replaced
                             it, what it replaced, which is current)
  dependents <id>            what would be affected by forgetting this entry
                             (evidence back-links, state_key orphans, lineage)
  forget <id> [--yes]        delete via the running service (dependents shown
                             first unless --yes)
  correct <id> <new_text>    supersede via the running service: store new text,
                             mark old superseded (the L6 anchored protocol)

Offline reads use a snapshot copy; they may lag the live store by up to
one save cycle. Writes require the service up.
"""
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

PKL = os.path.join(_HERE, "flymemory", "flymemory_v3.pkl")
SERVICE = "http://127.0.0.1:8765/mcp"

_SNAP_MEM = None


def _snapshot_path():
    snap = os.path.join(tempfile.gettempdir(), "fm_cli_snapshot.pkl")
    if not os.path.exists(snap) or \
            os.path.getmtime(snap) < os.path.getmtime(PKL):
        shutil.copy(PKL, snap)
    return snap


def _load_snapshot():
    global _SNAP_MEM
    if _SNAP_MEM is None:
        from flymemory.v3 import load
        _SNAP_MEM = load(_snapshot_path(), enable_hopfield=False)
    return _SNAP_MEM


def _snapshot_mem_by_id():
    d = pickle.load(open(_snapshot_path(), "rb"))
    return {m["memory_id"]: m for m in d["memories"]}


def service_call(tool, args):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": args}}).encode()
    req = urllib.request.Request(SERVICE, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read().decode("utf-8", "replace")
    for line in data.splitlines():
        if line.startswith("data:"):
            line = line[5:].strip()
        try:
            obj = json.loads(line)
        except Exception:
            continue
        content = obj.get("result", {}).get("content") or []
        text = "\n".join(c.get("text", "") for c in content
                         if isinstance(c, dict))
        if text.strip():
            return text
    return "(no output)"


def service_up():
    try:
        req = urllib.request.Request(SERVICE, method="POST", data=b"{}",
                                     headers={"Content-Type": "application/json",
                                              "Accept": "application/json, text/event-stream"})
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def fmt_age(ts):
    dt = time.time() - (ts or 0)
    if dt < 86400:
        return f"{int(dt // 3600)}h前"
    return f"{int(dt // 86400)}天前"


def lineage_of(mid, mems_by_id):
    seen = {mid}
    frontier = [mid]
    while frontier:
        cur = frontier.pop()
        m = mems_by_id.get(cur)
        if not m:
            continue
        sb = m.get("superseded_by")
        if sb and sb != "None" and sb not in seen:
            seen.add(sb)
            frontier.append(sb)
    for other in mems_by_id.values():
        if other.get("superseded_by") == mid and \
                other["memory_id"] not in seen:
            seen.add(other["memory_id"])
            frontier.append(other["memory_id"])
    return seen


def dependents_of(mid, mems_by_id):
    deps = {"evidence_backlinks": [], "replaced_by": None,
            "replaces": [], "state_key": None}
    m = mems_by_id.get(mid)
    if not m:
        return deps
    for other in mems_by_id.values():
        eids = other.get("evidence_ids") or []
        if mid in eids:
            deps["evidence_backlinks"].append(other["memory_id"])
    sb = m.get("superseded_by")
    if sb and sb != "None":
        deps["replaced_by"] = sb
    for other in mems_by_id.values():
        if other.get("superseded_by") == mid:
            deps["replaces"].append(other["memory_id"])
    if m.get("state_key"):
        deps["state_key"] = m["state_key"]
    return deps


def cmd_stats():
    d = pickle.load(open(_snapshot_path(), "rb"))
    mems = d["memories"]
    active = [m for m in mems if not m.get("superseded_by")
              or m.get("superseded_by") == "None"]
    superseded = len(mems) - len(active)
    keyed = sum(1 for m in active if m.get("state_key"))
    ev = sum(1 for m in mems if m.get("evidence_ids"))
    print(f"entries: {len(mems)} (active {len(active)}, superseded {superseded})")
    print(f"state_keyed active: {keyed} | entries with evidence links: {ev}")
    print(f"model: {d.get('embedder')}")
    print(f"decay_tau: {d.get('decay_tau')} days")


def cmd_search(query, top=5, comp=None):
    mem = _load_snapshot()
    results = mem.recall(query, top_k=top, compartment=comp or None)
    if not results:
        print("no relevant memories found")
        return
    for entry, sim, eff in results:
        state = "superseded" if entry.superseded_by else "active"
        comp_tag = next((t[5:] for t in entry.tags
                         if t.startswith("comp:")), "")
        print(f"[#{entry.memory_id} | {state} | sim={sim:.2f} | "
              f"{fmt_age(entry.timestamp)} | {entry.source}"
              + (f" | comp:{comp_tag}" if comp_tag else "") + "] "
              + entry.text[:100])


def cmd_history(mid):
    mems_by_id = _snapshot_mem_by_id()
    lin = sorted(lineage_of(mid, mems_by_id),
                 key=lambda x: mems_by_id[x].get("timestamp") or 0)
    if not lin:
        print(f"#{mid} not found")
        return
    print(f"lineage around #{mid} ({len(lin)} members, time order):")
    for x in lin:
        m = mems_by_id[x]
        cur = " <- current" if not m.get("superseded_by") else ""
        arrow = (" -> superseded by #" + str(m.get("superseded_by"))
                 if m.get("superseded_by") else "")
        print(f"  #{x} [{fmt_age(m.get('timestamp'))}] "
              f"{m['text'][:80]}{cur}{arrow}")


def cmd_dependents(mid):
    mems_by_id = _snapshot_mem_by_id()
    deps = dependents_of(mid, mems_by_id)
    if not mems_by_id.get(mid):
        print(f"#{mid} not found")
        return
    print(f"dependents of #{mid}:")
    print(f"  evidence back-links (entries citing this as evidence): "
          f"{deps['evidence_backlinks'] or 'none'}")
    print(f"  replaced by: #{deps['replaced_by']}" if deps["replaced_by"]
          else "  replaced by: nothing (this is current)")
    print(f"  replaces: {deps['replaces'] or 'nothing'}")
    print(f"  state_key: {deps['state_key'] or 'none'}")
    if deps["evidence_backlinks"]:
        print("  NOTE: entries above stand on this one as evidence — "
              "forgetting cleans the links but weakens their grounding.")


def cmd_forget(mid, yes=False):
    if not service_up():
        print("service is down (127.0.0.1:8765) — writes must go through "
              "the running service; start it first")
        return
    cmd_dependents(int(mid))
    if not yes:
        print("\n(dry run — re-run with --yes to forget via the service)")
        return
    print(service_call("flymemory_forget", {"memory_id": int(mid)}))


def cmd_correct(mid, new_text):
    if not service_up():
        print("service is down — cannot correct")
        return
    resp = service_call("flymemory_remember",
                        {"text": new_text, "source": "model"})
    print(resp)
    import re as _re
    m = _re.search(r"#(\d+)", resp)
    if not m:
        print("could not parse the new entry id — supersede manually with "
              "flymemory_supersede(old, new)")
        return
    new_id = int(m.group(1))
    if new_id == int(mid):
        print("(remember strengthened/merged into the same entry — "
              "no supersede needed)")
        return
    print(service_call("flymemory_supersede",
                       {"old_memory_id": int(mid),
                        "new_memory_id": new_id}))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "stats":
        cmd_stats()
    elif cmd == "search" and len(args) >= 2:
        top = 5
        comp = None
        if "--top" in args:
            top = int(args[args.index("--top") + 1])
        if "--comp" in args:
            comp = args[args.index("--comp") + 1]
        cmd_search(args[1], top, comp)
    elif cmd == "history" and len(args) >= 2:
        cmd_history(int(args[1]))
    elif cmd == "dependents" and len(args) >= 2:
        cmd_dependents(int(args[1]))
    elif cmd == "forget" and len(args) >= 2:
        cmd_forget(args[1], yes="--yes" in args)
    elif cmd == "correct" and len(args) >= 3:
        cmd_correct(int(args[1]), " ".join(args[2:]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
