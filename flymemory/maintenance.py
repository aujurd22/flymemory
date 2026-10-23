"""FlyMemory maintenance: rebuildable store architecture (borrowed from
scope-recall-hermes, 2026-10-24).

Architecture principle: FACTS are the authority (text + metadata +
lineage), embeddings are a REBUILDABLE companion derived from facts.
Two operations follow:

  export-facts   pkl -> SQLite facts.db
                 Every durable field except embeddings; human-inspectable,
                 portable, model-independent.

  rebuild-embeddings   facts.db (or the pkl itself) -> re-embed every text
                 with the CURRENT model and write a fresh pkl, preserving
                 memory ids, timestamps, tags and supersede lineage.

This solves two standing pains:
  - switching FLYMEMORY_MODEL used to mean manual delete + re-import
    (lineage lost); rebuild keeps ids and lineage intact.
  - pkl corruption: facts survive in SQLite; the vector companion is
    re-derivable.

SAFETY: both operations refuse to write while the service is running on
127.0.0.1:8765 (the service holds the authoritative copy — the
改库先停服 iron rule). Stop the service first, then run, then restart.

Run:
  py -3.13 flymemory/maintenance.py export-facts  [--pkl PATH] [--to PATH]
  py -3.13 flymemory/maintenance.py rebuild-embeddings [--from facts.db] [--yes]
"""
import json
import os
import pickle
import sqlite3
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # repo root: import flymemory.v3
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

_REPO = os.path.dirname(_HERE)          # repo root (package dir's parent)
PKL = os.path.join(_REPO, "flymemory", "flymemory_v3.pkl")
FACTS_DEFAULT = os.path.join(_REPO, "flymemory", "facts.db")
SERVICE = "http://127.0.0.1:8765/mcp"

FIELDS = ["memory_id", "text", "response", "timestamp", "last_accessed",
          "access_count", "tags", "source", "superseded_by", "evidence_ids",
          "updated_at", "valid_from", "valid_to", "state_key", "state_value"]


def service_running():
    # port probe, not HTTP: the MCP endpoint 4xx-rejects bare POSTs, which
    # urllib raises as HTTPError — a live service would read as down.
    import socket
    try:
        with socket.create_connection(("127.0.0.1", 8765), timeout=2):
            return True
    except Exception:
        return False


def load_entries(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["memories"], d


def export_facts(pkl_path, out_path):
    if service_running():
        print("REFUSING: the flymemory service is running on 127.0.0.1:8765."
              " Stop it first (改库先停服), then export.")
        return 1
    mems, _ = load_entries(pkl_path)
    if os.path.exists(out_path):
        os.remove(out_path)
    con = sqlite3.connect(out_path)
    cols = ", ".join(FIELDS)
    con.execute(f"CREATE TABLE entries ({cols})")
    con.execute("""CREATE TABLE meta(
        key TEXT PRIMARY KEY, value TEXT)""")
    for m in mems:
        row = []
        for f in FIELDS:
            v = m.get(f)
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            row.append(v)
        con.execute(f"INSERT INTO entries VALUES ({','.join('?' * len(FIELDS))})",
                    row)
    con.execute("INSERT OR REPLACE INTO meta VALUES('entry_count', ?)",
                (str(len(mems)),))
    con.execute("INSERT OR REPLACE INTO meta VALUES('exported_at', ?)",
                (str(time.time()),))
    con.commit()
    con.close()
    print(f"exported {len(mems)} entries -> {out_path} "
          f"({os.path.getsize(out_path)//1024} KB; no embeddings — "
          f"facts are the authority)")


def rebuild_embeddings(facts_path, out_path, model=None):
    if service_running():
        print("REFUSING: the flymemory service is running on 127.0.0.1:8765."
              " Stop it first, rebuild, then restart.")
        return 1
    con = sqlite3.connect(facts_path)
    rows = con.execute(f"SELECT {', '.join(FIELDS)} FROM entries").fetchall()
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    con.close()
    print(f"rebuilding embeddings for {len(rows)} entries "
          f"(model: {model or 'current FLYMEMORY_MODEL'})", flush=True)

    from flymemory.v3 import SmartMemory, _embed  # noqa: E402
    mem = SmartMemory(n_bits=4096, enable_hopfield=False)
    # re-embed in entry order; preserve ids by appending then renumbering
    t0 = time.time()
    new_entries = []
    for ri, row in enumerate(rows):
        rec = dict(zip(FIELDS, row))
        for f in ("tags", "evidence_ids"):
            v = rec.get(f)
            if isinstance(v, str) and v:
                try:
                    rec[f] = json.loads(v)
                except Exception:
                    rec[f] = v
        emb = _embed(rec["text"])
        rec["embedding"] = emb
        new_entries.append(rec)
        if (ri + 1) % 1000 == 0:
            print(f"  embedded {ri+1}/{len(rows)} ({time.time()-t0:.0f}s)",
                  flush=True)
    # write the pkl directly: ids and lineage are preserved verbatim
    out = {"schema": "3.2", "embedder": model or
           "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
           "config": {"enable_hopfield": False, "two_stage": False,
                      "enable_rerank": False},
           "memories": new_entries,
           "_next_id": max(r["memory_id"] for r in new_entries) + 1,
           "n_bits": 4096,
           "decay_tau": 2592000.0, "decay_half_life": 7776000.0}
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {out_path} ({os.path.getsize(out_path)//1024//1024} MB) "
          f"({time.time()-t0:.0f}s total)")
    print("NOTE: start the service and verify recall before trusting this "
          "rebuild; keep the old pkl as backup.")
    return 0


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    pkl = PKL
    if "--pkl" in args:
        pkl = args[args.index("--pkl") + 1]
    if cmd == "export-facts":
        out = (args[args.index("--to") + 1] if "--to" in args
               else FACTS_DEFAULT)
        sys.exit(export_facts(pkl, out))
    elif cmd == "rebuild-embeddings":
        facts = (args[args.index("--from") + 1] if "--from" in args
                 else FACTS_DEFAULT)
        model = (args[args.index("--model") + 1] if "--model" in args
                 else None)
        out = (args[args.index("--out") + 1] if "--out" in args
               else pkl + ".rebuilt")
        sys.exit(rebuild_embeddings(facts, out, model))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
