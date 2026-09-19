"""One-off importer: seed a flymemory library from a directory of markdown notes.

Each .md file becomes one memory entry (frontmatter stripped, filename kept as
a tag). Run with the server STOPPED — the running service holds an in-memory
copy and would overwrite the file on its next save.

Usage:
  python import_zcode_memories.py --src ./path/to/notes [--db ./flymemory_v3.pkl]
"""
import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sentence_transformers  # noqa: F401  import torch in the main thread first
from flymemory.v3 import SmartMemory, load, save


def main():
    ap = argparse.ArgumentParser(description="Seed a flymemory library from markdown notes")
    ap.add_argument("--src", default="./zcode_memories",
                    help="directory of .md files to import (default: ./zcode_memories)")
    ap.add_argument("--db", default=os.path.join(_HERE, "flymemory_v3.pkl"),
                    help="target library file")
    ap.add_argument("--half-life", type=float, default=2592000.0,
                    help="decay tau in seconds (default: 30 days)")
    args = ap.parse_args()

    mem = load(args.db) if os.path.exists(args.db) else SmartMemory(
        n_bits=4096, decay_tau=args.half_life)
    mem.decay_tau = args.half_life

    for f in sorted(glob.glob(os.path.join(args.src, "*.md"))):
        base = os.path.splitext(os.path.basename(f))[0]
        if base.lower() == "index":
            continue  # index files would only create low-granularity duplicates
        text = open(f, encoding="utf-8").read()
        text = re.sub(r"^---.*?---\s*", "", text, flags=re.S).strip()
        if not text:
            continue
        r = mem.remember(text, tags=["import", base], source="import")
        print(f"{base:35s} {r['action']:13s} #{r.get('memory_id', '?')}")

    save(mem, args.db)
    print(f"done, total memories: {mem.size}")


if __name__ == "__main__":
    main()
