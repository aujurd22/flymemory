"""One-shot repair: backfill last_accessed/timestamp on longmemeval_s_bench.pkl.

The S-edition library was built with every haystack date parse silently
failing ('2023/05/20 (Sat) 02:21' split at the mid-string weekday paren),
leaving 199,509 entries with last_accessed=None -> decay weights all-NaN.
Entries carry their session id in tags[1]; map sid -> session date from the
source JSON and backfill in place. No re-embedding needed.

Run: python repair_lme_s_timestamps.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from bench_longmemeval_s import parse_lme_date  # noqa: E402
from flymemory.v3 import load, save  # noqa: E402


def main():
    with open(os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    sid_ts = {}
    for q in data:
        for sid, ds in zip(q["haystack_session_ids"], q["haystack_dates"]):
            if sid not in sid_ts:
                sid_ts[sid] = parse_lme_date(ds)
    print(f"session dates parsed: {len(sid_ts)}", flush=True)

    mem = load(os.path.join(_HERE, "longmemeval_s_bench.pkl"), enable_hopfield=False)
    fixed = missing = 0
    for m in mem.memories:
        sid = next((t for t in m.tags if t != "lme"), None)
        ts = sid_ts.get(sid)
        if ts is None:
            missing += 1
            continue
        m.last_accessed = ts
        m.timestamp = ts
        fixed += 1
    print(f"backfilled {fixed}, unmapped {missing} / {mem.size}", flush=True)
    assert missing == 0, "unmapped sessions remain -- do not save"
    save(mem, os.path.join(_HERE, "longmemeval_s_bench.pkl"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
