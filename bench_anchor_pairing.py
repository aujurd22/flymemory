"""P-ANCH-1: anchor pairing experiment (pre-registered, see
research/RESEARCH.md P-2026-09-28-ANCH1 -- criteria locked BEFORE this ran).

Turns the L6 anchoring observation (atomic arms A 81.2% vs B 100% were
different tasks) into a PAIRED controlled experiment: the SAME 48 turns,
two conditions, one dependent variable -- "does this message invalidate
the stored state?"

  set (n=48)
    12 balanced-stale  (anchor gold YES -- real state updates)
    12 supersede-type  (anchor gold YES, from the v1.5 dataset via
                        gold.supersede[0].old_id)
    12 balanced-resist (anchor gold NO  -- considerations/questions)
    12 noop-type       (anchor gold NO  -- facts about OLD objects; the
                        structural false-positive source)

  condition U (unanchored): turn alone -> asserts_fact true/false.
      A naive rule asserts_fact -> invalidate formalizes the
      "classify-then-decide-freely" caller strategy. Structural ceiling
      36/48 = 75% (noop class always asserts a fact yet never invalidates).
  condition A (anchored): (stored entry, turn) -> invalidates true/false.

Model: deepseek-chat, temperature 0. 96 calls total.

Run:  py -3.13 bench_anchor_pairing.py
"""
import json
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ACTOR_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

SYSTEM_U = """You are a fact-assertion detector. Given a user message, decide
whether it asserts something as CURRENTLY TRUE.
  - true  : the message states a fact in the indicative ("I switched to X",
            "I still keep the old Y around", "I fixed Z yesterday")
  - false : the message asserts no current fact (pure questions,
            considerations, hypotheses -- "I'm thinking of", "should I?",
            "maybe")
Respond with ONLY: {"asserts_fact": true/false}"""

SYSTEM_A = """You are a state-change detector. Given ONE stored memory entry
and the user's new message, decide whether the message INVALIDATES the
stored fact:
  - true  : the message states (not merely plans or considers) a
            replacement for the stored fact
  - false : the stored fact remains valid (unconfirmed considerations,
            questions, mentions of old objects, unrelated facts)
Respond with ONLY: {"invalidates": true/false}"""


def ds_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ds_chat(client, system, user):
    r = client.chat.completions.create(
        model=ACTOR_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=100)
    return r.choices[0].message.content or ""


def parse_bool(raw, key):
    try:
        return bool(json.loads(raw)[key])
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return bool(json.loads(raw[start:end + 1])[key])
            except Exception:
                pass
    return None


def load_set():
    """48 (case_id, entry_text, turn, anchor_gold, cls) from the v1.5 set."""
    path = os.path.join(_HERE, "data", "memory_judgment.json")
    cases = json.load(open(path, encoding="utf-8"))["cases"]
    items = []
    for cls, types, cap, pick_entry in [
        ("stale", {"balanced-stale"}, 12,
         lambda c: c["memories"][0]["text"]),
        ("supersede", {"supersede"}, 12,
         lambda c: next(m["text"] for m in c["memories"]
                        if m["id"] == c["gold"]["supersede"][0]["old_id"])),
        ("resist", {"balanced-resist"}, 12,
         lambda c: c["memories"][0]["text"]),
        ("noop", {"noop"}, 12, lambda c: c["memories"][0]["text"]),
    ]:
        got = [c for c in cases if c["type"] in types][:cap]
        assert len(got) == cap, f"{cls}: only {len(got)} cases"
        anchor = "NO" if cls in ("resist", "noop") else "YES"
        for c in got:
            items.append((c["case_id"], pick_entry(c), c["turn"],
                          anchor, cls))
    return items


def main():
    items = load_set()
    client = ds_client()
    traces = []
    pair = Counter()          # (U_correct, A_correct) -> n
    cls_err = {"U": Counter(), "A": Counter()}
    acc = {"U": 0, "A": 0}

    for i, (cid, entry, turn, anchor, cls) in enumerate(items):
        u_raw = ds_chat(client, SYSTEM_U, f"User message:\n\"{turn}\"")
        asserts = parse_bool(u_raw, "asserts_fact")
        u_verdict = "YES" if asserts else ("NO" if asserts is not None
                                           else "PARSE_FAIL")

        a_raw = ds_chat(client, SYSTEM_A,
                        f"Stored memory entry:\n\"{entry}\"\n\n"
                        f"User's new message:\n\"{turn}\"")
        inv = parse_bool(a_raw, "invalidates")
        a_verdict = "YES" if inv else ("NO" if inv is not None
                                       else "PARSE_FAIL")

        u_ok = u_verdict == anchor
        a_ok = a_verdict == anchor
        acc["U"] += u_ok
        acc["A"] += a_ok
        pair[(u_ok, a_ok)] += 1
        if not u_ok:
            cls_err["U"][cls] += 1
        if not a_ok:
            cls_err["A"][cls] += 1
        traces.append({"case_id": cid, "cls": cls, "anchor_gold": anchor,
                       "u_verdict": u_verdict, "a_verdict": a_verdict,
                       "u_ok": u_ok, "a_ok": a_ok})
        print(f"[{i+1}/48] {cid:14s} ({cls:9s}) gold={anchor:3s} "
              f"U={u_verdict:3s}{'✗' if not u_ok else ' '} "
              f"A={a_verdict:3s}{'✗' if not a_ok else ' '}", flush=True)

    n = len(items)
    print(f"\n=== P-ANCH-1 | n={n} | deepseek-chat temp 0 ===")
    print(f"acc(U unanchored+naive rule) = {acc['U']}/{n} = {acc['U']/n:.1%}"
          f"   [structural ceiling 75%]")
    print(f"acc(A anchored)              = {acc['A']}/{n} = {acc['A']/n:.1%}")
    b = pair[(True, False)]   # U right, A wrong
    c = pair[(False, True)]   # U wrong, A right (discordant, McNemar)
    print(f"discordant pairs: U-right/A-wrong={b}  U-wrong/A-right={c}")
    print(f"error class breakdown: U {dict(cls_err['U'])} | A {dict(cls_err['A'])}")

    # pre-registered verdict
    res = "n/a"
    resist_a_ok = sum(1 for t in traces
                      if t["cls"] == "resist" and t["a_ok"])
    if acc["A"] >= 44 and resist_a_ok >= 10:
        res = "SUPPORTED"
    elif acc["A"] < 38 or resist_a_ok < 8:
        res = "NOT SUPPORTED"
    else:
        res = "PARTIAL"
    print(f"resist-class A repair: {resist_a_ok}/12")
    print(f"PRE-REGISTERED VERDICT: {res}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"anchor_pairing_{ts}.json")
    json.dump({"experiment": "P-ANCH-1", "n": n, "acc_U": acc["U"],
               "acc_A": acc["A"], "discordant": {"b": b, "c": c},
               "resist_a_ok": resist_a_ok, "verdict": res,
               "traces": traces},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
