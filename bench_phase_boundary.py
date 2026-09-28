"""P-PHASE-1: anchor phase boundary (pre-registered, research/RESEARCH.md
P-2026-09-28-PHASE1 -- verdict PENDING there; criteria locked before run).

Borrows P47's phase-boundary method from intuition-mechanism: the SAME
anchored-judgment task at graded task difficulty, everything else fixed.

Task per trial: a 6-entry store (entry #1 = anchor, #2-6 distractors) and
one user message; the model answers whether the message INVALIDATES the
fact stored in entry #1 (same prompt shape as P-ANCH-1 condition A).

36 trials:
  18 YES -- 6 SMALL  (digit/date swap inside a near-identical sentence;
                      the L4 small-edit class)
           6 MID    (value swap + natural rewording)
           6 LARGE  (different wording + value domain; the B-arm class)
  18 NO  -- 6 consideration, 6 question,
            6 old-object mention with a near-collision distractor entry

Frozen at construction (before any LLM call): delta_emb = 1 - cos(A, E)
on YES trials; sim-gap = cos(A,E) - max_j cos(A,D_j) on all trials.

Run:  py -3.13 bench_phase_boundary.py
"""
import json
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
try:
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 4))
except ImportError:
    pass
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

ACTOR_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

FILLERS = [
    "The user enjoys cycling on weekends.",
    "The user likes hiking in autumn.",
    "The user designs circuit boards at work.",
    "The user reads on a tablet in the evening.",
    "The user writes two blog posts per month.",
    "The user attends a weekly badminton game.",
]

# (trial_id, level, gold, anchor, assertion, near_collision_distractor)
TRIALS = [
    # ---- SMALL: digit/date swap, near-identical sentence ----
    ("sm01", "SMALL", True,
     "The user's weekly team sync is on Monday at 15:00.",
     "The weekly team sync is on Monday at 16:00 now.", None),
    ("sm02", "SMALL", True,
     "The user's phone number is 138-0000-1111.",
     "My phone number is 138-0000-8888.", None),
    ("sm03", "SMALL", True,
     "The user's database is PostgreSQL 15.",
     "The database is on PostgreSQL 16.", None),
    ("sm04", "SMALL", True,
     "The user lives at 21 Riverside Road, apartment 4.",
     "I live at 21 Riverside Road, apartment 7 now.", None),
    ("sm05", "SMALL", True,
     "The user's office is in Building 3.",
     "My office is in Building 5.", None),
    ("sm06", "SMALL", True,
     "The user's flight home is on the 12th.",
     "My flight home is on the 14th.", None),
    # ---- MID: value swap + natural rewording ----
    ("md01", "MID", True,
     "The user's database is PostgreSQL 15.",
     "We upgraded the whole stack to Postgres 16 over the weekend.", None),
    ("md02", "MID", True,
     "The user's manager is Manager Zhang.",
     "Since the reorg I report to Manager Liu.", None),
    ("md03", "MID", True,
     "The user drives an old gasoline hatchback.",
     "I got rid of the hatchback -- driving an EV now.", None),
    ("md04", "MID", True,
     "The user studies Japanese with a paper textbook.",
     "I switched my Japanese study to a spaced-repetition app.", None),
    ("md05", "MID", True,
     "The user's editor is PyCharm.",
     "These days I do all my editing in Neovim.", None),
    ("md06", "MID", True,
     "The user hosts their blog on a rented VPS.",
     "The blog moved to a static hosting platform in July.", None),
    # ---- LARGE: different wording + value domain (the B-arm class) ----
    ("lg01", "LARGE", True,
     "The user's favourite editor is VS Code.",
     "I switched to Zed as my daily editor.", None),
    ("lg02", "LARGE", True,
     "The user's morning commute is the subway.",
     "I started biking to work instead of taking the subway.", None),
    ("lg03", "LARGE", True,
     "The user's cloud provider is AWS.",
     "I migrated my workloads to GCP.", None),
    ("lg04", "LARGE", True,
     "The user's keyboard is a HHKB Professional 2.",
     "I switched to a Nuphy Air75 keyboard.", None),
    ("lg05", "LARGE", True,
     "The user prefers dark mode in every app.",
     "I switched to light mode after my eye surgery last month.", None),
    ("lg06", "LARGE", True,
     "The user tracks tasks in Notion.",
     "I moved my task tracking to Obsidian.", None),
    # ---- NO: consideration ----
    ("nc01", "CONSIDER", False,
     "The user tracks tasks in Obsidian.",
     "I'm considering switching back to Notion for tasks.", None),
    ("nc02", "CONSIDER", False,
     "The user drinks cold brew every morning.",
     "Maybe I should go back to pour-over -- the cold brew might have been a phase.", None),
    ("nc03", "CONSIDER", False,
     "The user's favourite podcast is Developer Tea.",
     "I'm thinking of resubscribing to the Lex Fridman podcast.", None),
    ("nc04", "CONSIDER", False,
     "The user bikes to work these days.",
     "I'm considering taking the subway again -- my bike commute might have been a mistake.", None),
    ("nc05", "CONSIDER", False,
     "The user's cloud is GCP now.",
     "Should I move my workloads back to AWS from GCP?", None),
    ("nc06", "CONSIDER", False,
     "The user's keyboard is a Nuphy Air75.",
     "Should I sell the Nuphy and go back to my old HHKB?", None),
    # ---- NO: question ----
    ("nq01", "QUESTION", False,
     "The user's monitor is a 32-inch 5K.",
     "Is my current monitor the 5K one?", None),
    ("nq02", "QUESTION", False,
     "The user's database is PostgreSQL 16.",
     "Is my database still on PostgreSQL 15? I forgot.", None),
    ("nq03", "QUESTION", False,
     "The user tracks tasks in Obsidian.",
     "Was I using Obsidian or Notion for tasks?", None),
    ("nq04", "QUESTION", False,
     "The user lives in Hangzhou now.",
     "Did I move to Hangzhou two months ago or three?", None),
    ("nq05", "QUESTION", False,
     "The user's flight home is on the 14th.",
     "What date is my flight home again?", None),
    ("nq06", "QUESTION", False,
     "The user's office is in Building 5.",
     "Which building is my office in?", None),
    # ---- NO: old-object mention with near-collision distractor ----
    ("nx01", "NEAR-COLL", False,
     "The user runs Fedora Linux now.",
     "I spent this morning fixing something on my old Windows VM.",
     "The user keeps a Windows VM for legacy testing."),
    ("nx02", "NEAR-COLL", False,
     "The user's current phone is 139-9999-8888.",
     "I found my old business cards with the 138 number while cleaning.",
     "The user's old business cards show the 138-0000-1111 number."),
    ("nx03", "NEAR-COLL", False,
     "The user drives an electric car now.",
     "Petrol prices went up again -- glad I don't care anymore.",
     "The user used to complain about petrol prices."),
    ("nx04", "NEAR-COLL", False,
     "The user's blog is on a static hosting platform.",
     "I still have the old VPS backup archive on my drive.",
     "The user keeps an old VPS backup archive on disk."),
    ("nx05", "NEAR-COLL", False,
     "The user studies Japanese with a spaced-repetition app.",
     "The paper textbook is still sitting on my desk, never got around to tossing it.",
     "The user's paper Japanese textbook sits on the desk."),
    ("nx06", "NEAR-COLL", False,
     "The user's team is called Team Kestrel.",
     "I drank coffee from my old Falcon mug today.",
     "A Falcon logo is still printed on the user's mug."),
]

SYSTEM = """You are a state-change detector. You see a set of stored memory
entries and the user's new message. Decide whether the message INVALIDATES
the fact stored in entry #1 (the anchor):
  - true  : the message states (not merely plans or considers) a
            replacement for entry #1's fact
  - false : entry #1 remains valid (unconfirmed considerations, questions,
            mentions of old objects, unrelated facts)
Other entries are context only -- judge entry #1.
Respond with ONLY: {"invalidates": true/false}"""


def ds_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ds_chat(client, user):
    r = client.chat.completions.create(
        model=ACTOR_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
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


def cos(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-8)
                 / (np.linalg.norm(b) + 1e-8))


def build_user(anchor, assertion, distractors):
    lines = [f"#1: {anchor}"]
    for i, d in enumerate(distractors, start=2):
        lines.append(f"#{i}: {d}")
    return ("Stored memory entries:\n" + "\n".join(lines) +
            f"\n\nUser's new message:\n\"{assertion}\"\n\n"
            "Does the message invalidate entry #1? Answer now.")


def main():
    # ---- frozen geometric measurements (construction QC, pre-LLM) ----
    from flymemory.v3 import _embed  # noqa: E402

    # trial texts are single sentences; _embed directly
    emb = _embed

    frozen = []
    for tid, level, gold, anchor, assertion, near in TRIALS:
        distractors = [near] if near else []
        for f in FILLERS:
            if len(distractors) >= 5:
                break
            if f != anchor:
                distractors.append(f)
        e_a, e_e = emb(assertion), emb(anchor)
        sims_d = [cos(e_a, emb(d)) for d in distractors]
        frozen.append({
            "trial_id": tid, "level": level, "gold": gold,
            "delta_emb": round(1 - cos(e_a, e_e), 4),
            "sim_gap": round(cos(e_a, e_e) - max(sims_d), 4),
            "anchor": anchor, "assertion": assertion,
            "distractors": distractors,
        })

    print("frozen measurements (pre-LLM):")
    for lv in ("SMALL", "MID", "LARGE"):
        ds = [f["delta_emb"] for f in frozen if f["level"] == lv]
        print(f"  {lv:6s} delta_emb mean {sum(ds)/len(ds):.3f} "
              f"(n={len(ds)})")
    nc = [f["sim_gap"] for f in frozen if f["level"] == "NEAR-COLL"]
    print(f"  NEAR-COLL sim_gap mean {sum(nc)/len(nc):.3f}")

    # ---- LLM run ----
    client = ds_client()
    traces = []
    acc_by = {"YES": {"SMALL": [0, 0], "MID": [0, 0], "LARGE": [0, 0]},
              "NO": {"CONSIDER": [0, 0], "QUESTION": [0, 0],
                     "NEAR-COLL": [0, 0]}}
    for i, f in enumerate(frozen):
        raw = ds_chat(client, build_user(f["anchor"], f["assertion"],
                                         f["distractors"]))
        inv = parse_bool(raw, "invalidates")
        verdict = "YES" if inv else ("NO" if inv is not None
                                     else "PARSE_FAIL")
        ok = verdict == ("YES" if f["gold"] else "NO")
        key = f["level"]
        slot = acc_by["YES" if f["gold"] else "NO"][key]
        slot[1] += 1
        slot[0] += ok
        traces.append({**f, "verdict": verdict, "ok": ok,
                       "raw": raw[:200]})
        print(f"[{i+1}/36] {f['trial_id']} ({key:9s}) gold="
              f"{'YES' if f['gold'] else 'NO ':4s} pred={verdict:4s} "
              f"{'OK' if ok else 'MISS'}", flush=True)

    print("\n=== P-PHASE-1 | deepseek-chat temp 0 ===")
    for lv in ("SMALL", "MID", "LARGE"):
        c, n = acc_by["YES"][lv]
        print(f"YES {lv:6s}: {c}/{n}")
    for lv in ("CONSIDER", "QUESTION", "NEAR-COLL"):
        c, n = acc_by["NO"][lv]
        print(f"NO  {lv:9s}: {c}/{n} correct "
              f"(FP {n-c}/{n})")
    c_s, n_s = acc_by["YES"]["SMALL"]
    c_l, n_l = acc_by["YES"]["LARGE"]
    print(f"gradient large-small gap: {c_l/n_l - c_s/n_s:+.1%}")
    fp_near = acc_by["NO"]["NEAR-COLL"][1] - acc_by["NO"]["NEAR-COLL"][0]
    fp_other = sum(n - c for lv, (c, n) in acc_by["NO"].items()
                   if lv != "NEAR-COLL")
    n_other = sum(n for lv, (c, n) in acc_by["NO"].items()
                  if lv != "NEAR-COLL")
    print(f"NO FP rate: near-collision {fp_near}/6 vs others {fp_other}/{n_other}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"phase_{ts}.json")
    json.dump({"experiment": "P-PHASE-1", "frozen": frozen,
               "acc_by": acc_by, "traces": traces},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
