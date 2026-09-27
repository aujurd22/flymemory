"""Atomic discretion benchmark -- decompose memory policy into three
cognitive primitives, benchmarked SEPARATELY (the next-phase direction
from the round-9 cross-review).

Motivation: end-to-end judgment (168 cases) shows supersede ~1.00 P/R and
stale 12/12, with failures concentrated at the RESIST/intent boundary. The
Arm-S transplant (3f87703) showed a single extraction scaffold does not fix
that. Hypothesis: the monolithic memory-policy prompt conflates several
cognitive primitives; measuring them separately locates which primitive
actually owns the errors, and whether any has an objective sufficient
statistic (P32-i style) waiting to be found.

Arms:
  A  assertion extraction    turn -> [{text, kind: fact|intent|question}]
     (representation problem, P32-i-analogous)
  B  state-change detection  (entry, assertion) -> CHANGED|UNCHANGED|UNKNOWN
     (does this assertion invalidate the stored state?)
  C  memory-worthiness       (assertion, context) -> KEEP|DISCARD|EPHEMERAL
     (even if true, worth keeping long-term?)

Key attribution questions:
  - A: do intent-statements like "I'm switching back to X" get labeled
    fact? (direct measurement of the bal_resist_01 root cause)
  - B: given a FACT assertion, is CHANGED/UNCHANGED easy? (if yes, the
    end-to-end failures own entirely to A)
  - C: is EPHEMERAL distinguished from KEEP, or is everything true kept?

Run:  py -3.13 bench_atomic_judgment.py            (all arms)
      DEEPSEEK_API_KEY=... py -3.13 bench_atomic_judgment.py --arm A
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ACTOR_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
DAY = 86400.0

# --------------------------------------------------------------------------
# ARM A -- assertion extraction. 32 turns. gold kinds annotated by hand.
# The interesting rows are the RESIST intent statements; STALE turns are
# third-person fact statements (trivially fact) and serve as fillers.
# --------------------------------------------------------------------------
ARM_A = [
    # --- 12 STALE turns: gold fact (explicit state updates) ---
    ("a_st01", "The user switched to Zed as their daily editor.", "fact"),
    ("a_st02", "The user migrated their database to PostgreSQL 16.", "fact"),
    ("a_st03", "The user started biking to work instead of taking the subway.", "fact"),
    ("a_st04", "The user migrated their workloads to GCP.", "fact"),
    ("a_st05", "The user's favourite sushi place is now Sushi Ken.", "fact"),
    ("a_st06", "The user moved their task tracking from Notion to Obsidian.", "fact"),
    ("a_st07", "The user changed their desktop background to an ocean photo.", "fact"),
    ("a_st08", "The user switched from pour-over to cold brew every morning.", "fact"),
    ("a_st09", "The user's tests moved from pytest to vitest-style runner.", "fact"),
    ("a_st10", "The user upgraded to a 32-inch 5K monitor.", "fact"),
    ("a_st11", "The user's current favourite podcast is Developer Tea.", "fact"),
    ("a_st12", "The user switched to a Nuphy Air75 keyboard.", "fact"),
    # --- 12 RESIST turns: the boundary under measurement ---
    ("a_rs01", "I'm switching back to VS Code -- Zed kept crashing.", "intent"),
    ("a_rs02", "Is my database still on PostgreSQL 15? Thinking of downgrading anyway.", "question"),
    ("a_rs03", "I'm considering taking the subway again -- my bike commute might have been a mistake.", "intent"),
    ("a_rs04", "Should I move my workloads back to AWS from GCP?", "question"),
    ("a_rs05", "I miss Sushi Iwa -- should I go back there for lunch today?", "question"),
    ("a_rs06", "I'm thinking of going back to Notion for tasks -- Obsidian has been fine though.", "intent"),
    ("a_rs07", "That mountain photo background I used to have -- should I switch back to it?", "question"),
    ("a_rs08", "Maybe I should go back to pour-over -- the cold brew might have been a phase.", "intent"),
    ("a_rs09", "Should I revert the frontend tests to pytest? The vitest-style runner might have been a mistake.", "question"),
    ("a_rs10", "The 27-inch 4K might have been better -- should I go back to it?", "question"),
    ("a_rs11", "I'm thinking of resubscribing to the Lex Fridman podcast.", "intent"),
    ("a_rs12", "Should I sell the Nuphy Air75 and go back to my old HHKB? The HHKB is in the closet.", "question"),
    # --- 4 first-person FACT updates (non-trivial fact class) ---
    ("a_ft01", "I moved all my daily editing to Neovim recently.", "fact"),
    ("a_ft02", "I updated my contact number to 139-9999-8888 in June.", "fact"),
    ("a_ft03", "I sold the hatchback and bought an electric car in May.", "fact"),
    ("a_ft04", "I migrated the blog to a static hosting platform in July.", "fact"),
    # --- 4 no-op turns: fact assertions that do NOT update stored state ---
    ("a_np01", "I fixed something on my old Windows VM today.", "fact"),
    ("a_np02", "The old 138 number still appears on some printed business cards.", "fact"),
    ("a_np03", "I complained about petrol prices out of habit yesterday.", "fact"),
    ("a_np04", "An old VPS backup archive still exists on my disk.", "fact"),
]

SYSTEM_A = """You are an assertion extractor. Given a user message, list EVERY
factual claim it asserts, each tagged with its kind:
  - "fact"    : stated as currently true ("I moved to Hangzhou",
                "I'm switching back to VS Code" if the user declares they
                ARE doing it)
  - "intent"  : only a plan/consideration/hypothesis ("I'm thinking of",
                "I'm considering", "maybe I should")
  - "question": a pure question, asserting nothing
If the message asserts nothing, return an empty list.

Respond with ONLY this JSON:
{"assertions": [{"text": "...", "kind": "fact|intent|question"}]}"""

# --------------------------------------------------------------------------
# ARM B -- state-change detection. (entry, assertion) pairs.
# CHANGED half is extracted from real supersede scenarios; UNCHANGED half
# is paraphrase-kept / offhand-mention / unrelated assertion.
# --------------------------------------------------------------------------
ARM_B = [
    # CHANGED: assertion invalidates the entry's state
    ("b_ch01", "The user's favourite editor is VS Code.", "I switched to Zed as my daily editor.", "CHANGED"),
    ("b_ch02", "The user's database is PostgreSQL 15.", "I migrated my database to PostgreSQL 16.", "CHANGED"),
    ("b_ch03", "The user's morning commute is the subway.", "I started biking to work instead of taking the subway.", "CHANGED"),
    ("b_ch04", "The user's cloud provider is AWS.", "I migrated my workloads to GCP.", "CHANGED"),
    ("b_ch05", "The user tracks tasks in Notion.", "I moved my task tracking to Obsidian.", "CHANGED"),
    ("b_ch06", "The user drinks pour-over coffee every morning.", "I switched from pour-over to cold brew every morning.", "CHANGED"),
    ("b_ch07", "The user's main monitor is a 27-inch 4K.", "I upgraded to a 32-inch 5K monitor.", "CHANGED"),
    ("b_ch08", "The user's keyboard is a HHKB Professional 2.", "I switched to a Nuphy Air75 keyboard.", "CHANGED"),
    ("b_ch09", "The user writes most backend code in Python.", "I migrated my backend services to Rust in March.", "CHANGED"),
    ("b_ch10", "The user lives in Shenzhen.", "I moved to Hangzhou for a new job two months ago.", "CHANGED"),
    ("b_ch11", "The user's phone number is 138-0000-1111.", "I updated my contact number to 139-9999-8888 in June.", "CHANGED"),
    ("b_ch12", "The user reports to Manager Zhang.", "My new manager is Manager Liu since the reorg.", "CHANGED"),
    ("b_ch13", "The user hosts their blog on a rented VPS.", "I migrated the blog to a static hosting platform in July.", "CHANGED"),
    ("b_ch14", "The user prefers dark mode in every app.", "I switched to light mode after eye surgery last month.", "CHANGED"),
    ("b_ch15", "The user drives an old gasoline hatchback.", "I sold the hatchback and bought an electric car in May.", "CHANGED"),
    # UNCHANGED: paraphrase-kept
    ("b_uc01", "The user's favourite editor is Zed.", "Zed is my editor of choice these days.", "UNCHANGED"),
    ("b_uc02", "The user bikes to work these days.", "I still commute by bike every day.", "UNCHANGED"),
    ("b_uc03", "The user's database is PostgreSQL 16.", "My database is on Postgres 16.", "UNCHANGED"),
    # UNCHANGED: offhand mention of an old state (not an update)
    ("b_uc04", "The user runs Fedora Linux now.", "I fixed something on my old Windows VM today.", "UNCHANGED"),
    ("b_uc05", "The user's current phone is 139-9999-8888.", "The old 138 number still appears on some printed business cards.", "UNCHANGED"),
    ("b_uc06", "The user drives an electric car now.", "I complained about petrol prices out of habit yesterday.", "UNCHANGED"),
    # UNCHANGED: unrelated assertion
    ("b_uc07", "The user's favourite sushi place is Sushi Ken.", "I also enjoy cycling on weekends.", "UNCHANGED"),
    ("b_uc08", "The user tracks tasks in Obsidian.", "I read on a tablet in the evening.", "UNCHANGED"),
    # UNCHANGED: hypothetical/conditional about the stored state itself
    ("b_uc09", "The user's cloud is GCP.", "If GCP ever gets too expensive I might reconsider.", "UNCHANGED"),
    ("b_uc10", "The user's keyboard is a Nuphy Air75.", "The Nuphy Air75 has been working fine so far.", "UNCHANGED"),
    # UNCHANGED: question about the stored state
    ("b_uc11", "The user tracks tasks in Obsidian.", "Was I using Obsidian or Notion for tasks? I forgot.", "UNCHANGED"),
    ("b_uc12", "The user's monitor is a 32-inch 5K.", "Is my current monitor the 5K one?", "UNCHANGED"),
    # UNCHANGED: consideration-only statements (the RESIST class)
    ("b_uc13", "The user's favourite podcast is Developer Tea.", "I'm thinking of resubscribing to the Lex Fridman podcast.", "UNCHANGED"),
    ("b_uc14", "The user drinks cold brew every morning.", "Maybe I should go back to pour-over -- the cold brew might have been a phase.", "UNCHANGED"),
    ("b_uc15", "The user's favourite editor is Zed.", "I'm considering switching back to VS Code -- Zed kept crashing.", "UNCHANGED"),
]

SYSTEM_B = """You are a state-change detector. Given one stored memory entry
and one assertion from the user's new message, decide whether the assertion
INVALIDATES the stored state:
  - "CHANGED"  : the assertion is stated as currently true and contradicts/
                 replaces the entry's fact
  - "UNCHANGED": the entry remains valid (paraphrase, offhand mention of an
                 old state, unrelated fact, question, or unconfirmed
                 consideration/hypothesis)
  - "UNKNOWN"  : genuinely undecidable from the given information
Considerations, plans and hypotheticals do NOT change state.

Respond with ONLY this JSON: {"verdict": "CHANGED|UNCHANGED|UNKNOWN",
"reason": "one short sentence"}"""

# --------------------------------------------------------------------------
# ARM C -- memory-worthiness. (assertion, context) -> KEEP|DISCARD|EPHEMERAL
# --------------------------------------------------------------------------
ARM_C = [
    # KEEP: durable new facts
    ("c_kp01", "I got a new job as a site reliability engineer at a logistics company.", "{}", "KEEP"),
    ("c_kp02", "I'm building a connectome analysis toolkit in the evenings.", "{}", "KEEP"),
    ("c_kp03", "My daughter just started primary school this September.", "{}", "KEEP"),
    ("c_kp04", "I study Japanese with a spaced-repetition app now.", "{}", "KEEP"),
    ("c_kp05", "I design circuit boards at work.", "{}", "KEEP"),
    ("c_kp06", "We moved to Hangzhou last month for my new job.", "{}", "KEEP"),
    ("c_kp07", "I'm allergic to shellfish.", "{}", "KEEP"),
    ("c_kp08", "My main machine runs Fedora Linux now.", "{}", "KEEP"),
    ("c_kp09", "I write two blog posts per month.", "{}", "KEEP"),
    ("c_kp10", "I attend the weekly team sync every Monday.", "{}", "KEEP"),
    # DISCARD: small talk / one-off events / already-known
    ("c_dc01", "Thanks, that helped a lot!", "{}", "DISCARD"),
    ("c_dc02", "I had a sandwich for lunch today.", "{}", "DISCARD"),
    ("c_dc03", "The weather is nice today.", "{}", "DISCARD"),
    ("c_dc04", "I accidentally closed the wrong tab earlier.", "{}", "DISCARD"),
    ("c_dc05", "My train was delayed twenty minutes this morning.", "{}", "DISCARD"),
    ("c_dc06", "Zed is my editor of choice these days.", "Memory already contains: The user's favourite editor is Zed.", "DISCARD"),
    ("c_dc07", "I commute by bike every day.", "Memory already contains: The user bikes to work these days.", "DISCARD"),
    ("c_dc08", "My database is Postgres 16.", "Memory already contains: The user's database is PostgreSQL 16.", "DISCARD"),
    ("c_dc09", "Sorry for the late reply, I was in a meeting.", "{}", "DISCARD"),
    ("c_dc10", "I'll send you that file later today.", "{}", "DISCARD"),
    # EPHEMERAL: transient states / unconfirmed considerations
    ("c_ep01", "I'm so tired this week.", "{}", "EPHEMERAL"),
    ("c_ep02", "I'm thinking of switching back to Notion for tasks.", "{}", "EPHEMERAL"),
    ("c_ep03", "Maybe I should go back to pour-over coffee.", "{}", "EPHEMERAL"),
    ("c_ep04", "I'm considering selling the Nuphy keyboard.", "{}", "EPHEMERAL"),
    ("c_ep05", "I might take a day off next Friday.", "{}", "EPHEMERAL"),
    ("c_ep06", "This cold brew might have been a phase.", "{}", "EPHEMERAL"),
    ("c_ep07", "I'm in a mood to rewatch old sci-fi movies lately.", "{}", "EPHEMERAL"),
    ("c_ep08", "Should I go back to the Lex Fridman podcast? Not sure yet.", "{}", "EPHEMERAL"),
    ("c_ep09", "I'm tempted to buy the new 6K monitor when it drops.", "{}", "EPHEMERAL"),
    ("c_ep10", "Feeling a bit burned out on side projects these days.", "{}", "EPHEMERAL"),
]

SYSTEM_C = """You are a memory-worthiness classifier. Given an assertion from
the user's message (and optionally what memory already contains), decide:
  - "KEEP"      : a durable fact about the user worth keeping long-term
                  (job, projects, family, preferences, skills, environment)
  - "DISCARD"   : small talk, one-off events, or facts memory already
                  contains (a rephrasing of an existing entry adds nothing)
  - "EPHEMERAL" : transient states, moods, unconfirmed considerations or
                  plans that are true now but will likely stop being true
                  or never were confirmed
Respond with ONLY this JSON: {"verdict": "KEEP|DISCARD|EPHEMERAL",
"reason": "one short sentence"}"""


def ds_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ds_chat(client, system, user, max_tokens=400):
    r = client.chat.completions.create(
        model=ACTOR_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=max_tokens)
    return r.choices[0].message.content or ""


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def run_arm(name, system, items, build_user, extract_gold, extract_pred):
    client = ds_client()
    traces, confusion = [], Counter()
    for i, (cid, *payload) in enumerate(items):
        gold = extract_gold(payload)
        raw = ds_chat(client, system, build_user(payload))
        dec = parse_json(raw)
        pred = extract_pred(dec)
        confusion[(gold, pred)] += 1
        traces.append({"case_id": cid, "gold": gold, "pred": pred,
                       "raw_decision": dec, "raw_output": raw[:500]})
        print(f"  [{i+1}/{len(items)}] {cid}: gold={gold} pred={pred}",
              flush=True)
    n = sum(confusion.values())
    correct = sum(v for (g, p), v in confusion.items() if g == p)
    print(f"\n  == Arm {name}: {correct}/{n} = "
          f"{correct/n:.1%} | confusion: "
          f"{ {f'{g}->{p}': v for (g, p), v in sorted(confusion.items())} }\n")
    return {"arm": name, "correct": correct, "n": n,
            "confusion": {f"{g}|{p}": v for (g, p), v in confusion.items()},
            "traces": traces}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "all"], default="all")
    args = ap.parse_args()

    results = {}
    if args.arm in ("A", "all"):
        results["A"] = run_arm(
            "A assertion-extraction", SYSTEM_A, ARM_A,
            lambda p: f"User message:\n\"{p[0]}\"\n\nExtract assertions now.",
            lambda p: p[1],
            lambda d: (d.get("assertions") or [{}])[0].get("kind",
                                                           "PARSE_FAIL")
            if d.get("assertions") else
            ("EMPTY_OK" if "assertions" in d else "PARSE_FAIL"))
    if args.arm in ("B", "all"):
        results["B"] = run_arm(
            "B state-change-detection", SYSTEM_B, ARM_B,
            lambda p: (f"Stored memory entry:\n\"{p[0]}\"\n\n"
                       f"New assertion:\n\"{p[1]}\"\n\nVerdict?"),
            lambda p: p[2],
            lambda d: d.get("verdict", "PARSE_FAIL"))
    if args.arm in ("C", "all"):
        results["C"] = run_arm(
            "C memory-worthiness", SYSTEM_C, ARM_C,
            lambda p: (f"Assertion:\n\"{p[0]}\"\n\n"
                       f"Context: {p[1].format('Nothing relevant stored.')}\n\n"
                       f"Verdict?"),
            lambda p: p[2],
            lambda d: d.get("verdict", "PARSE_FAIL"))

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"atomic_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmark": "atomic-discretion-v0", "ts": ts,
                   "results": results}, f, ensure_ascii=False, indent=1)
    print(f"trace: {out}")


if __name__ == "__main__":
    main()
