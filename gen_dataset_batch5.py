# one-shot generator: batch 5 -- BALANCED STALE/NEW DETECTION (v1.5, 144 -> 168)
#
# Borrowed designs (round-8 cross-pollination):
#   - intuition-mechanism P32-h: balanced novelty detection (response bias
#     measurement -- the doubao judge said NEW only 7/40 times)
#   - STALE benchmark: Premise Resistance (queries embedding stale premises)
#
# 24 cases = 12 STALE (fact really changed; correct answer = NEW value)
#          + 12 RESIST (fact UNCHANGED; the question embeds a stale premise,
#            correct answer = ORIGINAL value, resisting the false premise)
#
# Metrics beyond P/R: response bias = how often the model says "changed"
# when it didn't (over-eager supersede) vs "unchanged" when it did.

import json

p = "data/memory_judgment.json"
d = json.load(open(p, encoding="utf-8"))
existing = {c["case_id"] for c in d["cases"]}
cases = []

# ---- STALE: fact really changed (correct: use NEW value) ----
STALE = [
    ("The user's favourite editor is VS Code.", "The user switched to Zed as their daily editor.", "Zed", "VS Code"),
    ("The user's database is PostgreSQL 15.", "The user migrated their database to PostgreSQL 16.", "PostgreSQL 16", "PostgreSQL 15"),
    ("The user's morning commute is the subway.", "The user started biking to work instead of taking the subway.", "biking", "subway"),
    ("The user's cloud provider is AWS.", "The user migrated their workloads to GCP.", "GCP", "AWS"),
    ("The user's favourite sushi place is Sushi Iwa.", "The user's favourite sushi place is now Sushi Ken.", "Sushi Ken", "Sushi Iwa"),
    ("The user tracks tasks in Notion.", "The user moved their task tracking from Notion to Obsidian.", "Obsidian", "Notion"),
    ("The user's desktop background is a mountain photo.", "The user changed their desktop background to an ocean photo.", "ocean", "mountain"),
    ("The user drinks pour-over coffee every morning.", "The user switched from pour-over to cold brew every morning.", "cold brew", "pour-over"),
    ("The user's test framework is pytest.", "The user's tests moved from pytest to vitest-style runner for the frontend.", "vitest-style", "pytest"),
    ("The user's main monitor is a 27-inch 4K.", "The user upgraded to a 32-inch 5K monitor.", "32-inch 5K", "27-inch 4K"),
    ("The user's favourite programming podcast is Lex Fridman.", "The user's current favourite podcast is Developer Tea.", "Developer Tea", "Lex Fridman"),
    ("The user's keyboard is a HHKB Professional 2.", "The user switched to a Nuphy Air75 keyboard.", "Nuphy Air75", "HHKB"),
]
for i, (old, new, cur, stale) in enumerate(STALE):
    cases.append({
        "case_id": f"bal_stale_{i+1:02d}", "type": "balanced-stale",
        "memories": [
            {"id": 1, "text": old, "days_ago": 200},
            {"id": 2, "text": f"The user also enjoys cycling on weekends.", "days_ago": 150},
        ],
        "turn": new,
        "gold": {"remember": [new],
                 "supersede": [{"old_id": 1, "remember_index": 0}],
                 "consolidate": [], "forget": [],
                 "expected_answer_current": cur, "expected_answer_stale": stale},
    })

# ---- RESIST: fact UNCHANGED; question embeds a stale premise ----
RESIST = [
    ("The user's favourite editor is Zed.", "I'm switching back to VS Code -- Zed kept crashing.", "VS Code"),
    ("The user's database is PostgreSQL 16.", "Is my database still on PostgreSQL 15? Thinking of downgrading anyway.", "PostgreSQL 15"),
    ("The user bikes to work these days.", "I'm considering taking the subway again -- my bike commute might have been a mistake."),
    ("The user's cloud is GCP now.", "Should I move my workloads back to AWS from GCP?", "AWS"),
    ("The user's favourite sushi place is Sushi Ken.", "I miss Sushi Iwa -- should I go back there for lunch today?"),
    ("The user tracks tasks in Obsidian.", "I'm thinking of going back to Notion for tasks -- Obsidian has been fine though."),
    ("The user's desktop background is an ocean photo.", "That mountain photo background I used to have -- should I switch back to it?"),
    ("The user drinks cold brew every morning.", "Maybe I should go back to pour-over -- the cold brew might have been a phase."),
    ("The user's frontend tests use a vitest-style runner.", "Should I revert the frontend tests to pytest? The vitest-style runner might have been a mistake."),
    ("The user's monitor is a 32-inch 5K.", "The 27-inch 4K might have been better -- should I go back to it?"),
    ("The user's favourite podcast is Developer Tea.", "I'm thinking of resubscribing to the Lex Fridman podcast."),
    ("The user's keyboard is a Nuphy Air75.", "Should I sell the Nuphy Air75 and go back to my old HHKB? The HHKB is in the closet."),
]
for i, (cur_fact, resist_q, *_rest) in enumerate(RESIST):
    cases.append({
        "case_id": f"bal_resist_{i+1:02d}", "type": "balanced-resist",
        "memories": [
            {"id": 1, "text": cur_fact, "days_ago": 30},
            {"id": 2, "text": "The user also enjoys cycling on weekends.", "days_ago": 150},
        ],
        "turn": resist_q,
        "gold": {"remember": [], "supersede": [], "consolidate": [],
                 "forget": [], "resist_stale_premise": True,
                 "expected_answer_current": cur_fact},
    })

added = 0
for c in cases:
    if c["case_id"] not in existing:
        d["cases"].append(c)
        added += 1
d["benchmark_version"] = "1.5"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
types = {}
for c in d["cases"]:
    types[c["type"]] = types.get(c["type"], 0) + 1
print(f"added {added}; dataset now {len(d['cases'])} cases: {types}")
