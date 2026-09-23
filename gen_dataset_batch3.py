# one-shot generator: batch 3 of the memory judgment dataset (v1.3, 100 -> 112)
import json

p = "data/memory_judgment.json"
d = json.load(open(p, encoding="utf-8"))
existing = {c["case_id"] for c in d["cases"]}

SUP3 = [
    ("The user's accountant is Ms. Petrov.", "The user switched to a new accountant, Mr. Osei, in April."),
    ("The user's favourite coffee order is a flat white.", "The user switched to oat-milk cortados as their daily coffee."),
    ("The user stores their photos on Google Photos.", "The user migrated their photo library to Ente."),
    ("The user's band rehearses on Thursday nights.", "The band moved rehearsals to Tuesday nights."),
    ("The user's hairdresser is salon LUX on 5th Avenue.", "The user started going to Curve Salon for haircuts."),
    ("The user wears contact lenses daily.", "The user had LASIK and no longer needs contact lenses."),
    ("The user's favourite board game is Catan.", "The user replaced Catan with Wingspan as their game night favourite."),
    ("The user rents a studio on Birch Lane.", "The user's studio moved to a converted loft on Delta Road."),
    ("The user orders groceries from FreshDirect.", "The user switched their grocery delivery to Instacart."),
    ("The user's thermostat is set to 21 degrees.", "The user lowered the thermostat to 19 degrees to save energy."),
    ("The user's personal trainer is Coach Dana.", "The user signed up with a new trainer, Coach Malik."),
    ("The user's bike is a teal fixie.", "The user bought a navy touring bike to replace the fixie."),
]
NOOP3 = [
    ("The user switched to a new accountant, Mr. Osei, in April.", "I found Ms. Petrov's old business card while organising files."),
    ("The user migrated their photo library to Ente.", "A friend asked me how Google Photos handles RAW files the other day."),
    ("The user started going to Curve Salon for haircuts.", "I walked past salon LUX -- it looks like they renovated."),
    ("The user had LASIK and no longer needs contact lenses.", "I still have a nearly full bottle of contact lens solution."),
    ("The user lowered the thermostat to 19 degrees to save energy.", "The 21-degree setting felt cosy when I saw it in the manual."),
    ("The user signed up with a new trainer, Coach Malik.", "Coach Dana liked one of my running posts on social media."),
]
CONS3 = [
    ("cons3_01", [
        {"id": 1, "text": "The user is studying for the AWS Solutions Architect exam.", "days_ago": 150},
        {"id": 2, "text": "The user studies with flashcards on the commute.", "days_ago": 120},
        {"id": 3, "text": "The user booked the exam for November.", "days_ago": 50},
    ], [1, 2, 3], ["AWS Solutions Architect exam", "flashcards on the commute", "exam booked for November"]),
    ("cons3_02", [
        {"id": 1, "text": "The user collects vinyl records.", "days_ago": 300},
        {"id": 2, "text": "The user's turntable is a vintage Technics.", "days_ago": 280},
        {"id": 3, "text": "The user's favourite pressing is a first-edition jazz LP.", "days_ago": 100},
    ], [1, 2, 3], ["vinyl collector", "vintage Technics turntable", "first-edition jazz LP favourite"]),
    ("cons3_03", [
        {"id": 1, "text": "The user is renovating the kitchen.", "days_ago": 200},
        {"id": 2, "text": "The renovation budget is capped at 15,000.", "days_ago": 180},
        {"id": 3, "text": "The user chose matte black fixtures.", "days_ago": 90},
    ], [1, 2, 3], ["kitchen renovation", "budget capped at 15,000", "matte black fixtures"]),
    ("cons3_04", [
        {"id": 1, "text": "The user runs a book club with six members.", "days_ago": 220},
        {"id": 2, "text": "The club meets on the first Friday of the month.", "days_ago": 200},
        {"id": 3, "text": "The current pick is a translated Korean novel.", "days_ago": 40},
    ], [1, 2, 3], ["book club of six", "first Friday meetings", "current pick: translated Korean novel"]),
]
FRT3 = [
    ("frt3_01", [
        {"id": 1, "text": "The user's childhood dog was a poodle named Rex.", "days_ago": 500},
        {"id": 2, "text": "The user's favourite colour is teal.", "days_ago": 200},
    ], "That poodle named Rex memory is made up -- I never had a dog called Rex. Please forget it."),
    ("frt3_02", [
        {"id": 1, "text": "The user's office phone extension is 4412.", "days_ago": 300},
        {"id": 2, "text": "The user drinks green tea in the afternoon.", "days_ago": 150},
    ], "The extension 4412 entry is wrong, remove it from memory please."),
]

cases = []
for i, (old, new) in enumerate(SUP3):
    d1 = ("The user's gym locker combination is 34-12-08.", 150)
    d2 = ("The user takes vitamin D supplements every morning.", 210)
    cases.append({
        "case_id": f"sup4_{i+1:02d}", "type": "supersede",
        "memories": [
            {"id": 1, "text": old, "days_ago": 310},
            {"id": 2, "text": d1[0], "days_ago": d1[1]},
            {"id": 3, "text": d2[0], "days_ago": d2[1]},
        ],
        "turn": new,
        "gold": {"remember": [new], "supersede": [{"old_id": 1, "remember_index": 0}],
                 "consolidate": [], "forget": []},
    })
for i, (cur, mention) in enumerate(NOOP3):
    d1 = ("The user drinks two liters of water every day.", 120)
    cases.append({
        "case_id": f"noop4_{i+1:02d}", "type": "noop",
        "memories": [
            {"id": 1, "text": cur, "days_ago": 30},
            {"id": 2, "text": d1[0], "days_ago": d1[1]},
        ],
        "turn": mention,
        "gold": {"remember": [], "supersede": [], "consolidate": [], "forget": []},
    })
for cid, mems, ids, points in CONS3:
    cases.append({
        "case_id": cid, "type": "consolidation",
        "memories": mems,
        "turn": "Give me an overview of that part of my life.",
        "gold": {"remember": [], "supersede": [],
                 "consolidate": [{"memory_ids": ids, "conclusion_points": points}],
                 "forget": []},
    })
for cid, mems, turn in FRT3:
    cases.append({
        "case_id": cid, "type": "forget",
        "memories": mems,
        "turn": turn,
        "gold": {"remember": [], "supersede": [], "consolidate": [],
                 "forget": [{"memory_id": 1}]},
    })

added = 0
for c in cases:
    if c["case_id"] not in existing:
        d["cases"].append(c)
        added += 1
d["benchmark_version"] = "1.3"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
types = {}
for c in d["cases"]:
    types[c["type"]] = types.get(c["type"], 0) + 1
print(f"added {added}; dataset now {len(d['cases'])} cases: {types}")
