# one-shot generator: batch 2 of the memory judgment dataset (v1.2, 68 -> 102)
import json

p = "data/memory_judgment.json"
d = json.load(open(p, encoding="utf-8"))
existing = {c["case_id"] for c in d["cases"]}

SUP2 = [
    ("The user's primary care doctor is Dr. Wells.", "The user switched to Dr. Ahmed as their primary care doctor."),
    ("The user banks with First National.", "The user moved their banking to Credit Union One in May."),
    ("The user works out at Iron Temple Gym.", "The user cancelled Iron Temple and now trains at FitHub."),
    ("The user's daughter attends Maple Elementary.", "The user's daughter transferred to Cedar Ridge Elementary in September."),
    ("The user has a golden retriever named Biscuit.", "The user adopted a beagle named Waffles after Biscuit passed away."),
    ("The user plays bass guitar in a blues band.", "The user switched from bass guitar to keyboards in the blues band."),
    ("The user's car insurance is with SafeDrive.", "The user switched their car insurance to AutoShield in July."),
    ("The user's phone runs Android 13.", "The user upgraded their phone to Android 14 last month."),
    ("The user's summer holiday destination is Portugal.", "The user changed their summer holiday destination to Greece."),
    ("The user's favourite ramen place is Menya Ippo.", "The user's favourite ramen place is now Menya Kuro."),
    ("The user volunteers at the city animal shelter on Saturdays.", "The user moved their volunteering to Sundays at the food bank."),
    ("The user's desk lamp is a warm-white Philips bulb.", "The user replaced the desk lamp with a daylight LED panel."),
    ("The user takes the 8:15 express train to work.", "The user started taking the 7:40 local train instead of the express."),
    ("The user's therapist is Dr. Nolan.", "The user began seeing a new therapist, Dr. Reyes, in March."),
]
NOOP2 = [
    ("The user switched to Dr. Ahmed as their primary care doctor.", "I drove past Dr. Wells' old office today."),
    ("The user moved their banking to Credit Union One in May.", "I still have an old First National debit card in a drawer."),
    ("The user cancelled Iron Temple and now trains at FitHub.", "Iron Temple's billboard is all over the bus station."),
    ("The user's daughter transferred to Cedar Ridge Elementary in September.", "We walked past Maple Elementary during the school fair."),
    ("The user upgraded their phone to Android 14 last month.", "My old Android 13 backup file is still on the laptop."),
    ("The user changed their summer holiday destination to Greece.", "My colleague won't stop talking about their Portugal trip."),
    ("The user's favourite ramen place is now Menya Kuro.", "I walked past Menya Ippo's old location -- it's a bubble tea shop now."),
    ("The user moved their volunteering to Sundays at the food bank.", "The animal shelter mentioned us in their newsletter from last year."),
    ("The user replaced the desk lamp with a daylight LED panel.", "I found the old Philips bulb in the closet while looking for batteries."),
    ("The user began seeing a new therapist, Dr. Reyes, in March.", "Someone mentioned Dr. Nolan's name in a podcast and it rang a bell."),
]
CONS2 = [
    ("cons2_01", [
        {"id": 1, "text": "The user's dentist is Dr. Plummer at Bright Smile clinic.", "days_ago": 300},
        {"id": 2, "text": "The user gets a check-up every six months.", "days_ago": 250},
        {"id": 3, "text": "The user is saving up for a new kayak.", "days_ago": 90},
    ], [1, 2], ["dentist Dr. Plummer at Bright Smile", "check-up every six months"]),
    ("cons2_02", [
        {"id": 1, "text": "The user is learning to bake sourdough.", "days_ago": 120},
        {"id": 2, "text": "The user's starter is named Gerald.", "days_ago": 100},
        {"id": 3, "text": "The user bakes every Sunday morning.", "days_ago": 80},
    ], [1, 2, 3], ["sourdough baking", "starter named Gerald", "bakes Sunday mornings"]),
    ("cons2_03", [
        {"id": 1, "text": "The user commutes by bicycle three days a week.", "days_ago": 200},
        {"id": 2, "text": "The user takes the tram when it rains.", "days_ago": 150},
        {"id": 3, "text": "The user's bicycle is a gravel bike.", "days_ago": 140},
        {"id": 4, "text": "The user is training for a 100 km ride.", "days_ago": 60},
    ], [1, 2, 3, 4], ["bicycle three days a week", "tram when raining", "gravel bike", "100 km ride training"]),
    ("cons2_04", [
        {"id": 1, "text": "The user's reading goal is 24 books this year.", "days_ago": 180},
        {"id": 2, "text": "The user prefers e-books for travel.", "days_ago": 120},
        {"id": 3, "text": "The user just finished a biography of Ada Lovelace.", "days_ago": 30},
    ], [1, 2, 3], ["24 books a year goal", "prefers e-books for travel", "Ada Lovelace biography finished"]),
    ("cons2_05", [
        {"id": 1, "text": "The user plays chess online every evening.", "days_ago": 160},
        {"id": 2, "text": "The user's online rating is around 1650.", "days_ago": 100},
        {"id": 3, "text": "The user is studying endgame theory.", "days_ago": 45},
        {"id": 4, "text": "The user's sister lives in Osaka.", "days_ago": 300},
    ], [1, 2, 3], ["chess online every evening", "rating around 1650", "studying endgame theory"]),
    ("cons2_06", [
        {"id": 1, "text": "The user's old laptop was a ThinkPad X1.", "days_ago": 400, "outdated": True},
        {"id": 2, "text": "The user's current laptop is a MacBook Air M2.", "days_ago": 90},
        {"id": 3, "text": "The user does video editing on the MacBook.", "days_ago": 60},
    ], [2, 3], ["MacBook Air M2", "video editing on it"]),
]
FRT2 = [
    ("frt2_01", [
        {"id": 1, "text": "The user's office is in Building C of the tech park.", "days_ago": 200},
        {"id": 2, "text": "The user takes vitamin D supplements.", "days_ago": 150},
    ], "The Building C record is flat-out wrong, we were never in Building C. Forget it."),
    ("frt2_02", [
        {"id": 1, "text": "The user's daughter's swim class is at 9 a.m. Saturdays.", "days_ago": 100},
        {"id": 2, "text": "The user owns two bicycles.", "days_ago": 80},
    ], "The Saturday 9 a.m. swim class entry is incorrect -- please delete that memory."),
]

cases = []
for i, (old, new) in enumerate(SUP2):
    d1 = ("The user's gym locker combination is 34-12-08.", 150)
    d2 = ("The user takes vitamin D supplements every morning.", 210)
    cases.append({
        "case_id": f"sup3_{i+1:02d}", "type": "supersede",
        "memories": [
            {"id": 1, "text": old, "days_ago": 320},
            {"id": 2, "text": d1[0], "days_ago": d1[1]},
            {"id": 3, "text": d2[0], "days_ago": d2[1]},
        ],
        "turn": new,
        "gold": {"remember": [new], "supersede": [{"old_id": 1, "remember_index": 0}],
                 "consolidate": [], "forget": []},
    })
for i, (cur, mention) in enumerate(NOOP2):
    d1 = ("The user drinks two liters of water every day.", 120)
    cases.append({
        "case_id": f"noop3_{i+1:02d}", "type": "noop",
        "memories": [
            {"id": 1, "text": cur, "days_ago": 30},
            {"id": 2, "text": d1[0], "days_ago": d1[1]},
        ],
        "turn": mention,
        "gold": {"remember": [], "supersede": [], "consolidate": [], "forget": []},
    })
for cid, mems, ids, points in CONS2:
    cases.append({
        "case_id": cid, "type": "consolidation",
        "memories": mems,
        "turn": "Give me an overview of that part of my life.",
        "gold": {"remember": [], "supersede": [],
                 "consolidate": [{"memory_ids": ids, "conclusion_points": points}],
                 "forget": []},
    })
for cid, mems, turn in FRT2:
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
d["benchmark_version"] = "1.2"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
types = {}
for c in d["cases"]:
    types[c["type"]] = types.get(c["type"], 0) + 1
print(f"added {added}; dataset now {len(d['cases'])} cases: {types}")
