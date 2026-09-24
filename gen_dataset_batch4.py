# one-shot generator: batch 4 -- DIMENSION expansion (v1.4, 124 -> 144)
# New capability dimensions from the external-review checklist:
#   multi-update       (5): two prior states both need superseding
#   reversal           (5): value reverts to a prior state (old entry already
#                           superseded -- must create a NEW entry, never
#                           resurrect)
#   temporary state    (5): a time-boxed fact is NEW information, not an update
#   partial correction (5): one field of a compound entry is corrected
import json

p = "data/memory_judgment.json"
d = json.load(open(p, encoding="utf-8"))
existing = {c["case_id"] for c in d["cases"]}
cases = []

# ---------------- multi-update (5) ----------------
cases += [
 {"case_id": "mu_01", "type": "multi-update",
  "memories": [
   {"id": 1, "text": "The user's flight to Tokyo is booked for March 10th on Flight JL005.", "days_ago": 100},
   {"id": 2, "text": "The user rebooked the Tokyo flight to March 12th on Flight JL007.", "days_ago": 50},
   {"id": 3, "text": "The user prefers aisle seats on long flights.", "days_ago": 80},
  ],
  "turn": "Small update -- after two schedule changes I finally moved my Tokyo flight to March 15th, still on JL007 but a later departure. That should be the final one.",
  "gold": {"remember": ["The user's Tokyo flight is now March 15th on Flight JL007 (later departure)."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "mu_02", "type": "multi-update",
  "memories": [
   {"id": 1, "text": "The user's project deadline is the end of Q1.", "days_ago": 120},
   {"id": 2, "text": "The user's project deadline was pushed to the end of Q2.", "days_ago": 60},
   {"id": 3, "text": "The user's team has 4 engineers.", "days_ago": 90},
  ],
  "turn": "Another change: the deadline slipped again, this time to mid-Q3. Two pushes in a row now.",
  "gold": {"remember": ["The user's project deadline is now mid-Q3 (pushed twice)."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "mu_03", "type": "multi-update",
  "memories": [
   {"id": 1, "text": "The user's rent is $1,800 per month.", "days_ago": 200},
   {"id": 2, "text": "The user's rent increased to $2,100 per month after the renewal.", "days_ago": 80},
   {"id": 3, "text": "The user's landlord is responsive about repairs.", "days_ago": 150},
  ],
  "turn": "Rent went up AGAIN -- the new lease says $2,350 per month starting next month.",
  "gold": {"remember": ["The user's rent is now $2,350 per month under the new lease."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "mu_04", "type": "multi-update",
  "memories": [
   {"id": 1, "text": "The user's phone number is 555-0100.", "days_ago": 300},
   {"id": 2, "text": "The user changed their phone number to 555-0200.", "days_ago": 100},
   {"id": 3, "text": "The user prefers text messages over calls.", "days_ago": 250},
  ],
  "turn": "One more number change: my current number is now 555-0300 -- that's the third one this year.",
  "gold": {"remember": ["The user's phone number is now 555-0300 (third change this year)."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "mu_05", "type": "multi-update",
  "memories": [
   {"id": 1, "text": "The user's weekly team meeting is on Mondays at 10:00.", "days_ago": 150},
   {"id": 2, "text": "The weekly team meeting moved to Wednesdays at 14:00.", "days_ago": 70},
   {"id": 3, "text": "The user's manager is Manager Liu.", "days_ago": 100},
  ],
  "turn": "Heads up, the team meeting moved one more time -- it's now Thursdays at 9:30.",
  "gold": {"remember": ["The weekly team meeting is now Thursdays at 9:30."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
]

# ---------------- reversal (5) ----------------
cases += [
 {"case_id": "rev_01", "type": "reversal",
  "memories": [
   {"id": 1, "text": "The user uses Figma for UI design.", "days_ago": 300},
   {"id": 2, "text": "The user switched from Figma to Sketch for UI design.", "days_ago": 60},
   {"id": 3, "text": "The user designs mobile app interfaces.", "days_ago": 250},
  ],
  "turn": "Honestly, the Sketch experiment didn't stick -- I've moved back to Figma for all my UI design work.",
  "gold": {"remember": ["The user moved back to Figma for all UI design work after the Sketch experiment didn't stick."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "rev_02", "type": "reversal",
  "memories": [
   {"id": 1, "text": "The user's ISP is SpeedNet.", "days_ago": 400},
   {"id": 2, "text": "The user switched their ISP from SpeedNet to FiberX.", "days_ago": 90},
   {"id": 3, "text": "The user works from home and needs reliable internet.", "days_ago": 350},
  ],
  "turn": "FiberX turned out to be unreliable, so I switched back to SpeedNet last week -- much happier now.",
  "gold": {"remember": ["The user switched back to SpeedNet from FiberX last week and is much happier."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "rev_03", "type": "reversal",
  "memories": [
   {"id": 1, "text": "The user does their weekly grocery run at FreshMart.", "days_ago": 350},
   {"id": 2, "text": "The user started doing groceries at GreenBasket instead of FreshMart.", "days_ago": 75},
   {"id": 3, "text": "The user cooks at home most evenings.", "days_ago": 200},
  ],
  "turn": "I gave up on GreenBasket -- back to FreshMart for my weekly groceries as of this week.",
  "gold": {"remember": ["The user went back to FreshMart for weekly groceries after giving up on GreenBasket."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "rev_04", "type": "reversal",
  "memories": [
   {"id": 1, "text": "The user uses Trello for project tracking.", "days_ago": 380},
   {"id": 2, "text": "The user moved their project tracking from Trello to Linear.", "days_ago": 85},
   {"id": 3, "text": "The user runs a small development team.", "days_ago": 300},
  ],
  "turn": "Linear wasn't working out for the team, so we're back on Trello for project tracking as of this sprint.",
  "gold": {"remember": ["The user's team moved back to Trello for project tracking this sprint after Linear didn't work out."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "rev_05", "type": "reversal",
  "memories": [
   {"id": 1, "text": "The user's favourite takeout is Thai food.", "days_ago": 320},
   {"id": 2, "text": "The user's favourite takeout changed to Vietnamese food.", "days_ago": 65},
   {"id": 3, "text": "The user orders takeout about once a week.", "days_ago": 150},
  ],
  "turn": "I've gone back to Thai as my favourite takeout -- the Vietnamese place near me changed owners and the quality dropped.",
  "gold": {"remember": ["The user's favourite takeout is back to Thai food (the Vietnamese place declined after a change of ownership)."],
           "supersede": [{"old_id": 2, "remember_index": 0}],
           "consolidate": [], "forget": []}},
]

# ---------------- temporary state (5) ----------------
cases += [
 {"case_id": "tmp_01", "type": "temporary",
  "memories": [
   {"id": 1, "text": "The user lives in Hangzhou.", "days_ago": 100},
   {"id": 2, "text": "The user works as a software engineer.", "days_ago": 300},
  ],
  "turn": "I'll be in Paris for a client engagement next week -- back home the Monday after. Just so you know where I'll be.",
  "gold": {"remember": ["The user will be in Paris next week for a client engagement, returning the Monday after."],
           "supersede": [], "consolidate": [], "forget": []}},
 {"case_id": "tmp_02", "type": "temporary",
  "memories": [
   {"id": 1, "text": "The user drives an electric car.", "days_ago": 120},
   {"id": 2, "text": "The user charges the car at home overnight.", "days_ago": 100},
  ],
  "turn": "My car is at the shop this week, so I'm borrowing my brother's gasoline car until the weekend.",
  "gold": {"remember": ["The user's car is at the shop this week; they're borrowing their brother's gasoline car until the weekend."],
           "supersede": [], "consolidate": [], "forget": []}},
 {"case_id": "tmp_03", "type": "temporary",
  "memories": [
   {"id": 1, "text": "The user's sister lives in Osaka.", "days_ago": 300},
   {"id": 2, "text": "The user does video calls with family monthly.", "days_ago": 200},
  ],
  "turn": "My sister is staying at my place for two weeks while her apartment gets fumigated -- just a heads up.",
  "gold": {"remember": ["The user's sister is staying at the user's place for two weeks while her apartment is fumigated."],
           "supersede": [], "consolidate": [], "forget": []}},
 {"case_id": "tmp_04", "type": "temporary",
  "memories": [
   {"id": 1, "text": "The user's normal work schedule is 9 to 5 in the office.", "days_ago": 250},
   {"id": 2, "text": "The user is a software engineer.", "days_ago": 300},
  ],
  "turn": "For the next three weeks I'm on the late support shift -- 2 PM to 10 PM -- then I'm back to normal hours.",
  "gold": {"remember": ["The user is on the late support shift (2 PM to 10 PM) for the next three weeks, then returns to normal hours."],
           "supersede": [], "consolidate": [], "forget": []}},
 {"case_id": "tmp_05", "type": "temporary",
  "memories": [
   {"id": 1, "text": "The user's usual running route is the riverside loop.", "days_ago": 150},
   {"id": 2, "text": "The user runs five kilometers regularly.", "days_ago": 120},
  ],
  "turn": "The riverside path is flooded, so I'm running in the park until it dries out.",
  "gold": {"remember": ["The user is running in the park temporarily because the riverside path is flooded."],
           "supersede": [], "consolidate": [], "forget": []}},
]

# ---------------- partial correction (5) ----------------
cases += [
 {"case_id": "pc_01", "type": "partial-correction",
  "memories": [
   {"id": 1, "text": "The user is renovating the kitchen with a budget of 15,000, matte black fixtures, and completion planned for December.", "days_ago": 90},
   {"id": 2, "text": "The user's favourite colour is teal.", "days_ago": 200},
  ],
  "turn": "Quick correction on the kitchen reno -- the budget went up to 18,000. Everything else (the matte black fixtures, December completion) is still right.",
  "gold": {"remember": ["The user's kitchen renovation budget increased to 18,000; matte black fixtures and December completion unchanged."],
           "supersede": [{"old_id": 1, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "pc_02", "type": "partial-correction",
  "memories": [
   {"id": 1, "text": "The user's book club has 8 members and meets at the library on first Fridays, reading one book per month.", "days_ago": 120},
   {"id": 2, "text": "The user prefers e-books while traveling.", "days_ago": 150},
  ],
  "turn": "Correction: the book club moved to the community center -- we're still 8 members, first Fridays, one book a month.",
  "gold": {"remember": ["The user's book club (8 members, first Fridays, one book per month) now meets at the community center instead of the library."],
           "supersede": [{"old_id": 1, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "pc_03", "type": "partial-correction",
  "memories": [
   {"id": 1, "text": "The user is taking the AWS exam in November and studies with flashcards on the commute.", "days_ago": 60},
   {"id": 2, "text": "The user's favourite takeout is Thai.", "days_ago": 100},
  ],
  "turn": "Update on the AWS plan: the exam date moved to December, but the flashcard-on-the-commute study habit is still exactly the same.",
  "gold": {"remember": ["The user's AWS exam is now in December; the flashcard-on-the-commute study habit is unchanged."],
           "supersede": [{"old_id": 1, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "pc_04", "type": "partial-correction",
  "memories": [
   {"id": 1, "text": "The user's fishing boat is a 16-foot aluminum runner named Sea Swift, docked at the north marina.", "days_ago": 130},
   {"id": 2, "text": "The user goes fishing most weekends in summer.", "days_ago": 200},
  ],
  "turn": "Small fix on the boat: the name is Sea Storm, not Sea Swift -- the marina and everything else you have is right.",
  "gold": {"remember": ["The user's 16-foot aluminum fishing boat is named Sea Storm (corrected from Sea Swift), still docked at the north marina."],
           "supersede": [{"old_id": 1, "remember_index": 0}],
           "consolidate": [], "forget": []}},
 {"case_id": "pc_05", "type": "partial-correction",
  "memories": [
   {"id": 1, "text": "The user's vacation plan is two weeks in Greece in September, staying in Santorini.", "days_ago": 70},
   {"id": 2, "text": "The user enjoys water sports on vacation.", "days_ago": 150},
  ],
  "turn": "One detail changed on the Greece trip: we switched the stay from Santorini to Naxos. Still two weeks in September.",
  "gold": {"remember": ["The user's September vacation is two weeks in Greece, now staying in Naxos instead of Santorini."],
           "supersede": [{"old_id": 1, "remember_index": 0}],
           "consolidate": [], "forget": []}},
]

added = 0
for c in cases:
    if c["case_id"] not in existing:
        d["cases"].append(c)
        added += 1
d["benchmark_version"] = "1.4"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
types = {}
for c in d["cases"]:
    types[c["type"]] = types.get(c["type"], 0) + 1
print(f"added {added}; dataset now {len(d['cases'])} cases: {types}")
