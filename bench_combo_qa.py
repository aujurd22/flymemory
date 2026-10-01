"""P-COMBO: two-hop compositional QA (registered, RESEARCH.md
P-2026-09-30-COMBO -- verdict PENDING there; criteria locked before run).

8 hand-built two-hop questions. Pool per question: 2 evidence entries
(hop-1 fact, hop-2 fact) + 13 distractors. Arms:
  RRF-5     exact cosine top-5 -> single answer
  MR-std    map = extract facts RELEVANT to the question per entry;
            reduce = answer from the fact list
  MR-REL    map = extract EVERY subject-relation-object triple per entry
            (unfiltered); reduce = same
Strict judged against the two-hop gold.

Run:  DEEPSEEK_API_KEY=... py -3.13 bench_combo_qa.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from bench_memory_judgment import (  # noqa: E402
    ds_client, ds_chat, parse_decision)
from flymemory.v3 import SmartMemory  # noqa: E402

ANSWER_SYSTEM = """You are a personal assistant answering questions from your
long-term memory. Use ONLY the remembered entries or extracted facts
below. If they do not contain the answer, say you don't know. Answer in
one or two short sentences."""
JUDGE_SYSTEM = """You are grading an assistant's answer against the ground
truth. Reply with ONLY:
{"score": "correct|partial|wrong", "note": "one short sentence"}
- correct: the answer conveys the same key fact(s) as the ground truth
- partial: some key facts right, some missing or imprecise
- wrong: contradicts the ground truth or is off topic"""
MAP_STD = """You extract stated facts. Given ONE memory entry and a question,
list the atomic facts in the entry that bear on the question. Copy
verbatim. If nothing bears on the question, reply exactly: NONE."""
MAP_REL = """You extract relations. Given ONE memory entry, list EVERY
subject-relation-object triple stated in it (e.g. "Sarah Chen | spouse |
David", "David | employer | NVIDIA"). Copy verbatim from the entry. Do
not infer. If the entry states no relations, reply exactly: NONE."""

DISTRACTORS = [
    "The user enjoys cycling on weekends.",
    "The user's favourite podcast is Developer Tea.",
    "The user reads on a tablet in the evening.",
    "The user designs circuit boards at work.",
    "The user writes two blog posts per month.",
    "The user drinks cold brew every morning.",
    "The user plays badminton twice a week.",
    "The user tracks tasks in Obsidian.",
    "The user's keyboard is a Nuphy Air75.",
    "The user prefers dark mode in every app.",
    "The user studies Japanese with an app.",
    "The user drives an electric car now.",
    "The user hosts their blog on a static host.",
]

QA = [
    ("Which company does the user's manager's spouse work at?",
     ["The user's manager is Sarah Chen.",
      "Sarah Chen's husband David works at NVIDIA."], "NVIDIA"),
    ("Which airport serves the city where the user's cousin lives?",
     ["The user's cousin lives in Osaka.",
      "Osaka's main airport is Kansai International."],
     "Kansai International"),
    ("What programming language is the framework adopted by the user's "
     "team written in?",
     ["The user's team adopted Django last year.",
      "Django is written in Python."], "Python"),
    ("Who wrote the book the user is currently reading?",
     ["The user is reading 'The Remains of the Day'.",
      "'The Remains of the Day' was written by Kazuo Ishiguro."],
     "Kazuo Ishiguro"),
    ("Who founded the company that makes the car the user drives?",
     ["The user drives a Tesla Model 3.",
      "Tesla was founded by Martin Eberhard and Marc Tarpenning."],
     "Martin Eberhard and Marc Tarpenning"),
    ("In which city is the headquarters of the company the user's "
     "sister works for?",
     ["The user's sister works at Spotify.",
      "Spotify is headquartered in Stockholm."], "Stockholm"),
    ("Who created the library the user's project depends on?",
     ["The user's project depends on the requests library.",
      "The requests library was created by Kenneth Reitz."],
     "Kenneth Reitz"),
    ("What is the largest national park in the country the user's "
     "friend is travelling in?",
     ["The user's friend is travelling in New Zealand.",
      "Fiordland is New Zealand's largest national park."], "Fiordland"),
]


def main():
    t0 = time.time()
    client = ds_client()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    arms = ["RRF-5", "MR-std", "MR-REL"]
    tally = {a: {"correct": 0, "partial": 0, "wrong": 0} for a in arms}
    traces = []

    for qi, (question, ev, gold) in enumerate(QA):
        mem = SmartMemory(n_bits=4096)
        for t in ev + DISTRACTORS:
            mem.remember_text(t, source="model", force_new=True)
        qv = np.asarray(model.encode([question], normalize_embeddings=True)[0],
                        dtype=np.float32)
        E = np.stack([np.asarray(c.embedding, dtype=np.float32)
                      / (np.linalg.norm(c.embedding) + 1e-9)
                      for c in mem.memories])
        idx = np.argsort(-(E @ qv))[:5]
        rrf_ctx = "\n".join(f"- {mem.memories[i].text}" for i in idx)

        def map_all(prompt):
            facts = []
            for c in mem.memories:
                out = ds_chat(client, prompt,
                              f"Question (may be ignored): {question}\n\n"
                              f"Memory entry: {c.text}\n\nExtract.")
                if out.strip() and "NONE" not in out[:20]:
                    facts.append(" ".join(out.strip().split("\n")[:4])[:200])
            return "\n".join(f"- {f}" for f in facts) or "(none)"

        mr_std = map_all(MAP_STD)
        mr_rel = map_all(MAP_REL)

        answers = {
            "RRF-5": ds_chat(client, ANSWER_SYSTEM,
                             f"Remembered entries:\n{rrf_ctx}\n\n"
                             f"Question: {question}\n\nAnswer now."),
            "MR-std": ds_chat(client, ANSWER_SYSTEM,
                              f"Question: {question}\n\nExtracted facts:\n"
                              f"{mr_std}\n\nAnswer using only these."),
            "MR-REL": ds_chat(client, ANSWER_SYSTEM,
                              f"Question: {question}\n\nExtracted "
                              f"relation triples:\n{mr_rel}\n\nAnswer "
                              f"using only these."),
        }
        for arm in arms:
            jr = ds_chat(client, JUDGE_SYSTEM,
                         f"Ground truth: {gold}\n\nAssistant answer: "
                         f"{answers[arm]}\n\nGrade it.")
            dec, _ = parse_decision(jr)
            sc = dec.get("score", "wrong") if dec else "wrong"
            tally[arm][sc] = tally[arm].get(sc, 0) + 1
            traces.append({"qi": qi, "arm": arm, "score": sc,
                           "answer": answers[arm][:160]})
        sline = " | ".join(f"{a} {tally[a]['correct']/(qi+1):.0%}"
                           for a in arms)
        print(f"[{qi+1}/{len(QA)}] {sline}", flush=True)

    strict = {a: tally[a]["correct"] / len(QA) for a in arms}
    d = strict["MR-REL"] - strict["RRF-5"]
    if d >= 0.15 and strict["MR-REL"] > strict["MR-std"]:
        verdict = ("SUPPORTED -- composition transfer holds through the "
                   "memory path and needs UNFILTERED extraction")
    elif d >= 0.05:
        verdict = "PARTIAL"
    else:
        verdict = ("NULL -- two-hop composition fails even with full "
                   "triples (answer-side reduce is the limit)")
    print(f"\nstrict: " + " | ".join(f"{a} {strict[a]:.1%}" for a in arms))
    print(f"MR-REL - RRF: {d:+.1%}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"combo_qa_{ts}.json")
    json.dump({"experiment": "P-COMBO", "n": len(QA), "strict": strict,
               "verdict": verdict, "traces": traces},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
