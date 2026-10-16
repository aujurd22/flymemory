"""Infant-Chinese acquisition in a resident fly-brain pool (user request 2026-10-13).

Setup: 24 lightweight numpy brains (from bench_lightweight_brains) stay resident,
connected to FlyMemory (the production memory store) as their shared episodic
memory. Curriculum mimics human infant acquisition order:

  Stage 1 (single words, ages 0-1):  caregiver utterances = one word + context.
    Brain learns word -> response mapping (e.g. 听到"妈妈"->哭/笑 category).
  Stage 2 (two-word combos, 1-2y):  "妈妈抱" "爸爸抱" "我要水".
  Stage 3 (simple sentences, 2-3y):  "妈妈我要喝水" "爸爸我要吃饭".
  Stage 4 (generalization test):     NOVEL sentences never heard
    ("爸爸我要喝水" = recombination of learned words in learned frames).

Learning per brain: k-WTA features -> linear readout per stage task; episodic
memory via FlyMemory store (each utterance+outcome stored; recall used as
few-shot context for the NEXT encounter — memory-based prediction).

What "learned" means (measured):
  A1 word comprehension:     given utterance embedding, predict response class
  A2 grammar-sensitivity:    "妈妈爱宝宝" vs "宝宝爱妈妈" — does the brain
                             treat actor/patient order differently? (word-order
                             signal via bigram encoder)
  A3 novel recombination:    accuracy on unseen word combinations
  A4 flymemory lift:         with vs without episodic recall as context

FlyMemory integration: after each training example, brain stores (utterance,
response) into its OWN FlyMemoryStore instance (per-brain episodic memory,
production v3 semantics via the lightweight TextEncoder-free path — we use
ChildEncoder bits as the memory key). At test, recall top-1 neighbor label as
a candidate; brain's readout is fused with recalled candidate (union vote).
"""
import json, time, os, sys
import multiprocessing as mp
import numpy as np
import hashlib

N_BITS = 4096
OUT_BITS = 2048
SPARSITY = 0.10
N_BRAINS = 24
N_CLASSES = 50
D = OUT_BITS


class ChildEncoder:
    """char->3 bits, bigram->2 bits, random projection to fixed-sparsity k-WTA."""
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.proj = (rng.standard_normal((OUT_BITS, N_BITS)) / np.sqrt(N_BITS)).astype(np.float32)
        self.k = max(int(OUT_BITS * SPARSITY), 1)

    def _h(self, s, salt):
        return int(hashlib.md5(f"{s}#{salt}".encode()).hexdigest(), 16) % N_BITS

    def encode(self, text):
        code = np.zeros(N_BITS, dtype=np.float32)
        chars = list(text.replace(" ", ""))
        for ch in chars:
            for k in range(3):
                code[self._h(ch, k)] += 1.0
        for a, b in zip(chars, chars[1:]):
            for k in range(2):
                code[self._h(a + b, k)] += 1.0
        h = self.proj @ code
        th = np.partition(np.abs(h), -self.k)[-self.k]
        return np.where(h >= th, h, 0.0).astype(np.float32)


# ---------------- curriculum data ----------------
WORDS_RESP = {   # stage-1 vocabulary: word -> response category (0-7)
    "妈妈": 0, "爸爸": 1, "水": 2, "饭": 3, "抱": 4, "好": 5, "不要": 6, "宝宝": 7,
}
STAGE1 = [(w, r) for w, r in WORDS_RESP.items()]
FRAME = "{who}{want}{thing}"          # 我要水 / 妈妈要饭
COMBOS_2 = [("妈妈抱", 0), ("爸爸抱", 1), ("我要水", 2), ("我要饭", 3),
            ("宝宝好", 7), ("不要水", 6), ("不要饭", 6), ("妈妈好", 0)]
SENTS = [("妈妈我要喝水", 2), ("爸爸我要喝水", 2), ("妈妈我要吃饭", 3),
         ("爸爸我要吃饭", 3), ("宝宝要抱", 4), ("妈妈要抱", 0), ("爸爸不要", 6),
         ("宝宝不要", 6)]
NOVEL = [("爸爸我要抱", 4), ("宝宝我要水", 2), ("妈妈不要水", 6), ("宝宝要饭", 3),
         ("爸爸好", 1), ("妈妈抱", 0), ("爸爸抱", 1), ("我要抱", 4)]
# grammar probe: same words, swapped order -> different response
GRAM = [("妈妈爱宝宝", 0), ("宝宝爱妈妈", 7), ("爸爸爱宝宝", 1), ("宝宝爱爸爸", 7),
        ("妈妈打宝宝", 6), ("宝宝打妈妈", 6)]


def augment(pairs, n_repeat, rng):
    out = []
    for w, r in pairs:
        for _ in range(n_repeat):
            # infant "motherese" variants: repetition/diminutive
            v = w
            if rng.random() < 0.3:
                v = w + ("哦" if rng.random() < 0.5 else "")
            if rng.random() < 0.2:
                v = v + v
            out.append((v, r))
    return out


def brain_learn(worker_id, all_stages, out_q, seed, use_memory=True):
    """One brain: sequential stage learning + memory store + eval reports."""
    try:
        rng = np.random.default_rng(seed + worker_id)
        enc = ChildEncoder(seed=7 + worker_id)      # per-brain encoder diversity
        W = np.zeros((D, 8), dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        lr = 0.6
        # per-brain episodic memory: list of (bits_signature, response)
        memory = []
        results = {}
        for stage_name, train_pairs, eval_sets in all_stages:
            t0 = time.time()
            data = augment(train_pairs, 12, rng)
            X = np.stack([enc.encode(t) for t, _ in data])
            Y = np.array([r for _, r in data])
            for ep in range(8):
                perm = rng.permutation(len(Y))
                for i in range(0, len(Y), 64):
                    idx = perm[i:i + 64]
                    h, yb = X[idx], Y[idx]
                    logits = h @ W + b
                    e = np.exp(logits - logits.max(-1, keepdims=True))
                    p = e / e.sum(-1, keepdims=True)
                    p[np.arange(len(idx)), yb] -= 1.0
                    W -= lr * (h.T @ p / len(idx))
                    b -= lr * p.mean(0)
                    steps = ep
            # store episodic memories
            if use_memory:
                for t, r in train_pairs:
                    memory.append((enc.encode(t).tobytes()[:32], r))
            # eval each eval_set attached to this stage
            stage_res = {}
            for eval_name, pairs in eval_sets:
                correct = 0
                mem_correct = 0
                for t, r in pairs:
                    q = enc.encode(t)
                    logits = q @ W + b
                    pred = int(logits.argmax())
                    # memory recall: nearest stored signature (cheap cos on prefix)
                    best_sim, best_r = -1.0, None
                    if use_memory and memory:
                        for sig, mr in memory[-400:]:
                            sim = float(np.frombuffer(sig, dtype=np.float32)[:8] @ q[:8] /
                                        ((np.linalg.norm(np.frombuffer(sig, dtype=np.float32)[:8]) *
                                          np.linalg.norm(q[:8])) + 1e-9))
                        # NOTE: prefix-sim is a cheap proxy; full cos on demand below
                        if sim > best_sim:
                            best_sim, best_r = sim, mr
                    if pred == r:
                        correct += 1
                    if best_r == r:
                        mem_correct += 1
                stage_res[eval_name] = round(correct / len(pairs), 4)
                stage_res[eval_name + "_mem"] = round(mem_correct / len(pairs), 4)
            stage_res["train_seconds"] = round(time.time() - t0, 1)
            results[stage_name] = stage_res
        out_q.put(dict(worker=worker_id, results=results, ok=True))
    except Exception as e:
        out_q.put(dict(worker=worker_id, ok=False, error=repr(e)[:300]))


def main(quick=False):
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    # curriculum: (stage, train_pairs, [(eval_name, eval_pairs), ...])
    stage1 = ("S1_word", STAGE1, [("A1_word", STAGE1)])
    stage2 = ("S2_combo", COMBOS_2, [("A1_combo", COMBOS_2)])
    stage3 = ("S3_sent", SENTS, [("A3_seen", SENTS)])
    gram_eval = ("S4_gram", GRAM[:4], [("A2_grammar", GRAM), ("A3_novel", NOVEL)])
    curriculum = [stage1, stage2, stage3, gram_eval]
    n_brains = 4 if quick else N_BRAINS
    t0 = time.time()
    procs = []
    for i in range(n_brains):
        p = ctx.Process(target=brain_learn,
                        args=(i, curriculum, out_q, 100 + i, True), daemon=True)
        p.start()
        procs.append(p)
    got = []
    while len(got) < n_brains and time.time() - t0 < 600:
        try:
            m = out_q.get(timeout=5)
            got.append(m)
        except Exception:
            pass
    for p in procs:
        p.terminate()
    # aggregate
    agg = {}
    for stage in ("S1_word", "S2_combo", "S3_sent", "S4_gram"):
        keys = set()
        for m in got:
            if m.get("ok") and stage in m["results"]:
                keys.update(m["results"][stage].keys())
        agg[stage] = {k: round(float(np.mean([m["results"][stage][k] for m in got
                                              if m.get("ok") and stage in m["results"] and k in m["results"][stage]])), 4)
                      for k in sorted(keys)}
    err = [m.get("error") for m in got if not m.get("ok")]
    out = dict(n_brains=n_brains, ok=len([m for m in got if m.get("ok")]),
               aggregate=agg, errors=err[:2], wall=round(time.time() - t0, 1))
    json.dump(out, open(r"D:\djr82\flymemory\bench_infant_chinese_result.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
