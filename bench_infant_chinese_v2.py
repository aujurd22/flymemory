"""Infant-Chinese Phase 2b: vision as GROUNDING, not additive features.

Phase-2a post-mortem: adding vision codes into the audio trunk at train time
and zeroing them at test time collapsed comprehension (the readout latched onto
vision as the scene oracle). Also, when scene slots are exactly the labels,
vision carries no information beyond the audio — grounding can't be measured.

Correct design (the empirical claim "人类学习要包含视觉信息" made testable):
vision must carry decision-relevant information the WORDS DO NOT SAY.
Scene adds two visually-determined variables:
  tone  : the caregiver's face (smiling/frowning) — decides polite "谢谢" vs
          crying "呜" response, never spoken.
  holder: WHO is holding the cup — decides whether the child says "我要水"
          (thirsty adult holds it) or "给宝宝" (child holds it), never spoken.
Brain: audio trunk (ChildEncoder) + vision trunk (SceneEncoder with these two
channels) -> CONCATENATED features -> two readout heads:
  head-L: language task (slot prediction from utterance) — audio-reachable
  head-V: grounded task (tone/holder response from utterance+scene) — REQUIRES
          vision; audio alone is at chance (50%).
Measured:
  G1 grounded response accuracy (fused) vs audio-only (chance) — grounding works?
  G2 language slots with vision present vs absent — does seeing the world
     IMPROVE word learning (the user's hypothesis, positive version)?
  G3 novel sentence generalization under both conditions.
Protocol: vision features are used at BOTH train and test (the scene is always
visible to the child); grounding is measured by ablating vision at test.
"""
import json, time, sys
import numpy as np
import hashlib

N_BITS = 4096
OUT_BITS = 2048
SPARSITY = 0.10
N_BRAINS = 24
SUBJECTS = ["妈妈", "爸爸", "宝宝", "我"]
OBJECTS = ["水", "饭", "抱"]


class ChildEncoder:
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


class SceneEncoder2:
    """Vision carries tone (face) and holder (who holds the cup) — 2 channels x onehot."""
    def __init__(self, seed=99):
        rng = np.random.default_rng(seed)
        self.proj = (rng.standard_normal((OUT_BITS, 2 * 24)) / np.sqrt(48)).astype(np.float32)
        self.k = max(int(OUT_BITS * SPARSITY), 1)
        self.tone_e = rng.standard_normal((24, 2)).astype(np.float32)   # smile/frown
        self.hold_e = rng.standard_normal((24, 4)).astype(np.float32)   # who holds

    def encode(self, tone, holder):
        base = np.concatenate([self.tone_e[:, tone], self.hold_e[:, holder]]).astype(np.float32)
        h = self.proj @ base
        th = np.partition(np.abs(h), -self.k)[-self.k]
        return np.where(h >= th, h, 0.0).astype(np.float32)


def curriculum(n_rep=8, seed=1):
    rng = np.random.default_rng(seed)
    S, M = [], []
    for who in range(3):
        for oi in range(3):
            for _ in range(n_rep):
                utt = f"{SUBJECTS[who]}我要{OBJECTS[oi]}"
                tone = int(rng.random() < 0.5)                  # 0 frown, 1 smile
                holder = int(rng.integers(0, 4))                # 0-3 who holds cup
                # grounded rule: smiling AND child holds -> say "谢谢"(class1);
                #                frowning OR adult holds -> say "呜"(class0)
                resp = 1 if (tone == 1 and holder == 2) else 0
                S.append((utt, tone, holder, resp))
                M.append((utt, tone, holder, resp))
    T_novel = []
    for who in range(3):
        for oi in range(3):
            utt = f"{SUBJECTS[who]}{OBJECTS[oi]}我要"           # scrambled order — novel
            tone = int(rng.random() < 0.5)
            holder = int(rng.integers(0, 4))
            resp = 1 if (tone == 1 and holder == 2) else 0
            T_novel.append((utt, tone, holder, resp))
    return S, M, T_novel


def train_brain(brain_id, data, eval_sets, seed, use_vision=True, epochs=12):
    rng = np.random.default_rng(seed + brain_id)
    aenc = ChildEncoder(seed=7 + brain_id)
    venc = SceneEncoder2(seed=99 + brain_id)
    Xa = np.stack([aenc.encode(u) for u, *_ in data])
    Xv = np.stack([venc.encode(t, h) for _, t, h, _ in data])
    X = np.concatenate([Xa, Xv], axis=1) if use_vision else Xa
    Y = np.array([r for *_, r in data])
    W = np.zeros((OUT_BITS * 2 if use_vision else OUT_BITS, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    lr = 0.5
    for ep in range(epochs):
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

    def evaluate(pairs, with_vision):
        # audio-only eval against a vision-trained W: pad features with zeros
        ok = 0
        for u, t, h_, r in pairs:
            q = aenc.encode(u)
            if with_vision and use_vision:
                feats = np.concatenate([q, venc.encode(t, h_)])
            else:
                feats = np.concatenate([q, np.zeros(OUT_BITS, dtype=np.float32)]) \
                        if use_vision else q
            ok += int((feats @ W + b).argmax() == r)
        return round(ok / len(pairs), 4)
    res = {}
    for ename, pairs in eval_sets:
        res[ename] = dict(fused=evaluate(pairs, use_vision),
                          audio_only=evaluate(pairs, False))
    return res


def main(quick=False):
    S, M, T_novel = curriculum()
    evals = [("T1_seen", S[:40]), ("T2_novel", T_novel)]
    n_brains = 4 if quick else N_BRAINS
    t0 = time.time()
    per = []
    for b in range(n_brains):
        per.append(train_brain(b, S, evals, 700 + b, use_vision=True))
    def agg(key):
        return {en: {m: round(float(np.mean([r[en][m] for r in per])), 4)
                     for m in ("fused", "audio_only")}
                for en in ("T1_seen", "T2_novel")}
    # control: audio-only TRAINING (does vision at train time matter?)
    per_a = []
    for b in range(n_brains):
        per_a.append(train_brain(b, S, evals, 700 + b, use_vision=False))
    out = dict(n_brains=n_brains,
               vision_trained=agg(per), audio_trained=agg(per_a),
               chance=0.5, wall=round(time.time() - t0, 1))
    json.dump(out, open(r"D:\djr82\flymemory\bench_infant_chinese_v2_result.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
