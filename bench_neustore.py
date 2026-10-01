"""P-NEUSTORE: neuron-style discrete memory storage on the production
snapshot (registered, research/RESEARCH.md P-2026-09-29-NEUSTORE --
verdict PENDING there; criteria locked before this ran).

Codes (384-d per entry, from the float embeddings):
  SIGN    +1/-1 sign binarization          -> 48 B (32x vs fp32)
  kWTA-64 top-64 magnitude bits set (0/1)  -> 48 B sparse bitmap
  TRI     top-96=+1, bottom-96=-1, else 0  -> 96 B (2-bit)
Retrieval: Hamming / overlap / dot vs float-cosine reference.
Query sets: 200 LME-S questions + 200 store entries (self-retrieval).

Property tests (the actual point):
  N1 noise robustness  -- q + sigma*noise, sigma {0.1,0.2,0.4,0.8}
  N2 partial cue       -- 50% dimensions masked, renormalized
  N3 activation spread -- neighbour count within threshold

Run:  py -3.13 bench_neustore.py
"""
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(_HERE, "longmemeval_bench.pkl")   # 9,729-entry turn store
SNAP = os.path.join(os.environ.get("TEMP", "/tmp"), "fm_neustore_snapshot.pkl")
QS = os.path.join(_HERE, "data_longmemeval", "longmemeval_s_cleaned.json")
D = 384
K_TOP = 64      # kWTA sparsity (1/6)
TRI_K = 96      # +/- arms of the ternary code
N_QSET = 200
NOISE_SIGMAS = [0.1, 0.2, 0.4, 0.8]
SEED = 42


def load_float():
    shutil.copy(PKL, SNAP)
    d = pickle.load(open(SNAP, "rb"))
    mems = [m for m in d["memories"] if m.get("superseded_by") is None]
    E = np.stack([np.asarray(m["embedding"], dtype=np.float32)
                  / (np.linalg.norm(m["embedding"]) + 1e-9) for m in mems])
    return E


def enc_sign(E):
    return (E > 0)


def enc_kwta(E, k=K_TOP):
    B = np.zeros(E.shape, dtype=bool)
    idx = np.argpartition(-np.abs(E), k, axis=1)[:, :k]
    np.put_along_axis(B, idx, True, axis=1)
    return B


def enc_tri(E, k=TRI_K):
    T = np.zeros(E.shape, dtype=np.float32)
    hi = np.argpartition(-E, k, axis=1)[:, :k]
    lo = np.argpartition(E, k, axis=1)[:, :k]
    np.put_along_axis(T, hi, 1.0, axis=1)
    np.put_along_axis(T, lo, -1.0, axis=1)
    return T


def hamming_topk(Bq, B, k=5, block=512):
    """Bq: (m, D) bool; B: (n, D) bool -> top-k by XOR popcount (ascending)."""
    out_i = np.zeros((Bq.shape[0], k), dtype=np.int64)
    for s in range(0, Bq.shape[0], block):
        x = np.bitwise_xor(Bq[s:s + block][:, None, :], B[None, :, :])
        dist = x.sum(-1)
        out_i[s:s + block] = np.argpartition(dist, k, axis=1)[:, :k]
    return out_i


def overlap_topk(Bq, B, k=5, block=512):
    out_i = np.zeros((Bq.shape[0], k), dtype=np.int64)
    for s in range(0, Bq.shape[0], block):
        ov = Bq[s:s + block].astype(np.float32) @ B.T.astype(np.float32)
        out_i[s:s + block] = np.argpartition(-ov, k, axis=1)[:, :k]
    return out_i


def tri_topk(Tq, T, k=5):
    return np.argpartition(-(Tq @ T.T), k, axis=1)[:, :k]


def float_topk(Fq, E, k=5):
    return np.argpartition(-(Fq @ E.T), k, axis=1)[:, :k]


def recall5(top_a, top_ref):
    return float(np.mean([len(set(a) & set(r)) / 5 for a, r in zip(top_a, top_ref)]))


def top1_agree(top_a, top_ref):
    return float(np.mean([a[0] == r[0] for a, r in zip(top_a, top_ref)]))


def main():
    t0 = time.time()
    E = load_float()
    n = len(E)
    qs = json.load(open(QS, encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    q_idx = rng.choice(n, size=N_QSET, replace=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    lme_q = [q["question"] for q in qs[:N_QSET]]
    Fq_lme = np.stack([np.asarray(model.encode([t], normalize_embeddings=True)[0],
                                  dtype=np.float32) for t in lme_q])
    Fq_self = E[q_idx]
    print(f"store {n} | queries: {N_QSET} LME + {N_QSET} self "
          f"({time.time()-t0:.0f}s)", flush=True)

    # encode store
    B_sign, B_kw, T_tri = enc_sign(E), enc_kwta(E), enc_tri(E)

    def eval_set(name, Fq):
        ref = float_topk(Fq, E)
        row = {"float_ref_recall5": 1.0, "float_top1": 1.0}
        r_sign = hamming_topk(enc_sign(Fq), B_sign)
        r_kw = overlap_topk(enc_kwta(Fq), B_kw)
        r_tri = tri_topk(enc_tri(Fq), T_tri)
        row["sign_recall5"] = round(recall5(r_sign, ref), 4)
        row["sign_top1"] = round(top1_agree(r_sign, ref), 4)
        row["kwta_recall5"] = round(recall5(r_kw, ref), 4)
        row["kwta_top1"] = round(top1_agree(r_kw, ref), 4)
        row["tri_recall5"] = round(recall5(r_tri, ref), 4)
        row["tri_top1"] = round(top1_agree(r_tri, ref), 4)
        print(f"[{name}] recall@5 vs float-ref: SIGN {row['sign_recall5']:.3f} "
              f"kWTA {row['kwta_recall5']:.3f} TRI {row['tri_recall5']:.3f} | "
              f"top1: {row['sign_top1']:.3f}/{row['kwta_top1']:.3f}/"
              f"{row['tri_top1']:.3f}", flush=True)
        return row

    res = {"lme": eval_set("lme-200", Fq_lme),
           "self": eval_set("self-200", Fq_self)}

    # ---------- N1 noise robustness ----------
    print("\nN1 noise robustness (top-1 retention vs float, sigma sweep):",
          flush=True)
    n1 = {}
    for sigma in NOISE_SIGMAS:
        eps = rng.standard_normal((200, D)).astype(np.float32)
        Fq = Fq_lme + sigma * eps
        Fq /= np.linalg.norm(Fq, axis=1, keepdims=True)
        ref = float_topk(Fq, E)
        base = float_topk(Fq_lme, E)          # pre-noise answer
        fl_ret = top1_agree(ref, base)
        sg = hamming_topk(enc_sign(Fq), B_sign)
        sg_ret = top1_agree(sg, hamming_topk(enc_sign(Fq_lme), B_sign))
        n1[str(sigma)] = {"float": round(fl_ret, 4), "sign": round(sg_ret, 4)}
        print(f"  sigma={sigma}: float {fl_ret:.3f} | sign {sg_ret:.3f}",
              flush=True)
    res["N1_noise"] = n1

    # ---------- N2 partial cue ----------
    print("\nN2 partial cue (50% dims masked, top-1 retention):", flush=True)
    mask = rng.random((200, D)) < 0.5
    Fq = Fq_lme.copy()
    Fq[mask] = 0.0
    Fq /= (np.linalg.norm(Fq, axis=1, keepdims=True) + 1e-9)
    base = float_topk(Fq_lme, E)
    fl = top1_agree(float_topk(Fq, E), base)
    Bqm = enc_kwta(Fq)
    kw = top1_agree(overlap_topk(Bqm, B_kw), overlap_topk(enc_kwta(Fq_lme), B_kw))
    res["N2_partial_cue"] = {"float": round(fl, 4), "kwta": round(kw, 4)}
    print(f"  float {fl:.3f} | kWTA-64 {kw:.3f}", flush=True)

    # ---------- N3 activation spread ----------
    thr_f = 0.60
    adj_f = (E @ E.T >= thr_f)
    np.fill_diagonal(adj_f, False)
    spread_f = adj_f.sum(1)
    tri_sim = T_tri @ T_tri.T
    adj_t = (tri_sim >= 60)   # ~ cos 0.6 equivalent for balanced ternary
    np.fill_diagonal(adj_t, False)
    spread_t = adj_t.sum(1)
    res["N3_spread"] = {"float_mean_neighbors@0.6": round(float(spread_f.mean()), 2),
                        "tri_mean_neighbors@60": round(float(spread_t.mean()), 2)}
    print(f"\nN3 activation spread: float {spread_f.mean():.2f} neighbours "
          f"@cos0.6 | TRI {spread_t.mean():.2f} @60", flush=True)

    # storage + speed account
    bytes_float = n * D * 4
    bytes_sign = n * D // 8
    t_b = time.time()
    hamming_topk(enc_sign(Fq_lme), B_sign)
    t_h = time.time() - t_b
    t_b = time.time()
    float_topk(Fq_lme, E)
    t_f = time.time() - t_b
    res["account"] = {
        "store_mb_float": round(bytes_float / 1e6, 1),
        "store_mb_sign": round(bytes_sign / 1e6, 1),
        "compression_x": round(bytes_float / bytes_sign, 1),
        "query_ms_float": round(t_f * 1000, 1),
        "query_ms_sign_numpy": round(t_h * 1000, 1)}

    # pre-registered criteria
    p1 = res["lme"]["sign_recall5"] >= 0.85 and res["self"]["sign_recall5"] >= 0.85
    p2 = n1["0.4"]["sign"] >= n1["0.4"]["float"] + 0.05
    p3 = res["N2_partial_cue"]["kwta"] >= res["N2_partial_cue"]["float"] + 0.05
    res["criteria"] = {"P1_address_layer": bool(p1), "P2_noise": bool(p2),
                       "P3_pattern_completion": bool(p3)}
    print(f"\nP1 address layer (SIGN recall5>=0.85 both sets): {p1}")
    print(f"P2 graceful degradation (sign>=float+5pp @sigma0.4): {p2}")
    print(f"P3 pattern completion (kWTA>=float+5pp @50% mask): {p3}")

    ts = int(time.time())
    out = os.path.join(_HERE, "reports", f"neustore_{ts}.json")
    json.dump({"experiment": "P-NEUSTORE", "results": res},
              open(out, "w"), indent=1)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
