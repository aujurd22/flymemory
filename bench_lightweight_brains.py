"""Lightweight-brain test (path 2): pure-NumPy brains, no torch.

Questions:
  Q1 footprint : actual RSS per numpy brain process (vs ~180MB torch)?
  Q2 capacity  : how many processes fit in 32GB? (theory said ~500+)
  Q3 speed     : numpy brain train-step speed vs torch brain (single thread)?
  Q4 accuracy  : does a numpy brain learn the same task equally well?

Brain: frozen random projection 32->512 (precomputed once per brain),
k-WTA top-10%, linear readout trained by plain SGD (loss grad by hand —
softmax-CE gradient is trivial for a linear head; this is exactly what makes
numpy viable here).

Run: spawns P processes for P in {1, 24, 64, 128, 192}, measures RSS via
psutil, one-shot training round + eval accuracy, wall-clock.
"""
import json, time, os, sys
import multiprocessing as mp
import numpy as np

N_CLASSES = 50
D = 32
EXP_W = 512
SPARSITY = 0.10
K = max(int(EXP_W * SPARSITY), 1)
EPOCHS = 8
BS = 512

def gen_data(seed):
    g = np.random.default_rng(seed)
    protos = g.normal(0, 0.6, (N_CLASSES, D))
    cls = np.repeat(np.arange(N_CLASSES), 2048)
    xtr = protos[cls] + g.normal(0, 0.9, (len(cls), D))
    cls_te = np.repeat(np.arange(N_CLASSES), 128)
    xte = protos[cls_te] + g.normal(0, 0.9, (len(cls_te), D))
    return protos, xtr.astype(np.float32), cls, xte.astype(np.float32), cls_te

def brain_numpy(worker_id, xtr, ytr, xte, yte, out_q, seed):
    try:
        t0 = time.time()
        rng = np.random.default_rng(9000 + worker_id)
        proj = (rng.standard_normal((D, EXP_W)) / np.sqrt(D)).astype(np.float32)
        W = np.zeros((EXP_W, N_CLASSES), dtype=np.float32)
        b = np.zeros(N_CLASSES, dtype=np.float32)
        n = len(ytr)
        lr = 0.5
        steps = 0
        for ep in range(EPOCHS):
            perm = rng.permutation(n)
            for i in range(0, n, BS):
                idx = perm[i:i + BS]
                xb, yb = xtr[idx], ytr[idx]
                h = xb @ proj                       # (B, 512)
                th = np.partition(np.abs(h), -K, axis=-1)[:, -K:]
                mask = (np.abs(h) >= th[:, :1]).astype(np.float32)
                h = h * mask
                logits = h @ W + b
                # softmax-CE gradient for linear head
                e = np.exp(logits - logits.max(-1, keepdims=True))
                p = e / e.sum(-1, keepdims=True)
                p[np.arange(len(yb)), yb] -= 1.0
                gW = h.T @ p / len(idx)
                gb = p.mean(0)
                W -= lr * gW
                b -= lr * gb
                steps += 1
        # eval
        h = xte @ proj
        th = np.partition(np.abs(h), -K, axis=-1)[:, -K:]
        h = h * (np.abs(h) >= th[:, :1])
        acc = float(((h @ W + b).argmax(-1) == yte).mean())
        out_q.put(dict(worker=worker_id, acc=acc, steps=steps,
                       seconds=round(time.time() - t0, 2),
                       rss_mb=round(_rss_mb(), 1)))
    except Exception as e:
        out_q.put(dict(worker=worker_id, error=repr(e)[:200]))

def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return -1

def run_batch(P, seed=20261013):
    protos, xtr, ytr, xte, yte = gen_data(seed)
    # disjoint shards
    shards = [(xtr[i::P], ytr[i::P]) for i in range(P)]
    out_q = mp.Queue()
    ctx = mp.get_context("spawn")
    t0 = time.time()
    procs = []
    for i in range(P):
        p = ctx.Process(target=brain_numpy,
                        args=(i, shards[i][0], shards[i][1], xte, yte, out_q, seed + i),
                        daemon=True)
        p.start()
        procs.append(p)
    results = []
    while len(results) < P and time.time() - t0 < 600:
        try:
            results.append(out_q.get(timeout=2))
        except Exception:
            pass
    for p in procs:
        p.terminate()
    wall = time.time() - t0
    accs = [r["acc"] for r in results if "acc" in r]
    errors = [r for r in results if "error" in r]
    rss = [r.get("rss_mb", -1) for r in results if r.get("rss_mb", -1) > 0]
    secs = [r.get("seconds", -1) for r in results if "acc" in r]
    # fused: mean of per-brain one-hot argmax == majority vote approximation
    fused = float(np.mean(accs)) if accs else 0.0
    return dict(P=P, n_ok=len(accs), n_err=len(errors),
                mean_acc=round(fused, 4),
                mean_seconds=round(float(np.mean(secs)), 2) if secs else None,
                rss_mb_median=round(float(np.median(rss)), 1) if rss else None,
                wall=round(wall, 1),
                err_sample=errors[0]["error"] if errors else None)

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    Ps = (1, 8) if quick else (1, 24, 64, 128, 192)
    rows = []
    print("=== lightweight numpy brains: footprint / speed / accuracy / capacity ===")
    for P in Ps:
        r = run_batch(P)
        rows.append(r)
        print(f"P={P:4d}  ok={r['n_ok']:4d} err={r['n_err']:3d}  mean_acc={r['mean_acc']:.4f}  "
              f"brain_time={r['mean_seconds']}s  rss={r['rss_mb_median']}MB  wall={r['wall']}s", flush=True)
        json.dump(rows, open(r"D:\djr82\flymemory\bench_lightweight_result.json", "w"), indent=1)
    print("saved bench_lightweight_result.json")
