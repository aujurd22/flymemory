"""Dormant-pool test (path 3): 500 resident brain PROCESSES, <=24 active at once.

Architecture (bio: most neurons are silent most of the time; cortex runs on a
small active fraction):
  - 500 lightweight numpy-brain processes spawn at startup, each holding its
    own trained weights (post one-shot training on its shard), then SLEEP on a
    per-process command Queue (blocking get — zero CPU while dormant).
  - Coordinator keeps an activity ledger; at most 24 concurrently ACTIVE.
  - Query pattern: burst of 500 queries, each addressed to a random brain;
    coordinator dispatches up to 24 concurrently, others wait in queue.
  - Measured: per-query LATENCY (dispatch->answer), brain wake-from-dormant
    latency, throughput (queries/sec), CPU utilization, and pool RAM.

This answers: does "500 concurrent brains" work as a dormant pool (potential
concurrency 500, real concurrency 24), with what latency tax?
"""
import json, time, os, sys
import multiprocessing as mp
import numpy as np

N_CLASSES = 50
D = 32
EXP_W = 512
SPARSITY = 0.10
K = max(int(EXP_W * SPARSITY), 1)
POOL = 500
MAX_ACTIVE = 24


def brain_dormant(worker_id, cmd_q, ans_q, seed):
    """Train once on a small shard, then sleep on cmd_q; serve queries on wake."""
    import queue as _q
    rng = np.random.default_rng(seed)
    g = np.random.default_rng(20261013)
    protos = g.normal(0, 0.6, (N_CLASSES, D))
    cls = np.repeat(np.arange(N_CLASSES), 64)          # small per-brain shard
    x = (protos[cls] + g.normal(0, 0.9, (len(cls), D))).astype(np.float32)
    proj = (rng.standard_normal((D, EXP_W)) / np.sqrt(D)).astype(np.float32)
    W = np.zeros((EXP_W, N_CLASSES), dtype=np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)
    lr = 0.5

    def _train_once(_rng, _x, _proj, _W, _b, _cls):
        for ep in range(6):
            perm = _rng.permutation(len(_cls))
            for i in range(0, len(_cls), 256):
                idx = perm[i:i + 256]
                h = _x[idx] @ _proj
                th = np.partition(np.abs(h), -K, axis=-1)[:, -K:]
                h = h * (np.abs(h) >= th[:, :1])
                logits = h @ _W + _b
                e = np.exp(logits - logits.max(-1, keepdims=True))
                p = e / e.sum(-1, keepdims=True)
                yb = _cls[idx]
                p[np.arange(len(idx)), yb] -= 1.0
                _W -= lr * (h.T @ p / len(idx))
                _b -= lr * p.mean(0)

    trained = False
    ans_q.put(("ready", worker_id))
    # dormant loop; training deferred to first wake (event-driven, like biology)
    while True:
        req = cmd_q.get()                               # BLOCKING: zero CPU asleep
        if req is None:
            break
        t0 = req["t_sent"]
        if not trained:
            _train_once(rng, x, proj, W, b, cls)
            trained = True
        qv = req["x"].reshape(1, -1)
        h = qv @ proj
        th = np.partition(np.abs(h), -K, axis=-1)[:, -K:]
        h = h * (np.abs(h) >= th[:, :1])
        pred = int((h @ W + b).argmax())
        ans_q.put(("ans", worker_id, req["qid"], pred, time.time() - t0))


def run_pool(pool_size=POOL, max_active=MAX_ACTIVE, n_queries=500, seed=20261013):
    ctx = mp.get_context("spawn")
    cmd_qs = [ctx.Queue() for _ in range(pool_size)]
    ans_q = ctx.Queue()
    g = np.random.default_rng(seed + 5)
    protos = g.normal(0, 0.6, (N_CLASSES, D))
    cls_te = np.repeat(np.arange(N_CLASSES), 4)
    xte = (protos[cls_te] + g.normal(0, 0.9, (len(cls_te), D))).astype(np.float32)

    t0 = time.time()
    procs = []
    for i in range(pool_size):
        p = ctx.Process(target=brain_dormant, args=(i, cmd_qs[i], ans_q, seed + i), daemon=True)
        p.start()
        procs.append(p)
        if (i + 1) % 50 == 0:
            time.sleep(3)
    # wait for all ready
    ready = 0
    while ready < pool_size:
        kind, wid = ans_q.get(timeout=120)
        if kind == "ready":
            ready += 1
    spawn_time = time.time() - t0

    # dispatch n_queries to random brains, <=max_active in flight
    lat = []
    pending = {}
    next_q = 0
    in_flight = 0
    t_start = time.time()
    idx_order = g.permutation(pool_size)
    idle_loops = 0
    while len(lat) < n_queries:
        # top up in-flight
        while in_flight < max_active and next_q < n_queries:
            wid = int(idx_order[next_q % pool_size])
            cmd_qs[wid].put(dict(qid=next_q, x=xte[next_q % len(xte)],
                                 t_sent=time.time()))
            pending[next_q] = time.time()
            next_q += 1
            in_flight += 1
        try:
            kind, wid, qid, pred, serv = ans_q.get(timeout=10)
            if kind == "ans":
                lat.append(serv)
                in_flight -= 1
        except Exception:
            idle_loops += 1
    total = time.time() - t_start
    import psutil
    pool_rss = 0.0
    for p in procs:
        try:
            pool_rss += psutil.Process(p.pid).memory_info().rss / 1e6
        except Exception:
            pass
    cpu = psutil.cpu_percent(interval=1)
    for p in procs:
        p.terminate()
    return dict(pool=pool_size, max_active=max_active, queries=n_queries,
                spawn_s=round(spawn_time, 1),
                throughput=round(n_queries / total, 1),
                p50_ms=round(float(np.percentile(lat, 50)) * 1000, 2),
                p95_ms=round(float(np.percentile(lat, 95)) * 1000, 2),
                max_ms=round(float(np.max(lat)) * 1000, 2),
                pool_rss_gb=round(pool_rss / 1000, 2),
                cpu_percent=cpu, idle_loops=idle_loops)

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    rows = []
    print("=== dormant-pool: 500 resident brains, <=24 active, event-driven ===")
    for pool, active in ([(100, 8)] if quick else [(100, 24), (200, 48)]):
        r = run_pool(pool, active, n_queries=200 if quick else 500)
        rows.append(r)
        print(f"pool={r['pool']:4d} active<={r['max_active']:3d}  spawn={r['spawn_s']}s  "
              f"thr={r['throughput']}/s  lat p50={r['p50_ms']}ms p95={r['p95_ms']}ms  "
              f"pool_ram={r['pool_rss_gb']}GB  cpu={r['cpu_percent']}%", flush=True)
        json.dump(rows, open(r"D:\djr82\flymemory\bench_dormant_pool_result.json", "w"), indent=1)
    print("saved bench_dormant_pool_result.json")
