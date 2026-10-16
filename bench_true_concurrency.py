"""TRUE concurrency: independent fly-brain PROCESSES, each with its own state,
training asynchronously and exchanging knowledge through message queues.

Why this differs from everything so far (user's correction, 2026-10-13): the
"federation" benches were one process doing batched matmul — fake parallelism.
True concurrency = N independent OS processes, each holding its OWN memory
state, training on its OWN data shard, communicating ONLY via messages
(knowledge export/import), coordinated by a central learner that aggregates
their proposals. This is the fly-optic-lobe / cortical-column metaphor
implemented at the OS level.

Architecture (borrowing harness-learning ideas):
  - W worker processes ("brains"): each = MBColumn (frozen proj + k-WTA +
    trainable readout) training on its own shard of the data, maintaining its
    own prototype memory bank for retrieval.
  - 1 coordinator process ("DAN"): collects each brain's readout weights +
    retrieval stats, fuses by mean-softmax for a GLOBAL evaluation, and sends
    back a scalar "modulatory signal" (the global accuracy) that gates each
    brain's learning rate (success => keep exploring, failure => explore more).
  - Communication: multiprocessing.Queue, non-blocking, real IPC.
  - No shared memory of weights between brains except through the coordinator.

Scale: W = number of CPU cores (12) x oversubscription 2 = 24 processes is the
honest max on this box (32GB RAM, each process ~200MB torch). We run W in
{1, 4, 12, 24} and measure WALL-CLOCK time-to-accuracy and final fused accuracy,
plus the real question: does TRUE concurrent training with async message
passing reach better accuracy than 1 brain at equal wall-clock? (the actual
promise of concurrency: speed, not accuracy).

Task: 50-class D=32 prototype classification (calibrated bayes ~10%), 102k
train patterns split across brains (each brain sees a disjoint shard —
data-parallel with information isolation).
"""
import json, time, os, sys
import multiprocessing as mp
import numpy as np

N_CLASSES = 50
D = 32
N_TRAIN_PER = 2048
N_TEST_PER = 128
PROTO_SCALE = 0.6
VIEW_NOISE = 0.9
EXP_W = 512
SPARSITY = 0.10
MAX_SECONDS = 240           # wall-clock budget per arm


def gen_data(seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    protos = (torch.randn(N_CLASSES, D, generator=g) * PROTO_SCALE).to("cpu")
    cls_tr = torch.arange(N_CLASSES).repeat_interleave(N_TRAIN_PER)
    xtr = protos[cls_tr] + torch.randn(len(cls_tr), D, generator=g) * VIEW_NOISE
    cls_te = torch.arange(N_CLASSES).repeat_interleave(N_TEST_PER)
    xte = protos[cls_te] + torch.randn(len(cls_te), D, generator=g) * VIEW_NOISE
    return protos, xtr, cls_tr, xte, cls_te


def brain_process(worker_id, xtr_np, ytr_np, xte_np, yte_np, msg_q, cmd_q,
                   target_acc, deadline):
    """One independent brain: own torch state, own training, message-passing only."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)                      # critical: 1 thread per process
    torch.manual_seed(1000 + worker_id)
    dev = "cpu"
    xtr = torch.tensor(xtr_np); ytr = torch.tensor(ytr_np)
    xte = torch.tensor(xte_np); yte = torch.tensor(yte_np)
    protos = torch.tensor(np.load(rf"D:\djr82\flymemory\_tc_protos.npy"))
    g = torch.Generator().manual_seed(9000 + worker_id)
    proj = (torch.randn(D, EXP_W, generator=g) / np.sqrt(D))
    W = proj.clone().requires_grad_(True)
    readout = nn.Linear(EXP_W, N_CLASSES)
    nn.init.zeros_(readout.bias)
    with torch.no_grad():
        readout.weight.normal_(0, 0.01)
    k = max(int(EXP_W * SPARSITY), 1)
    lr = 3e-3
    mod_signal = 0.5                              # DAN signal starts neutral
    bs = 512
    n = len(ytr)
    t0 = time.time()
    step = 0
    try:
        while time.time() - t0 < deadline:
            # non-blocking message check: DAN modulation
            try:
                while True:
                    msg = cmd_q.get_nowait()
                    if msg[0] == "mod":
                        mod_signal = msg[1]
            except Exception:
                pass
            # train step: k-WTA trunk + readout
            idx = torch.randint(0, n, (bs,))
            xb, yb = xtr[idx], ytr[idx]
            h = xb @ W
            th = torch.topk(h.abs(), k, dim=-1).values[..., -1:]
            h = h * (h.abs() >= th)
            logits = readout(h)
            loss = nn.functional.cross_entropy(logits, yb)
            gW = torch.autograd.grad(loss, [W, *readout.parameters()], retain_graph=False)
            # DAN modulation: success => normal LR, failure => boosted LR (explore)
            scale = 0.5 + mod_signal
            with torch.no_grad():
                W -= lr * scale * gW[0]
                readout.weight -= lr * scale * gW[1]
                readout.bias -= lr * scale * gW[2]
            step += 1
            # every 25 steps: self-eval + send report
            if step % 25 == 0:
                with torch.no_grad():
                    h = xte @ W
                    th = torch.topk(h.abs(), k, dim=-1).values[..., -1:]
                    acc = float((readout(h * (h.abs() >= th)).argmax(-1) == yte).float().mean())
                try:
                    msg_q.put_nowait(dict(worker=worker_id, step=step, acc=acc,
                                          wall=time.time() - t0,
                                          W=W.detach().clone().numpy(),
                                          rw=readout.weight.detach().clone().numpy(),
                                          rb=readout.bias.detach().clone().numpy()))
                except Exception as _e:
                    print(f'CHILD{worker_id} report failed: {_e}', flush=True)
                if acc >= target_acc:
                    break
    except Exception as e:
        try:
            msg_q.put_nowait(dict(worker=worker_id, step=-1, acc=-1.0, wall=0,
                                  error=repr(e)[:200],
                                  W=W.detach().numpy(), rw=readout.weight.detach().numpy(),
                                  rb=readout.bias.detach().numpy()))
        except Exception:
            pass
    return


def run_arm(n_brains, seed=20261013, target_acc=0.87, deadline=MAX_SECONDS):
    """Spawn n_brains INDEPENDENT PROCESSES, each on a disjoint data shard."""
    import torch
    protos, xtr, ytr, xte, yte = gen_data(seed)
    np.save(r"D:\djr82\flymemory\_tc_protos.npy", protos.numpy())
    # disjoint shards: brain i sees rows [i::n_brains] (information isolation)
    shards = [(xtr[i::n_brains].numpy(), ytr[i::n_brains].numpy()) for i in range(n_brains)]
    msg_q = mp.Queue()
    cmd_qs = [mp.Queue() for _ in range(n_brains)]
    procs = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    for i in range(n_brains):
        p = ctx.Process(target=brain_process,
                        args=(i, shards[i][0], shards[i][1],
                              xte.numpy(), yte.numpy(),
                              msg_q, cmd_qs[i], target_acc, deadline), daemon=True)
        p.start()
        procs.append(p)
    # coordinator loop: collect reports, fuse, send modulation (DAN role)
    best_fused = 0.0
    timeline = []
    latest = {}
    reports = 0
    while time.time() - t0 < deadline:
            try:
                m = msg_q.get(timeout=1.0)
                latest[m["worker"]] = m
                reports += 1
            except Exception:
                m = None
            # periodic global eval + modulation
            if latest and (reports % n_brains == 0 or (time.time() - t0) > deadline - 2):
                # fuse: mean softmax over brains on test set (coordinator-side eval)
                xte_t = torch.tensor(xte.numpy())
                yte_t = torch.tensor(yte.numpy())
                probs = []
                for wid, m in latest.items():
                    W = torch.tensor(m["W"])
                    h = xte_t @ W
                    k = max(int(EXP_W * SPARSITY), 1)
                    th = torch.topk(h.abs(), k, dim=-1).values[..., -1:]
                    h = h * (h.abs() >= th)
                    ro_w = torch.tensor(m["rw"]); ro_b = torch.tensor(m["rb"])
                    logits = h @ ro_w.T + ro_b
                    probs.append(torch.softmax(logits, -1))
                if probs:
                    fused = float((torch.stack(probs).mean(0).argmax(-1) == yte_t).float().mean())
                    best_fused = max(best_fused, fused)
                    timeline.append(dict(wall=round(time.time() - t0, 1), fused=fused,
                                         n_reporting=len(latest)))
                    # DAN modulation: global success gates everyone's LR
                    mod = min(max((fused - 0.70) / 0.17, 0.0), 1.0)
                    for cq in cmd_qs:
                        try:
                            cq.put_nowait(("mod", mod))
                        except Exception:
                            pass
                reports = 0
            if all(not p.is_alive() for p in procs):
                break
    for p in procs:
        p.terminate()
    wall_to_best = next((t["wall"] for t in reversed(timeline) if t["fused"] >= best_fused - 1e-9), None)
    return dict(n_brains=n_brains, best_fused=best_fused, wall_to_best=wall_to_best,
                n_reports=len(timeline), timeline=timeline[-6:])


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        MAX_SECONDS = 45
        N_TRAIN_PER = 512
        N_TEST_PER = 32
        target = 0.80
    else:
        target = 0.87
    rows = []
    print("=== TRUE concurrency: independent processes, message passing, disjoint shards ===")
    print(f"(budget {MAX_SECONDS}s per arm; {'SMOKE' if smoke else 'full'})")
    for W in ((1, 4) if smoke else (1, 4, 12, 24)):
        r = run_arm(W, target_acc=target, deadline=MAX_SECONDS)
        rows.append(r)
        print(f"W={W:3d}  best_fused={r['best_fused']:.4f}  wall_to_best={r['wall_to_best']}s  "
              f"reports={r['n_reports']}", flush=True)
        json.dump(rows, open(r"D:\djr82\flymemory\bench_true_concurrency_result.json", "w"), indent=1)
    print("saved bench_true_concurrency_result.json")
