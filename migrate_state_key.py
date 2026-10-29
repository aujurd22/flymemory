"""
flymemory state_key 迁移: 规范化 + 消除 split-brain

背景: v3.py 的 _norm_state_key 修复只对**新写入**生效。历史库里已经存在的
      'User.Phone' / 'user.phone ' 之类的写法仍是独立 key，并且可能有多个
      ACTIVE 条目共存（静默状态分裂）。

做两件事:
  1. 把所有 state_key 规范化 (strip + casefold)
  2. 对规范化后仍有多条 ACTIVE 的同名 key：保留**最新**一条，其余 supersede
     （不删除，history 仍可查）

默认 dry-run。加 --apply 才写盘，写前自动备份。
"""
import os, sys, time, shutil
from collections import defaultdict

sys.path.insert(0, r"D:\djr82\dsh-flymemory\python")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from flymemory.v3 import load, save, _norm_state_key

LIB = r"D:\djr82\flymemory\flymemory\flymemory_v3.pkl"


def analyze(path):
    mem = load(path, enable_hopfield=False)
    keyed = [m for m in mem.memories if m.state_key]
    unnorm = [m for m in keyed if m.state_key != _norm_state_key(m.state_key)]

    groups = defaultdict(list)
    for m in keyed:
        if m.superseded_by is None:
            groups[_norm_state_key(m.state_key)].append(m)
    splits = {k: v for k, v in groups.items() if len(v) > 1}
    return mem, keyed, unnorm, groups, splits


def main():
    apply = "--apply" in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = argv[0] if argv else LIB
    t0 = time.time()
    mem, keyed, unnorm, groups, splits = analyze(path)
    print(f"[load] {path}")
    print(f"       entries={len(mem.memories)}  load={time.time()-t0:.1f}s", flush=True)

    print(f"\n[现状]")
    print(f"  带 state_key 的条目        {len(keyed)}")
    print(f"  非规范写法 (需改写)        {len(unnorm)}")
    print(f"  ACTIVE 的规范化 key 数     {len(groups)}")
    print(f"  **split-brain 组数**       {len(splits)}")

    if unnorm:
        print(f"\n  非规范写法样例 (最多 15 条):")
        for m in unnorm[:15]:
            print(f"    #{m.memory_id:<7} {m.state_key!r}  ->  {_norm_state_key(m.state_key)!r}")

    if splits:
        n_extra = sum(len(v) - 1 for v in splits.values())
        print(f"\n  split-brain 明细 (最多 20 组, 共需 supersede {n_extra} 条):")
        for k, v in list(splits.items())[:20]:
            v = sorted(v, key=lambda x: (x.valid_from or x.timestamp))
            keep = v[-1]
            print(f"    key={k!r}  共 {len(v)} 条 ACTIVE")
            for x in v:
                tag = "  <- 保留(最新)" if x is keep else "     supersede"
                print(f"      #{x.memory_id:<7} {x.state_value!r}{tag}")
    else:
        print("\n  无 split-brain，只需改写写法。")

    if not apply:
        print("\n[dry-run] 未写盘。确认无误后加 --apply 执行。")
        return

    bak = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, bak)
    print(f"\n[backup] {bak}  ({os.path.getsize(bak)/1e6:.1f} MB)")

    for m in keyed:
        m.state_key = _norm_state_key(m.state_key)

    n_sup = 0
    for k, v in splits.items():
        v = sorted(v, key=lambda x: (x.valid_from or x.timestamp))
        keep = v[-1]
        for x in v[:-1]:
            x.superseded_by = keep.memory_id
            x.valid_to = keep.valid_from or keep.timestamp
            mem._unindex(x.memory_id)
            n_sup += 1

    save(mem, path)
    print(f"[apply] 改写 state_key {len(unnorm)} 条, supersede {n_sup} 条")
    print(f"[verify] 重新加载校验 ...", flush=True)
    del mem
    mem2, keyed2, unnorm2, groups2, splits2 = analyze(path)
    print(f"  非规范写法剩余  {len(unnorm2)}")
    print(f"  split-brain 剩余 {len(splits2)}")
    print(f"  entries={len(mem2.memories)}")


if __name__ == "__main__":
    main()
