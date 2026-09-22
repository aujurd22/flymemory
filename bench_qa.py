"""Mini QA benchmark: user-facing recall quality on the real library.

Ten questions a user might realistically ask, each with a signature substring
that identifies the correct answer entry. Measures: correct entry in top-3
(the production injection size). This is the LongMemEval-style "knowledge
update / temporal reasoning / extraction" mini-suite, flymemory edition.

Run: python bench_qa.py [--pkl PATH] [--topk 3]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flymemory"))

from flymemory.v3 import load, split_chunks, _embed  # noqa: E402

QA = [
    ("小说《固定的比特》写到哪里了", "写作进度", "创作进度类"),
    ("伏笔总账里登记了哪些伏笔", "伏笔总账", "小说设定类"),
    ("色带耗材的库存预警怎么样了", "色带", "运维耗材类"),
    ("GSO1S615F00NT83 这个货号关联了几件", "GSO1S615F00NT83", "精确货号类"),
    ("打印桥的 9100 RAW 部署架构是什么", "9100 RAW", "部署架构类"),
    ("flymemory 的压缩恢复包是怎么工作的", "压缩", "自我机制类"),
    ("两段式检索的使用前提是什么", "two_stage", "自我机制类"),
    ("果蝇视觉系统的假设讨论过吗", "果蝇", "跨会话记忆类"),
    ("HKT 登录流程现在什么状态", "HKT", "跨项目记忆类"),
    ("DA 门控衰减是什么意思", "DA", "自我机制类"),
    ("flymemory 的 CI 和打包现在什么状态", "GitHub Actions", "自我机制类"),
    ("两段式检索为什么被禁用了", "two_stage", "自我机制类"),
    ("和 OpenAI Dreaming 对比后的定位结论是什么", "状态机", "定位结论类"),
    ("温度缩放补齐校准之前做过没有", "温度缩放", "跨会话记忆类"),
    ("随机跳批对照的裁决是什么", "随机跳批", "FlyPoet 结论类"),
    ("connectome-kit 是什么工具", "connectome", "工具链类"),
    ("BRE 训练投影的结论是什么", "BRE", "负结果类"),
    ("rehearsal 排练规则的缺陷是什么", "排练", "自我机制类"),
    ("HippoRAG 和 A-MEM 的对比有结论吗", "HippoRAG", "外部格局类"),
    ("定向遗忘淋浴的结论被推翻了吗", "定向遗忘", "FlyPoet 结论类"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(_HERE, "flymemory", "flymemory_v3.pkl"))
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    mem = load(args.pkl)
    print(f"library: {mem.size} entries, top_k={args.topk}\n")

    passed = 0
    for q, sig, cat in QA:
        hits = mem.recall(q, top_k=args.topk)
        texts = [h[0].text for h in hits]
        ok = any(sig in t for t in texts)
        passed += ok
        mark = "PASS" if ok else "MISS"
        best = hits[0][0].text[:40] if hits else "-"
        print(f"[{mark}] {q[:28]:30s} (签名: {sig[:12]})  top1: {best[:36]}")

    print(f"\n{passed}/{len(QA)} = {passed/len(QA)*100:.0f}%  "
          f"（正确条目进入 top-{args.topk} 的比率）")


if __name__ == "__main__":
    main()
