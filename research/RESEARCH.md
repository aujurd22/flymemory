# The Mushroom-Body Program

**Selection, Compression, and Emergent Structure**

> 大问题:**When does compression induce structure?** —— 当系统用有损压缩(记忆整合)或
> 稀疏竞争(k-WTA)替换"保留全部"的基线时,下游能力何时上升、何时崩溃、以什么形态上升?

果蝇的蘑菇体(Mushroom Body)是神经科学中"稀疏高维编码 + 选择性压缩 → 泛化记忆"的经典
结构。本纲领主张:这个生物学回路是一类工程问题的抽象原型——**用简单的局部选择与压缩机制,
换取全局能力**——而这类机制何时有效、何时失效,是可实验测量的。

## Testbeds(同一问题的两条实验轴)

| Testbed | 压缩形态 | 局部机制 | 下游能力 |
|---|---|---|---|
| **FlyMemory** | 记忆压缩(dedup / supersede / merge / consolidation) | 相似度选择 + 时间衰减 | 问答正确率、stale 率 |
| **FlyPoet** | 激活压缩(k-WTA 稀疏竞争) | 胜者通吃 + 梯度优化 | 数学/推理任务表现 |

## Laws(已确立,均可一键复现)

**L1 · 形态定律** — 抽象必须叠加,不能替换。
替换式整合 −10pp(22% vs 32%),overlay 式 +5.8pp(42.8% vs 37.0%,n=500)。
复现:`bench_granularity.py --mode replace|overlay`

**L2 · 二阶定律** — 整合的表述形态(prose vs 结构化时间线)在叠加前提下是二阶变量
(41.6% vs 42.8%,n=500,统计持平)。一阶变量是"是否叠加"本身。
复现:`bench_timeline.py --sample 500` vs `bench_granularity.py --mode overlay --sample 500`

**L3 · 相变定律** — 稀疏竞争的涌现窗口存在,但相变式且随规模非单调(k-WTA 甜点在
216M 反转)。复现:FlyPoet 仓库 sweet-spot 网格实验。

**L4 · 失效定律** — 选择机制会静默吞掉"小编辑型状态更新"(20 对中 5 对被 dedup
丢弃;改动仅一个数字/时间时),必须由 lineage(tombstone/supersede)兜底,否则状态
丢失且不可恢复。复现:`bench_state_fidelity.py`(修复前 git 历史)

## Registered Predictions(先注册,后实验;注册时间戳为证)

**P-2026-09-24-TL2 · 结构化时间线 overlay**
- 注册:2026-09-24(见 git 历史,先于实验运行)
- 干预:整合条目从散文摘要改为结构化格式(`SUBJECT: attr = value (date)`,逐条、保留数字与专名)
- 动机:LME e2e 归因显示 265 个 wrong 中 temporal-reasoning 占 106(40%),且 8 题人工
  复核中 6/8 需要跨 turn 时间/数量运算——散文摘要恰好丢这类结构
- **P1(方向)**:structured timeline > prose timeline(50 题 strict,seed 7 同题)
- **P2(幅度)**:temporal-reasoning 类错误率相对下降 > 非 temporal 类
- **P3(安全)**:总体 strict 不低于 turn-only 基线(32%)
- 证伪条件:structured ≤ prose timeline,则 L2 修订为"任何整合形态均为二阶,仅叠加有效"
- 状态:**判定完毕(2026-09-24 15:20)——P1 证伪,P3 成立,L2 修订**
  - 结果:structured overlay 38.0% strict / 40.0% weighted(n=50, seed 7)
  - 对比:turn-only 32.0% | prose overlay 38.0% | timeline overlay 44.0%
  - P1 证伪:structured(38.0%)≤ prose(38.0%),未超 timeline(44.0%)
  - P3 成立:38.0% > 32%(overlay 一阶效应第三次确认,+6pp)
  - **L2 修订(触发注册的证伪条件)**:三种整合形态(散文/时间线/结构化)
    互比无稳定差异——形态整体为二阶,仅叠加有效。timeline 的 50 题 44%
    为小样本乐观(全量 41.6%,与 prose 持平)。
  - 附加数据点:结构化 JSON 生成的 session 成功率 52%(488/940)显著低于
    散文的 74.8%——输出格式越严格,整合覆盖率越低,是工程成本维度
  - trace:reports/structured_50_1790233989.json

## Reproduction

全部基准脚本在仓库根,固定 seed,数据集版本化(`data/memory_judgment.json` v1.3)。
复现索引见 README 与 `OVERNIGHT_20260924.md`。

## Related

- FlyPoet(k-WTA×Transformer,216M 反转):架构轴实验记录
- FlyMemory README:记忆系统工程文档(本纲领的 testbed 之一)
- LongMemEval / LongMemEval-V2:外部基准(接入待办)
