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

**P-2026-09-24-AGG1 · 聚合型条目 overlay**
- 注册:2026-09-24(先于实验)
- 干预:在 consolidated 之外新增聚合统计条目(清单/汇总式:"TOPIC: item1;
  item2 (total: N)")——直接对应 85 个 full-miss 中的聚合统计类(数医生/
  累加花费/特定人物活动),这类问题 best evidence rank 1708-11867、结构性
  超出 top-k 单条语义检索
- **P1(方向)**:turn+cons+agg 三层 overlay 的总体 strict ≥ turn+cons 双层
  (38.0%,50 题 seed 7)
- **P2(定向)**:multi-session / temporal-reasoning / knowledge-update 类
  错误数下降
- **P3(安全)**:细节类(single-session-*)不降(L1 叠加原则)
- 证伪条件:三层 ≤ 双层,则聚合条目在当前生成质量下无增益(可能因
  session 内聚合≠跨 session 聚合,后者才是 85 miss 的真实需求)
- 状态:**判定完毕(2026-09-24 16:58)——P1 弱确认,P3 成立**
  - 结果:三层(turns+cons+agg)40.0% strict(n=50, seed 7)
  - 对比:双层(turns+cons)38.0% | turn-only 32.0% | timeline 44.0%
  - P1 弱确认:+2pp(2 题,n=50 噪声边缘,方向符合预测但幅度小)
  - P3 成立:single-session 类无退化(assistant 4/5、user 6/9)
  - 关键洞察:聚合条目只覆盖 session 内聚合(20% sessions 有可数主题),
    而 85 miss 的真实需求是 **跨 session 聚合**(evidence rank 1708-11867
    分散在千名开外)——session 内清单救不了它。**确认 README 判断:
    residual 的下一个杠杆在 answer 侧(查询时聚合计算/工具),不在
    ingest 侧的更多整合形态**(第 4 个探针的 4th 负结果+1 弱正)
  - trace:reports/aggregate_50_1790240186.json

**P-2026-09-24-TOOL1 · Agentic answer(检索工具化)**
- 注册:2026-09-24(先于实验)
- 干预:answer LLM 带 `search_memory(query)` 工具(function calling 循环),
  可自主多轮、多措辞检索,收集证据后作答——针对"聚合统计题需要多角度
  多轮检索而单次 top-5 装不下"的瓶颈(85 miss 归因 + AGG1 判定共同指向)
- 库:三层(turns+cons+agg,与 AGG1 同)
- **P1(方向)**:agentic answer 的 strict ≥ 单轮三层(40.0%,50 题 seed 7)
- **P2(定向)**:multi-session / temporal-reasoning 错误数下降
- 证伪条件:≤ 40.0%,则确认瓶颈在 answer 模型聚合能力本身而非检索形态
  ——路线转向"换更强 answer 模型"或接受边界
- 状态:**判定完毕(2026-09-24)——P1 成立,P2 成立**
  - 结果:agentic answer **46.0% strict / 51.0% weighted**(n=50, seed 7,
    平均 3.7 次 search/题)
  - 对比:单轮三层 40.0% strict / 40.0% weighted → **+6pp strict / +11pp weighted**
  - P2 成立:temporal-reasoning 8/14 correct(57%,vs structured 版 36%);
    multi-session wrong 9→7
  - **结论:agentic retrieval(多轮自主检索)打破单次 top-5 的结构限制**
    ——与 85 miss 归因预测一致("聚合统计题需要多角度多轮检索")。完整
    提升链(同 50 题):turn-only 32% → 双层 38% → 三层 40% → agentic 46%
  - **L5 · 聚合定律(新)**:聚合统计类问题("数 X""累加 Y")的答案
    分散于多条 turn,单次 top-k 检索结构性不足;agentic 多轮检索
    (自主改写查询)可将 strict 从 40% 提升至 46%——检索形态必须匹配
    问题类型(lookup 用 top-k,aggregation 用多轮工具化检索)
  - 注:answer/judge 同模型(DeepSeek),交叉 judge 校准待补
  - trace:reports/toolanswer_50_1790263915.json

**P-2026-09-24-TOOL2 · time-scoped search**
- 注册:2026-09-24(先于实验)
- 干预:search_memory 工具加 time_range 参数("YYYY-MM..YYYY-MM" 过滤),
  system prompt 指示对时间限定问题使用 scoping;对照 TOOL1(46.0% strict,
  temporal 切片 8/14 correct)
- **P1(定向)**:temporal-reasoning wrong 数下降(TOOL1: 6/14)
- **P2(安全)**:总体 strict 不低于 TOOL1(46.0%)
- 证伪条件:temporal 不降,则"时间结构检索"假设出清,v4 RFC 检索侧核心
  收缩为纯 agentic(无 time_range)
- 状态:**PENDING**

## Reproduction

全部基准脚本在仓库根,固定 seed,数据集版本化(`data/memory_judgment.json` v1.3)。
复现索引见 README 与 `OVERNIGHT_20260924.md`。

## Related

- FlyPoet(k-WTA×Transformer,216M 反转):架构轴实验记录
- FlyMemory README:记忆系统工程文档(本纲领的 testbed 之一)
- LongMemEval / LongMemEval-V2:外部基准(接入待办)
