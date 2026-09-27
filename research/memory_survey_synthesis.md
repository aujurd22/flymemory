# GitHub 调研综合报告:记忆系统 × AI × 生物启发(2026-09-25)

> 规模:4577 仓库索引(8 桶 58 查询)→ 16 仓深度精读(行业 10 + 学术 3 + 教学/实验 3)
> 原始索引:reports/github_survey_raw.json | 精读笔记:memory_survey_notes.md | 打分:github_survey_top400.json

---

## 一、行业机制分类学(六大流派)

| 流派 | 代表 | 核心主张 | 对 flymemory |
|---|---|---|---|
| 检索智能派 | mem0 | 写入 ADD-only,时效性交给检索端时间重排 | 哲学对立面(我们写入维护)——可做 stale 率对照实验 |
| 记忆 OS 派 | MemOS | 记忆=系统资源;Neo4j+Qdrant;L1/L2/L3 分层+Skills 结晶 | 工程重;L1/L2/L3 分类学与我们的层级互证 |
| 自编辑派 | letta/MemGPT | 模型用工具自主增删改记忆;core/recall/archival 三层;sleep-time compute | 哲学同源(模型判断);缺常驻 persona 块是我们的差异 |
| hook 自动化派 | claude-mem、**flymemory** | 生命周期钩子自动捕获+压缩+注入 | claude-mem 的三层渐进披露值得借鉴 |
| 图谱派 | cognee、Zep/Graphiti、m_flow | 实体-关系图;graph-as-scoring-engine(m_flow) | m_flow 的粒度匹配检索是 85-miss 聚合题的已实现解 |
| 结构化状态派 | Memori、INITE、yantrikdb | 类型化记忆/双时间线/矛盾状态机 | **与 V4 方向最接近,借鉴最多的一派** |

## 二、精华机制 Top 10(可借鉴,按价值排序)

1. **双时间线**(INITE):有效时间(现实成立)+ 知识时间(系统何时知道)——v4-rfc 的 valid_from/valid_to 应升级为双时间线
2. **矛盾检测的诚实边界**(yantrikdb):只检测结构化单值声明的极性/时序冲突,策略 ask_user;COMPETING 状态保留未解决分歧——flymemory 缺失的机械矛盾层,且边界声明诚实
3. **主动洞察触发器**(yantrikdb think()):待解矛盾/临近 deadline/跨域模式/即将衰减的高价值记忆/目标追踪——从被动召回升级为主动推送
4. **四层金字塔+确定性下钻**(TencentDB):L0 对话→L1 原子→L2 场景→L3 画像;Persona→Scenario→Atom→Conversation 沿 node_id 下钻,注入仅几百 token
5. **三层渐进披露注入**(claude-mem):search(索引 ~100tok)→ timeline → 全文(~1000tok),省 10× token——把注入预算变成两阶段交互
6. **粒度匹配检索+图传播**(m_flow):精确线索命中原子层、宽泛主题命中 Episode 层——85-miss 聚合题的已实现解
7. **共指消解在摄取时**(m_flow):代词→实体,避免检索锚点丢失——85-miss 的隐性原因之一
8. **L1 抽取节奏控制**(TencentDB):每 N 轮触发、每批上限、空闲触发、最小间隔——比我们每轮 hook 更省
9. **sleep-time compute**(letta):空闲时后台整理巩固——与粒度 A/B 的 consolidation 同向,有 idle-time agent 实现可参照
10. **COGX 记忆互导格式**(cognee):Mem0/Letta/Zep/Graphiti 记忆可互导——行业已开始标准化,导出格式可参考

## 三、负结果与诚实声明(同行的,与我们文化共鸣)

- mem0:开源版效果仅 "directionally similar" 托管版
- yantrikdb:公开 12 题小样本局限、承认某操作符"数学上无法翻转决策"的负面结果
- MHN:自我定位 research 级非生产替代
- INITE:证据不足可弃权(abstain)

## 四、格式谱系定稿(flymemory 本轮实验)

narrative turns(32%)→ key-value entity-state(36%,过度压缩有害)→ prose overlay(38%)→ **timeline overlay(44% @50题 / 41.6% @500题,与 prose 持平)**
→ 一阶杠杆 = overlay 本身;条目格式是二阶。

## 五、flymemory 独占定位(四象限之外)

行业四象限(检索智能/记忆OS/自编辑/hook自动化)+ 图谱派 + 结构化状态派,无一家同时具备 flymemory 的五件事:
1. 写入侧状态机的**机械不变量**(tombstone/lineage/evidence 卫生,测试锁定)
2. **果蝇机制实验源头**(k-WTA/小室/门控的 ML 翻译实验链)
3. **预注册 benchmark**(判据先锁,防 HARKing)
4. **负结果文化**(每个失败都进 README)
5. 本地单文件 + 零依赖

## 六、项目推进建议(按优先级)

### 短期(下个会话可做)
1. **矛盾检测最小版**:借鉴 yantrikdb 边界——只检测结构化单值声明的极性冲突;检测到→提示模型走 supersede(补全"感知→操作"环路)
2. **主动洞察触发器最小版**:每 N 次整合后生成"待解矛盾/即将衰减高价值记忆"摘要注入
3. **渐进披露注入**:recall 先返回一句话索引,模型要详情再拉(需 MCP 交互循环)
4. **README 定位语升级**:采用评审建议的 thesis——"An evidence-preserving temporal state machine for long-term agent memory"

### 中期(v4 实现)
5. v4-rfc.md 修订:采纳双时间线(INITE)、COMPETING 状态、Beliefs 层评估
6. Entity-State 条目从 LME haystack 同源生成(修正 B2 设计),与 timeline 做 500 题对照
7. 数据集维度轴(v1.4)+ 类型轴(Memori 八类)组成完整矩阵

### 长期
8. LME-V2 能力分类学采纳(state tracking/workflow/gotchas/premise);数据集接入在 V4 schema 落地后
9. 跨 actor/judge 三角验证(需第二家 API key)
10. 论文:以"预注册负结果文化 + 状态机不变量"为方法学贡献,以 144 场景判断基准 + 500 题 e2e 五臂为实证

## 七、Round-8 交叉传粉:P32-h 方法学移植(2026-09-28)

把 intuition-mechanism P32-h(balanced novelty detection)的平衡设计与 STALE
的 Premise Resistance 轴移植进判断基准 → 数据集 v1.5(168 场景),新增
balanced-stale / balanced-resist 各 12 条(gen_dataset_batch5.py)。

**结果(DeepSeek, temperature 0)**:
- balanced-STALE 12/12 —— 真变化零漏检,无保守偏差
- balanced-RESIST 9/12 —— 3 条越界,分两类:
  ① 1 条真 Premise-Resistance 失守:意图声明("I'm switching back to VS Code")
     被直接 supersede 成状态变化
  ② 2 条边界 remember:状态未动(正确),但把"考虑中"本身存成了新条目
     ——可辩解为真,但严格 no-op gold 下是噪音

**Response bias 结论**:DeepSeek 的偏差方向是 **over-eager 而非 conservative**
(对照:P32-h 里 doubao judge NEW 仅 7/40,严重保守)。集中出现在
intent-statement 场景;模型 reason 里明确写出"only considering, not
confirmed"却仍 2/12 次写入——概念存在,执行不稳。

方法学价值:单侧 P/R 看不见偏差方向;平衡设计两侧一测,方向+大小同时出来。
这是三项目母题(稀疏选择→压缩→稳定结构→再利用)之外的第二条通用方法:
**balanced probes dissociate recognition from bias**——P32-h 用它分离
recognition(80%)与 novelty(随机),这里用它分离"知道变了"与"该不该写"。

### 7.1 Arm-S 移植负结果(P32-i → flymemory,2026-09-28)

P32-i 证明其新颖性缺陷是提取失败(Arm S 供提取程序 20/40→39/40,Arm T 截断
只换偏置 21/40)。移植 two-step 协议(强制断言提取 fact/intent/question)到
判断基准 168 场景,预注册判据判定 **NOT SUPPORTED**:

- bal_resist_01 未修复:模型提取正确但把 "I'm switching back to VS Code"
  标为 fact——失败在**分类裁量**,不在提取
- RESIST mutation 3→4(08 修复,11/12 新增协议允许的 consideration 写入)
- 噪声回归:noop 0/36→1/36、supersede R 1.00→0.989(sup_06)、
  consolidation 时机 3 例改变

**与 P32-i 的调和(跨界线发现)**:P32-i 的程序有效因为其充分统计量是客观的
(与 6/20/70 的算术比较+符号翻转);memory judgment 没有机械充分统计量——
什么算 update、什么值得存是语义裁量。**脚手架修复适用于"客观签名的提取瓶颈",
不适用于"语义边界的裁量瓶颈"**。三方对照:P32-i=提取瓶颈(可脚手架修复)、
flymemory=裁量瓶颈(不可)、Arm T 式"换偏置"在两域都出现过(P32-i 21/40;
此处 STALE 未翻转因为协议保留了 fact-only supersede 规则)。这加深 flymemory
核心论点:judgment lives in the caller——裁量原语不能被流水线脚手架机械化。

## 八、Round-9 评审:law 正式化 + 三裁量原子拆分(2026-09-28,用户贴文)

**Program-level law(两仓交叉验证,用户正式化)**:
- Scaffolding helps when the target has an objective sufficient statistic
- Scaffolding does not automatically solve semantic boundary judgment

**对 P32-i 的方法学警告(留给 intuition-mechanism 侧)**:Arm S 同时供给
representation+classifier rule(签名怎么算+类别怎么比),39/40 证明的是
"供给正确结构表示及其判别方式后任务变易",**抽取与比较两因果因素未分离**。
建议 P32-j 两阶段:阶段一只输出 signature(H/A/B/C)不给映射规则;阶段二
把结构化 signature 交给通用 comparator——真正拆开
raw→representation 与 representation→decision。
(该实验属 intuition-mechanism 仓库,由其并行会话执行;此处存档设计。)

**对 flymemory README 措辞的逻辑边界(已采纳修正)**:
"no mechanical sufficient statistic" → "this schema supplies no comparable
mechanical sufficient statistic"(关于本 schema 的陈述,非存在性证明);
判断"原则上能否分解为客观子问题"恰是下一阶段研究对象。

**三裁量原子拆分(下一阶段主实验,bench_atomic_judgment.py v0 已建)**:
- A Assertion extraction:turn → [{text, kind: fact|intent|question}]
  ——表征问题,P32-i 同构
- B State-change detection:(entry, assertion) → CHANGED|UNCHANGED|UNKNOWN
- C Memory-worthiness:(assertion, context) → KEEP|DISCARD|EPHEMERAL
目标:把 168 端到端的错误归因到原子段,寻找 flymemory 的机械充分统计量
在哪个原子存在。关键预期:A 段在"I'm switching back"类语句上 fact/intent
标注分歧=resist_01 根因的直接测量;B 段若给定 fact 断言后 CHANGED 判定
容易,则端到端失败完全归属 A 段。

**瓶颈收敛判定(用户)**:supersede ~1.0 P/R + stale 12/12 意味着
"明确的 state-change"已解,瓶颈从 memory management 收敛为
**semantic boundary detection**(用户只是说起可能性但尚未真改状态的边界)。

### 8.1 三原子首轮结果(2026-09-28,atomic_1790532587.json)

| 原子 | 分数 | 混淆 |
|---|---|---|
| A 断言提取 | 81.2% (26/32) | 全部单向偏 fact:intent 4/5(a_rs01 声明式进行时→fact)、question 2/7(3→fact,2→intent) |
| B 状态变化检测 | **100%** (30/30) | 无——含 3 条 resist 类考虑语句全部正确 UNCHANGED |
| C 记忆价值 | 86.7% (26/30) | KEEP 10/10、EPHEMERAL 10/10;4 条一次性事件→EPHEMERAL(无害) |

**分解归因(本轮最重要发现)**:
1. 端到端 RESIST 失败完全归属 A 段;B 段给锚点(存储值 vs 断言值)后
   恰恰是那些失败语句全部可解
2. **锚点=半客观充分统计量的来源**:同一条"I'm considering switching
   back to VS Code",在 A 段(无锚点,纯语言行为分类)守不住 fact/intent
   边界;在 B 段(有 entry 具体值可比)100% 正确。语义裁量并非均匀地难——
   **有锚点的裁量近乎机械,无锚点的裁量才真是裁量**
3. question→fact 3 例:A 段 prompt 明说 question 不作断言仍被吸成 fact
   ——向 fact 的拉力比指令强,这是单向偏置的又一证据
4. C 段错误全部无害方向(宁存勿删,不产错误状态)
5. 工程处方(下一阶段候选):调用方协议改为"对每条已存条目问 B 问题
   (锚定、近机械)",而非"先自由分类 turn 再决策"(无锚)
