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
