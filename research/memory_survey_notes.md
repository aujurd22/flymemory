# GitHub 调研笔记:记忆系统 × AI × 生物启发(2026-09-25)

> 方法:小批量搜索 → 逐仓精读 README → 提取机制/差异/可借鉴点。
> 索引面:research/github_survey_raw.json(4577 仓,8 桶)/ top400 打分版。
> 目的:为 flymemory V4(Temporal-Evidence Memory,v4-rfc.md)收集设计输入。

---

## 一、LLM 记忆系统(直接竞品/同类)

### mem0ai/mem0 (★66k)
- **机制**:2026-04 新算法 = **ADD-only 单遍抽取**(一次 LLM 调用抽事实,无 UPDATE/DELETE);检索端三路融合(向量+BM25+实体链接)+ 时间感知重排序(当前/过去/未来查询对带日期实例重排)。
- **哲学**:写入极简,复杂性转移到检索端;时效性是查询属性而非存储属性。
- **对比 flymemory**:与我们的路线相反——我们写入端做 supersede/tombstone(状态机),检索端简单 RRF。mem0 用"检索智能"解决我们用"写入维护"解决的问题。**启示:两者可以对照测**——stale 答案率在我们 v1.4 场景上 vs mem0 的 ADD-only+时间重排,谁的 stale 更低?这是可做的对照实验。
- 弱点:无显式遗忘、无 lineage、无整合;基准分数来自托管版(开源版 "directionally similar")。

### MemTensor/MemOS (★11.6k)
- **机制**:记忆当 OS 资源;Neo4j 图 + Qdrant 向量;多模态(文本/图像/工具轨迹/persona);Multi-Cube 隔离共享;MemScheduler 异步摄取;L1 轨迹/L2 策略/L3 世界模型分层 + Skills 结晶化。
- **对比**:工程重(Neo4j+Qdrant+云端),与 flymemory 单文件本地路线相反。**可借鉴:L1/L2/L3 分层**(轨迹→策略→世界模型)与我们的 L0 原始/L1 原子/整合条目分类学互证;"记忆当系统资源"的叙事框架。
- 无自动遗忘/衰减——显式删除+反馈修正。

### TencentCloud/TencentDB-Agent-Memory (★27k)
- **机制**:**四层金字塔 L0 对话→L1 原子事实→L2 场景块→L3 画像**,上层保判断、下层保证据,"Persona→Scenario→Atom→Conversation"确定性下钻链;检索 = BM25(jieba)+向量+**RRF 融合**(与我们同款);短期上下文三级压缩(原始 refs→JSONL 摘要→Mermaid 画布,按 node_id 下钻)。
- **哲学**:"记忆不是囤积一切,而是让人不必重复自己";白盒可调试,拒绝黑盒向量分。
- **对比**:与我们最接近的工程哲学(白盒+确定性下钻+RRF)。**可借鉴:①L1 抽取节奏控制**(每 N 轮触发、每批上限、空闲触发——比我们的每轮 hook 更省);②Mermaid 画布做任务态(我们 RECENT 通道的结构化升级方向);③"不可逆有损摘要"的明确拒绝与我们 tombstone 原则同构。
- 数字:WideSearch token −61%、PersonaMem 48%→76%。

### yantrikos/yantrikdb (★63, Rust)
- **机制**:嵌入式 SQLite 单文件;五索引(HNSW/实体图/时间/衰减堆/KV);**decay heap** + `stale(days=14)`(高价值近期未访问);**矛盾检测只做结构化单值声明**(极性/时序冲突,策略 ask_user,诚实声明不做通用矛盾检查);`think()` 四步自主整合(合并+冲突扫描+跨域模式挖掘+**触发器评估**:待解矛盾/临近 deadline/跨域模式/即将衰减高价值记忆);`recall_as_of(t)` 历史时点查询;CRDT 多设备同步。
- **对比**:机制与我们高度重叠(decay/整合/历史查询),但多了**主动洞察触发器**(flymemory 没有:我们被动召回,它主动推送洞察)和**矛盾检测**(我们靠模型自觉 supersede,它有机械检测+ask_user 流程)。**两个最值得借鉴的机制**。
- 工程过度部分(双层 LSM/openraft 集群)与个人定位无关。

---

## 待精读队列
inite-ai/inite-brain-service(bitemporal KG)、FlowElement-xinliuyuansu/m_flow(★4.5k bio-inspired)、shahzebqazi/mhn-ai-agent-memory(Hopfield)、NirDiamant/Agent_Memory_Techniques(★1k 教学全景)、GMvandeVen/continual-learning(★1.9k)、NawrotLab/KC_KC_lateral_interactions(KC 侧向交互,学术)、Tanvrit/smritidb(associative memory standard)、memvid(★16.5k 单文件)、letta(★24.9k)、cognee(★31k)

### inite-ai/inite-brain-service (★40, AGPL, Node/SurrealDB) —— **与 V4 方向最接近的已实现系统**
- **双时间线**:有效时间(validFrom,事实在现实世界成立)+ 知识时间(系统何时知道,自动记录)——比 v4-rfc 的 valid_from/valid_to 更完整,值得采纳为"双时间线"。
- **写入状态机**:每条事实经冲突解决后显式返回 INSERTED / SUPERSEDED / **COMPETING**(未解决分歧保留,供 detect_contradiction/get_competing_facts 检查);撤销保留历史;管理性遗忘删记录留 tombstone。
- **五层记忆**:Facts / Episodes / Scenes / Beliefs / Evidence。我们缺 Beliefs 层(从证据推断的信念,与证据分离)。
- **answer 带 citations 与 evidenceCitations 分离,证据不足可弃权(abstain)**——我们 Phase 2 judge 有 unknown 判定,但 answer 侧没有弃权机制。
- **领域包(Domain Packs)**:版本化 JSON manifest 声明类型化谓词的冲突语义/衰减规则/场景模式,按租户安装不 fork 引擎——冲突语义可配置的优雅解法。
- **对 v4-rfc 的修正输入**:①双时间线取代单一 valid_to;②COMPETING 状态(矛盾不必即时解决);③Beliefs 与 Facts 分层;④弃权是 answer 侧行为,不是记忆侧行为。

### FlowElement-xinliuyuansu/m_flow (★4.5k) —— 四层锥形图,已实现的 V4 多粒度
- **四层锥形图**:Episode(事件)→ Facet(维度)→ FacetPoint(原子事实)→ Entity(跨事件实体),查询按粒度落层(精确线索→FacetPoint,宽泛主题→Episode),图传播路由到 Episode bundle。
- **图即评分引擎**(path-cost retrieval):沿语义类型化边传播证据,按最强证据链打分——区分"相似"(空间距离)与"相关"(连贯证据链连接)。多数 GraphRAG 图只是辅助,它让图拓扑决定得分。
- **共指消解在摄取时**(代词→实体,避免检索时锚点丢失)——我们 hook 写入无共指消解,"它/她"指代丢失,这是 85-miss 里部分聚合题的隐性原因。
- 自报:LoCoMo-10 81.8%、LongMemEval 89%(自报协议未验证,谨慎)。
- **对 85-miss 的启示**:聚合统计题的已实现解 = 粒度匹配检索 + 图传播,比 top-k 重;v4 若做,granularity-aware retrieval 是核心。

### shahzebqazi/mhn-ai-agent-memory (★5, research) —— MHN = 单步 attention
- Modern Hopfield 检索步 `x' = X @ softmax(β·Xᵀ·x)` 与 transformer attention 数学同构(Ramsauer 2020)——**flypoet 的 attention 主干本身就是联想记忆检索步**,理论与我们的架构天然连接。
- 零匹配三信号检测(max_sim + gap + sentinel weight,返回 None 而非强制最近模式)——比我们 sim>0.4 单阈值精细。
- 排斥性注意力(能量景观加对比山丘)多跳收敛 17×;冷热分层(Hopfield 热+FAISS 冷)。
- 自我定位诚实:research 级,非生产替代。对 flymemory:MHN 一步集中检索可作为 RRF+CE 的实验分支(低成本 numpy),但 RRF+CE 已工作,记录为理论连接。

### NirDiamant/Agent_Memory_Techniques (★1.1k) —— 行业分类学参照
30 个可运行 notebook:conversation buffer / vector RAG memory / summary memory / entity memory / knowledge-graph memory / 等等。作为"行业怎么做"的分类学索引,与我们"研究怎么做"互补。

### thedotmack/claude-mem (★94.6k, TypeScript) —— 与 flymemory 高度同构
- hook 驱动(SessionStart/UserPromptSubmit/PostToolUse/Stop/SessionEnd——比我们多 PostToolUse/Stop 的工具使用观察);SQLite+FTS5+Chroma 混合检索;语义摘要压缩。
- **三层渐进披露**:search(索引 ~50-100 tok/条)→ timeline(时间线)→ get_observations(全文 ~500-1000 tok/条),自报省 10× token——两阶段注入(先索引后详情)是我们没有的"注入预算分层"。
- 多语言模式、本地 SQLite、可选云同步。对我们的启示:recall 注入可先给一句话索引,模型感兴趣再拉全文(需要交互循环,MCP 下可行)。

### letta-ai/letta (★24.9k, MemGPT 后继) —— 模型自编辑记忆的鼻祖产品化
- **三层记忆**:core(常驻上下文,persona+human 两块)/ recall(全对话史)/ archival(外部知识库)。
- **记忆自编辑**:模型通过工具调用(core_memory_append/replace、archival_memory_insert)自主决定记什么——与我们"模型判断 supersede"哲学同源;差异是他们连"记"也交给模型,我们是 hook 机械捕获+模型只做修正。
- **sleep-time compute**:空闲时后台整理巩固记忆(类睡眠固化)——与我们粒度 A/B 的 consolidation 同向,他们有 idle-time agent 实现可参照。
- 常驻 persona/human core 块我们没有——我们的等价物是 recalled+RECENT 注入,无常驻身份块。v4 可考虑固定 persona 条目常驻注入。

### 行业格局速写(基于本批+此前)
- 大厂/平台:mem0(检索智能派)、MemOS(OS 派)、letta(自编辑派)、claude-mem(hook 派——与我们同类)、cognee(图派)、m_flow(图评分派)
- flymemory 差异化坐标:**本地单文件+机械不变量+预注册 benchmark+果蝇机制源头**。行业在"检索智能/分层/工具化"上卷,flymemory 的不可替代性在"写入侧状态机+可证明不变量+负结果文化"。

### GMvandeVen/continual-learning (★1.9k, NMI 2022 官方实现) —— 与 flypoet comp 臂同构
- 实现 12 种持续学习方法(EWC/SI/LwF/DGR/BI-R/ER/A-GEM/iCaRL/生成分类器/**XdG** 等)× 三场景(task/domain/class-incremental)。
- **XdG = 上下文相关门控**(每任务掩蔽不同神经元子集,训练测试都只激活该子集)——与 flypoet comp 臂(每域随机 30% 权重)同思想;论文结论:XdG 只适用 task-incremental(需知任务身份),与 SI/EWC 组合更好。
- **场景分类学**(重要坐标系):flypoet 的 4 域顺序微调 = domain-incremental;该场景下论文结论"正则化退化、回放类最好"——与 fly 臂(参数分区+门控)仍胜形成张力:小模型+门控在大模型正则化失效处仍有效,或因 92.6M 参数量下分区足够粗粒度。
- **总体定律**:跨场景最稳健 = 回放+正则化组合。flypoet 睡眠循环三 seed 无差异(+0.003)与"回放有效"主流相反——我们的淋浴剂量(375 步,~+3% 算力)可能低于有效剂量,这是未解差异。
- XdG 与果蝇的对应:蘑菇体小室 = 硬件化的 XdG(域→小室分配在连接组里固定),FlyPoet 用随机 mask 模拟——差异是果蝇的小室有气味→小室的映射结构,非随机。

### NawrotLab/KC_KC_lateral_interactions (学术, Current Biology 2026) —— flypoet 未测变量
- KC-KC **侧向交互**(KC 间的直接局部连接,区别于 APL 全局抑制)对嗅觉学习效率与特异性的作用;速率模型拟合 Manoim et al. 2022 钙成像。论文: doi.org/10.1016/j.cub.2026.01.014
- 研究维度:侧向交互对稀疏性/气味表征去相关/模式分离/记忆特异性的贡献。
- **flypoet 下一个实验设计**:k-WTA 是全局竞争(所有 KC 一起排名);果蝇另有 KC-KC 局部侧向抑制。可测:在 k-WTA 之外加"局部侧向抑制层"(仅相似 KC 间互抑)是否比全局竞争产生更好的模式分离——特别是相似气味(高重叠 odor 对)的分辨。这是把"全局 vs 局部抑制"变成受控实验。

### topoteretes/cognee (★31k, 图派) —— 待读
### Tanvrit/smritidb —— "biology-inspired associative memory standard, remembers by partial cue" —— 待读

### topoteretes/cognee (★31k, 图+向量混合) —— 图谱派代表
- **混合存储**:图谱(实体-关系)+ 向量块 + 会话存储,检索自动路由;可全跑单一 Postgres。
- **四操作生命周期**:remember/recall/improve(session distillation——把会话中被采纳的经验教训提炼进持久图谱)/forget。
- **无 LLM 也能工作**:本地 GLiNER 抽取+本地嵌入;崩溃后管线可恢复;数据集绑定嵌入模型防不匹配。
- **COGX 交换格式**:支持从 Mem0/Letta/Zep/Graphiti 导入既有记忆——**行业已在标准化记忆互导**(我们若做导出格式可参考)。
- 学术背书:arXiv 2505.24478 + BEAM 评测(100K token 0.79 / 10M 探索性 0.67)。
- 对比:图派重"关系结构",flymemory 重"状态机+不变量"——不同哲学,但 cognee 的 session distillation 与我们的 consolidation 同构。

### memvid (★16.5k, Rust) —— 单文件记忆
- 记忆编码进单个视频文件(二维码帧),无服务器无数据库,即时检索。工程奇观,机制参考价值低;但"记忆=可携带单文件"的定位与 flymemory 单文件 pkl 哲学同向。

### MemoriLabs/Memori (★16.9k) / EverMind-AI/EverOS (★13.1k) —— 简记
- Memori:agent-native 结构化持久状态(与我们的 entity-state 方向同行),LLM-agnostic。
- EverOS:本地优先、Markdown-native、用户自有、自我演化——**与 flymemory "本地+可审计"定位最接近的叙事**;Markdown-native(人类可读)是我们 pkl 之外的另一个取舍点。

### 调研阶段小结(12 仓精读 + 4577 索引)
四象限定位:检索智能(mem0)/记忆OS(MemOS)/自编辑(letta)/hook自动化(claude-mem、flymemory)/图谱(cognee、Zep)/生物机制(flypoet、MHN、果蝇学术)。flymemory 独占的组合:**写入侧状态机(机械不变量)+ 果蝇机制实验源头 + 预注册 benchmark + 负结果文化**。行业可借鉴 top3:①YantrikDB 主动洞察触发器+矛盾 ask_user;②TencentDB L1抽取节奏控制;③claude-mem 三层渐进披露注入。

### Tanvrit/smritidb —— Kanerva SDM/超维计算的标准化工尝试
- **机制**:二元超维计算(Kanerva 1988)——数据编码为 10000 维二进制向量,**内容即地址**;查询用 partial cue 编码按余弦 top-K 召回,无需精确匹配。
- **独特性质**:①全息退化——记录分散在大量存储单元,丢部分基底=整体变模糊而非条目丢失(向量库按分片丢数据是真丢);②Hebbian 自组织——频繁共访项自动绑定,冷数据摘要压缩(类比海马体→皮层固化);③跨语言字节级一致(KMF 线格式+一致性语料+BLAKE3 决胜)。
- **制度设计**:Apache-2.0 + 不可撤销专利授权——"开放标准的价值在于透明,黑盒化会让标准崩塌"。
- **与 flymemory**:SDM 是我们 .bio 方向的理论源头(flymemory 的 dense matrix + cosine 检索就是简化版 SDM);flypoet 的 address book(Hamming hit@10)与 smritidb 同理论家族。全息退化是我们没有的性质。

### NirDiamant/Agent_Memory_Techniques (★1.1k) —— 行业教学分类学(30 notebook)
覆盖:conversation buffer、summary memory、vector RAG memory、entity memory、knowledge-graph memory、episodic memory 等的行业标准做法。用途:作为"行业怎么做"的对照系;flymemory 的差异化(写入侧状态机+不变量+预注册)对照这份清单更清晰。

### MemoriLabs/Memori (★16.9k) —— 八类结构化记忆 + 执行轨迹记忆
- **三追踪层级**(entity/process/session)× **八类结构化记忆**:attributes/events/facts/people/preferences/relationships/rules/skills——不只对话文本,还从 **agent 执行轨迹**(tool calls/decisions/outcomes)提取记忆("做了什么"而非只"说了什么")。
- **框架观察**:记忆"类型"(what)与"更新维度"(how it changes:multi-update/reversal/temporary/partial-correction,我们 v1.4 的轴)是两个正交轴——行业分类学+我们的维度轴可组成完整矩阵。
- **token 效率卖点**:LoCoMo 87%,每查询 721 tokens(全量上下文的 2.8%)——结构化记忆替代长上下文的成本论证值得引用。

### EverMind-AI/EverOS (★13.1k) —— 本地 Markdown-native 叙事同类
本地优先、Markdown-native、用户自有、自我演化——与 flymemory"本地+可审计"定位最接近的叙事;Markdown-native(人类可读)vs 我们的 pkl 是另一个取舍轴。
