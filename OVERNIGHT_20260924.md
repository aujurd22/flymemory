# 过夜研究交接 2026-09-24(凌晨)

指令:flymemory 接续任务,研究到 9 点。当前 02:20,已完成核心成果链,文档供早上接续。

## 一夜成果链(全部推送,CI 绿)

| 提交 | 内容 |
|---|---|
| 0b773f2 | cross-encoder rerank 集成(oracle +4.4pp)+ S 库时间戳修复 |
| 9ee71d0 | S 库 20 万条全量六臂:rerank 45.8% vs RRF 40.8%(+5.0pp) |
| 84bbe63 | 外部评审五项修复(consolidate/supersede/forget/文案/评分语义) |
| 4218e53 | state-aware 重排负结果(融合后 eff 重排有害) |
| 7f3cc1e + 4ad09f7 | README 语义自洽修正(production=RRF,decay=maintenance) |
| f2a6af3 | **Phase 1 Memory Judgment Benchmark**(38 场景) |
| a5293ab | **Phase 2 五臂端到端**(answer 级) |
| 7388e86 | **engine dedup 修复**(5/20 丢状态更新 → 0/20) |
| c278bac + 2f6dbbe + 84324d9 | LME e2e + 粒度 A/B + 五臂扩展 |

## 核心数字(全部可复现,脚本在仓库根)

- 检索层(S 库 20 万条,evidence-hit@3):dense 27.4 / bm25 33.6 / RRF 40.8 / **RRF+CE 45.8**,oracle@10 55.4
- 判断层(38 场景):DeepSeek supersede P/R **1.00/1.00**(执行感知)、forget 1.00/1.00、**unnecessary mutation 0/10**
- 端到端(30 状态场景):naive RAG **13-17% stale 答案**;FlyMemory(oracle 与 autonomous)**均 0% stale、87% current**;no-memory 93% 只会答"不知道"
- 粒度 A/B(50 题):turn-only 32% / **替换式整合 22%(负结果)** / **叠加式 38%(+6pp)**
- **粒度 overlay 全量 500 题验证:42.8% strict / 44.9% 加权,对 turn-only 37.0%/39.6% = +5.8pp strict,全量保持**(9d75eb7)
- LME e2e 全量 500 题:strict 37.0%(185/500)、加权 39.6%;归因 hit@5=73.2%,命中后 43% vs 未命中 22%
- LME e2e 归因:hit@5=64% × 回答转化 50% = 32% strict;top-k 10 无增益
- 状态保真审计:修复前 5/20 真实状态更新被 dedup 静默丢弃;修复后 20/20 正确

## 设计原则获得测量支持

1. "Raw entries are never deleted — abstraction without loss":替换式整合 22% vs 叠加式 38%,原则被实验证实。
2. "Decay 是维护信号不是排序信号":融合后 eff 重排有害(oracle −0.8/−2.0pp)。
3. "能机械验证的不交给模型":机械有效性走代码,unsupported inference 才走 judge。

## 已知问题 / 下一步(按优先级)

1. **judge 无 stale 维度**:cons_08 把已取代的旧耗材写进"当前状态"总结,judge 判 supported(字面忠实≠当前正确)。修法:data/memory_judgment.json 的 consolidation 场景加 stale_evidence 标注,judge prompt 加维度。
2. **flymemory_remember 工具回喂不区分 new/merged**:Phase 1.5 的模型困惑源(mcp_v3 已区分 action,但 merged 返回的 id 是已有条目)。修 mcp_v3.py 回喂文本。
3. **merge_inplace 的历史丢失**(粒度 A/B 与 dedup 修复的副产品,6/20):旧文本被原地改写,include_superseded 救不回。设计权衡:可给 merge 加 supersede lineage(旧文本存 tombstone)。
4. **judgment 数据集扩到 100+** → **已完成(v1.2,100 场景 = 48 sup/30 noop/14 cons/8 frt,ed9a763)**:DeepSeek supersede P/R 1.00/1.00(n=50)、forget 1.00/1.00、mutation 0/30;consolidation evidence 8/14(57%,仍是改进目标)。
4b. **merge lineage tombstone 已实现**(ad54abe,从 RFC 转正):rewriting merge 保留旧文本为 superseded tombstone,include_superseded 可恢复历史;52 测试 + fidelity/contradiction/QA 回归全绿,服务已重启生效。
4c. **dream.py 已实现并真实验证**(09-25 凌晨):idle-time consolidation(rfc §9)——读 RECENT 90 分钟窗口 → DeepSeek 蒸馏 → 逐条幻觉审计门禁 → 经 MCP flymemory_remember 写入(生产正确路径,直写 pkl 会与服务内存副本冲突)。实测 8 条整合条目入库(NEW/MERGED 正确)。部署建议:Windows 计划任务每小时跑一次(空闲时段),或手动。**注意**:90 分钟窗口可能混入其他并行会话的记忆(flymemory 是跨会话共享库)——dreaming 整理的是"全部最近记忆"而非单会话。
5. **Phase 4 长期运行验证**(评审定调的最终 thesis 检验)。

## 运维状态

- 服务:33920(flymemory-host.exe,8765 LISTENING),跑修复后 engine,守护已启用
- HEAD:c278bac;CI 全绿;生产库 pkl 未动(修复只影响未来写入,无需迁移)
- benchmark 纪律:engine 已再次冻结(state-fidelity 修复是正确性 bug 例外);reports/ 在 .gitignore

## 复现命令

```bash
PY="C:\Users\djr82\AppData\Local\Programs\Python\Python313\python.exe"
$PY bench_state_fidelity.py                      # 状态保真审计
$PY bench_memory_judgment.py --actor oracle      # harness 验证(应全 1.0)
DEEPSEEK_API_KEY=sk-... $PY bench_memory_judgment.py --actor deepseek --judge
DEEPSEEK_API_KEY=sk-... $PY bench_memory_judgment_tools.py
DEEPSEEK_API_KEY=sk-... $PY bench_e2e_answer.py
DEEPSEEK_API_KEY=sk-... $PY bench_lme_e2e.py --sample 50
DEEPSEEK_API_KEY=sk-... $PY bench_granularity.py --mode overlay
```
