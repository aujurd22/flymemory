"""Behavioral rule tests for FlyMemory v3.

These pin the memory *rules* (what to store, when to merge, when to supersede,
how to rank) so future refactors cannot silently change behavior.
"""
import time

import numpy as np
import pytest

from flymemory.v3 import SmartMemory, split_chunks, load, save


# ---------------------------------------------------------------- chunking
def test_split_chunks_one_sentence_per_chunk():
    chunks = split_chunks("第一句讲果蝇视觉。第二句讲打印机色带。第三句问明天天气。")
    assert len(chunks) == 3


def test_split_chunks_long_sentence_split_at_commas():
    text = "这是一个非常长的句子，" * 16 + "结尾。"
    chunks = split_chunks(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 120 for c in chunks)


def test_split_chunks_short_complete_sentence_stands_alone():
    """A short but complete sentence is its own topic — it must NOT be merged
    into the previous chunk just for being short (2026-09-19 regression)."""
    chunks = split_chunks("第一句讲果蝇视觉。第二句讲打印机色带。第三句问明天天气。")
    assert len(chunks) == 3


def test_split_chunks_true_fragments_merge():
    """Fragments of the same (over-long) sentence merge back; the final piece
    with terminal punctuation stands alone."""
    chunks = split_chunks("这是一个用于测试分块规则的相当长的句子，" * 8 + "结尾。")
    assert chunks[-1].endswith("。")
    assert all("结尾" not in c or c is chunks[-1] for c in chunks[:-1]) or len(chunks) == 1


# ------------------------------------------------------------ store / dedup
def test_store_and_recall_basic(mem, warm_model):
    r = mem.remember("用户经营一家 SHEIN 跨境电商店铺，主要卖女装", source="model")
    assert r["action"] == "new"
    hits = mem.recall("用户的电商店铺是卖什么的", top_k=1)
    assert hits and "SHEIN" in hits[0][0].text


def test_exact_duplicate_strengthens(mem, warm_model):
    mem.remember("用户喜欢简洁的界面设计")
    n_before = mem.size
    r = mem.remember("用户喜欢简洁的界面设计")
    assert r["action"] == "strengthened"
    assert mem.size == n_before


def test_paraphrase_may_stay_separate_by_design(mem, warm_model):
    """Measured on the multilingual model: a true paraphrase pair scores 0.622
    while a same-structure pair differing only in a detail ("明天/今天三点开会")
    scores 0.932. NO similarity threshold can merge the former without merging
    the latter — so thresholds only catch near-identical duplicates, semantic
    consolidation is left to the calling model (supersede), and paraphrase
    variants may coexist as separate entries. This test pins that contract."""
    mem.remember("用户喜欢简洁的界面设计，讨厌复杂的布局")
    r = mem.remember("用户偏好简单干净的界面布局")
    # either merged by a future smarter policy, or stored separately — but the
    # information must remain recallable either way
    assert r["action"] in ("new", "merged", "strengthened")
    hits = mem.recall("用户对界面设计有什么偏好", top_k=2)
    texts = " ".join(h[0].text for h in hits)
    assert "界面" in texts


def test_short_parallel_unrelated_sentences_both_stored(mem, warm_model):
    """Regression: on the old English-only model these two unrelated sentences
    scored 0.983 (structure parallel) and the second one was wrongly merged."""
    a = "这一句是关于果蝇视觉系统的假设"
    b = "另一句讲的是打印机色带耗材的库存预警"
    mem.remember(a)
    r = mem.remember(b)
    assert r["action"] == "new"
    assert mem.size == 2


def test_similar_but_opposite_events_both_stored(mem, warm_model):
    """Rule: events with similar wording but opposite meaning must both survive.
    Merging them would rewrite history."""
    e1 = "2026-08-18 SHEIN 卖家账号被黑客接管，后台完全无法登录"
    e2 = "2026-08-22 SHEIN 卖家账号找回成功，重新获得了后台访问权"
    mem.remember(e1)
    mem.remember(e2)
    hits = mem.recall("账号被盗无法登录", top_k=2)
    texts = " ".join(h[0].text for h in hits)
    assert "被黑客接管" in texts


def test_multi_topic_message_chunked(mem, warm_model):
    r = mem.remember_text("今天讨论果蝇的视觉系统假设。另外打印机色带的库存要预警了。顺便记录项目里程碑达成。")
    assert r["chunks"] == 3
    assert r["counts"]["new"] == 3
    top = mem.recall("色带库存预警", top_k=1)[0][0].text
    assert "色带" in top
    top = mem.recall("果蝇视觉系统", top_k=1)[0][0].text
    assert "果蝇" in top


def test_very_short_message_still_stored(mem, warm_model):
    r = mem.remember_text("好的明白了")
    assert r["stored"] is True


# ----------------------------------------------------------- recall ranking
def test_query_chunking_hits_per_topic_memory(mem, warm_model):
    mem.remember("蜜蜂采蜜路径优化算法讨论")
    mem.remember("打印机色带耗材的库存预警")
    hits = mem.recall("蜜蜂采蜜的路径怎么优化，另外色带库存怎么样了", top_k=2)
    texts = " ".join(h[0].text for h in hits)
    assert "蜜蜂" in texts and "色带" in texts


def test_vectorized_matches_bruteforce(mem, warm_model):
    """The vectorized recall must agree with a straightforward brute-force loop."""
    for t in ["果蝇蘑菇体的稀疏编码机制", "打印机色带耗材的库存预警阈值",
              "用户偏好深色主题界面", "跨境外币汇率对账的流程",
              "果蝇视觉系统中的位置编码假设", "周五下午三点的产品评审会"]:
        mem.remember(t)
    mem.remember("果蝇视觉系统的后续实验设计")
    q = "果蝇视觉系统还有什么要做的"
    hits = mem.recall(q, top_k=3)

    # brute force reference
    q_texts = split_chunks(q)
    q_embs = [mem._encode(qt)[1] for qt in q_texts]
    scored = []
    for m in mem.memories:
        sim = max(mem._semantic_similarity(qe, m.embedding) for qe in q_embs)
        scored.append((m, sim * mem._decay_weight(m)))
    scored.sort(key=lambda x: -x[1])
    brute_ids = [m.memory_id for m, _ in scored[:3]]
    vec_ids = [h[0].memory_id for h in hits]
    assert vec_ids == brute_ids


def test_superseded_excluded_but_semantically_close(mem, warm_model):
    old = mem.remember("用户当前使用 Windows 11 办公")["memory_id"]
    new = mem.remember("用户最近把主力系统换成了 Linux")["memory_id"]
    mem.supersede(old, new)
    hits = mem.recall("用户的电脑是什么系统", top_k=3)
    ids = [h[0].memory_id for h in hits]
    assert old not in ids


def test_include_superseded_recovers_history(mem, warm_model):
    old = mem.remember("用户当前使用 Windows 11 办公")["memory_id"]
    new = mem.remember("用户最近把主力系统换成了 Linux")["memory_id"]
    mem.supersede(old, new)
    hits = mem.recall("用户用过什么操作系统", top_k=5, include_superseded=True)
    ids = [h[0].memory_id for h in hits]
    assert old in ids and new in ids


def test_supersede_unknown_id_returns_false(mem):
    assert mem.supersede(99999, 100000) is False


# -------------------------------------------------------------- decay math
def test_power_law_retention_values():
    """R(t) = (1 + t/tau)^-0.5: R(tau)=0.707, R(3*tau)=0.5 — the true half-life
    is 3*tau, NOT tau. This test pins the math so nobody 'fixes' it wrongly."""
    mem = SmartMemory(n_bits=64, decay_tau=3600.0)
    now = time.time()
    e = mem.memories  # empty list ok, we craft entries manually
    from flymemory.v3 import MemoryEntry
    m = MemoryEntry(text="x", response="", embedding=np.zeros(384, dtype=np.float32),
                    timestamp=now, last_accessed=now - 3600.0, access_count=0,
                    tags=[], memory_id=0)
    assert abs(mem._decay_weight(m) - 0.7071) < 0.01
    m2 = MemoryEntry(text="y", response="", embedding=np.zeros(384, dtype=np.float32),
                     timestamp=now, last_accessed=now - 3 * 3600.0, access_count=0,
                     tags=[], memory_id=1)
    assert abs(mem._decay_weight(m2) - 0.5) < 0.01


def test_rehearsal_slows_decay():
    mem = SmartMemory(n_bits=64, decay_tau=3600.0)
    now = time.time()
    from flymemory.v3 import MemoryEntry
    fresh = MemoryEntry(text="a", response="", embedding=np.zeros(384, dtype=np.float32),
                        timestamp=now, last_accessed=now - 9 * 3600.0, access_count=0,
                        tags=[], memory_id=0)
    rehearsed = MemoryEntry(text="b", response="", embedding=np.zeros(384, dtype=np.float32),
                            timestamp=now, last_accessed=now - 9 * 3600.0, access_count=3,
                            tags=[], memory_id=1)
    assert mem._decay_weight(rehearsed) > mem._decay_weight(fresh)


def test_decay_tau_legacy_kwarg_alias():
    mem = SmartMemory(n_bits=64, decay_half_life=123.0)
    assert mem.decay_tau == 123.0


# ---------------------------------------------------------------- lexical
def test_exact_identifier_recall(mem, warm_model):
    """Part numbers / IDs are invisible to embeddings; the lexical channel must
    surface the right entry even when the query carries extra prose."""
    mem.remember("货号 GSO1S615F00NT83 已经关联了一件，等待核实")
    mem.remember("货号 QWZX8765MNK 完成了两件关联，可以发货")
    mem.remember("打印机色带耗材一般每月盘点一次")
    hits = mem.recall("GSO1S615F00NT83 这个货号关联了几件", top_k=2)
    assert "GSO1S615F00NT83" in hits[0][0].text


def test_lexical_channel_does_not_hijack_semantic_recall(mem, warm_model):
    mem.remember("果蝇蘑菇体的稀疏编码机制研究")
    mem.remember("打印机色带耗材的库存预警")
    hits = mem.recall("果蝇的稀疏编码", top_k=1)
    assert "稀疏编码" in hits[0][0].text


# ------------------------------------------------------- provenance source
def test_source_field_recorded(mem, warm_model):
    mem.remember("hook 机械捕获的一条消息", source="hook")
    mem.remember("模型判断值得精存的结论", source="model")
    srcs = {m.source for m in mem.memories}
    assert srcs == {"hook", "model"}


def test_recall_output_carries_source(mem, warm_model):
    mem.remember("模型精存的结论示例", source="model")
    entry, sim, eff = mem.recall("精存结论示例", top_k=1)[0]
    assert entry.source == "model"


# ------------------------------------------------------------ persistence
def test_save_load_roundtrip(mem, warm_model, tmp_path):
    a = mem.remember("第一条记忆：果蝇视觉系统假设", source="model")["memory_id"]
    b = mem.remember("第二条记忆：打印机色带库存预警", source="hook")["memory_id"]
    mem.supersede(a, b)
    p = str(tmp_path / "lib.pkl")
    save(mem, p)
    mem2 = load(p)
    assert mem2.size == 2
    by_id = {m.memory_id: m for m in mem2.memories}
    assert by_id[a].superseded_by == b
    assert by_id[a].source == "model"
    assert by_id[b].source == "hook"
    assert mem2._next_id == mem._next_id


def test_load_legacy_schema(tmp_path, warm_model):
    """Old libraries (decay_half_life key, no source/superseded_by) must load."""
    import pickle
    emb = np.zeros(384, dtype=np.float32)
    emb[0] = 1.0
    data = {
        "memories": [{"text": "旧格式条目", "response": "", "embedding": emb.tolist(),
                      "timestamp": time.time(), "last_accessed": time.time(),
                      "access_count": 1, "tags": ["legacy"], "memory_id": 0}],
        "_next_id": 1, "n_bits": 4096, "decay_half_life": 2592000.0,
    }
    p = str(tmp_path / "old.pkl")
    with open(p, "wb") as f:
        pickle.dump(data, f)
    mem = load(p)
    assert mem.size == 1
    assert mem.decay_tau == 2592000.0
    assert mem.memories[0].source == "auto"


def test_new_schema_written_with_legacy_alias(mem, warm_model, tmp_path):
    import pickle
    mem.remember("兼容性检查条目")
    p = str(tmp_path / "lib.pkl")
    save(mem, p)
    with open(p, "rb") as f:
        data = pickle.load(f)
    assert data["decay_tau"] == data["decay_half_life"]
