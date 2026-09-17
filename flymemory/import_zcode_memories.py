"""一次性导入：把 ZCode 文件记忆灌入 flymemory 常驻库（服务停机时离线执行）。

用法: python import_zcode_memories.py
"""
import sys, os, glob, re
sys.path.insert(0, r"D:\projects\flymemory")
sys.path.insert(0, r"D:\projects\flymemory\flymemory")
import sentence_transformers  # noqa: F401  主线程导入 torch，避免工作线程死锁
from flymemory.v3 import SmartMemory, load, save

DB = r"D:\projects\flymemory\flymemory\flymemory_v3.pkl"
MEM_DIR = r"C:\Users\<user>\.zcode\cli\memories\projects\default-6b038e47646e2d45\memory"
HALF_LIFE = 2592000.0  # 30 天

mem = load(DB) if os.path.exists(DB) else SmartMemory(n_bits=4096, decay_half_life=HALF_LIFE)
mem.decay_half_life = HALF_LIFE

for f in sorted(glob.glob(os.path.join(MEM_DIR, "*.md"))):
    base = os.path.splitext(os.path.basename(f))[0]
    if base == "MEMORY":
        continue  # 索引由文件本体导出，跳过避免低粒度重复
    text = open(f, encoding="utf-8").read()
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.S).strip()  # 去掉 frontmatter
    if not text:
        continue
    r = mem.remember(text, tags=["zcode-memory", base])
    print(f"{base:35s} {r['action']:13s} #{r.get('memory_id', '?')}")

save(mem, DB)
print(f"done, total memories: {mem.size}")
