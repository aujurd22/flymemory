"""One-shot: register the program's recurring pitfall lessons into
comp:lessons (via the RUNNING service's MCP endpoint -- never direct pkl
writes). Entries follow the LESSON RULE: trigger words first, declarative
form. Dedup ladder applies server-side, so re-running is safe.

Run:  py -3.13 batch_register_lessons.py
"""
import json
import urllib.request

URL = "http://127.0.0.1:8765/mcp"

LESSONS = [
    ("PowerShell/Git Bash: 在 bash 里内联 PowerShell 命令时引号转义会坏"
     "（曾损坏 server.log 配置导致静默卡死）。教训：复杂 PowerShell 一律"
     "写 .ps1 脚本文件或 py -3.13 -c 执行，绝不在 bash 内联长命令。"),
    ("进程清理/taskkill: 按进程名全杀 python.exe/pythonw.exe 曾连带杀掉"
     "记忆服务和并行会话任务（三次事故）。教训：只按精确 PID 杀，杀前用"
     "wmic 核对命令行；命令行含 flymemory 的进程是记忆服务本体，任何清理"
     "例程不得触碰。"),
    ("flymemory 改库: 服务持有 pkl 的权威内存副本，直写 pkl 会被服务的下次"
     "save 覆盖。教训：改库前 schtasks /End + /Change /DISABLE 停 FlyMemoryGuard"
     "（Git Bash 下双斜杠 //），改完 ENABLE；日常注入走 MCP 接口而非直写。"),
    ("bench harness: grep -c 无匹配时退出码 1，在 && 链里会静默截断后续命令"
     "（曾致补丁未落盘就重跑）。教训：grep 仅作存在性检查时用 if 或 || true "
     "兜底；关键补丁后 ast.parse/语法检查再运行。"),
    ("bench harness: bash heredoc 里的 python 补丁反复因 \\\\n 转义出错"
     "（f-string 被写成真换行）。教训：多行代码补丁用 Write 工具写独立"
     "脚本文件，不用 heredoc 内嵌 python。"),
    ("torch/并发: 满核训练任务存在时，torch 默认线程数会让 CE 前向活锁"
     "（220s+ vs 限 1-4 线程的 0.3s）。教训：bench 与推理脚本一律 "
     "OMP_NUM_THREADS=4 + torch.set_num_threads(4)，CUDA_VISIBLE_DEVICES= "
     "防抢占 GPU；重任务与训练串行。"),
    ("flymemory server.log: pythonw 下 sys.stdout/stderr 为 None，裸 print "
     "直接崩；buffering=0 二进制流让 sys.stderr 写 str 抛 TypeError 静默"
     "卡死。教训：pythonw 脚本顶部把 stdout/stderr 重定向到日志文件"
     "（文本模式、行缓冲）， dreaming 任务曾因此空转。"),
    ("计划任务: FlyMemoryDream 的上下文没有 shell 环境变量，DEEPSEEK_API_KEY "
     "缺失使自动运行每次在蒸馏前退出空转六天（手动带 env 跑掩盖了它）。教训："
     "定时任务验证要手动 Start-ScheduledTask 后看日志，不能只看任务 State。"),
    ("记忆索引: 详情文件持续更新但 MEMORY.md 索引行不同步，新会话按索引回忆"
     "接不上真实状态（intuition 行停在 P51 导致记忆混乱）。教训：重大进展"
     "落详情文件的同时必须刷新 MEMORY.md 对应索引行；索引行写指针+教训注记。"),
    ("跨会话引用: 兄弟仓库的结论可能已被同日会话撤回/降级（hub 曾宣传已撤回"
     "的 G4 复制）。教训：引用其他仓库判定前 git log 核对该仓最新 commit，"
     "不以记忆或转述为准。"),
    ("flymemory server.log: server.log 被残留句柄锁住时自动降级 server.<pid>"
     ".log；transformers 的 logger 在导入时绑定 stderr，事后重定向丢日志；"
     "HF_HUB_OFFLINE 是 huggingface_hub 导入时读取的，须同步改 constants。教训："
     "日志/离线开关都在模块顶部处理，导入后再改无效。"),
    ("评估铁律: 微调后必须在未参与训练的留出集上评估（b8 的 100% 实为 80%）；"
     "用户质疑指标时先怀疑数据泄漏；错标数据不能当评估真值。"),
    ("flymemory 排线: 服务刚启动的预热窗口（端口已开、嵌入器载入中）内工具"
     "调用会等 _mem_lock，hook 的超时可能静默错过该条消息——窄窗口可接受，"
     "但启动后第一条消息的召回缺失是预热不是故障。"),
    ("内存纪律: 32GB 红线——重任务（GPU 训练/大语料/多进程）一律串行，启动前"
     "用 python ctypes 查内存；0.5B 级模型在本机 batch<=4 且需长时预算。"),
    ("跨仓库引用: flyloop 与 flymemory 有同名实验（P-COMBO 两物），转述时"
     "曾混淆。教训：引用兄弟仓判定带仓库名前缀，先查该仓最新 commit。"),
    ("runner 脚本: run_*_with_key.py 系列硬编码 sys.argv 不转发参数（flag "
     "静默失效，跑错模式）。教训：runner 转发 sys.argv[1:]；改 runner 后 "
     "grep 核对。"),
]


def main():
    ok, dup_fail = 0, 0
    for text in LESSONS:
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "flymemory_remember",
                       "arguments": {"text": text, "source": "model",
                                     "tags": "lesson",
                                     "compartment": "lessons"}},
        }).encode()
        req = urllib.request.Request(URL, data=body, headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", "replace")
        text_out = ""
        for line in data.splitlines():
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                obj = json.loads(line)
            except Exception:
                continue
            content = obj.get("result", {}).get("content") or []
            text_out = "\n".join(c.get("text", "") for c in content
                                 if isinstance(c, dict))
        if "[REJECTED" in text_out or "SKIPPED" in text_out:
            dup_fail += 1
        else:
            ok += 1
        print(f"  {text_out[:90]}", flush=True)
    print(f"\nregistered {ok}, skipped/merged {dup_fail} of {len(LESSONS)}")


if __name__ == "__main__":
    main()
