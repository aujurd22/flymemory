"""端到端验证 flymemory 常驻 HTTP 服务：握手 + tools/list + 真实读写。"""
import asyncio, time
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

URL = "http://127.0.0.1:8765/mcp"

async def main():
    t0 = time.perf_counter()
    async with streamablehttp_client(URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await asyncio.wait_for(session.initialize(), timeout=10)
            print(f"[1] handshake OK in {time.perf_counter()-t0:.2f}s")
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"[2] tools({len(names)}):", ", ".join(names))
            stats = await session.call_tool("flymemory_stats", {})
            print("[3] stats:", stats.content[0].text)
            mem = await session.call_tool("flymemory_remember",
                                          {"text": "flymemory 常驻 HTTP 模式验证条目 2026-09-16"})
            print("[4] remember:", mem.content[0].text)
            rec = await session.call_tool("flymemory_recall",
                                          {"query": "常驻 HTTP 验证", "top_k": 3})
            print("[5] recall:\n", rec.content[0].text)
            t1 = time.perf_counter()
            await session.call_tool("flymemory_stats", {})
            print(f"[6] second call latency: {(time.perf_counter()-t1)*1000:.0f}ms")

asyncio.run(main())
