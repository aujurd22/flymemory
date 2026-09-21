"""ZCode SessionStart(compact) hook: compression-recovery pack.

Fires right after the host compresses a conversation — the moment when the
recent narrative thread is destroyed. Pulls the working trail plus the newest
model-stored conclusions from the flymemory server and injects them back as
additionalContext. Silent no-op on any error (never blocks the session).
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8765/mcp"


def main():
    try:
        raw = sys.stdin.read()  # hook payload; consumed for protocol compliance
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "flymemory_session_pack",
                       "arguments": {"minutes": 180}},
        }).encode()
        req = urllib.request.Request(URL, data=body, headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8", "replace")
        pack = ""
        for line in data.splitlines():
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                obj = json.loads(line)
            except Exception:
                continue
            content = obj.get("result", {}).get("content") or []
            pack = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        if pack.strip() and "Nothing to recover" not in pack:
            print(json.dumps(
                {"additionalContext": "[flymemory 压缩恢复包 — 压缩前的近期轨迹与结论]\n" + pack[:2500]},
                ensure_ascii=False))
    except Exception:
        pass  # silent no-op: the server may be down; never block the session


main()
