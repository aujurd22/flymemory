"""FlyMemory supervisor: keep the persistent MCP server alive.

Launched headless (pythonw / flymemory-host.exe) from a Startup shortcut or a
scheduled task. Behavior:
  - if the server port is already serving, idle (never double-launch);
  - if the child dies after running >5s (crash/killed) → restart within 3s;
  - if the child exits instantly (port busy / bad start) → back off 60s;
  - the supervisor itself never exits.
"""
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "mcp_v3.py")
HOST, PORT = "127.0.0.1", 8765


def port_busy() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def main():
    while True:
        try:
            if port_busy():
                time.sleep(30)  # an instance is already serving
                continue
            t0 = time.time()
            try:
                subprocess.run([sys.executable, CHILD, "--http"])
            except Exception:
                pass
            time.sleep(3 if time.time() - t0 > 5 else 60)
        except Exception:
            time.sleep(30)  # the supervisor itself must never die


main()
