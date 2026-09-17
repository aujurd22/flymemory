"""FlyMemory 常驻服务守护进程：拉起 mcp_v3.py --http，子进程退出后自动重启。

由启动文件夹 flymemory_http.lnk 以 pythonw 调起（无窗口）。
- 服务健康（8765 有监听）时只闲逛，不重复拉起；
- 子进程运行超过 5s 后退出（崩溃/被杀）→ 3s 内重启；
- 子进程秒退（端口被占/启动失败）→ 退避 60s 再试，避免疯狂循环。
"""
import socket, subprocess, sys, time

CHILD = r"D:\projects\flymemory\flymemory\mcp_v3.py"
HOST, PORT = "127.0.0.1", 8765


def port_busy() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def main():
    while True:
        try:
            if port_busy():
                time.sleep(30)  # 已有实例在服务
                continue
            t0 = time.time()
            try:
                subprocess.run([sys.executable, CHILD, "--http"])
            except Exception:
                pass
            time.sleep(3 if time.time() - t0 > 5 else 60)
        except Exception:
            time.sleep(30)  # 守护进程自身绝不退出


main()
