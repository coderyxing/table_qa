# 脚本用途：为当前 pip 进程的 HTTP 连接选择 IPv4，绕开服务器 IPv6 连接停滞。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Use IPv4 for pip's HTTP connections when a server's IPv6 DNS stalls.

No TLS changes, no system resolver changes. Arguments are the normal pip args.
"""
import socket
from pip._vendor.urllib3.util import connection
connection.allowed_gai_family=lambda:socket.AF_INET
from pip._internal.cli.main import main
raise SystemExit(main())
