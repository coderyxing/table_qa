# 脚本用途：检查下载端点与服务器连接情况，输出经过限制的网络诊断。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Read-only network diagnostics; deliberately never prints proxy credentials."""
import os
import ssl
import urllib.parse
import urllib.request
for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
    value=os.environ.get(name)
    if value: print(name,urllib.parse.urlsplit(value).hostname,flush=True)
for url in ('https://download.pytorch.org/whl/cu121/torch/','https://pypi.tuna.tsinghua.edu.cn/simple/torch/'):
    try:
        with urllib.request.urlopen(url,timeout=12) as r:
            print(url,r.status,len(r.read(1024)),flush=True)
    except Exception as exc: print(url,type(exc).__name__,flush=True)
