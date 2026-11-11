# 脚本用途：读取 PyPI 元数据，按 Linux 与 Python 标签选择匹配的 CUDA 轮子。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Resolve torch's pinned Linux CUDA wheels using PyPI metadata and wheel tags."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

# 读取候选依赖包的发布元数据。
# 输入：name；version。
def info(name,version):
    r=subprocess.run(['curl','-4','-fLsS','--connect-timeout','10','--max-time','30',f'https://pypi.org/pypi/{name}/{version}/json'],check=True,capture_output=True)
    return json.loads(r.stdout)

torch=info('torch','2.5.1'); wanted=[]
for text in torch['info']['requires_dist']:
    req=Requirement(text)
    if req.marker and not req.marker.evaluate(): continue
    if not (req.name.startswith('nvidia-') or req.name=='triton'): continue
    version=next(s.version for s in req.specifier if s.operator=='==')
    wanted.append((req.name,version))
tags=set(sys_tags())
# 根据目标平台和 Python 标签选择兼容轮子。
# 输入：pair。
# 返回形式含：字段 Path、Size、Sha256、DownloadURL。
def select(pair):
    name,version=pair; payload=info(name,version)
    for r in payload['urls']:
        if r['filename'].endswith('.whl') and tags.intersection(parse_wheel_filename(r['filename'])[3]):
            return {'Path':r['filename'],'Size':r['size'],'Sha256':r['digests']['sha256'],
                    'DownloadURL':r['url'].replace('https://files.pythonhosted.org/','https://pypi.tuna.tsinghua.edu.cn/')}
    raise RuntimeError('No compatible wheel '+name)
with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(select,wanted))
root=Path('runs/cuda-wheels'); root.mkdir(parents=True,exist_ok=True)
(root/'download-manifest.json').write_text(json.dumps(records,indent=2))
print('CUDA wheels',len(records),'bytes',sum(r['Size'] for r in records),flush=True)
