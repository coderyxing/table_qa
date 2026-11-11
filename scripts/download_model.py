# 脚本用途：从 ModelScope 元数据下载公开 Qwen 文件，逐文件验证大小与 SHA-256。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Download public Qwen files from ModelScope with per-file SHA256 checking.

Uses curl for network and standard library for verification. Never executes repo code.
Manifest records the file revisions returned by the API; each URL pins that revision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote


# 计算文件或产物内容摘要，供模型、输入和协议绑定。
# 输入：path（输入或输出路径）。
# 相关调用：h.hexdigest。
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：(root / 'DOWNLOAD_COMPLETE').write_text、manifest.write_text、manifest.read_text。
# 异常约束：Only public Qwen models enabled
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p=argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('output'); p.add_argument('--metadata-only',action='store_true'); args=p.parse_args()
    if not args.model.startswith('Qwen/'): raise ValueError('Only public Qwen models enabled')
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    manifest=root/'download-manifest.json'
    if manifest.exists(): records=json.loads(manifest.read_text())
    else:
        url=f'https://modelscope.cn/api/v1/models/{args.model}/repo/files?Revision=master&Recursive=true'
        result=subprocess.run(['curl','-fLsS','--connect-timeout','15','--max-time','60',url],check=True,capture_output=True)
        payload=json.loads(result.stdout); assert payload['Code']==200
        records=[r for r in payload['Data']['Files'] if r['Type']=='blob' and
                 (r['Path'].endswith(('.json','.safetensors','.txt','.md')) or r['Path']=='LICENSE')]
        if not records or not any(r['Path'].endswith('.safetensors') for r in records):
            raise ValueError('No model weights found')
        manifest.write_text(json.dumps(records,indent=2),encoding='utf-8')
    missing=sum(r['Size'] for r in records if not (root/r['Path']).exists())
    if shutil.disk_usage(root).free < missing+15*1024**3:
        raise ValueError('Insufficient disk: keep at least 15 GiB free')
    for r in records:
        if args.metadata_only and r['Path'].endswith('.safetensors'): continue
        path=(root/r['Path']).resolve()
        if not path.is_relative_to(root): raise ValueError('Unsafe repository path')
        expected=r['Sha256']
        if not expected: raise ValueError('Missing expected SHA256')
        if path.exists() and path.stat().st_size==r['Size'] and sha(path)==expected:
            print('verified',r['Path'],flush=True); continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.part')
        url=f"https://modelscope.cn/api/v1/models/{args.model}/repo?Revision={quote(r['Revision'],safe='')}&FilePath={quote(r['Path'],safe='')}"
        print('downloading',r['Path'],r['Size'],flush=True)
        # Restart requests from the current file size after a stalled connection.
        for attempt in range(8):
            result=subprocess.run(['curl','-fL','--connect-timeout','15','--max-time','1800',
                        '--speed-limit','1024','--speed-time','30',
                        '--continue-at','-','--output',str(temporary),url])
            if result.returncode==0: break
            print('retry',attempt+1,r['Path'],flush=True)
        else: raise RuntimeError(f"Download stalled repeatedly: {r['Path']}")
        if temporary.stat().st_size!=r['Size'] or sha(temporary)!=expected:
            raise ValueError(f"Size/SHA256 mismatch: {r['Path']}")
        temporary.replace(path)
    if args.metadata_only:
        print('METADATA_READY',root,flush=True); return
    (root/'DOWNLOAD_COMPLETE').write_text('Per-file SHA256 verified\n')
    print('COMPLETE',root,flush=True)


if __name__=='__main__': main()
