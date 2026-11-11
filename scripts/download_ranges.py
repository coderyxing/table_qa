# 脚本用途：按已登记元数据并行下载 HTTP 字节区间，校验分块后合并并核对完整哈希。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Verified HTTP range downloader for the existing ModelScope manifest.

Requires the manifest from download_model.py. Never executes model repository code.
Existing .part bytes are reused only for complete chunks and final SHA256 verifies all.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import quote

CHUNK=8*1024*1024

# 分块读取文件求 SHA-256，避免一次把大权重文件载入内存。
# 输入：path（输入或输出路径）。
# 相关调用：h.hexdigest。
def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''): h.update(b)
    return h.hexdigest()

# 按请求的 URL 或字节区间下载文件，交由后续大小与哈希检查验收。
# 输入：url；start；end；size；target。
# 相关调用：header.read_text。
def fetch(url,start,end,size,target):
    header=target.with_suffix('.headers')
    if target.exists() and target.stat().st_size==end-start+1 and not header.exists(): return
    for attempt in range(12):
        result=subprocess.run(['curl','-4','--http1.1','-fLsS','--connect-timeout','15','--max-time','120',
             '--speed-time','20','--speed-limit','1024','--range',f'{start}-{end}',
             '--max-filesize',str(end-start+1),'-D',str(header),'-o',str(target),url],capture_output=True)
        text=header.read_text(errors='replace') if header.exists() else ''
        ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',text,re.I)
        if result.returncode==0 and ranges and tuple(map(int,ranges[-1]))==(start,end,size) and target.stat().st_size==end-start+1:
            header.unlink(missing_ok=True); return
        print('range_retry',start,end,'attempt',attempt+1,'curl',result.returncode,flush=True)
        time.sleep(min(30,2**attempt))
    raise RuntimeError(f'Range failed or ignored: {start}-{end}; curl={result.returncode}')

# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：(root / 'DOWNLOAD_COMPLETE').write_text、(root / 'download-manifest.json').read_text。
# 异常约束：Unsafe path
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p=argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('directory'); p.add_argument('--workers',type=int,default=8); p.add_argument('--probe',action='store_true'); args=p.parse_args()
    root=Path(args.directory).resolve()
    records=json.loads((root/'download-manifest.json').read_text())
    for record in records:
        name=record['Path']; size=record['Size']; target=(root/name).resolve()
        if not target.is_relative_to(root): raise ValueError('Unsafe path')
        if target.exists() and target.stat().st_size==size and hash_file(target)==record['Sha256']: continue
        if size<CHUNK:
            if not record.get('DownloadURL'): continue
            temporary=target.with_name(target.name+'.part')
            subprocess.run(['curl','-4','-fLsS','--connect-timeout','10','--max-time','90','--retry','2',record['DownloadURL'],'-o',str(temporary)],check=True)
            if temporary.stat().st_size!=size or hash_file(temporary)!=record['Sha256']: raise ValueError('Small artifact hash mismatch')
            temporary.replace(target); continue
        url=record.get('DownloadURL') or f"https://modelscope.cn/api/v1/models/{args.model}/repo?Revision={quote(record['Revision'],safe='')}&FilePath={quote(name,safe='')}"
        folder=target.with_name(target.name+'.chunks'); folder.mkdir(exist_ok=True)
        if args.probe:
            t=time.monotonic(); fetch(url,0,1048575,size,folder/'probe')
            print('RANGE_PROBE_OK',name,'seconds',round(time.monotonic()-t,2),flush=True); return
        if shutil.disk_usage(root).free<2*size+10*1024**3: raise ValueError('Insufficient assembly space')
        partial=target.with_name(target.name+'.part')
        jobs=[(start,min(size-1,start+CHUNK-1),folder/f'{start:014d}') for start in range(0,size,CHUNK)]
        if partial.exists():
            with partial.open('rb') as f:
                for start,end,path in jobs:
                    if end>=partial.stat().st_size: break
                    if not path.exists():
                        f.seek(start); path.write_bytes(f.read(end-start+1))
        # 执行一个独立下载任务，供并行工作池调度。
        # 输入：job。
        def work(job):
            start,end,path=job; fetch(url,start,end,size,path); return end-start+1
        begin=time.monotonic(); done=0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for n in pool.map(work,jobs):
                done+=n
                if done%(CHUNK*8)==0 or done==size:
                    print(name,round(done/size*100,1),'%',round(done/1024**2/max(time.monotonic()-begin,.01),2),'MiB/s',flush=True)
        assembled=target.with_name(target.name+'.assembled')
        with assembled.open('wb') as out:
            for _,_,path in jobs:
                with path.open('rb') as inp: shutil.copyfileobj(inp,out,CHUNK)
        if hash_file(assembled)!=record['Sha256']: raise ValueError('Assembled SHA256 mismatch')
        assembled.replace(target)
        partial.unlink(missing_ok=True)
        for path in folder.iterdir(): path.unlink()
        folder.rmdir()
        print('VERIFIED',name,flush=True)
    missing=[r['Path'] for r in records if not (root/r['Path']).exists()]
    if missing: raise RuntimeError(f'Missing small files: {missing}; rerun download_model.py')
    (root/'DOWNLOAD_COMPLETE').write_text('Per-file SHA256 verified\n')
    print('DOWNLOAD_COMPLETE',flush=True)

if __name__=='__main__': main()
