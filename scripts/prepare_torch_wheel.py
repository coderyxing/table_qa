"""Select the Python 3.12 Linux PyPI wheel; trust the official PyPI digest."""
import json
from pathlib import Path
import subprocess
root=Path('runs/wheels'); root.mkdir(parents=True,exist_ok=True)
result=subprocess.run(['curl','-fLsS','--max-time','30','https://pypi.org/pypi/torch/2.5.1/json'],capture_output=True,check=True)
payload=json.loads(result.stdout)
name='torch-2.5.1-cp312-cp312-manylinux1_x86_64.whl'
record=next(r for r in payload['urls'] if r['filename']==name)
url=record['url'].replace('https://files.pythonhosted.org/','https://pypi.tuna.tsinghua.edu.cn/')
entry={'Path':name,'Size':record['size'],'Sha256':record['digests']['sha256'],'DownloadURL':url}
(root/'download-manifest.json').write_text(json.dumps([entry],indent=2))
print(name,record['size'],flush=True)
