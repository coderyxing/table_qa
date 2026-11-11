"""Extract data-only members from complete official repository archives."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('data/raw'); provenance=[]
for dataset,name in [('fetaqa','fetaqa-full.tar.gz'),('wikitq','wtq-full.tar.gz'),('tabfact','tabfact-full.tar.gz')]:
    archive=root/name
    if not archive.exists(): continue
    try:
        with tarfile.open(archive,'r:gz') as tf:
            members=[]
            for m in tf.getmembers():
                parts=Path(m.name).parts
                if len(parts)<3 or not m.isfile(): continue
                if dataset=='fetaqa': allowed=parts[1]=='data' and m.name.endswith(('.json','.jsonl'))
                elif dataset=='wikitq': allowed=parts[1] in {'data','csv'} and m.name.endswith(('.tsv','.csv'))
                else: allowed=(parts[1]=='tokenized_data' and m.name.endswith('.json')) or (parts[1:3]==('data','all_csv') and m.name.endswith('.csv'))
                if allowed: members.append(m)
            target=root/(dataset+'-official'); target.mkdir(exist_ok=True)
            tf.extractall(target,members=members,filter='data')
            files=list(target.rglob('*.jsonl'))+list(target.rglob('*.json')) if dataset in {'fetaqa','tabfact'} else list(target.rglob('*.tsv'))
            print(dataset,'extracted',len(members),'files',flush=True)
            print('candidate data files',[str(p) for p in files if '/data/' in str(p)][:25],flush=True)
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        provenance.append({'dataset':dataset,'archive':str(archive),'sha256':h.hexdigest()})
    except (tarfile.TarError,EOFError) as e:
        print(dataset,'ARCHIVE_INCOMPLETE',type(e).__name__,flush=True)
(root/'archive-provenance.json').write_text(json.dumps(provenance,indent=2))
