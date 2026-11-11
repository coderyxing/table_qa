"""Pin official source for inspection without installing or executing it.

Use already fetched GitHub commit/tree JSON. Download only code, README,
requirements and licenses; verify each Git blob ID and record SHA256 hashes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit_dir'); args = p.parse_args(); root = Path(args.audit_dir)
    commit = json.loads((root / 'upstream-commit.json').read_text(encoding='utf-8'))
    tree = json.loads((root / 'upstream-tree.json').read_text(encoding='utf-8'))
    sha = commit['sha']
    if not re.fullmatch(r'[0-9a-f]{40}', sha) or tree.get('truncated'):
        raise ValueError('Complete pinned commit/tree required')
    if tree['sha'] != commit['commit']['tree']['sha']:
        raise ValueError('Commit and tree identity do not match')
    entries = [e for e in tree['tree'] if e['type'] == 'blob' and
               (e['path'].endswith('.py') or PurePosixPath(e['path']).name in {'LICENSE', 'README.md', 'requirements.txt'})]
    destination = root / ('source-' + sha); destination.mkdir(parents=True, exist_ok=True)

    def download(entry):
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in entry['path']:
            raise ValueError('Unsafe source path')
        path = destination.joinpath(*relative.parts); path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/google-research/chain-of-table/{sha}/{relative}'
        if not path.exists():
            partial = path.with_name(path.name + '.partial')
            subprocess.run(['curl.exe', '-4', '--connect-timeout', '15', '--max-time', '60',
                            '--retry', '2', '--fail', '--location', '--silent', '--show-error',
                            url, '--output', str(partial)], check=True)
            data = partial.read_bytes()
        else:
            data = path.read_bytes(); partial = None
        git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if git_blob != entry['sha']:
            raise ValueError(f'Git blob verification failed: {relative}')
        if partial is not None:
            partial.replace(path)
        return {'path': str(relative), 'url': url, 'git_blob_sha1': git_blob,
                'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, entries))
    manifest = {'repository': 'https://github.com/google-research/chain-of-table',
                'commit': sha, 'tree': tree['sha'], 'upstream_commit_date': commit['commit']['committer']['date'],
                'inspection_only': True, 'source_executed': False, 'dependencies_installed': False,
                'data_zip_downloaded': False, 'files': sorted(files, key=lambda r: r['path'])}
    (root / 'source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'commit': sha, 'verified_files': len(files), 'bytes': sum(f['bytes'] for f in files)}))


if __name__ == '__main__': main()
