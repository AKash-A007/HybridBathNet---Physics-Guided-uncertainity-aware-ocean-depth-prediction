"""Resume the interrupted official wheel, verify SHA256, then install in this venv."""
import hashlib
import html
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote

import requests

name = 'torch-2.11.0+cu128-cp314-cp314-win_amd64.whl'
target = Path('runs/2026-11-04/dependencies') / name
target.parent.mkdir(parents=True, exist_ok=True)
old = Path('C:/Users/mujee/AppData/Local/Temp/vscode-agent-codex-sandbox-POWBOM/pip-unpack-7hiajyk4') / name
if not target.exists() and old.exists():
    shutil.copyfile(old, target)
index = requests.get('https://download.pytorch.org/whl/cu128/torch/', timeout=60)
index.raise_for_status()
link = next(html.unescape(value) for value in re.findall(r'href="([^"]+)"', index.text)
            if name in unquote(value))
url, expected = link.split('#sha256=')
offset = target.stat().st_size if target.exists() else 0
print(f'Resuming at {offset:,} bytes', flush=True)
response = requests.get(url, headers={'Range': f'bytes={offset}-'}, stream=True, timeout=120)
if response.status_code != 416:
    response.raise_for_status()
    mode = 'ab' if response.status_code == 206 else 'wb'
    if mode == 'ab' and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
        raise RuntimeError('Unexpected Content-Range')
    with target.open(mode) as out:
        for chunk in response.iter_content(1024 * 1024):
            out.write(chunk)
response.close()
with target.open('rb') as source:
    actual = hashlib.file_digest(source, 'sha256').hexdigest()
if actual != expected:
    raise RuntimeError(f'Wheel SHA256 mismatch: {actual}')
print('Official SHA256 verified; installing CUDA PyTorch', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', str(target),
                '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
print('CUDA installation complete', flush=True)
