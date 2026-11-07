"""Small persistent stage tracker for the authorized recovery components."""
from pathlib import Path
import datetime
import json
import os

ROOT=Path('runs/2026-11-06/recovery_v2')


def update(component,phase,**extra):
    ROOT.mkdir(parents=True,exist_ok=True)
    state={'component':component,'phase':phase,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    path=ROOT/f'{component}_status.json'
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(path)
    states=[]
    for name in ('atl24','pairing','visibility','uncertainty','tides'):
        p=ROOT/f'{name}_status.json'
        states.append(f"{name}: {json.loads(p.read_text(encoding='utf-8'))['phase'] if p.exists() else 'pending'}")
    marker='<!-- RECOVERY_V2_STATUS -->';end='<!-- /RECOVERY_V2_STATUS -->'
    note=marker+'\nRecovery study: **'+', '.join(states)+'**. '
    note+='[Work directory](runs/2026-11-06/recovery_v2/) | [Final report](runs/2026-11-06/recovery_v2/REPORT.md).\n'+end
    for p in (Path('README.md'),Path('PROJECT_CONTEXT.md')):
        text=p.read_text(encoding='utf-8')
        if marker in text:
            a=text.index(marker);b=text.index(end,a)+len(end);text=text[:a]+note+text[b:]
        else:
            a=text.index('\n')+1;text=text[:a]+'\n'+note+'\n'+text[a:]
        p.write_text(text,encoding='utf-8')

