"""Wait for the existing fold job, then run the short dependent stages once."""
import sys
import time
import json
import subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gebco_oof_common import ROOT,update


def main():
    if (ROOT/'results.json').exists():print('Study complete; no work duplicated');return
    print('Waiting for the already running fold job; no second depth training launched',flush=True)
    deadline=time.monotonic()+7200
    while True:
        stage=json.loads((ROOT/'status.json').read_text())['stage']
        if stage=='spatial_folds_failed':raise RuntimeError('Fold training failed; inspect existing log')
        if all((ROOT/f'fold_{k}/complete.json').exists() for k in range(3)) and stage=='spatial_folds_complete':break
        if time.monotonic()>deadline:raise TimeoutError('Fold job did not complete within2hours')
        time.sleep(30)
    subprocess.run([sys.executable,'-u','scripts/train_gebco_oof_head.py'],check=True)
    subprocess.run([sys.executable,'-u','scripts/report_gebco_oof_uncertainty.py'],check=True)
    subprocess.run([sys.executable,'-u','scripts/verify_plot_gebco_oof.py'],check=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update('dependent_stages_failed',error=type(exc).__name__);raise
