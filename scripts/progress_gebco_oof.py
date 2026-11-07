"""Refresh ten-epoch summaries from saved fold histories without GPU work."""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gebco_oof_common import ROOT


def main():
    lines=['# Spatial OOF uncertainty progress','','Fold validation selects checkpoints. OOF errors are evaluated only after each fold finishes.','']
    status={}
    for fold in range(3):
        root=ROOT/f'fold_{fold}';p=root/'checkpoints/training_history.json'
        if not p.exists():status[str(fold)]='pending';continue
        try:h=json.loads(p.read_text())
        except json.JSONDecodeError:continue
        best=min(h,key=lambda a:a['val_rmse'])
        status[str(fold)]={'epoch':h[-1]['epoch'],'best_epoch':best['epoch'],'best_val_rmse':best['val_rmse'],'complete':(root/'complete.json').exists()}
        lines += [f'## Fold {fold}','',f"Best validation {best['val_rmse']:.3f}m at epoch{best['epoch']}.",'',
            '| Epoch | Validation RMSE (m) |','|---|---:|']
        lines += [f"| {a['epoch']} | {a['val_rmse']:.3f} |" for a in h if a['epoch']%10==0 or a is h[-1]]
        lines.append('')
    hp=ROOT/'risk_head/training_history.json'
    if hp.exists():
        h=json.loads(hp.read_text());best=min(h,key=lambda a:a['validation_nll_no_constant'])
        status['risk_head']={'epoch':h[-1]['epoch'],'best_epoch':best['epoch'],'best_validation_nll_no_constant':best['validation_nll_no_constant']}
        lines+=['## Uncertainty correction head','', '| Epoch | Validation NLL, constant omitted |','|---|---:|']
        lines += [f"| {a['epoch']} | {a['validation_nll_no_constant']:.4f} |" for a in h if a['epoch']%10==0 or a is h[-1]]
    (ROOT/'LIVE_PROGRESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
