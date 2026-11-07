"""Read saved histories and refresh the recovery ten-epoch progress artifact."""
from pathlib import Path
import datetime
import json
ROOT=Path('runs/2026-11-06/recovery_v2')


def main():
    lines=['# Recovery experiment progress','',f'Updated UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
        '', 'Three GEBCO uncertainty seeds are complete. Official ATL24 paired arms use426 train/102 validation/171 test; all modules active.',
        'Validation metrics below are intermediate, not test results. Historical clear imagery is used for86/426 training and29/171 test points; other inputs are explicit composite fallbacks.','']
    summaries={}
    for arm in ('composite_control','matched_tide'):
        p=ROOT/f'atl24_paired/{arm}/checkpoints/training_history.json'
        if not p.exists():lines += [f'## {arm}', '', 'Pending.',''];continue
        try:h=json.loads(p.read_text())
        except json.JSONDecodeError:continue
        best=min(h,key=lambda r:r['val_rmse'])
        summaries[arm]={'epoch':h[-1]['epoch'],'validation_rmse':h[-1]['val_rmse'],'best_epoch':best['epoch'],'best_validation_rmse':best['val_rmse'],
            'complete':(p.parents[1]/'complete.json').exists()}
        lines += [f'## {arm}', '',f"Latest epoch{h[-1]['epoch']}; best validation{best['val_rmse']:.3f}m at{best['epoch']}.",
            '', '| Epoch | Validation RMSE | Depth loss | NLL | Extinction | Reconstruction |','|---|---:|---:|---:|---:|---:|']
        for row in h:
            if row['epoch']%10==0 or row is h[-1]:
                lines.append(f"| {row['epoch']} | {row['val_rmse']:.3f}m | {row['loss_depth']:.4f} | {row['loss_nll']:.4f} | {row['loss_ext']:.4f} | {row['loss_recon']:.4f} |")
        lines.append('')
    (ROOT/'LIVE_PROGRESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
