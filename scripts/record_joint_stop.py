"""Record an explicitly user-requested stop; never launch or resume training."""
import datetime
import json
from pathlib import Path

root=Path('runs/2026-11-05/joint_v1')
history=json.loads((root/'control/checkpoints/training_history.json').read_text())
best=min(history,key=lambda row:row['val_rmse'])
state={'stage':'stopped_by_user','experiment':'control','last_completed_epoch':history[-1]['epoch'],
       'best_control_validation_rmse':best['val_rmse'],'best_control_epoch':best['epoch'],
       'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'reason':'User requests stopping training and discussing ICESat-2/nearshore evaluation before proceeding',
       'not_started':['mask_aware','joint_composite','joint_full_coverage','full_joint','sentinel_only_training'],
       'preserved':['control/checkpoints/last_training.pt','control/checkpoints/best_ema.pt',
                    'features/patches.npz','features/patches_before_invalid_label_fill.npz'],
       'note':'Sentinel-only data preparation and backward smoke check completed, but no optimizer training run started.'}
(root/'status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
(root/'STOPPED.md').write_text('# Stopped at user request\n\n'+
    f"Control completed {history[-1]['epoch']} epochs; best validation RMSE {best['val_rmse']:.4f} m at epoch {best['epoch']}.\n\n"+
    'The control has reconstruction weight zero. No integrated variant/full model or Sentinel-only training was completed. '
    'Checkpoints and data are preserved. Resume only after a new explicit user instruction. '
    'This partial pilot is not a new selected result and has no new test evaluation.\n',encoding='utf-8')
marker='<!-- JOINT_JOB_STATUS -->'; end='<!-- /JOINT_JOB_STATUS -->'
for path in (Path('PROJECT_CONTEXT.md'),Path('README.md')):
    text=path.read_text(encoding='utf-8')
    note=f'{marker}\nJoint training **stopped at user request** during control epoch {history[-1]["epoch"]}. '
    note+='Checkpoints preserved; full joint and Sentinel-only training not started. '
    note+='Current scope is ICESat-2/nearshore discussion and read-only diagnostics. '
    note+='[Stop details](runs/2026-11-05/joint_v1/STOPPED.md).\n'+end
    if marker in text:
        a=text.index(marker); b=text.index(end,a)+len(end); text=text[:a]+note+text[b:]
    else: text=text.split('\n',1)[0]+'\n\n'+note+'\n'+text.split('\n',1)[1]
    path.write_text(text,encoding='utf-8')
print(json.dumps(state,indent=2))
