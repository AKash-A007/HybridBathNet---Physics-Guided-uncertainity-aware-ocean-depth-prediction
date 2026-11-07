"""Analytical metric checks, source hashes and report asset/link validation."""
from pathlib import Path
import sys
import json
import re
import hashlib
from html.parser import HTMLParser
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compute_study_report_metrics import ROOT,metric


def main():
    toy=metric(np.array([1.,2.,3.]),np.array([2.,4.,6.]),None,np.ones(3,dtype=bool),np.arange(3))
    np.testing.assert_allclose(toy['rmse_m'],np.sqrt(14/3));assert toy['mae_m']==2 and toy['r2']==-6
    np.testing.assert_allclose(toy['accuracy_within_1m_pct'],100/3)
    np.testing.assert_allclose(toy['accuracy_within_2m_pct'],200/3)
    assert toy['mean_predicted_sigma_m'] is None
    metrics=json.loads((ROOT/'evaluation_metrics.json').read_text());sources=json.loads((ROOT/'prediction_provenance.json').read_text())
    if (ROOT/'atl24_prediction_provenance.json').exists():sources+=json.loads((ROOT/'atl24_prediction_provenance.json').read_text())
    for a in sources:assert hashlib.sha256(Path(a['path']).read_bytes()).hexdigest()==a['sha256']
    full=metrics['Full v1']['splits']['test']['0-10'];assert full['n']==75 and full['count_within_1m']==41 and full['count_within_2m']==70
    old=json.loads(Path('runs/2026-11-05/nearshore_hybrid_v1/results.json').read_text())
    np.testing.assert_allclose(metrics['Full v1']['splits']['test']['0-20']['rmse_m'],old['test']['primary_0_20']['rmse'],atol=1e-12)
    for name in ('No uncertainty','Neither'):
        assert metrics[name]['splits']['test']['0-10']['mean_predicted_sigma_m'] is None
    for domain in ('0-10','0-20'):
        a=metrics['Ensemble raw']['splits']['test'][domain]
        for name in ('Ensemble scaled','OOF raw','OOF scaled'):
            np.testing.assert_allclose(metrics[name]['splits']['test'][domain]['rmse_m'],a['rmse_m'],atol=1e-12)
    links=[]
    for file in (ROOT/'HYBRIDBATHNET_STUDY_REPORT.md',ROOT/'LAYER_REFERENCE.md'):
        text=file.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://','#')):continue
            assert (file.parent/target).exists(),target;links.append(target)
    class Checker(HTMLParser):
        def __init__(self):super().__init__();self.tables=0;self.images=0
        def handle_starttag(self,tag,attrs):
            if tag=='table':self.tables+=1
            if tag=='img':self.images+=1
            if tag in ('a','img'):
                d=dict(attrs);value=d.get('href' if tag=='a' else 'src','')
                if value and not value.startswith(('http://','https://','#')):assert (ROOT/value).exists(),value
    check=Checker();check.feed((ROOT/'HYBRIDBATHNET_STUDY_REPORT.html').read_text(encoding='utf-8'))
    report={'analytical_metric_identity_and_tolerance_checks':True,'full_shallow_expected_counts':True,
        'historical_primary_rmse_reproduced':True,'uncertainty_off_reported_unavailable':True,
        'postprocessing_depth_metrics_unchanged':True,'prediction_hashes_unchanged':len(sources),
        'local_markdown_links_checked':len(links),'html_tables':check.tables,'html_images':check.images,
        'scope':'No training or source prediction/checkpoint modification; statistical/algebraic and document checks, not external causal validation'}
    (ROOT/'report_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
