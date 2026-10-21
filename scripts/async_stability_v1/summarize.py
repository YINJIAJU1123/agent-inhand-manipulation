"""Summarize completed local diagnostics without counting replay as online eval."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/async_stability_v1'
def main():
 result={'online_astra_updated':False,'historical_cold_drops':8,'unique_cold_states':2,'groups':{}}
 for directory in ('cold_diagnostic_20261021','causal_followup_20261021','startup_validation_20261021','oracle_validation_20261021','startup_validation_retry_20261021','placement_validation_20261021'):
  folder=BASE/directory
  if not folder.exists():continue
  rows=[]
  for path in sorted(folder.glob('*/report.json')):
   r=json.loads(path.read_text());a=r['metadata']['args'];row={'job':path.parent.name,'seed':r['metadata']['seed'],'episodes':len(r['episodes']),'drops':r['drops'],'successes':r['successes'],'mode':a['mode'],'initial_lift_m':a.get('initial_lift',0),'max_steps':a['max_steps'],'cases':r['episodes']}
   audit=path.with_name('prefix_audit.json')
   if audit.exists():row['prefix_audit']=json.loads(audit.read_text())
   for suffix in ('before','after'):
    audit=path.with_name(f'geometry_{suffix}.json')
    if audit.exists():
     g=json.loads(audit.read_text());row[f'geometry_{suffix}']=[{'case':x['case'],'convex_overlap_pieces':len(x['overlaps']),'max_intersection_radius_m':max([v['intersection_radius_m'] for v in x['overlaps']],default=0)} for x in g['cases']]
   rows.append(row)
  status=json.loads((folder/'progress.json').read_text())
  result['groups'][directory]={'status':status['status'],'rows':rows}
 pilot=BASE/'goal_switch_pilot_retry_20261021'
 if (pilot/'progress.json').exists():
  result['learning_pilot']={'progress':json.loads((pilot/'progress.json').read_text())}
  if (pilot/'summary.json').exists():result['learning_pilot']['summary']=json.loads((pilot/'summary.json').read_text())
 out=BASE/'local_summary_20261021.json';out.write_text(json.dumps(result,indent=2)+'\n')
 for name,g in result['groups'].items():
  print(name,g['status'])
  for r in g['rows']:print(' ',r['job'],f"success={r['successes']}/{r['episodes']} drop={r['drops']}/{r['episodes']}")
if __name__=='__main__':main()
