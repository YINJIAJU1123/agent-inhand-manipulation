"""Audit and summarize all four text-grounding closed-loop evaluations."""
import argparse
import json
from pathlib import Path
from text_grounding import HELDOUT_TEMPLATES, FACE_NAMES
p=argparse.ArgumentParser()
p.add_argument('--report-dir',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.report_dir)
cases=['canonical_seed123_offset0','heldout_seed123_offset0','heldout_seed456_offset0','heldout_seed123_offset1']
valid_heldout={t.format(color=c) for t in HELDOUT_TEMPLATES for c in FACE_NAMES}
rows=[]
reports=[]
for name in cases:
    report=json.loads((root/(name+'.json')).read_text())
    records=report['records']
    assert report['episodes']==len(records)>0
    assert report['grounding_uses_target_labels'] is False
    assert report['grounding_method']=='siglip2_text_classifier_v2'
    assert report['semantic_camera'] and report['cached_features'] is None
    if name.startswith('heldout'):
        assert all(r['instruction'] in valid_heldout for r in records)
    n=len(records)
    success=sum(r['success'] for r in records)
    drops=sum(r['drop'] for r in records)
    assert abs(success/n-report['success_rate'])<1e-12
    offset=report['instruction_color_offset']
    commands_correct=sum(r['predicted_face']==(r['face']+offset)%6 for r in records)
    rows.append(dict(case=name,episodes=n,successes=success,success_rate=success/n,
                     drops=drops,drop_rate=drops/n,
                     command_grounding_correct=commands_correct,command_grounding_accuracy=commands_correct/n,
                     unique_instructions=len({r['instruction'] for r in records}),
                     per_face=report['per_face']))
    reports.append(report)
for key in ('checkpoint_sha256','prompt_adapter_sha256','text_cache_sha256','evaluator_sha256','num_envs','max_steps','action_scale','vision_stride'):
    assert len({r[key] for r in reports})==1, key
positive=reports[1]['records']+reports[2]['records']
summary=dict(cases=rows,heldout_aggregate=dict(episodes=len(positive),successes=sum(r['success'] for r in positive),
    success_rate=sum(r['success'] for r in positive)/len(positive),
    drops=sum(r['drop'] for r in positive),drop_rate=sum(r['drop'] for r in positive)/len(positive),
    unique_instructions=len({r['instruction'] for r in positive})),
    negative_control_definition='Changed commanded color; success is scored against ORIGINAL simulator goal, not changed command.',
    integrity_checks_passed=True)
Path(a.output).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
