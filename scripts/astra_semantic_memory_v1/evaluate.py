"""RGB-only prediction first, then score frozen semantic memory on new episodes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
from memory import SemanticMemory
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise RuntimeError('Checkpoint changed')
    pack=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    if pack.get('architecture')!='verified' or not pack.get('offline_gate_passed'):
        raise RuntimeError('Expected the frozen verified model that passed its offline audit')
    model_source=ROOT/'scripts/astra_symbol_verified_v1/model.py'
    spec=importlib.util.spec_from_file_location('frozen_verified_symbol_model',model_source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.StickerReader().to(a.device).eval();model.load_state_dict(pack['model']);model.requires_grad_(False)
    # Only these RGB tensors and calibrated weights are read before predictions
    # are complete. Physics, visibility masks, layout and outcomes are absent.
    rgb_file=a.source/'sampled_rgb.pt'
    frames=torch.load(rgb_file,map_location='cpu',weights_only=True)
    steps=sorted(int(s) for s in frames)
    if not steps:raise ValueError('No sampled images')
    memory=SemanticMemory(len(frames[str(steps[0])]),pack['threshold'],a.device)
    predictions={};start=time.time()
    with torch.inference_mode():
        for step in steps:
            rgb=frames[str(step)].to(a.device).permute(0,3,1,2)
            output=model(rgb);probability=output['probability']
            snapshot=memory.update(probability,step)
            predictions[step]=dict(probability=probability.cpu(),boxes=output['boxes'].cpu(),
                memory={k:v.cpu() for k,v in snapshot.items()})
    torch.save(predictions,a.output/'predictions.pt')
    save(a.output/'inference.json',dict(status='completed',threshold=pack['threshold'],frames=len(steps),
        rgb_sha256=sha(rgb_file),prediction_sha256=sha(a.output/'predictions.pt'),
        checkpoint_sha256=sha(a.checkpoint),model_source_sha256=sha(model_source),
        memory_source_sha256=sha(Path(__file__).with_name('memory.py')),cloud_calls=0,
        inference_s=time.time()-start,input='Recorded RGB only; no layout or simulator state opened during inference.'))
    # Evaluator labels and survival masks are opened only after predictions save.
    protocol=json.loads((a.source/'protocol.json').read_text())
    state=torch.load(a.source/'states.pt',map_location='cpu',weights_only=True)
    layout=torch.tensor(protocol['original_layout'])
    per_case=[dict(case=i,frames=0,fresh_accepted=0,fresh_correct=0,memory_accepted=0,memory_correct=0,
                   fresh_unique=0,fresh_unique_correct=0,memory_unique=0,memory_unique_correct=0) for i in range(len(layout))]
    phases={};threshold=pack['threshold']
    for step,pred in predictions.items():
        metadata=state[str(step)];active=metadata['active']
        phase=metadata['phase'];counts=phases.setdefault(phase,dict(active_frames=0,fresh_accepted=0,fresh_correct=0,memory_accepted=0,memory_correct=0))
        confidence,fresh=pred['probability'][...,:6].max(-1)
        accepted=confidence>=threshold;stored=pred['memory']['symbol'];known=stored>=0
        for i in range(len(layout)):
            if not active[i]:continue
            r=per_case[i];r['frames']+=1;counts['active_frames']+=1
            for name,mask,values in [('fresh',accepted[i],fresh[i]),('memory',known[i],stored[i])]:
                count=int(mask.sum());correct=int(((values==layout[i])&mask).sum())
                r[name+'_accepted']+=count;r[name+'_correct']+=correct
                counts[name+'_accepted']+=count;counts[name+'_correct']+=correct
                # Public virtual query = symbol indexed by case, fixed before
                # any observation. This is identification, not task execution.
                colors=((values==i)&mask).nonzero().flatten()
                unique=len(colors)==1
                r[name+'_unique']+=int(unique)
                r[name+'_unique_correct']+=int(unique and int(layout[i,int(colors[0])])==i)
    totals={k:sum(r[k] for r in per_case) for k in per_case[0] if k!='case'}
    for name in ('fresh','memory'):
        totals[name+'_precision']=totals[name+'_correct']/max(1,totals[name+'_accepted'])
        totals[name+'_unique_precision']=totals[name+'_unique_correct']/max(1,totals[name+'_unique'])
        totals[name+'_query_coverage']=totals[name+'_unique']/max(1,totals['frames'])
    report=json.loads((a.source/'report.json').read_text())
    result=dict(status='completed',seed=protocol['seed'],threshold=threshold,nominal_cases=len(layout),
        sampled_active_frames=totals['frames'],totals=totals,by_case=per_case,by_phase=phases,
        early_drops=report['drops'],checkpoint_sha256=sha(a.checkpoint),cloud_calls=0,
        note='New physical seeds and unused permutations; frozen weights and thresholds. Binding accuracy and coverage on active frames only, with deaths separately retained. Fixed public skills, no Astra or autonomous semantic task success.')
    save(a.output/'result.json',result)
    print('FRESH_MEMORY_RESULT '+json.dumps({k:v for k,v in result.items() if k not in ('by_case','by_phase')}),flush=True)

if __name__=='__main__':main()
