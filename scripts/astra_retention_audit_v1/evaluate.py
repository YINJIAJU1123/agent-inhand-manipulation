"""Audit action values on the exact sensor-eligible runtime cohort, without refitting."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]/'astra_retention_v1'))
import value_model
from value_model import RetentionValue,RATES
p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--threshold',type=float,required=True);p.add_argument('--base-sha256',required=True);a=p.parse_args()
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=json.loads((a.training/'result.json').read_text());protocol=json.loads((a.training/'protocol.json').read_text())
assert result['checkpoint_sha256']==digest(a.training/'head.pt')
pack=torch.load(a.training/'head.pt',map_location='cpu',weights_only=True)
assert pack['base_sha256']==a.base_sha256 and tuple(pack['rates'])==RATES
assert protocol['model_source_sha256']==digest(value_model.__file__)
cache=torch.load(a.training/'features.pt',map_location='cpu',weights_only=True)
model=RetentionValue().eval();model.load_state_dict(pack['model']);torch.set_num_threads(2)
metrics={};selected_fixed=None
with torch.inference_mode():
    for split in ('calibration','audit'):
        data=cache['data'][split];feature=data['feature'];chosen=model(feature)['value'].argmax(-1)
        # Frozen features are query_hidden[96], ordinal[7], log_error[1], action_gap[21].
        # Runtime uses sigmoid(ordinal[2]); no ground-truth label affects eligibility.
        ready=feature[:,98].sigmoid()>=a.threshold
        quality=data['quality'];drops=data['drop'];rows=torch.arange(len(feature));groups={}
        for name,mask in [('all',torch.ones(len(feature),dtype=torch.bool)),('runtime_eligible',ready),('not_eligible',~ready)]:
            idx=rows[mask];count=len(idx)
            groups[name]=dict(states=count,chosen_quality=float(quality[idx,chosen[idx]].mean())if count else None,
                 chosen_drops=int(drops[idx,chosen[idx]].sum()),choices=[int((chosen[idx]==i).sum())for i in range(3)],
                 fixed_quality=quality[idx].mean(0).tolist()if count else None,fixed_drops=drops[idx].sum(0).tolist(),
                 oracle_quality=float(quality[idx].max(-1).values.mean())if count else None)
        if split=='calibration'and ready.any():selected_fixed=int(quality[ready].mean(0).argmax())
        by_seed={}
        for seed in sorted({k['seed']for k in cache['keys'][split]}):
            mask=torch.tensor([k['seed']==seed for k in cache['keys'][split]])&ready;idx=rows[mask]
            by_seed[str(seed)]=dict(eligible_states=len(idx),learned_quality=float(quality[idx,chosen[idx]].mean())if len(idx)else None,
                 fixed_quality=float(quality[idx,selected_fixed].mean())if len(idx)and selected_fixed is not None else None,
                 learned_drops=int(drops[idx,chosen[idx]].sum()))
        metrics[split]=dict(groups=groups,by_seed=by_seed)
cal=metrics['calibration']['groups']['runtime_eligible'];audit=metrics['audit']['groups']['runtime_eligible']
enough=cal['states']>=8 and audit['states']>=8
passed=bool(enough and selected_fixed is not None and audit['chosen_quality']>audit['fixed_quality'][selected_fixed]+.03 and audit['chosen_drops']<=audit['fixed_drops'][selected_fixed])
out=dict(metrics=metrics,threshold=a.threshold,calibration_selected_fixed_rate=RATES[selected_fixed]if selected_fixed is not None else None,
          original_offline_gate_passed=result['offline_gate_passed'],runtime_cohort_gate_passed=passed,sufficient_eligible_states=enough,
          checkpoint_sha256=result['checkpoint_sha256'],features_sha256=digest(a.training/'features.pt'),source_sha256=digest(__file__),
          cloud_calls=0,note='Additional diagnostic scoped to actual runtime activation. No model, threshold, checkpoint, or dataset has been refit. Does not replace closed-loop and live Astra validation.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print('RETENTION_RUNTIME_AUDIT '+json.dumps(out),flush=True)
