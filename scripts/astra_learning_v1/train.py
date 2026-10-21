"""Offline representation fitting, then advantage-weighted bounded regression.

Only historical TRAIN collections are loaded. Whole reset seed 930103 is a
DEVELOPMENT split, not an untouched test: collection policies evolved across
these seeds. Final claims require the separate fresh live Astra pilot.
"""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/agentic'))
import torch
from torch import nn
from recovery_controller_v2 import advantages
from model import LearnedController
DATA=ROOT/'outputs/recovery_factorial/ablation_20261019/train'
BASE=ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt'
ADAPTER=ROOT/'outputs/visual_student_v3/checkpoints/siglip2_text_classifier_v2.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(191019)
    state=dict(pid=os.getpid(),status='building_dataset',started=time.time())
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state);print(json.dumps(state),flush=True)
    allrows={k:[] for k in ['obs','goal','act','adv','returns','label','seed','episode']};manifest=[];episode=0
    for p in sorted(DATA.glob('*/*/draw_*.pt')):
        c=torch.load(p,map_location='cpu',weights_only=True);r=c['rollout']
        if c['seed'] not in [930100,930101,930102,930103]:raise ValueError('Unexpected training seed')
        audit=json.loads(p.with_suffix('').joinpath('eval_trials.json').read_text())
        tensors={k:torch.stack([t[k] for t in r]) for k in ['obs','goal','action','value','reward','done','valid']}
        adv=advantages(tensors['reward'],tensors['value'],tensors['done'],tensors['valid'])
        manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),seed=c['seed'],arm=c['training_arm']))
        for j,row in enumerate(audit):
            times=tensors['valid'][:,j].nonzero().flatten()
            if not len(times):continue
            # Identical number of sampled frames per episode prevents long waits
            # or failures from becoming implicit episode weights.
            times=times[torch.linspace(0,len(times)-1,24).round().long()]
            label=0 if row['success'] else (1 if row['wait_drop'] or row['resume_drop'] else 2)
            for key,source in [('obs','obs'),('goal','goal'),('act','action')]:allrows[key].append(tensors[source][times,j])
            allrows['adv'].append(adv[times,j]);allrows['returns'].append((adv+tensors['value'])[times,j]/12.)
            for k,v in [('label',label),('seed',c['seed']),('episode',episode)]:allrows[k].append(torch.full((24,),v,dtype=torch.long))
            episode+=1
    d={k:torch.cat(v) for k,v in allrows.items()};tr=d['seed']!=930103;va=~tr
    assert tr.any() and va.any() and not set(d['episode'][tr].tolist())&set(d['episode'][va].tolist())
    canonical=torch.load(ADAPTER,map_location='cpu',weights_only=True)['canonical_embeddings'].float()
    m=LearnedController(canonical)
    base=torch.load(BASE,map_location='cpu',weights_only=True);m.reference.load_state_dict(base['model'])
    m.encoder.obs_mean.copy_(d['obs'][tr].mean(0));m.encoder.obs_scale.copy_(d['obs'][tr].std(0).clamp_min(.05))
    with torch.no_grad():
        exact=torch.equal(m.raw_action(d['obs'][:48],d['goal'][:48]),m.reference.distribution(d['obs'][:48],d['goal'][:48]).mean)
    assert exact
    protocol=dict(version='temporal_outcome_awr_v1',train_seeds=[930100,930101,930102],development_seed=930103,frames_per_episode=24,
        episodes=episode,train_episodes=int(tr.sum())//24,development_episodes=int(va.sum())//24,
        reference_sha256=sha(BASE),adapter_sha256=sha(ADAPTER),inputs=['4 compact RGB-D frames','proprioception','student action proposal','command history','observed request clocks','delivered goal embedding'],
        excluded_inputs=['privileged simulator state','true target from scorer','future response delay','future frames'],
        outcome_classes=['success','drop','non_drop_failure'],
        caveat='Outcome labels are from mixed collection policies; development reset seeds are excluded from fitting but were seen by historical collection policies. No held-out task-success claim.',
        policy='Frozen temporal encoder after outcome/return supervision; advantage-weighted behavior regression plus V3 action regularization; bounded raw correction <=0.1.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},data=manifest)
    save(out/'protocol.json',protocol)
    trainids=tr.nonzero().flatten();validids=va.nonzero().flatten()
    prior=nn.functional.one_hot(d['label'][tr],3).float().mean(0)
    def metrics():
        loss=bs=vl=0.;n=0
        with torch.no_grad():
            for ids in validids.split(512):
                logits,v=m.encoder.predictions(d['obs'][ids],d['goal'][ids]);y=d['label'][ids];one=nn.functional.one_hot(y,3)
                loss+=float(nn.functional.cross_entropy(logits,y,reduction='sum'));bs+=float(((logits.softmax(-1)-one)**2).sum());vl+=float(((v-d['returns'][ids])**2).sum());n+=len(ids)
        return dict(ce=loss/n,brier=bs/n,value_mse=vl/n)
    baseline_brier=float(((prior-nn.functional.one_hot(d['label'][va],3))**2).sum(-1).mean())
    opt=torch.optim.Adam(m.encoder.parameters(),lr=3e-4);best=float('inf');history=[]
    for epoch in range(20):
        losses=[]
        for ids in trainids[torch.randperm(len(trainids))].split(512):
            logits,v=m.encoder.predictions(d['obs'][ids],d['goal'][ids])
            loss=nn.functional.cross_entropy(logits,d['label'][ids])+.1*nn.functional.smooth_l1_loss(v,d['returns'][ids])
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.encoder.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        met=metrics();history.append(dict(epoch=epoch,train_loss=sum(losses)/len(losses),**met))
        if met['ce']<best:best=met['ce'];best_state=copy.deepcopy(m.encoder.state_dict());best_epoch=epoch
        progress(status='representation_training',epoch=epoch,development=met)
    m.encoder.load_state_dict(best_state);m.encoder.requires_grad_(False)
    torch.save(dict(model=m.state_dict(),canonical=canonical,protocol=protocol),out/'representation_only.pt')
    # Frozen features and detached advantages cannot leak outcome/value gradients
    # into the actor. Use train statistics only for offline regression weights.
    z=[];refs=[]
    with torch.no_grad():
        for ids in torch.arange(len(d['obs'])).split(512):
            z.append(m.encoder(d['obs'][ids],d['goal'][ids]));refs.append(m.reference.distribution(d['obs'][ids],d['goal'][ids]).mean)
    z=torch.cat(z);ref=torch.cat(refs)
    amean=d['adv'][tr].mean();asd=d['adv'][tr].std().clamp_min(1e-5)
    weights=((d['adv']-amean)/asd).clamp(-2,2).exp()
    opt=torch.optim.Adam(m.correction.parameters(),lr=2e-4)
    def actor_loss(ids):
        delta=m.correction_from_latent(z[ids]);error=((ref[ids]+delta-d['act'][ids])/.05).square().mean(-1)
        return (weights[ids]*error).mean()+2*(delta/.05).square().mean()
    with torch.no_grad():actor_baseline=float(actor_loss(validids));best=actor_baseline;actor_best=copy.deepcopy(m.correction.state_dict());actor_epoch=-1
    ah=[]
    for epoch in range(12):
        for ids in trainids[torch.randperm(len(trainids))].split(512):
            loss=actor_loss(ids);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.correction.parameters(),1.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():score=float(actor_loss(validids));delta=m.correction_from_latent(z[validids]);rms=float(delta.square().mean().sqrt())
        ah.append(dict(epoch=epoch,development_objective=score,raw_delta_rms=rms))
        if score<best:best=score;actor_best=copy.deepcopy(m.correction.state_dict());actor_epoch=epoch
        progress(status='correction_training',epoch=epoch,development_objective=score,raw_delta_rms=rms)
    m.correction.load_state_dict(actor_best)
    with torch.no_grad():
        x=d['obs'][validids[:512]];g=d['goal'][validids[:512]]
        a0=m.raw_action(x,g);other=canonical[(torch.cdist(g,canonical).argmin(-1)+3)%6]
        goal_effect=float((m.raw_action(x,other)-a0).abs().mean())
        learned_goal_effect=float((m.correction_from_latent(m.encoder(x,other))-m.correction_from_latent(m.encoder(x,g))).abs().mean())
    result=dict(representation_epoch=best_epoch,development=metrics(),prior_brier=baseline_brier,actor_epoch=actor_epoch,
        actor_baseline_objective=actor_baseline,actor_selected_objective=best,zero_initial_correction_exact=exact,
        total_raw_goal_sensitivity=goal_effect,learned_correction_goal_sensitivity=learned_goal_effect,
        live_success_rate=None,interpretation='Development prediction/regression metrics only. Live control improvement remains untested.',representation_history=history,actor_history=ah)
    torch.save(dict(model=m.state_dict(),canonical=canonical,protocol=protocol,metrics=result),out/'controller_final.pt')
    save(out/'metrics.json',result);progress(status='completed',checkpoint=str(out/'controller_final.pt'),checkpoint_sha256=sha(out/'controller_final.pt'))
if __name__=='__main__':main()
