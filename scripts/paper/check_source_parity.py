from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import torch
parser=argparse.ArgumentParser();parser.add_argument('--research-root',type=Path,required=True);args=parser.parse_args()
release=Path(__file__).resolve().parents[2];research=args.research_root.resolve()
sys.path[:0]=[str(release/'scripts/semantic_completion_v1'),str(release/'scripts/agentic'),str(release/'source/BrainCo_DexHand')]
from common import load_student
from adapted_actor import AdaptedActionProposal
from BrainCo_DexHand.algo.agentic.visual_student import VisualStudentBatch
spec=importlib.util.spec_from_file_location('research_actor_adapter',research/'scripts/teacher_onpolicy_eval_v1/adapter.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2);source,text,pack=load_student('cpu')
original=m.FittedActors(source,pack,research/'outputs/teacher_onpolicy_followup_v2/continuation_20261025_r2/fit/train')
published=AdaptedActionProposal(source,pack,release/'checkpoints/student/adapted_action_student.pt')
inputs={};checks=0;observations=0
with torch.inference_mode():
 for seed in [146107,146108]:
  path=research/f'outputs/teacher_onpolicy_followup_v2/continuation_20261025_r2/collection/data/seed_{seed}/data.pt';d=torch.load(path,map_location='cpu',weights_only=True);inputs[str(path.relative_to(research))]=sha(path)
  for ids in (torch.arange(6),torch.arange(len(d['goal_id'])-6,len(d['goal_id']))):
   b=VisualStudentBatch(d['rgb_history'][ids].float(),text[d['goal_id'][ids].long()],d['proprio_history'][ids].float());output=source(b);choices=original.proposals(output,b);observations+=6
   for pattern in range(4):
    known=torch.tensor([bool((i+pattern)%2) for i in range(6)]);active=torch.tensor([bool((i+pattern)%3) for i in range(6)])
    for enabled in [False,True]:
     expected=original.select(choices,'mixed',known & active & enabled)
     actual=published.action(output,b,known,active,enabled=enabled)
     assert torch.equal(expected,actual);checks+=1
assert sha(release/'checkpoints/student/adapted_action_student.pt')=='b02d80830dcd4aebeec14c7d30e8645a47dc168a6ce059165d4744f683359862'
report=dict(status='passed',scope='CPU proposal parity on 24 recorded frame histories, not episodes or task performance. Full asynchronous evaluator parity is not claimed.',recorded_histories=observations,mask_batch_checks=checks,maximum_action_difference=0.,network_requests=0,checkpoint_sha256=published.checkpoint_sha256,reference_sha256=sha(release/'checkpoints/student/viserdex_rgbd_student.pt'),input_sha256=inputs,research_adapter_sha256=sha(research/'scripts/teacher_onpolicy_eval_v1/adapter.py'),published_adapter_sha256=sha(release/'scripts/agentic/adapted_actor.py'),check_script_sha256=sha(Path(__file__)))
(release/'docs/paper_action_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
