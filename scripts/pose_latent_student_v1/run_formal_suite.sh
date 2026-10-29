#!/usr/bin/env bash
# Run the registered 320-trial suite only after the full release is frozen.
# No retries, replacement episodes, checkpoint selection or service startup.
set -Eeuo pipefail
if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Required environment:
  RUNTIME_ROOT                 Independent Newton runtime directory
  SOURCE_DIR                   Frozen evaluator source directory
  FORMAL_RELEASE               Frozen eight-condition release JSON
  SUITE_ID                     Unique output prefix, letters/digits/_/- only
  EXECUTION_ACTOR_CHECKPOINT    Selected execution actor from the release
Optional:
  ACTOR_CHECKPOINT              Defaults to the existing v1 waiting actor
This runs GPU queues 0, 1 and 5 and serializes VLM conditions on GPU 0.
Existing development and qualitative tasks must finish first. Services must
already be running on the endpoints in formal_launch_matrix_20261029.json.
HELP
  exit 0
fi
: "${RUNTIME_ROOT:?}"
: "${SOURCE_DIR:?}"
: "${FORMAL_RELEASE:?}"
: "${SUITE_ID:?}"
: "${EXECUTION_ACTOR_CHECKPOINT:?}"
[[ "$SUITE_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid suite output prefix' >&2; exit 1; }
export ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-$RUNTIME_ROOT/outputs/control_modules_20261029_v1/fit/maintenance_actor.pt}"
export SOURCE_DIR EXECUTION_ACTOR_CHECKPOINT
suite_out="$RUNTIME_ROOT/outputs/$SUITE_ID"
suite_python="$RUNTIME_ROOT/.venv-newton/bin/python"
# This preflight completes before creating any trial directory or process.
"$suite_python" - "$RUNTIME_ROOT" "$SOURCE_DIR" "$FORMAL_RELEASE" "$SUITE_ID" "$ACTOR_CHECKPOINT" "$EXECUTION_ACTOR_CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
root,sources,release_path=map(Path,sys.argv[1:4]);name=sys.argv[4]
sys.path.insert(0,str(sources))
from formal_release import validate_contracts,digest,EXTERNAL_SOURCES
release=json.loads(release_path.read_text());validate_contracts(release)
if release['sources_sha256']!={p.name:digest(p) for p in sorted(sources.glob('*.py'))}:
    raise ValueError('Frozen evaluator source mismatch')
if release['external_sources_sha256']!={p:digest(root/'VLMmanipulation'/p) for p in EXTERNAL_SOURCES}:
    raise ValueError('Frozen client source mismatch')
reference=release['conditions']['astra_full']['manifest_contract']
for key,path in [('actor_sha256',Path(sys.argv[5])),('execution_actor_sha256',Path(sys.argv[6])),
    ('student_sha256',root/'outputs/pose_latent_20261029_v1/adapt/latent_student_00400.pt'),
    ('original_student_sha256',root/'prepared_release_20261029/student/student_1000.pt')]:
    if digest(path)!=reference[key]:raise ValueError('Selected checkpoint mismatch: '+key)
if not json.loads((root/'outputs/pose_latent_20261029_v1/adapt/completion.json').read_text())['completed']:
    raise ValueError('Latent-student training is incomplete')
for predecessor in ('agent_rule_execution_v6_20261029','agent_rule_video_20261029',
    'agent_vlm_execution_v5_20261029','agent_rule_execution_v5_20261029',
    'controller_replay_v10_20261029','controller_replay_fixed_wait_v6_20261029',
    'formal_runtime_prepare_astra_candidates_20261029'):
    status=(root/'outputs'/predecessor/'process_status.txt').read_text().splitlines()
    if 'stage=finished' not in status or 'exit_code=0' not in status:
        raise ValueError('Unfinished or failed predecessor: '+predecessor)
paths=[root/'outputs'/name,root/'outputs'/('pose_eval_'+name+'_low_level')]
paths += [root/'outputs'/(name+'_'+c) for c in release['conditions'] if c not in
          ('released_pose_goal_student','migrated_pose_latent_student')]
for path in paths:
    if path.exists():raise FileExistsError(path)
print('Frozen sources, checkpoints, protocol, predecessors and unused outputs verified.',flush=True)
PY
mkdir "$suite_out"
stage=launching
trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$suite_out/process_status.txt"' EXIT
cp "$FORMAL_RELEASE" "$suite_out/formal_release.json"
export FORMAL_RELEASE="$suite_out/formal_release.json"
cp "$0" "$suite_out/run_formal_suite.sh"
unset WAIT_FOR_RUN STOP_CHECKPOINT STOP_CALIBRATION FORMAL_CONDITION
export PREPARE_ONLY=0 COMPLETION_DURING_WAIT=0 STOP_MODE=measured
export VLM_URL=http://127.0.0.1:18794 RULE_URL=http://127.0.0.1:18795
agent() {
  local condition="$1" provider="$2" maintenance="$3" port="$4"
  env RUN_ID="${SUITE_ID}_${condition}" FORMAL_CONDITION="$condition" PROVIDER="$provider" \
    MAINTENANCE="$maintenance" ASTRA_BRIDGE_URL="http://127.0.0.1:$port" \
    bash "$SOURCE_DIR/run_agent_eval.sh"
}
queue0() {
  stage=low_level
  trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$suite_out/gpu0_status.txt"' EXIT
  export CUDA_VISIBLE_DEVICES=0 POSE_SOURCE=gs_cnn
  env RUN_ID="${SUITE_ID}_low_level" EVALUATOR_DIRECTORY="$SOURCE_DIR" \
    TRAINING_RUN="$RUNTIME_ROOT/outputs/pose_latent_20261029_v1" TRAINING_PID=105062 \
    bash "$SOURCE_DIR/run_development_eval.sh"
  "$suite_python" - "$RUNTIME_ROOT/outputs/pose_eval_${SUITE_ID}_low_level" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for policy in ('released','latent'):
    for seed in (183101,183102):
        result=json.loads((root/f'{policy}_{seed}'/'result.json').read_text())
        if (result['status']!='completed' or result['recorded_episodes']!=20
                or not result['formal_diagnostic_result']):
            raise ValueError('Incomplete formal policy diagnostic')
PY
  stage=finetuned_vlm_full
  agent finetuned_vlm_full vlm learned 18801
  stage=rule_vlm_full
  agent rule_vlm_full rule learned 18801
  stage=finished
}
queue1() {
  stage=astra_full
  trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$suite_out/gpu1_status.txt"' EXIT
  export CUDA_VISIBLE_DEVICES=1
  agent astra_full astra learned 18801
  stage=learned_waiting_only
  agent learned_waiting_only astra learned_wait_only 18803
  stage=finished
}
queue5() {
  stage=fixed_slow_no_execution_recovery
  trap 'result=$?; printf "stage=%s\nexit_code=%s\n" "$stage" "$result" > "$suite_out/gpu5_status.txt"' EXIT
  export CUDA_VISIBLE_DEVICES=5
  agent fixed_slow_no_execution_recovery astra fixed_slow 18802
  stage=learned_execution_only_with_fixed_slow_waiting
  agent learned_execution_only_with_fixed_slow_waiting astra learned_execution_only 18804
  stage=finished
}
(queue0) > "$suite_out/gpu0.log" 2>&1 & queue0_pid=$!
(queue1) > "$suite_out/gpu1.log" 2>&1 & queue1_pid=$!
(queue5) > "$suite_out/gpu5.log" 2>&1 & queue5_pid=$!
"$suite_python" - "$suite_out" "$SUITE_ID" "$queue0_pid" "$queue1_pid" "$queue5_pid" <<'PY'
import json,sys,hashlib
from datetime import datetime,timezone
from pathlib import Path
out=Path(sys.argv[1]);name=sys.argv[2]
conditions={'released_pose_goal_student':dict(root='pose_eval_'+name+'_low_level',prefix='released_',scope='low_level_diagnostic'),
 'migrated_pose_latent_student':dict(root='pose_eval_'+name+'_low_level',prefix='latent_',scope='low_level_diagnostic')}
for condition,provider in [('astra_full','astra'),('finetuned_vlm_full','vlm'),('rule_vlm_full','rule'),
 ('fixed_slow_no_execution_recovery','astra'),('learned_waiting_only','astra'),
 ('learned_execution_only_with_fixed_slow_waiting','astra')]:
    conditions[condition]=dict(root=name+'_'+condition,prefix=provider+'_',scope='autonomous_semantic_agent')
(out/'launch.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
 queues={str(gpu):int(pid) for gpu,pid in zip((0,1,5),sys.argv[3:])},
 release_sha256=hashlib.sha256((out/'formal_release.json').read_bytes()).hexdigest(),
 launcher_sha256=hashlib.sha256((out/'run_formal_suite.sh').read_bytes()).hexdigest(),
 conditions=conditions,planned_raw_trials=320,retries=False,replacement_trials=False),indent=2)+'\n')
PY
stage=running
suite_code=0
for queue_pid in "$queue0_pid" "$queue1_pid" "$queue5_pid"; do
  if wait "$queue_pid"; then :; else suite_code=1; fi
done
if [[ "$suite_code" == 0 ]]; then stage=finished; else stage=failed; fi
exit "$suite_code"
