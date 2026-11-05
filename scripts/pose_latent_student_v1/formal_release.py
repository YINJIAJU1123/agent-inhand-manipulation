"""Validate a frozen release before reserved-seed agent or policy rollouts.

Release selection is done after development. This file never chooses weights or
creates a release from the outcomes of formal trials.
"""
import hashlib
import json
from pathlib import Path

REQUIREMENTS_SHA256 = '85da9291e02596910cf0bad1e633b8dd3e5fd6500d7f43e083588b5dbbc3e594'
FORMAL_SEEDS = (183101,183102)
FORMAL_CONDITIONS = (
    'released_pose_goal_student','migrated_pose_latent_student',
    'astra_full','finetuned_vlm_full','rule_vlm_full',
    'fixed_slow_no_execution_recovery','learned_waiting_only',
    'learned_execution_only_with_fixed_slow_waiting',
)

BARE_VLM_ADDENDUM_FORMAT = 'pose_latent_bare_vlm_addendum_v1'
BARE_VLM_PARENT_SHA256 = '07922f1aa5522fd665b6fc0a70da6aa75e6d52801b387372427a18d6e3e0ef28'
BARE_VLM_CONDITIONS = {
    'finetuned_vlm_bare': 'finetuned_vlm_full',
    'rule_vlm_bare': 'rule_vlm_full',
}
BARE_VLM_CHANGED_FIELDS = frozenset((
    'maintenance', 'controller_modes', 'actor_sha256', 'execution_actor_sha256',
))

CONTRACT_FIELDS = (
    'student_sha256','original_student_sha256','physical_binding','pose_estimator',
    'camera','live_geometry','goal_manifest','explicit_task_overrides','goal_input',
    'capture','source_sha256','provider','upstream','maintenance','controller_modes','evaluator_sha256',
    'actor_sha256','execution_actor_sha256','completion_sha256','completion_during_wait',
    'completion_kind','stop_mode','measured_stop_threshold_rad','stop_calibration_sha256',
    'stopping','physics','request_budget','service_exclusion','inspection','fine_tuned_goal',
)
DIAGNOSTIC_FIELDS = (
    'version','policy','checkpoint_sha256','checkpoint','original_student_sha256',
    'physical_environment_binding','goal_bank','reset_seed_rule','initial_state_digest_scope',
    'goal_distribution_override','metric','stopping','initial_reach','simultaneous_drop_and_reach',
    'not_autonomous_semantic_success','teacher_used','pose_source','full_scene_rgb',
    'visual_noise_added_to_cnn_pose','noise_cfg','physical_randomization','step_dt',
    'external_max_steps','external_horizon_seconds','preserved_environment_episode_seconds',
    'policy_inputs','ground_truth_scope','source_sha256','estimator','camera',
)
CONDITION_MODES = {
    'astra_full': ('astra', 'learned', 'learned', 'learned'),
    'finetuned_vlm_full': ('vlm', 'learned', 'learned', 'learned'),
    'rule_vlm_full': ('rule', 'learned', 'learned', 'learned'),
    'fixed_slow_no_execution_recovery': ('astra', 'fixed_slow', 'fixed_slow', 'student'),
    'learned_waiting_only': ('astra', 'learned_wait_only', 'learned', 'student'),
    'learned_execution_only_with_fixed_slow_waiting': ('astra', 'learned_execution_only', 'fixed_slow', 'learned'),
}
SHARED_AGENT_FIELDS = tuple(key for key in CONTRACT_FIELDS
                           if key not in ('provider', 'upstream', 'maintenance', 'controller_modes'))

EXTERNAL_SOURCES = (
    'scripts/astra_transport_v2/bridge.py',
    'scripts/astra_transport_v2/subscription_v2.py',
    'scripts/agentic/latent_service_v7.py',
    'scripts/agentic/agent_observation_v4.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_contracts(release):
    """Check a frozen original suite or its separately registered bare-VLM addendum."""
    if release.get('format') == BARE_VLM_ADDENDUM_FORMAT:
        validate_bare_vlm_addendum(release)
        return
    if release.get('format')!='pose_latent_agent_release_v1' or release.get('status')!='frozen':
        raise ValueError('Formal release must be frozen before any rollout')
    requirements=release['frozen_requirements']
    actual=hashlib.sha256(json.dumps(requirements,sort_keys=True).encode()).hexdigest()
    if actual!=REQUIREMENTS_SHA256:
        raise ValueError('Frozen success/seed/budget/exclusion protocol changed')
    if set(release['conditions'])!=set(FORMAL_CONDITIONS):
        raise ValueError('Freeze all eight planned conditions before any formal trial')
    for name in FORMAL_CONDITIONS:
        fields=DIAGNOSTIC_FIELDS if name in FORMAL_CONDITIONS[:2] else CONTRACT_FIELDS
        if set(release['conditions'][name]['manifest_contract'])!=set(fields):
            raise ValueError('Incomplete or unexpected runtime contract fields: '+name)
    reference=release['conditions']['astra_full']['manifest_contract']
    for name in FORMAL_CONDITIONS[2:]:
        contract=release['conditions'][name]['manifest_contract']
        for key in ('student_sha256','original_student_sha256','actor_sha256','execution_actor_sha256'):
            if not reference[key]:
                raise ValueError('The selected shared policy is missing: '+key)
        for key in SHARED_AGENT_FIELDS:
            if contract[key]!=reference[key]:
                raise ValueError('Formal agent cells must share the downstream contract: '+name+' / '+key)
        provider,mode,waiting,execution=CONDITION_MODES[name]
        if (contract['provider']!=provider or contract['maintenance']!=mode
                or contract['controller_modes']!={'waiting':waiting,'execution':execution}):
            raise ValueError('The runtime does not implement the registered factorial cell: '+name)
        if provider=='astra' and contract['upstream']!=reference['upstream']:
            raise ValueError('Astra ablations must share the same upstream service contract')


def validate_bare_vlm_addendum(release):
    """Allow only the requested two bare baselines against the immutable parent.

    This is a supplemental comparison declared after parent outcomes existed.
    It neither changes the original eight-cell contract nor selects new weights.
    The embedded parent text binds its exact bytes rather than a mutable path.
    """
    if release.get('status') != 'frozen':
        raise ValueError('The bare-VLM addendum must be frozen before any rollout')
    parent_text = release.get('parent_release_json')
    if (not isinstance(parent_text, str)
            or release.get('parent_release_sha256') != BARE_VLM_PARENT_SHA256
            or hashlib.sha256(parent_text.encode('utf-8')).hexdigest() != BARE_VLM_PARENT_SHA256):
        raise ValueError('The addendum is not bound to the unchanged original release bytes')
    parent = json.loads(parent_text)
    if parent.get('format') != 'pose_latent_agent_release_v1':
        raise ValueError('The addendum parent must be the original eight-condition release')
    validate_contracts(parent)
    if release.get('frozen_requirements') != parent['frozen_requirements']:
        raise ValueError('The addendum must preserve all success, seed, budget and exclusion rules')
    declaration = release.get('supplemental_declaration', {})
    expected = dict(created_after_parent_trials_started=True,
        parent_outcomes_used_for_new_weight_selection=False,
        new_weights_selected=False, old_conditions_modified=False,
        planned_new_raw_trials=80, retries=False, replacement_trials=False)
    if (any(type(declaration.get(key)) is not type(value) or declaration.get(key) != value
            for key, value in expected.items())
            or not isinstance(declaration.get('rationale'), str)
            or not declaration['rationale'].strip()):
        raise ValueError('Declare the post-parent supplemental scope without new selection or replacements')
    if release.get('formal_trials_used_for_selection') is not False:
        raise ValueError('The addendum cannot select weights from formal outcomes')
    sources = release.get('sources_sha256', {})
    parent_sources = parent['sources_sha256']
    if set(sources) != set(parent_sources):
        raise ValueError('The addendum must preserve the exact parent source-file set')
    changed = [name for name in sources if sources[name] != parent_sources[name]]
    if changed != ['formal_release.py']:
        raise ValueError('Only the release guard may differ from the parent evaluator snapshot')
    guard_digest = sources['formal_release.py']
    if (not isinstance(guard_digest, str) or len(guard_digest) != 64
            or any(char not in '0123456789abcdef' for char in guard_digest)):
        raise ValueError('Invalid addendum release-guard digest')
    if release.get('external_sources_sha256') != parent['external_sources_sha256']:
        raise ValueError('The addendum cannot change model transport or observation sources')
    if set(release.get('conditions', {})) != set(BARE_VLM_CONDITIONS):
        raise ValueError('The addendum must contain exactly the two requested bare-VLM conditions')
    for name, parent_name in BARE_VLM_CONDITIONS.items():
        item = release['conditions'][name]
        if item.get('parent_condition') != parent_name:
            raise ValueError('Wrong parent condition for bare baseline: ' + name)
        contract = item['manifest_contract']
        if set(contract) != set(CONTRACT_FIELDS):
            raise ValueError('Incomplete or unexpected addendum runtime fields: ' + name)
        reference = parent['conditions'][parent_name]['manifest_contract']
        if any(contract[key] != reference[key] for key in CONTRACT_FIELDS
               if key not in BARE_VLM_CHANGED_FIELDS):
            raise ValueError('The bare baseline changed more than the removed control actors: ' + name)
        if (contract['maintenance'] != 'continue'
                or contract['controller_modes'] != {'waiting': 'student', 'execution': 'student'}
                or any(contract[key] is not None for key in ('actor_sha256',
                    'execution_actor_sha256', 'completion_sha256', 'completion_kind',
                    'stop_calibration_sha256'))
                or contract['completion_during_wait'] is not False
                or contract['stop_mode'] != 'measured'
                or contract['measured_stop_threshold_rad'] != .16):
            raise ValueError('A bare baseline must use only student actions and the common measured gate: ' + name)


def load_release(args,source_directory,kind='agent'):
    if getattr(args,'prepare_only',False) and (args.formal_release or args.formal_condition or args.seed in FORMAL_SEEDS):
        raise ValueError('Runtime preparation uses non-formal seeds without a formal release')
    if bool(args.formal_release) != bool(args.formal_condition):
        raise ValueError('A formal release and condition must be supplied together')
    if args.formal_release is None:
        if args.seed in FORMAL_SEEDS:
            raise ValueError('Reserved formal seeds require a frozen release')
        return None
    release=json.loads(args.formal_release.read_text())
    validate_contracts(release)
    if release.get('format') == BARE_VLM_ADDENDUM_FORMAT:
        if (kind != 'agent' or args.formal_condition not in BARE_VLM_CONDITIONS
                or getattr(args, 'maintenance', None) != 'continue'
                or getattr(args, 'stop_mode', None) != 'measured'
                or getattr(args, 'completion_during_wait', False)
                or any(getattr(args, key, None) is not None for key in (
                    'actor_checkpoint', 'execution_actor_checkpoint',
                    'stop_checkpoint', 'stop_calibration'))):
            raise ValueError('Bare-VLM launch must omit every control/completion checkpoint and use continue/measured')
    if args.seed not in FORMAL_SEEDS or getattr(args,'start_episode',0)+args.episodes!=20:
        raise ValueError('Formal launch requires a scheduled twenty-trial seed or its unvisited contiguous tail')
    condition=release['conditions'][args.formal_condition]
    if kind=='agent':
        if args.provider=='replay' or args.record_video:
            raise ValueError('Formal semantic evaluation requires live inference without video rendering')
        if condition['manifest_contract']['provider']!=args.provider:
            raise ValueError('Provider does not match the frozen condition')
    elif kind=='diagnostic':
        if args.pose_source!='gs_cnn' or condition['manifest_contract']['policy']!=args.policy:
            raise ValueError('Formal policy diagnostics require the frozen policy and measured RGB-CNN pose')
    else:
        raise ValueError('Unknown formal evaluation scope')
    sources={p.name:digest(p) for p in sorted(source_directory.glob('*.py'))}
    if sources!=release['sources_sha256']:
        raise ValueError('Evaluator source snapshot differs from the frozen release')
    if kind=='agent':
        external={name:digest(args.project_root/name) for name in EXTERNAL_SOURCES}
        if external!=release['external_sources_sha256']:
            raise ValueError('Model transport or observation source changed after freeze')
    return release


def verify_runtime(release,condition,manifest,kind='agent'):
    fields=CONTRACT_FIELDS if kind=='agent' else DIAGNOSTIC_FIELDS
    expected=release['conditions'][condition]['manifest_contract']
    if set(expected)!=set(fields):
        raise ValueError('Frozen runtime contract is incomplete or has unexpected fields')
    # Match the JSON representation saved by preparation and release writers.
    actual=json.loads(json.dumps({key:manifest[key] for key in fields},allow_nan=False))
    changed=[key for key in fields if actual[key]!=expected[key]]
    if changed:
        raise ValueError('Runtime differs from the frozen release: '+', '.join(changed))


def save_preparation(output, source_directory, kind, project_root=None):
    """Record a loaded runtime before any scheduled trial or model inference."""
    value=dict(format='pose_latent_runtime_preparation_v1', kind=kind,
        manifest_sha256=digest(output/'manifest.json'),
        sources_sha256={p.name:digest(p) for p in sorted(source_directory.glob('*.py'))},
        external_sources_sha256=({name:digest(project_root/name) for name in EXTERNAL_SOURCES}
                                 if kind=='agent' else {}),
        recorded_episodes=0, actual_model_calls=0, formal_trials_used=False)
    (output/'preparation.json').write_text(json.dumps(value,indent=2)+'\n')
