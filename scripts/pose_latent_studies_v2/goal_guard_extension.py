"""Append-only admission for matched actor goal ablations, preserving cube-v1 scoring."""
_GOAL_LEGACY_VALIDATE = validate_contracts
_GOAL_LEGACY_LOAD = load_release
_GOAL_LEGACY_VERIFY = verify_runtime
GOAL_ABLATION_FORMAT = 'cube_controller_goal_ablation_v1'
GOAL_COHORT = dict(seeds=[183421,183422],episodes_per_seed=20,raw_trials_per_condition=40)
GOAL_CONDITIONS = dict(correct_active_goal=('active','none'),
    retrained_no_explicit_goal=('center','none'),shuffled_controller_goal=('active','cyclic_shift'))
GOAL_RUNTIME_FIELDS = CONTRACT_FIELDS + ('controller_goal_intervention',)
GOAL_CHANGED_FIELDS = ('actor_sha256','execution_actor_sha256','evaluator_sha256','controller_goal_intervention')


def goal_intervention_contract(mode, intervention):
    return dict(mode=intervention,mapping=[1,2,3,4,5,0] if intervention=='cyclic_shift' else None,
        waiting_goal_input=mode,execution_goal_input=mode,
        scope='actor explicit goal only; student and independent score unchanged')


def validate_contracts(release):
    if release.get('format')!=GOAL_ABLATION_FORMAT:
        return _GOAL_LEGACY_VALIDATE(release)
    text=release['parent_addendum_json']
    if hashlib.sha256(text.encode()).hexdigest()!=release['parent_addendum_sha256']:
        raise ValueError('Parent bytes mismatch')
    parent=json.loads(text)
    if parent.get('format')!=BARE_VLM_ADDENDUM_FORMAT:
        raise ValueError('Goal study requires original bare addendum')
    _GOAL_LEGACY_VALIDATE(parent)
    reference=json.loads(parent['parent_release_json'])['conditions']['astra_full']['manifest_contract']
    if release['status']!='frozen' or release['frozen_requirements']!=parent['frozen_requirements']:
        raise ValueError('Scoring and physical contract must remain frozen')
    if release['cohort']!=GOAL_COHORT or set(release['conditions'])!=set(GOAL_CONDITIONS):
        raise ValueError('Unregistered goal cohort or conditions')
    if release['formal_trials_used_for_selection'] is not False:
        raise ValueError('Formal trials cannot select weights')
    old=parent['sources_sha256'];src=release['sources_sha256']
    allowed={'formal_release.py','control_modules.py','evaluate_agent.py'}
    if set(src)!=set(old) or any(src[k]!=old[k] for k in old if k not in allowed):
        raise ValueError('Goal study changed unrelated runtime sources')
    if release['external_sources_sha256']!=parent['external_sources_sha256']:
        raise ValueError('Upstream clients changed')
    for name,(mode,intervention) in GOAL_CONDITIONS.items():
        c=release['conditions'][name]['manifest_contract']
        if set(c)!=set(GOAL_RUNTIME_FIELDS):raise ValueError('Incomplete goal runtime contract')
        if any(c[k]!=reference[k] for k in CONTRACT_FIELDS if k not in GOAL_CHANGED_FIELDS):
            raise ValueError('Non-goal runtime change: '+name)
        if c['controller_goal_intervention']!=goal_intervention_contract(mode,intervention):
            raise ValueError('Wrong goal intervention: '+name)
        if c['evaluator_sha256']!=src['evaluate_agent.py']:
            raise ValueError('Evaluator identity mismatch')
        for phase,field in [('waiting','actor_sha256'),('execution','execution_actor_sha256')]:
            expected=release['training_evidence']['models'][phase+'_'+mode]['sha256']
            if c[field]!=expected:raise ValueError('Wrong matched training checkpoint')
    if release['conditions']['correct_active_goal']['manifest_contract']['actor_sha256']==release['conditions']['retrained_no_explicit_goal']['manifest_contract']['actor_sha256']:
        raise ValueError('Center branch must be separately retrained')


def load_release(args,source_directory,kind='agent'):
    if args.formal_release is None:
        if args.seed in GOAL_COHORT['seeds']:
            raise ValueError('Reserved goal study seeds require a release')
        return _GOAL_LEGACY_LOAD(args,source_directory,kind)
    release=json.loads(args.formal_release.read_text())
    if release.get('format')!=GOAL_ABLATION_FORMAT:
        return _GOAL_LEGACY_LOAD(args,source_directory,kind)
    validate_contracts(release)
    if (kind!='agent' or args.prepare_only or args.record_video or args.provider!='astra'
            or args.maintenance!='learned' or args.stop_mode!='measured'):
        raise ValueError('Goal study requires live Astra with shared measured gate and both actors')
    if args.formal_condition not in GOAL_CONDITIONS or args.seed not in GOAL_COHORT['seeds']:
        raise ValueError('Unregistered condition/seed')
    if args.start_episode!=0 or args.episodes!=20:
        raise ValueError('Expect complete scheduled twenty-trial seed')
    expected=GOAL_CONDITIONS[args.formal_condition][1]
    if args.controller_goal_intervention!=expected:
        raise ValueError('Wrong launch intervention')
    if {p.name:digest(p) for p in source_directory.glob('*.py')}!=release['sources_sha256']:
        raise ValueError('Goal evaluator snapshot changed')
    if {name:digest(args.project_root/name) for name in EXTERNAL_SOURCES}!=release['external_sources_sha256']:
        raise ValueError('External clients changed')
    return release


def verify_runtime(release,condition,manifest,kind='agent'):
    if release.get('format')!=GOAL_ABLATION_FORMAT:
        return _GOAL_LEGACY_VERIFY(release,condition,manifest,kind)
    actual=json.loads(json.dumps({k:manifest[k] for k in GOAL_RUNTIME_FIELDS},allow_nan=False))
    expected=release['conditions'][condition]['manifest_contract']
    changed=[k for k in GOAL_RUNTIME_FIELDS if actual[k]!=expected[k]]
    if changed:raise ValueError('Goal runtime differs from freeze: '+', '.join(changed))
