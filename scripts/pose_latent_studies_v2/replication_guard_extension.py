"""Append to an immutable parent release guard for a new, registered cohort.

Only the guard differs. All evaluator, sensor, action and scoring code is bound
byte-for-byte to the completed bare-baseline addendum.
"""
_LEGACY_VALIDATE = validate_contracts
_LEGACY_LOAD = load_release
REPLICATION_FORMAT = 'cube_v1_independent_replication_v1'
REPLICATION_NAMES = ('astra_full','finetuned_vlm_bare','rule_vlm_bare')


def replication_references(release):
    text=release['parent_addendum_json']
    if digest_bytes(text)!=release['parent_addendum_sha256']:
        raise ValueError('Parent bytes mismatch')
    addendum=json.loads(text)
    if addendum.get('format')!=BARE_VLM_ADDENDUM_FORMAT:
        raise ValueError('Expected immutable bare baseline parent')
    _LEGACY_VALIDATE(addendum)
    parent=json.loads(addendum['parent_release_json'])
    return addendum,dict(astra_full=parent['conditions']['astra_full'],
        finetuned_vlm_bare=addendum['conditions']['finetuned_vlm_bare'],
        rule_vlm_bare=addendum['conditions']['rule_vlm_bare'])


def digest_bytes(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_contracts(release):
    if release.get('format')!=REPLICATION_FORMAT:
        return _LEGACY_VALIDATE(release)
    addendum,references=replication_references(release)
    if release['status']!='frozen' or release['new_weight_selection'] is not False:
        raise ValueError('Replication must not select new weights')
    if release['frozen_requirements']!=addendum['frozen_requirements']:
        raise ValueError('Success/budget/physics rules differ from parent')
    if release['cohort']!={'seeds':[183411,183412],'episodes_per_seed':20,'raw_trials_per_condition':40}:
        raise ValueError('Unregistered replication cohort')
    if set(release['conditions'])!=set(REPLICATION_NAMES):
        raise ValueError('Expected all three primary systems')
    for name in REPLICATION_NAMES:
        if release['conditions'][name]['manifest_contract']!=references[name]['manifest_contract']:
            raise ValueError('Replication runtime contract changed: '+name)
    src=release['sources_sha256'];old=addendum['sources_sha256']
    if set(src)!=set(old) or any(src[k]!=old[k] for k in old if k!='formal_release.py'):
        raise ValueError('Only the cohort guard may change')
    if release['external_sources_sha256']!=addendum['external_sources_sha256']:
        raise ValueError('Transport/observation sources differ')


def load_release(args,source_directory,kind='agent'):
    if args.formal_release is None:
        if args.seed in (183411,183412):
            raise ValueError('Replication seeds require preregistered release')
        return _LEGACY_LOAD(args,source_directory,kind)
    release=json.loads(args.formal_release.read_text())
    if release.get('format')!=REPLICATION_FORMAT:
        return _LEGACY_LOAD(args,source_directory,kind)
    validate_contracts(release)
    if kind!='agent' or args.prepare_only or args.record_video or args.provider=='replay':
        raise ValueError('Replication requires live semantic trials')
    if args.formal_condition not in REPLICATION_NAMES or args.seed not in release['cohort']['seeds']:
        raise ValueError('Unregistered condition/seed')
    if args.start_episode!=0 or args.episodes!=20:
        raise ValueError('Initial replication launch requires twenty scheduled trials')
    if args.provider!=release['conditions'][args.formal_condition]['manifest_contract']['provider']:
        raise ValueError('Wrong provider')
    if {p.name:digest(p) for p in sorted(source_directory.glob('*.py'))}!=release['sources_sha256']:
        raise ValueError('Evaluator snapshot changed')
    if {name:digest(args.project_root/name) for name in EXTERNAL_SOURCES}!=release['external_sources_sha256']:
        raise ValueError('External clients changed')
    return release
