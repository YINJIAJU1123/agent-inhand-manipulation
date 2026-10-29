"""Assemble a formal release from eight completed runtime preparations.

Run only after controller selection is recorded from completed development
experiments. This command starts no simulator and makes no inference request.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from formal_release import (CONTRACT_FIELDS, DIAGNOSTIC_FIELDS, EXTERNAL_SOURCES,
                            FORMAL_CONDITIONS, digest, validate_contracts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared_map', type=Path, required=True,
                        help='JSON mapping of all eight condition names to preparation directories')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--selection_record', type=Path, required=True,
                        help='JSON with status=selected, formal_trials_used=false, rationale and the selected actor hashes')
    parser.add_argument('--service_inventory', type=Path, required=True,
                        help='Public frozen source, checkpoint, backbone and service inventory, without credentials')
    parser.add_argument('--source_directory', type=Path, required=True)
    parser.add_argument('--project_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection_record.read_text())
    if (selection.get('status') != 'selected' or selection.get('formal_trials_used') is not False
            or not selection.get('rationale') or not selection.get('completed_development_evidence')):
        raise ValueError('A documented development-only selection is required before freeze')
    inventory = json.loads(args.service_inventory.read_text())
    if (inventory.get('credentials_copied') is not False
            or not all(inventory.get(key) for key in ('source_sha256', 'checkpoint_sha256', 'backbone_files', 'services'))):
        raise ValueError('A public source and weight inventory without copied credentials is required')
    sources = {p.name: digest(p) for p in sorted(args.source_directory.glob('*.py'))}
    external = {name: digest(args.project_root / name) for name in EXTERNAL_SOURCES}
    paths = json.loads(args.prepared_map.read_text())
    if set(paths) != set(FORMAL_CONDITIONS):
        raise ValueError('Prepare exactly the eight registered conditions')
    conditions = {}
    for name in FORMAL_CONDITIONS:
        root = Path(paths[name])
        if not root.is_absolute():
            root = args.prepared_map.parent / root
        prepared = json.loads((root / 'preparation.json').read_text())
        result = json.loads((root / 'result.json').read_text())
        manifest = json.loads((root / 'manifest.json').read_text())
        validity = json.loads(root.with_suffix('.validity.json').read_text())
        diagnostic = name in FORMAL_CONDITIONS[:2]
        kind = 'diagnostic' if diagnostic else 'agent'
        if (prepared.get('format') != 'pose_latent_runtime_preparation_v1'
                or prepared.get('kind') != kind
                or prepared.get('formal_trials_used') is not False
                or manifest.get('formal_trials_used') is not False
                or manifest.get('preparation_only') is not True
                or result.get('status') != 'prepared'
                or result.get('recorded_episodes') != 0
                or result.get('actual_model_calls', 0) != 0
                or prepared.get('recorded_episodes') != 0
                or prepared.get('actual_model_calls') != 0
                or validity.get('physical_log_valid') is not True
                or validity.get('exit_code') != 0):
            raise ValueError('Preparation did not finish cleanly without trials: ' + name)
        if any(root.glob('episode_*')) or (root / 'episodes.jsonl').exists():
            raise ValueError('Preparation unexpectedly contains trial records: ' + name)
        if prepared['manifest_sha256'] != digest(root / 'manifest.json'):
            raise ValueError('Prepared manifest was modified: ' + name)
        if prepared['sources_sha256'] != sources:
            raise ValueError('Preparation and release sources differ: ' + name)
        if not diagnostic and prepared['external_sources_sha256'] != external:
            raise ValueError('Preparation and release clients differ: ' + name)
        fields = DIAGNOSTIC_FIELDS if diagnostic else CONTRACT_FIELDS
        conditions[name] = dict(manifest_contract={key: manifest[key] for key in fields},
            preparation_evidence={file: digest(root / file) for file in ('preparation.json', 'manifest.json', 'result.json')},
            physics_validity_sha256=digest(root.with_suffix('.validity.json')))
    reference = conditions['astra_full']['manifest_contract']
    for key in ('actor_sha256', 'execution_actor_sha256'):
        if selection[key] != reference[key]:
            raise ValueError('Selected actor differs from the prepared actor: ' + key)
    for name, policy in zip(FORMAL_CONDITIONS[:2], ('released', 'latent')):
        contract = conditions[name]['manifest_contract']
        expected_checkpoint = reference['original_student_sha256'] if policy == 'released' else reference['student_sha256']
        if (contract['policy'] != policy or contract['checkpoint_sha256'] != expected_checkpoint
                or contract['original_student_sha256'] != reference['original_student_sha256']):
            raise ValueError('Prepared low-level diagnostic does not use the shared policy: ' + name)
    release = dict(format='pose_latent_agent_release_v1', status='frozen',
        created_at=datetime.now(timezone.utc).isoformat(),
        frozen_requirements=json.loads(args.protocol.read_text())['frozen_requirements'],
        sources_sha256=sources, external_sources_sha256=external, conditions=conditions,
        selection_record=selection, selection_record_sha256=digest(args.selection_record),
        service_inventory=inventory, service_inventory_sha256=digest(args.service_inventory),
        formal_trials_used_for_selection=False)
    validate_contracts(release)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(release, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(release=str(args.output), sha256=digest(args.output),
                         conditions=len(conditions), formal_trials_started=False)))


if __name__ == '__main__':
    main()
