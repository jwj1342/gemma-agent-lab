"""Package paired C2/checkpoint arms without launching an experiment.

Only the exact historical C2 source is accepted. Seed changes are identical in
both arms; every paired package must differ in the system prompt alone.
"""
import argparse
import json
from pathlib import Path
import shutil

from scripts.build_diagnosis_candidate import build_candidate, digest
from scripts.build_submission import build, collect_files


SPEC = Path(__file__).resolve().parents[1] / 'configs/experiments/observation-checkpoint.json'


def inventory(source):
    root, files = collect_files(source)
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in files}


def build_pilot(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a fresh output outside the control')
    spec = json.loads(SPEC.read_text())
    if inventory(source) != spec['control_members']:
        raise ValueError('Source must be the exact unseeded frozen C2 package')
    # The existing builder verifies the original inventory and one-paragraph delta.
    base = build_candidate(source, output / 'base', SPEC)
    records = []
    for seed in spec['evaluation']['seeds']:
        pair = {}
        for arm, template in [('control', source), ('candidate', output / 'base/agent')]:
            directory = output / f'seed-{seed}' / arm
            agent = directory / 'agent'
            shutil.copytree(template, agent)
            if seed != 42:
                sampling = agent / 'configs/sampling.yaml'
                # C2's exact hash above guarantees this has no seed key.
                sampling.write_bytes(sampling.read_bytes() + f'seed: {seed}\n'.encode())
            archive = build(agent, directory / 'submission.zip')
            pair[arm] = {'members': inventory(agent), 'archive_sha256': archive['sha256']}
        changed = [p for p in pair['control']['members']
                   if pair['control']['members'][p] != pair['candidate']['members'][p]]
        if changed != [spec['prompt_path']]:
            raise ValueError('Paired arms must differ in the system prompt alone')
        records.append({'seed_label': seed, 'sampling_seed': None if seed == 42 else seed,
                        'changed_members': changed, **pair})
    receipt = {'schema_version': 1, 'name': spec['name'],
               'spec_sha256': base['spec_sha256'], 'evaluation': spec['evaluation'],
               'pairs': records,
               'status': 'CPU packages frozen; runtime preflight and complete run-input manifest pending'}
    (output / 'pilot.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_pilot(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
