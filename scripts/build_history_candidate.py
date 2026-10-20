"""Build a hash-pinned C2 ablation that omits retained thought parts."""
import argparse
import json
from pathlib import Path
import re
import shutil

import yaml

from build_submission import build, collect_files, sha256

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / 'configs/experiments/c2-r5-control.json'


def build_candidate(source, output, expected):
    """Require the frozen control; create a new run with agent, ZIP and receipt."""
    source, files = collect_files(source)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('Use a new run directory outside the control agent')
    actual = {p.relative_to(source).as_posix(): sha256(p) for p in files}
    if actual != expected:
        raise ValueError('Control does not match the frozen file inventory and hashes')
    sampling_path = 'configs/sampling.yaml'
    original = (source / sampling_path).read_bytes()
    config = yaml.safe_load(original)
    thinking = config.get('thinking_config', {})
    if thinking.get('include_thoughts') is not True or thinking.get('thinking_budget') != 4096:
        raise ValueError('Expected retained thoughts and positive 4096-token thinking budget')
    changed, count = re.subn(rb'(?m)^(\s*include_thoughts:\s*)true([ \t]*\r?)$',
                             rb'\g<1>false\2', original)
    if count != 1:
        raise ValueError('Expected exactly one plain include_thoughts: true setting')
    candidate_config = yaml.safe_load(changed)
    candidate_config['thinking_config']['include_thoughts'] = True
    if candidate_config != config:
        raise ValueError('Candidate must change only include_thoughts')
    output.mkdir(parents=True)
    agent = output / 'agent'
    shutil.copytree(source, agent)
    (agent / sampling_path).write_bytes(changed)
    _, candidate_files = collect_files(agent)
    candidate = {p.relative_to(agent).as_posix(): sha256(p) for p in candidate_files}
    if {p for p in actual if actual[p] != candidate[p]} != {sampling_path}:
        raise ValueError('Unexpected changed files')
    archive = build(agent, output / 'submission.zip')
    receipt = {'control_files': actual, 'candidate_files': candidate,
               'intervention': 'include_thoughts true -> false; thinking_budget remains 4096',
               'archive_sha256': archive['sha256'],
               'status': 'CPU packaging only; official compilation and matched inference pending'}
    (output / 'ablation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Frozen r5 c2/agent directory')
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    args = parser.parse_args()
    expected = json.loads(DEFAULT_MANIFEST.read_text())['files']
    print(json.dumps(build_candidate(args.source, args.output, expected), indent=2))


if __name__ == '__main__':
    main()
