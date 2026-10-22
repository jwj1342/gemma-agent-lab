"""Build one prompt-only ablation from the exact frozen C2 member inventory.

No model, compiler, inference or submission is launched. Requires private control
files; the public experiment specification contains hashes and generic text only.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from scripts.build_submission import build, collect_files


DEFAULT_SPEC = Path(__file__).resolve().parents[1] / 'configs/experiments/diagnosis-checkpoint.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_candidate(source, output, spec_path=DEFAULT_SPEC):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a new run directory outside the control')
    spec_bytes = Path(spec_path).read_bytes()
    spec = json.loads(spec_bytes)
    root, files = collect_files(source)
    members = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    before = {name: digest(data) for name, data in members.items()}
    controls = {'c2': spec['control_members'], **spec.get('alternate_controls', {})}
    matched = [name for name, hashes in controls.items() if before == hashes]
    if len(matched) != 1:
        raise ValueError('Control inventory or bytes differ from the frozen C2 specification')
    prompt_path = spec['prompt_path']
    prompt = members[prompt_path].decode('utf-8')
    old, new = spec['old_paragraph'], spec['new_paragraph']
    if not old or old == new or prompt.count(old) != 1:
        raise ValueError('Require exactly one nonempty paragraph replacement')
    members[prompt_path] = prompt.replace(old, new, 1).encode('utf-8')
    after = {name: digest(data) for name, data in members.items()}
    changed = [name for name in before if before[name] != after[name]]
    if changed != [prompt_path]:
        raise ValueError('Candidate must change only the system prompt')
    for name in members:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe control member')
    output.mkdir(parents=True)
    for name, data in members.items():
        target = output / 'agent' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    archive = build(output / 'agent', output / 'submission.zip')
    receipt = {'name': spec['name'], 'control_variant': matched[0], 'spec_sha256': digest(spec_bytes),
               'control_members': before, 'candidate_members': after, 'changed_members': changed,
               'archive_sha256': archive['sha256'], 'weight_updates': 0,
               'status': 'CPU packaging only; official compilation, inference and repair evaluation pending'}
    (output / 'candidate.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.source, args.output, args.spec), indent=2))


if __name__ == '__main__':
    main()
