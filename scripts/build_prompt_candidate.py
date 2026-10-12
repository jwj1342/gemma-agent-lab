"""Build one bounded prompt ablation without models, data or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


VARIANTS = {'workflow': ['workflow'], 'protocol': ['workflow', 'protocol'], 'icl': ['workflow', 'fewshot'],
            'reflect': ['workflow', 'fewshot', 'reflection']}


def build_candidate(source, output, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown prompt variant')
    if output.exists():
        raise ValueError('Preserve existing candidate artifacts')
    shutil.copytree(source, output)
    prompt = '\n\n'.join((source / 'prompts' / (part + '.md')).read_text().strip()
                         for part in VARIANTS[variant]) + '\n'
    (output / 'prompts/system.md').write_text(prompt)
    if variant != 'reflect':
        config = output / 'agent.yaml'
        config.write_text(config.read_text().replace('skills:\n  - skills/repair-playbook\n', ''))
        shutil.rmtree(output / 'skills')
    for part in {part for parts in VARIANTS.values() for part in parts}:
        (output / 'prompts' / (part + '.md')).unlink()
    receipt = {'variant': variant, 'weight_updates': 0, 'prompt_characters': len(prompt),
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'memory_scope': 'within-task conversation only',
        'status': 'Prepared candidate; compile and repair evaluation pending'}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('configs/prompt-icl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.source, args.output, args.variant), indent=2))


if __name__ == '__main__':
    main()
