"""Package a completed training export with the unchanged baseline agent policy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, required=True, help='Directory containing completed.json and adapter/')
    p.add_argument('--baseline', type=Path, default=Path('configs/baseline'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    completion = json.loads((args.training / 'completed.json').read_text())
    if completion['samples_seen'] != completion['expected_samples']:
        raise ValueError('Complete-epoch data coverage is missing')
    adapter = args.training / 'adapter'
    for name, expected in completion['export_sha256'].items():
        if digest(adapter / name) != expected:
            raise ValueError('Trained export changed: ' + name)
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if config['peft_type'] != 'LORA' or config.get('use_dora') or config.get('modules_to_save'):
        raise ValueError('Unexpected adapter implementation')
    if args.output.exists():
        raise ValueError('Preserve the existing candidate')
    shutil.copytree(args.baseline, args.output)
    shutil.copytree(adapter, args.output / 'adapters/swe_lora_v1')
    agent = args.output / 'agent.yaml'
    if 'adapter:' in agent.read_text():
        raise ValueError('Baseline unexpectedly already declares an adapter')
    agent.write_text(agent.read_text() + 'adapter: swe_lora_v1\n')
    receipt = {'training_completion_sha256': digest(args.training / 'completed.json'),
        'selected_checkpoint': completion['selected'], 'adapter_sha256': completion['export_sha256'],
        'policy': 'Baseline prompt, tools, sampling and budgets; root agent receives the trained adapter',
        'status': 'Exported; official compilation and serving/repair validation are separate gates'}
    args.output.with_name(args.output.name + '-export.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
