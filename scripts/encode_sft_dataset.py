"""Encode a frozen, deduplicated action corpus on an allocated node."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from src.sft_data import encode_example, read_jsonl
from src.sft_subset import select_training_subset


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--swe', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--terminal', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Encode training data in an allocation')
    import torch
    from transformers import AutoTokenizer
    if args.output.exists():
        raise ValueError('Preserve the existing encoded corpus')
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    stop_token = '<|tool_response>'
    stop_id = tokenizer.convert_tokens_to_ids(stop_token)
    generation = json.loads((args.base / 'generation_config.json').read_text())
    if stop_id not in generation['eos_token_id']:
        raise ValueError('Tool handoff token is absent from the model stop-token configuration')
    buckets = {'train': [], 'validation': []}
    seen, cross_split, rejected, files, stats = {}, set(), Counter(), {}, {}
    terminal_prompt = None
    if config.get('training_terminal_system_prompt'):
        prompt_path = Path(config['training_terminal_system_prompt'])
        terminal_prompt = prompt_path.read_text()
        if not terminal_prompt.strip():
            raise ValueError('The training terminal system prompt must be nonempty')
        files[str(prompt_path.resolve())] = digest(prompt_path)
    for split in buckets:
        for kind in ('swe', 'graph', 'terminal'):
            path = getattr(args, kind) / (split + '.jsonl')
            files[str(path.resolve())] = digest(path)
            for sample in read_jsonl(path):
                # Match the updated harness descriptions, while retaining upstream observation provenance.
                sample['messages'][0]['content'] = (
                    'You repair Python repositories using the available tools. Inspect relevant code, '
                    'make a focused change, run a regression test, and submit the patch. Use exact '
                    'argument names. After a successful submission, return a brief final completion '
                    'message. Use graph observations to guide source inspection. An empty graph result '
                    'warrants a repository search. Some demonstrations retain upstream tool output formats.')
                if split == 'train' and kind == 'terminal' and terminal_prompt is not None:
                    # Align completion exercises with the actual agent policy.
                    # Keep held-out contexts and their encoded identities unchanged.
                    sample['messages'][0]['content'] = terminal_prompt
                for tool in sample['tools']:
                    function = tool['function']
                    if function['name'] == 'get_code_neighbors':
                        function['description'] = 'Find incoming and outgoing neighbors of a graph symbol.'
                    elif function['name'] == 'search_similar_code':
                        function['description'] = 'Retrieve related code using cached embeddings or lexical fallback.'
                encoded = encode_example(sample, tokenizer, config['max_length'],
                                         config['history_actions'], config['excerpt_chars'])
                if encoded is None:
                    rejected[split + ':' + kind + ':length'] += 1
                    continue
                identity = hashlib.sha256(json.dumps(encoded['input_ids']).encode()).hexdigest()
                if identity in seen:
                    if seen[identity] != split:
                        # Identical graph exercises can arise at different snapshots.
                        # Reserve their encoded context for validation and purge training copies.
                        cross_split.add(identity)
                        rejected['train:cross-split encoded duplicate'] += 1
                    else:
                        rejected[split + ':' + kind + ':duplicate'] += 1
                        continue
                seen[identity] = split
                target_start = next(i for i, t in enumerate(encoded['labels']) if t != -100)
                if sample['messages'][-1].get('tool_calls'):
                    if encoded['input_ids'][-1] != stop_id or encoded['labels'][-1] != stop_id:
                        raise ValueError('Tool handoff stop token is missing from supervised targets')
                buckets[split].append({'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.int32),
                    'target_start': target_start, 'kind': kind, 'sample_id': identity,
                    'target_tool': (sample['messages'][-1]['tool_calls'][0]['function']['name']
                                    if sample['messages'][-1].get('tool_calls') else 'final_answer'),
                    'metadata': encoded['metadata']})
                if len(buckets[split]) % 500 == 0:
                    print(split, len(buckets[split]), flush=True)
    buckets['train'] = [s for s in buckets['train'] if s['sample_id'] not in cross_split]
    if {s['sample_id'] for s in buckets['train']} & {s['sample_id'] for s in buckets['validation']}:
        raise ValueError('Encoded example leakage after duplicate removal')
    train_groups = {s['metadata']['group'].lower() for s in buckets['train']}
    val_groups = {s['metadata']['group'].lower() for s in buckets['validation']}
    if train_groups & val_groups:
        raise ValueError('Cross-source snapshot leakage')
    subset = None
    if config.get('training_subset'):
        buckets['train'], coverage = select_training_subset(
            buckets['train'], config['training_subset'], config['seed'])
        subset = {'budgets': config['training_subset'], 'seed': config['seed'],
                  'method': 'Round-robin over repository/commit and target-tool strata; hashed order',
                  'coverage': coverage, 'validation_population_unchanged': True,
                  'sample_ids': [row['sample_id'] for row in buckets['train']]}
    protocol_examples = []
    for split, samples in buckets.items():
        for sample in samples:
            # The native template closes a final answer with <turn|> followed
            # by a newline. Pending calls end directly at <|tool_response>.
            expected = ([stop_id] if sample['metadata']['target_kind'] == 'tool'
                        else tokenizer.encode('<turn|>\n', add_special_tokens=False))
            if (not 0 < sample['target_start'] < len(sample['input_ids'])
                    or sample['input_ids'][-len(expected):].tolist() != expected):
                raise ValueError('Native-tokenizer target boundary or termination contract failed')
        for kind in ('swe', 'graph', 'terminal'):
            rows = [s for s in samples if s['kind'] == kind]
            if not rows:
                raise ValueError(f'No {split} {kind} examples')
            lengths = sorted(len(s['input_ids']) for s in rows)
            stats[split + ':' + kind] = {'actions': len(rows),
                'trajectories': len({s['metadata'].get('traj_id', s['metadata'].get('symbol', s['metadata']['group'])) for s in rows}),
                'groups': len({s['metadata']['group'] for s in rows}),
                'input_tokens': sum(lengths),
                'assistant_tokens': sum(len(s['input_ids']) - s['target_start'] for s in rows),
                'length_p50': lengths[len(lengths)//2], 'length_p95': lengths[int(len(lengths)*.95)],
                'length_max': lengths[-1]}
            representative = rows[0]
            protocol_examples.append({'split': split, 'kind': kind, 'sample_id': representative['sample_id'],
                'target_start': representative['target_start'],
                'target_prefix': tokenizer.decode(representative['input_ids'][representative['target_start']:representative['target_start'] + 24].tolist()),
                'target_suffix': tokenizer.decode(representative['input_ids'][-24:].tolist()),
                'final_token_id': int(representative['input_ids'][-1])})
        torch.save(samples, args.output / (split + '.pt'))
    manifest = {'config': config, 'source_files_sha256': files, 'statistics': stats,
        'encoding_protocol': {
            'tool_handoff_stop_token': stop_token, 'tool_handoff_stop_id': stop_id,
            'tool_handoff_is_supervised': True,
            'source_sha256': {str(p): digest(p) for p in
                             (Path('src/sft_data.py'), Path('src/sft_subset.py'), Path('scripts/encode_sft_dataset.py'))},
            'base_protocol_sha256': {name: digest(args.base / name) for name in
                                    ('chat_template.jinja', 'tokenizer.json', 'generation_config.json')},
            'tool_targets_by_split': {split: sum(s['metadata']['target_kind'] == 'tool' for s in samples)
                                     for split, samples in buckets.items()}},
        'training_subset': subset,
        'rejected': dict(rejected), 'sampling': 'Every selected training example once per epoch, no replacement',
        'split': 'Disjoint repository/base-commit groups; protected benchmark snapshots excluded upstream',
        'encoded_files_sha256': {s: digest(args.output / (s + '.pt')) for s in buckets},
        'caveat': 'SWE success is an upstream label; graph/terminal examples are executed component exercises.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'protocol-examples.json').write_text(json.dumps(protocol_examples, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
