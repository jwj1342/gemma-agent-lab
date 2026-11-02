"""Count pinned shard coverage on CPU; never export source text or task identities."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.prepare_sft_data import DATASET, REVISION, SHARD_SHA256
from src.sft_data import UnsupportedAction, convert_trajectory, text_content


def xml_only(message):
    return (message.get('role') == 'assistant' and not message.get('tool_calls')
            and bool(re.search(r'<function\s*=', text_content(message.get('content')))))


def coverage(rows):
    counts = Counter()
    errors = Counter()
    for row in rows:
        counts['rows'] += 1
        if row.get('resolved') is not True:
            continue
        counts['resolved_trajectories'] += 1
        messages = json.loads(row['messages'])
        assistants = [m for m in messages if m.get('role') == 'assistant']
        counts['resolved_assistant_messages'] += len(assistants)
        xml = sum(xml_only(m) for m in assistants)
        counts['xml_only_messages'] += xml
        counts['trajectories_with_xml_only'] += bool(xml)
        counts['assistant_messages_without_native_calls'] += sum(not m.get('tool_calls') for m in assistants)
        terminal = (messages[-1] if messages and messages[-1].get('role') == 'assistant'
                    and not messages[-1].get('tool_calls') and not xml_only(messages[-1])
                    and text_content(messages[-1].get('content')).strip() else None)
        if terminal is not None:
            counts['source_terminal_messages'] += 1
            counts['source_terminals_with_explicit_reasoning'] += any(
                isinstance(terminal.get(k), str) and bool(terminal[k]) for k in ('reasoning', 'reasoning_content'))
        stats = {}
        try:
            converted = convert_trajectory(row, preserve_reasoning=True, export_stats=stats)
        except (UnsupportedAction, KeyError, ValueError, TypeError) as error:
            # Exception messages can contain upstream identifiers: publish only types.
            errors[type(error).__name__] += 1
            continue
        counts['converter_accepted_trajectories'] += 1
        counts['fully_convertible_trajectories'] += not bool(xml)
        counts['accepted_with_xml_omissions'] += bool(xml)
        counts['terminal_reasoning_dropped_in_accepted'] += stats.get('terminal_reasoning_dropped', 0)
        counts['emitted_terminal_messages'] += bool(converted[-1]['role'] == 'assistant' and not converted[-1].get('tool_calls'))
    def ratio(a, b):
        return counts[a] / counts[b] if counts[b] else None
    return {'counts': dict(counts), 'conversion_errors_by_type': dict(errors),
            'fully_convertible_fraction_of_resolved': ratio('fully_convertible_trajectories', 'resolved_trajectories'),
            'xml_only_fraction_of_assistant_messages': ratio('xml_only_messages', 'resolved_assistant_messages'),
            'source_terminal_explicit_reasoning_fraction': ratio('source_terminals_with_explicit_reasoning', 'source_terminal_messages'),
            'definitions': {
                'fully_convertible': 'Converter accepts with preserve_reasoning=True and no XML-only assistant actions omitted; no tool replay or target-equivalence claim.',
                'xml_only': 'Assistant without native tool_calls containing a <function=... action; standalone thought/final prose is not XML.',
                'source_terminal': 'Last source message is nonempty assistant prose without native tool_calls or XML action.',
                'reasoning': 'Nonempty explicit reasoning/reasoning_content, never inferred from terminal content.',
                'scope': 'All resolved rows before deduplication, reserved-snapshot filtering or export caps; not a training selection.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior receipts')
    digest = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    if digest != SHARD_SHA256:
        raise ValueError('Shard hash mismatch')
    import pyarrow.parquet as pq
    result = coverage(pq.read_table(args.parquet).to_pylist())
    result.update(dataset=DATASET, revision=REVISION, shard_sha256=digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
