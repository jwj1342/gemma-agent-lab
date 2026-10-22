"""Check real smoke artifacts without confusing model failures with environment failures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def ordinary_test_failure(result):
    return (result.get('resolved') is False
            and result.get('test_exit_code') == 1
            and not result.get('error_message')
            and re.search(r'\b[1-9][0-9]* failed\b', result.get('test_output', '')) is not None
            and 'error during collection' not in result.get('test_output', '').lower())


def control_checks(empty, gold, task_id):
    if empty.get('task_id') != task_id or gold.get('task_id') != task_id:
        raise ValueError('Control results refer to different tasks')
    return {'empty_patch_fails_tests': ordinary_test_failure(empty),
            'gold_patch_passes_tests': (gold.get('resolved') is True
                                       and gold.get('test_exit_code') == 0
                                       and not gold.get('error_message'))}


def summarize(run_dir, controls_only=False):
    empty = read(run_dir / 'verify/empty/result.json')
    gold = read(run_dir / 'verify/gold/result.json')
    checks = control_checks(empty, gold, gold['task_id'])
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'task_id': gold['task_id'],
              'checks': checks, 'controls_valid': all(checks.values())}
    if not controls_only:
        compilation = read(run_dir / 'compile/validation.json')
        prediction = read(run_dir / 'infer/prediction.json')
        agent = read(run_dir / 'verify/agent/result.json')
        if prediction['task_id'] != gold['task_id'] or agent['task_id'] != gold['task_id']:
            raise ValueError('Agent results refer to a different task')
        checks['official_compiler_passed'] = compilation.get('status') == 'compiled'
        checks['real_nonempty_prediction'] = bool(prediction.get('agent_patch')) and not prediction.get('error')
        checks['llm_was_called'] = prediction.get('llm_calls', 0) > 0
        checks['agent_tests_executed'] = (agent.get('resolved') is True
                                          and agent.get('test_exit_code') == 0
                                          and not agent.get('error_message')) or ordinary_test_failure(agent)
        result['agent_resolved'] = agent.get('resolved')
        result['inference_seconds'] = prediction.get('duration_seconds')
        result['pipeline_executed'] = all(checks.values())
        result['scope'] = 'One public task. Not a benchmark score, training validation, L4 timing result, or Kaggle submission.'
    else:
        result['scope'] = 'Empty/gold controls only. No inference evidence in this check.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--controls-only', action='store_true')
    args = parser.parse_args()
    result = summarize(args.run_dir, args.controls_only)
    output = args.run_dir / ('controls-summary.json' if args.controls_only else 'smoke-summary.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not all(result['checks'].values()):
        raise SystemExit('Smoke validation failed; inspect the individual test and inference logs.')
