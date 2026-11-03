"""Render resolved SWE-smith teacher trajectories (SWE-agent XML calls) as SFT rows in our harness's frame.

Unlike src/sft_data.convert_trajectory, the teacher's text before each call is kept as `reasoning`, and the system and
user prompts are the budget-fit agent's own (issue in the harness template, budget lines included, no directory tree),
so a row looks like what our agent sees at inference. Tools map through src/sft_data.convert_call; upstream
observations are wrapped in our tools' JSON envelopes. Only short trajectories are kept so they fit our 28-call budget.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sft_data import UnsupportedAction, action_examples, convert_call, text_content  # noqa: E402

CALL = re.compile(r'<function=([\w-]+)>(.*?)</function>', re.DOTALL)
PARAMETER = re.compile(r'<parameter=(\w+)>(.*?)</parameter>', re.DOTALL)
ISSUE = re.compile(r'<pr_description>\n?(.*?)\n?</pr_description>', re.DOTALL)
EXCLUDED_REPOS = ('textualize__rich', 'fastapi__fastapi', 'psf__requests', 'encode__httpx')


def local(text):
    return text.replace('/testbed/', '').replace('/testbed', '.')


def parse_call(content, index):
    """The reasoning before the single XML call and the call in OpenAI form (SWE-agent emits one call per turn)."""
    calls = CALL.findall(content)
    if len(calls) != 1:
        raise UnsupportedAction('Expected exactly one call per assistant turn')
    name, body = calls[0]
    arguments = {key: value.strip('\n') if key not in ('old_str', 'new_str', 'file_text') else value
                 for key, value in PARAMETER.findall(body)}
    if 'view_range' in arguments:
        arguments['view_range'] = json.loads(arguments['view_range'])
    call = convert_call({'id': f'call_{index}', 'function': {'name': name, 'arguments': arguments}})
    if call['function']['name'] == 'read_file' and '.' not in Path(call['function']['arguments']['filepath']).name:
        raise UnsupportedAction('Directory view has no read_file equivalent')
    for key in ('command', 'old_string', 'new_string', 'content'):
        if key in call['function']['arguments']:
            call['function']['arguments'][key] = local(call['function']['arguments'][key])
    return content[:content.index('<function=')].strip(), call


def envelope(call, observation):
    name, arguments = call['function']['name'], call['function']['arguments']
    if name == 'run_command':
        return {'status': 'ok', 'stdout': observation[:5000], 'stderr': ''}
    if name in ('read_file', 'edit_file', 'write_file'):
        return {'status': 'ok', 'filepath': arguments['filepath'], 'content': observation[:10000]}
    return {'status': 'ok'}


def convert(row, system, template, max_calls):
    if row['resolved'] is not True or row['instance_id'].lower().startswith(EXCLUDED_REPOS):
        raise UnsupportedAction('Unresolved or a competition repository')
    originals = [{**m, 'content': text_content(m.get('content'))} for m in json.loads(row['messages'])]
    issue = ISSUE.search(originals[1]['content'])
    if not issue:
        raise UnsupportedAction('No PR description')
    repo = row['instance_id'].split('.')[0].replace('__', '/')
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': template.format(repo=repo, issue=local(issue.group(1)))}]
    pending = None
    for original in originals[2:]:
        content = original['content']
        if original['role'] == 'assistant':
            if '<function=' not in content:
                if pending is None and messages[-1]['role'] == 'tool' and messages[-1]['name'] == 'submit_patch':
                    messages.append({'role': 'assistant', 'content': content.strip()})
                break
            reasoning, pending = parse_call(content, len(messages))
            messages.append({'role': 'assistant', 'content': '', 'reasoning': reasoning, 'tool_calls': [pending]})
        elif original['role'] == 'user' and pending:
            observation = local(content.removeprefix('OBSERVATION:').strip('\n'))
            messages.append({'role': 'tool', 'tool_call_id': pending['id'], 'name': pending['function']['name'],
                             'content': json.dumps(envelope(pending, observation))})
            pending = None
    calls = [m for m in messages if m.get('tool_calls')]
    if not calls or len(calls) > max_calls or calls[-1]['tool_calls'][0]['function']['name'] != 'submit_patch':
        raise UnsupportedAction('Too long or does not end with a submission')
    if any(m['tool_calls'][0]['function']['name'] == 'write_file' for m in calls):
        # The teacher writes reproduction scripts into the repository; our prompt forbids scratch files there.
        raise UnsupportedAction('Creates files')
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, nargs='+', required=True)
    parser.add_argument('--prompt-trace', type=Path, required=True,
                        help='One of our harness traces; its system prompt and user template are reused')
    parser.add_argument('--prompt-issue', required=True, help='The issue text inside that trace\'s user message')
    parser.add_argument('--max-calls', type=int, default=28)
    parser.add_argument('--max-trajectories', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    trace = json.loads(args.prompt_trace.read_text())
    user = trace['steps'][1]['message']
    head, tail = user.split(args.prompt_issue)
    head = head.replace(head.split('repository ', 1)[1].split('.\n', 1)[0], '{repo}', 1)
    template = head.replace('{', '{{').replace('}', '}}').replace('{{repo}}', '{repo}') + '{issue}' + \
        tail.split('## Workspace Layout')[0].replace('{', '{{').replace('}', '}}')
    rows, rejected, tasks = [], {}, set()
    for path in args.parquet:
        for row in pq.read_table(path).to_pylist():
            if len(tasks) >= args.max_trajectories:
                break
            if row['instance_id'] in tasks:
                continue
            try:
                messages = convert(row, trace['steps'][0]['message'], template, args.max_calls)
            except (UnsupportedAction, KeyError, ValueError) as error:
                rejected[str(error)] = rejected.get(str(error), 0) + 1
                continue
            tasks.add(row['instance_id'])
            metadata = {'source': 'SWE-bench/SWE-smith-trajectories', 'license': 'MIT', 'trace': row['traj_id'],
                        'task': 'teacher_' + row['instance_id'], 'teacher_model': row['model']}
            rows.extend(action_examples(messages, metadata))
    with args.output.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    print(json.dumps({'trajectories': len(tasks), 'rows': len(rows), 'rejected': rejected}, indent=2))


if __name__ == '__main__':
    main()
