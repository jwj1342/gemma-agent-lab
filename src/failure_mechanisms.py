"""Read-only release signals; source exposure never establishes semantic diagnosis.

Reference hunks include context: overlap is only an exposure proxy. Successful
editor calls, attempted edits and final patch paths are deliberately separate.
All case data and reasoning exports belong in an ignored private directory.
"""
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def path_normalize(path):
    return str(path).removeprefix('/workspace/').removeprefix('./')


def events(trace):
    result = []
    for si, step in enumerate(trace.get('steps', [])):
        calls = step.get('tool_calls') or []
        for call in calls:
            args = decode(call.get('arguments', {}))
            obs = decode((step.get('observation') or {}).get('content', {}))
            # No ambiguous assignment of a shared observation to multiple calls.
            if len(calls) != 1 or not isinstance(obs, dict):
                obs = {}
            result.append(dict(call=len(result) + 1, step_id=step.get('step_id'),
                step_index=si, tool=call.get('function_name'),
                args=args if isinstance(args, dict) else {}, obs=obs,
                thinking=step.get('message', ''),
                elapsed_s=(call.get('extra') or {}).get('elapsed_s')))
    return result


def views(event):
    """Only returned source with explicit positions; do not infer reads from args.

    Numbered grep lines are single-line evidence. Arbitrary shell and unnumbered
    output are retained for manual review rather than guessed into ranges.
    """
    a, o = event['args'], event['obs']
    if event['tool'] == 'read_file' and o.get('status') == 'ok':
        text = o.get('content', '')
        start = o.get('start_line')
        if isinstance(start, int) and text:
            end = min(o.get('end_line', start + len(text.splitlines()) - 1),
                      start + len(text.splitlines()) - 1)
            return [(path_normalize(o.get('filepath', a.get('filepath', ''))),
                     start, end, 'read_file', text)]
    if event['tool'] != 'run_command':
        return []
    out = o.get('stdout', o.get('details', {}).get('stdout', ''))
    command = a.get('command', '')
    # A single explicit .py path can qualify `123:source` grep output.
    paths = set(path_normalize(p) for p in re.findall(
        r'(?<![\w.])(?:/workspace/)?(?:[\w.-]+/)*[\w.-]+\.py\b', command))
    found = []
    for line in out.splitlines():
        m = re.match(r'^((?:/workspace/)?[\w./-]+\.py)[:\-](\d+)[:\-](.*)$', line)
        if m:
            found.append((path_normalize(m[1]), int(m[2]), int(m[2]),
                          'numbered_shell', m[3]))
        else:
            m = re.match(r'^(\d+)[:\-](.*)$', line)
            if m and len(paths) == 1 and re.search(r'\b(?:grep|rg)\b', command):
                found.append((next(iter(paths)), int(m[1]), int(m[1]),
                              'numbered_shell', m[2]))
    return found


def overlaps(start, end, hunk):
    return (hunk['old_len'] > 0 and start <= hunk['old_start'] + hunk['old_len'] - 1
            and end >= hunk['old_start'])


def diff_records(patch):
    """Parse headers only outside counted unified-diff hunks."""
    old_path = new_path = None
    old_left = new_left = 0
    old_line = 0
    for line in patch.splitlines():
        if old_left or new_left:
            if line.startswith('\\ No newline'):
                continue
            kind = line[:1]
            if kind not in (' ', '-', '+'):
                raise ValueError('Malformed or truncated diff hunk')
            yield ('body', old_path, old_line, kind)
            if kind in (' ', '-'):
                old_left -= 1
                old_line += 1
            if kind in (' ', '+'):
                new_left -= 1
            if old_left < 0 or new_left < 0:
                raise ValueError('Diff hunk exceeds declared lengths')
            continue
        if line.startswith('diff --git '):
            old_path = new_path = None
        elif line.startswith(('--- ', '+++ ')):
            raw = line[4:].split('\t', 1)[0]
            prefix = 'a/' if line.startswith('--- ') else 'b/'
            if raw == '/dev/null':
                path = None
            elif raw.startswith(prefix):
                path = raw[2:]
            else:
                raise ValueError('Unsupported reference diff path')
            if prefix == 'a/':
                old_path, new_path = path, None
            else:
                new_path = path
            yield ('path', path, 0, '')
        elif line.startswith('@@ '):
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if not match or (old_path is None and new_path is None):
                raise ValueError('Malformed reference hunk/path')
            old_line = int(match[1])
            old_left, new_left = int(match[2] or 1), int(match[4] or 1)
            yield ('hunk', old_path or new_path, old_line, old_left)
    if old_left or new_left:
        raise ValueError('Truncated diff hunk')


def patch_paths(patch):
    return sorted({path for kind, path, _, _ in diff_records(patch)
                   if kind == 'path' and path is not None})


def changed_old_lines(patch):
    result = {}
    for kind, path, line, operation in diff_records(patch):
        if kind == 'body' and operation == '-' and path is not None:
            result.setdefault(path, set()).add(line)
    return result


def reference_hunks(patch):
    result = {}
    for kind, path, line, length in diff_records(patch):
        if kind == 'hunk':
            result.setdefault(path, []).append({'old_start': line, 'old_len': length})
    return result


def prediction_paths(prediction):
    """Missing or invalid patch evidence is unknown, never an empty diff."""
    if not isinstance(prediction, dict) or not isinstance(prediction.get('agent_patch'), str):
        return None
    return patch_paths(prediction['agent_patch'])


def statement_exposure(trace, patch):
    changed = changed_old_lines(patch)
    covered = {path: set() for path in changed}
    for event in events(trace):
        for path, start, end, _, _ in views(event):
            if path in changed:
                covered[path].update(n for n in changed[path] if start <= n <= end)
    return {'changed_old_lines': sum(map(len, changed.values())),
            'exposed_changed_old_lines': sum(map(len, covered.values())),
            'scope': 'Position-based old-line coverage; no new-code or semantic diagnosis claim'}


CATEGORIES = ('no_decisive_source_exposure', 'diagnosis_unresolved',
              'recognized_not_applied', 'wrong_or_incomplete_patch',
              'verification_or_completion', 'insufficient_evidence')


def apply_annotation(row, annotation, trace):
    """Require an identified reviewer, frozen identity and exact excerpts for labels."""
    if annotation is None:
        return {'category': 'insufficient_evidence', 'reviewed': False}
    category = annotation.get('category')
    if category not in CATEGORIES:
        raise ValueError('Unknown mechanism category')
    if annotation.get('trace_sha256') != row.get('trace_sha256') or trace is None:
        raise ValueError('Annotation trace identity mismatch or missing trace')
    if annotation.get('prediction_sha256') != row.get('prediction_sha256'):
        raise ValueError('Annotation prediction identity mismatch')
    if not annotation.get('reviewer', '').strip() or not annotation.get('rationale', '').strip():
        raise ValueError('Annotations need a reviewer and rationale')
    evidence = annotation.get('evidence', [])
    if not evidence:
        raise ValueError('Annotations need exact trace evidence, even for uncertainty')
    for item in evidence:
        field, quote = item.get('field'), item.get('quote')
        if field not in ('message', 'observation') or not isinstance(quote, str) or not quote.strip():
            raise ValueError('Invalid annotation evidence')
        matching = [step for step in trace.get('steps', []) if step.get('step_id') == item.get('step_id')]
        if len(matching) != 1:
            raise ValueError('Evidence step must identify one trace step')
        value = matching[0].get(field, '')
        if field == 'observation':
            value = value.get('content', '') if isinstance(value, dict) else ''
        if not isinstance(value, str) or quote not in value:
            raise ValueError('Annotation quote is absent from the cited field')
    if category == 'recognized_not_applied' and annotation.get('source_supported') is not True:
        raise ValueError('Recognized changes require explicit source-support review')
    if category == 'recognized_not_applied' and {e['field'] for e in evidence} != {'message', 'observation'}:
        raise ValueError('Recognized changes need both reasoning and source-observation evidence')
    if category == 'wrong_or_incomplete_patch' and not row.get('final_patch_paths'):
        raise ValueError('Wrong/incomplete patch label requires a nonempty observed final patch')
    return {**annotation, 'reviewed': True}


def terminal_flags(row):
    """Endpoint metadata, kept independent from the human mechanism label."""
    error = str(row['outcome'].get('error') or '')
    return {'reported_timeout': bool(re.search(r'timeout|time limit', error, re.I)),
            'budget_rejection': row.get('observed_budget_rejection', False),
            'reported_error': error or None,
            'scope': 'Only recorded metadata; false is not proof the limit was not reached'}


def audit(trace, prediction, reference):
    ev = events(trace)
    exposure, attempts, shell_flags = [], [], []
    for e in ev:
        for path, start, end, kind, content in views(e):
            hs = [h for h in reference.get(path, []) if overlaps(start, end, h)]
            if hs:
                exposure.append(dict(call=e['call'], step_id=e['step_id'], path=path,
                    start=start, end=end, kind=kind, hunks=hs,
                    full_hunk=any(start <= h['old_start'] and
                        end >= h['old_start'] + h['old_len'] - 1 for h in hs)))
        if e['tool'] in ('edit_file', 'write_file'):
            attempts.append(dict(call=e['call'], step_id=e['step_id'],
                path=path_normalize(e['args'].get('filepath', '')),
                status=e['obs'].get('status', 'unknown'),
                error_type=e['obs'].get('error_type'),
                error_message=e['obs'].get('error_message')))
        if e['tool'] == 'run_command' and re.search(
                r'write_text|write_bytes|\.write\(|sed\s+-i|git\s+(?:apply|checkout|restore)|open\(',
                e['args'].get('command', '')):
            shell_flags.append(dict(call=e['call'], command=e['args']['command'],
                                    status=e['obs'].get('status', 'unknown')))
    readpaths = {x['path'] for x in exposure}
    attempted = {x['path'] for x in attempts}
    edited = {x['path'] for x in attempts if x['status'] == 'ok'}
    final = prediction_paths(prediction)
    return dict(recorded_calls=len(ev), reference_exposures=exposure,
        observed_budget_rejection=any(e['obs'].get('error_type') == 'BudgetExceeded' for e in ev),
        editor_attempts=attempts, possible_shell_mutations=shell_flags,
        final_patch_paths=final,
        read_reference=bool(readpaths),
        read_reference_via_read_file=any(x['kind'] == 'read_file' for x in exposure),
        attempted_reference=bool(attempted & reference.keys()),
        successful_editor_reference=bool(edited & reference.keys()),
        final_patch_reference=(bool(set(final or []) & reference.keys()) if final is not None else None),
        read_no_reference_attempt=bool(readpaths) and not bool(attempted & reference.keys()),
        read_no_reference_patch=(bool(readpaths) and not bool(set(final or []) & reference.keys()) if final is not None else None),
        exposed_paths_not_in_final_patch=(sorted(readpaths - set(final)) if final is not None else None),
        prompt_tokens=trace.get('final_metrics', {}).get('total_prompt_tokens'),
        completion_tokens=trace.get('final_metrics', {}).get('total_completion_tokens'),
        graph_calls=[e['call'] for e in ev if e['tool'] in (
            'get_code_neighbors', 'search_similar_code', 'get_code_subgraph')])


def load_run(root, name, reference):
    run = root / name
    summary_path = run / ('cohort-summary.json' if (run / 'cohort-summary.json').exists() else 'summary.json')
    summary = json.loads(summary_path.read_text())
    rows = summary.get('tasks', [])
    # r4 stores the same outcomes as a per-task dictionary.
    if isinstance(rows, dict):
        rows = [dict(v, task_id=k) for k, v in rows.items()]
    results = []
    seen = set()
    for task in rows:
        tid = task['task_id']
        if not isinstance(tid, str) or Path(tid).name != tid or tid in ('.', '..') or tid in seen:
            raise ValueError('Unsafe or duplicate task identity in run summary')
        seen.add(tid)
        d = run / 'tasks' / tid
        tr, pred = d / 'trace.json', d / 'prediction.json'
        result = dict(run=name, task_id=tid, outcome=task,
                      trace_present=tr.exists(), prediction_present=pred.exists())
        prediction = json.loads(pred.read_text()) if pred.exists() else None
        if tr.exists():
            result.update(audit(json.loads(tr.read_text()),
                prediction, reference[tid]))
            result['trace_sha256'] = sha(tr)
        elif prediction is not None:
            result['final_patch_paths'] = prediction_paths(prediction)
            result['final_patch_reference'] = (bool(set(result['final_patch_paths']) & reference[tid].keys())
                                                if result['final_patch_paths'] is not None else None)
        if pred.exists():
            result['prediction_sha256'] = sha(pred)
        results.append(result)
    return results
