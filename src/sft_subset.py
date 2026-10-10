"""Select a bounded training subset without changing the held-out population."""
from collections import defaultdict, deque
import hashlib


def select_training_subset(samples, budgets, seed):
    """Round-robin over snapshot/tool strata with deterministic hashed ordering."""
    if not budgets or any(type(n) is not int or n < 1 for n in budgets.values()):
        raise ValueError('Subset budgets must be positive integers')
    if set(budgets) != {row['kind'] for row in samples}:
        raise ValueError('Specify a budget for every training source')
    ids = [row['sample_id'] for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('Deduplicate the corpus before subset selection')

    def order(value):
        return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()

    selected = []
    coverage = {}
    for kind, limit in sorted(budgets.items()):
        population = [row for row in samples if row['kind'] == kind]
        if len(population) < limit:
            raise ValueError(f'Insufficient {kind} actions for the registered budget')
        strata = defaultdict(list)
        for row in population:
            key = (row['metadata']['group'].lower(), row['target_tool'])
            strata[key].append(row)
        queues = deque(deque(sorted(strata[key], key=lambda row: order(row['sample_id'])))
                       for key in sorted(strata, key=order))
        chosen = []
        while len(chosen) < limit:
            bucket = queues.popleft()
            chosen.append(bucket.popleft())
            if bucket:
                queues.append(bucket)
        selected.extend(chosen)
        coverage[kind] = {
            'population_actions': len(population), 'selected_actions': len(chosen),
            'population_strata': len(strata),
            'selected_strata': len({(r['metadata']['group'].lower(), r['target_tool']) for r in chosen}),
            'selected_groups': len({r['metadata']['group'].lower() for r in chosen}),
            'tools': sorted({r['target_tool'] for r in chosen}),
        }
    return sorted(selected, key=lambda row: order(row['sample_id'])), coverage
