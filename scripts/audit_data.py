"""Inspect competition data structure without running any supplied code.

Run on a datamover: archive headers require reading all compressed snapshots.
"""
import collections
import datetime
import email
import json
from pathlib import Path
import struct
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/gemma-4-developer-agent"
OUT = ROOT / "resource/manifests"


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


tasks = [json.loads(line) for line in (DATA / "tasks.jsonl").read_text().splitlines()]
summary = {"retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "tasks": len(tasks), "repos": dict(collections.Counter(t["repo"] for t in tasks)), "unique_repo_commits": len({(t['repo'], t['base_commit']) for t in tasks})}
task_rows = []
for t in tasks:
    stem = t['repo'].split('/')[-1] + '_' + t['base_commit']
    task_rows.append({**{k: t[k] for k in ['instance_id', 'repo', 'base_commit', 'created_at']}, "prompt_chars": len(t['problem_statement']), "hint_chars": len(t['hints_text']), "patch_chars": len(t['patch']), "test_patch_chars": len(t['test_patch']), "graph": 'graphs/' + stem + '.json', "graph_exists": (DATA / 'graphs' / (stem + '.json')).exists(), "embedding_exists": (DATA / 'embeddings' / (stem + '.npz')).exists(), "snapshot_exists": (DATA / 'snapshots' / (t['instance_id'] + '.tgz')).exists()})
save('tasks-audit.json', task_rows)

graphs = []
for p in sorted((DATA / 'graphs').glob('*.json')):
    g = json.loads(p.read_text())
    nodes = g.get('nodes', [])
    edges = g.get('edges', g.get('links', []))
    graphs.append({'file': p.name, 'keys': list(g), 'nodes': len(nodes), 'edges': len(edges), 'node_fields': sorted({k for n in nodes for k in n}), 'edge_types': dict(collections.Counter(str(e.get('type')) for e in edges))})
save('graphs-audit.json', graphs)

embeddings = []
for p in sorted((DATA / 'embeddings').glob('*.npz')):
    with zipfile.ZipFile(p) as z:
        headers = collections.Counter()
        for n in z.namelist():
            with z.open(n) as f:
                prefix = f.read(8)
                if prefix[:6] != b'\x93NUMPY':
                    raise ValueError('Invalid NPY header: ' + str(p))
                count = struct.unpack('<H' if prefix[6] == 1 else '<I', f.read(2 if prefix[6] == 1 else 4))[0]
                headers[f.read(count).decode().strip()] += 1
        embeddings.append({'file': p.name, 'vectors': len(z.namelist()), 'array_headers': dict(headers)})
save('embeddings-audit.json', embeddings)

wheels = []
for p in sorted((DATA / 'wheels').glob('*.whl')):
    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
        m = email.message_from_bytes(z.read(name))
        wheels.append({'file': p.name, 'package': m['Name'], 'version': m['Version'], 'requires_python': m['Requires-Python'], 'requires_dist': m.get_all('Requires-Dist', [])})
save('wheels-audit.json', wheels)

snapshots = []
for i, p in enumerate(sorted((DATA / 'snapshots').glob('*.tgz'))):
    count = size = links = 0
    top = collections.Counter()
    with tarfile.open(p, 'r|gz') as archive:
        for member in archive:
            count += 1
            size += member.size
            links += member.issym() or member.islnk()
            top[member.name.lstrip('./').split('/')[0]] += 1
    snapshots.append({'file': p.name, 'members': count, 'expanded_bytes': size, 'links': links, 'top_level_counts': dict(top)})
    if i % 20 == 0:
        print('snapshots', i + 1, flush=True)
save('snapshots-audit.json', snapshots)
summary.update(graphs=len(graphs), embeddings=len(embeddings), wheels=len(wheels), snapshots=len(snapshots), snapshot_expanded_bytes=sum(x['expanded_bytes'] for x in snapshots), missing_task_assets=[t['instance_id'] for t in task_rows if not all(t[k] for k in ['graph_exists', 'embedding_exists', 'snapshot_exists'])])
save('data-audit.json', summary)
print(json.dumps(summary), flush=True)
