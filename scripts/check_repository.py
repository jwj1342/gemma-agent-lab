"""Lightweight checks of versioned team code, agent packages, and local documentation links."""
import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit

import yaml

from build_submission import build


ROOT = Path(__file__).resolve().parents[1]


class Include(str):
    """An inert agent !include reference, not executable YAML."""


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor('!include', lambda loader, node: Include(loader.construct_scalar(node)))


def references(value, seen=None):
    """Yield local file references without following YAML alias cycles."""
    seen = set() if seen is None else seen
    if isinstance(value, Include):
        yield str(value)
    elif isinstance(value, (dict, list)) and id(value) not in seen:
        seen.add(id(value))
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'config_path' and isinstance(child, str):
                    yield child
                else:
                    yield from references(child, seen)
        else:
            for child in value:
                yield from references(child, seen)


def markdown_targets(text):
    """Read inline destinations and reference definitions outside fenced/inline code."""
    lines, fence = [], None
    for line in text.splitlines():
        match = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    prose = re.sub(r'`+[^`\n]*`+', '', '\n'.join(lines))
    targets = re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', prose)
    targets += re.findall(r'^\s*\[[^\]\n]+\]:\s*(<[^>]+>|\S+)', prose, re.MULTILINE)
    for target in targets:
        target = target.strip()
        if target.startswith('<'):
            yield target[1:target.index('>')]
        elif target:
            yield target.split()[0]


def tracked_destination(root, source, target, tracked, directories=False):
    """Require a local link/reference to exist in Git, not just in this checkout."""
    url = urlsplit(target)
    if url.scheme or url.netloc or not url.path:
        return True
    raw_path = unquote(url.path)
    path = (root / raw_path.lstrip('/') if raw_path.startswith('/') else source.parent / raw_path).resolve()
    if not path.is_relative_to(root):
        return False
    relative = path.relative_to(root)
    if relative in tracked and path.is_file():
        return True
    return directories and path.is_dir() and any(p.is_relative_to(relative) for p in tracked)


def team_markdown(path):
    return path.suffix == '.md' and (path.parts[0] != 'resource' or path.name == 'README.md')


def check(root):
    output = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'])
    tracked = {Path(name.decode()) for name in output.split(b'\0') if name}
    errors = []
    counts = {'python': 0, 'shell': 0, 'configuration': 0, 'markdown': 0, 'agent_packages': 0}
    for relative in sorted(tracked):
        path = root / relative
        try:
            if relative.parts[0] == 'slides':
                raise ValueError('Slides belong in the separate Overleaf workspace, outside Git')
            if relative.parts[0] in {'scripts', 'src', 'tests'}:
                if path.suffix == '.py':
                    ast.parse(path.read_text(), filename=str(relative))
                    counts['python'] += 1
                elif path.suffix == '.sh':
                    result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                    if result.returncode:
                        raise ValueError(result.stderr.strip())
                    counts['shell'] += 1
            if relative.parts[0] in {'configs', '.github'}:
                if path.suffix == '.json':
                    json.loads(path.read_text())
                    counts['configuration'] += 1
                elif path.suffix in {'.yml', '.yaml'}:
                    config = yaml.load(path.read_text(), Loader=ConfigLoader)
                    for target in references(config):
                        included = (path.parent / target).resolve()
                        if (not included.is_relative_to(root) or included.relative_to(root) not in tracked
                                or not included.is_file()):
                            raise ValueError(f'Untracked or missing configuration reference: {target}')
                    counts['configuration'] += 1
            if team_markdown(relative):
                for target in markdown_targets(path.read_text()):
                    if not tracked_destination(root, path, target, tracked, directories=True):
                        errors.append(f'{relative}: untracked or missing local link: {target}')
                counts['markdown'] += 1
        except (OSError, ValueError, SyntaxError, yaml.YAMLError) as error:
            errors.append(f'{relative}: {error}')
    with tempfile.TemporaryDirectory(prefix='gemma-ci-') as temporary:
        for index, relative in enumerate(sorted(tracked)):
            if relative.parts[0] == 'configs' and relative.name == 'agent.yaml':
                try:
                    agent = root / relative.parent
                    for path in agent.rglob('*'):
                        if path.is_file() and path.relative_to(root) not in tracked:
                            raise ValueError(f'Untracked package member: {path.relative_to(root)}')
                    first = build(agent, Path(temporary) / f'{index}-first.zip')
                    second = build(agent, Path(temporary) / f'{index}-second.zip')
                    if first['sha256'] != second['sha256']:
                        raise ValueError('Identical agent input produced different ZIP bytes')
                    counts['agent_packages'] += 1
                except (OSError, ValueError, AssertionError) as error:
                    errors.append(f'{relative.parent}: {error}')
    if not counts['agent_packages']:
        errors.append('No versioned agent package found')
    print(json.dumps(counts, indent=2))
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Repository checks passed. GPU/runtime validation remains a separate experiment.')


if __name__ == '__main__':
    check(ROOT)
