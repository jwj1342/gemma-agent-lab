"""Read-only, partial Python source graph for teacher mechanism experiments.

Edges resolve same-file functions and explicit self/Class methods only. Dynamic receivers,
imports and nested function bodies are deliberately not guessed. This is not the official ADK graph.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time


class Calls(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self.bound = set()

    def visit_Call(self, node):
        self.calls.append((ast.unparse(node.func), node.lineno))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.bound.add(node.id)

    def visit_Import(self, node):
        self.bound.update(alias.asname or alias.name.split('.')[0] for alias in node.names)

    def visit_ImportFrom(self, node):
        self.bound.update(alias.asname or alias.name for alias in node.names)


def build_graph(root):
    nodes, edges = {}, set()
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if any(part.startswith('.') or part in {'venv', '__pycache__'} for part in relative.parts):
            continue
        try:
            source = path.read_text(encoding='utf-8-sig')
            tree = ast.parse(source)
        except (UnicodeError, SyntaxError):
            continue
        definitions = []
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.append(('', item))
            elif isinstance(item, ast.ClassDef):
                definitions.extend((item.name, method) for method in item.body
                                   if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)))
        for class_name, function in definitions:
            symbol = class_name + '.' + function.name if class_name else function.name
            identity = relative.as_posix() + '::' + symbol
            visitor = Calls()
            visitor.bound.update(arg.arg for arg in [*function.args.posonlyargs, *function.args.args,
                                                     *function.args.kwonlyargs])
            for statement in function.body:
                visitor.visit(statement)
            nodes[identity] = {'id': identity, 'symbol': symbol, 'path': relative.as_posix(),
                               'line': function.lineno, 'end_line': function.end_lineno,
                               'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'class_name': class_name, 'calls': visitor.calls,
                               'bound_names': visitor.bound}
    for identity, node in nodes.items():
        unresolved = []
        bound_names = node.pop('bound_names')
        for token, line in node.pop('calls'):
            candidate = token
            if token.startswith('self.') and node['class_name']:
                candidate = node['class_name'] + token[4:]
            target = node['path'] + '::' + candidate
            if target in nodes and token.split('.')[0] not in bound_names - {'self'}:
                edges.add((identity, target))
            else:
                unresolved.append({'expression': token, 'line': line})
        node['unresolved_calls'] = unresolved
    return nodes, edges


def describe(node):
    return {key: node[key] for key in ('id', 'symbol', 'path', 'line', 'end_line')}


def query(root, symbol=None, node_id=None):
    nodes, edges = build_graph(root)
    result = {'scope': 'Partial syntactic same-file graph; missing edges do not imply absent calls.',
              'nodes': len(nodes), 'syntactic_edges': len(edges)}
    if symbol:
        matches = [node for node in nodes.values() if symbol.lower() in node['symbol'].lower()]
        result['matches'] = [describe(node) for node in matches[:30]]
        result['match_count'] = len(matches)
    else:
        if node_id not in nodes:
            raise ValueError('Unknown node; find its exact ID with --symbol first')
        node = nodes[node_id]
        result.update(node=describe(node), source_sha256=node['source_sha256'],
                      callers=[describe(nodes[a]) for a, b in sorted(edges) if b == node_id][:30],
                      callees=[describe(nodes[b]) for a, b in sorted(edges) if a == node_id][:30],
                      unresolved_calls=node['unresolved_calls'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--symbol')
    selection.add_argument('--node')
    parser.add_argument('--trace', type=Path, help='Append request/result evidence to this experiment JSONL')
    args = parser.parse_args()
    started = time.perf_counter()
    result = query(args.repo.resolve(strict=True), args.symbol, args.node)
    result['seconds'] = time.perf_counter() - started
    text = json.dumps(result, ensure_ascii=False)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps({'symbol': args.symbol, 'node': args.node,
                                     'output_bytes': len(text.encode()), 'result': result}) + '\n')
    print(text)


if __name__ == '__main__':
    main()
