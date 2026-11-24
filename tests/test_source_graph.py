"""Check the teacher graph's useful links and conservative handling of ambiguous calls."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from source_graph import build_graph, query


class SourceGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'sample.py').write_text('''
def helper():
    return 1
def consumer():
    return helper()
def shadowed(helper):
    return helper()
def dynamic(obj):
    return obj.helper()
def outer():
    def inner():
        return helper()
    return inner()
class Engine:
    def helper(self):
        return 2
    def run(self):
        return self.helper()
''')

    def test_same_file_edges_do_not_guess_dynamic_or_shadowed_receivers(self):
        nodes, edges = build_graph(self.root)
        self.assertIn(('sample.py::consumer', 'sample.py::helper'), edges)
        self.assertIn(('sample.py::Engine.run', 'sample.py::Engine.helper'), edges)
        self.assertNotIn(('sample.py::shadowed', 'sample.py::helper'), edges)
        self.assertNotIn(('sample.py::dynamic', 'sample.py::helper'), edges)
        self.assertNotIn(('sample.py::outer', 'sample.py::helper'), edges)
        self.assertEqual(nodes['sample.py::dynamic']['unresolved_calls'][0]['expression'], 'obj.helper')

    def test_ambiguous_search_returns_explicit_ids_and_unknown_nodes_fail(self):
        result = query(self.root, symbol='helper')
        self.assertEqual(result['match_count'], 2)
        self.assertEqual([r['id'] for r in query(self.root, node_id='sample.py::helper')['callers']],
                         ['sample.py::consumer'])
        with self.assertRaises(ValueError):
            query(self.root, node_id='missing')


if __name__ == '__main__':
    unittest.main()
