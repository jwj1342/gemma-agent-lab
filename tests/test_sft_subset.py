"""Check bounded sampling, diversity, and reproducibility without model dependencies."""
import unittest
from src.sft_subset import select_training_subset


class SubsetTests(unittest.TestCase):
    def population(self):
        return [{'sample_id': f'{kind}:{group}:{tool}:{n}', 'kind': kind,
                 'target_tool': tool, 'metadata': {'group': group}}
                for kind in ('swe', 'graph') for group in ('repo.a', 'repo.b')
                for tool in ('read', 'write') for n in range(5)]

    def test_budgets_diversity_and_order_invariance(self):
        rows = self.population()
        chosen, coverage = select_training_subset(rows, {'swe': 8, 'graph': 4}, 42)
        reverse, _ = select_training_subset(list(reversed(rows)), {'graph': 4, 'swe': 8}, 42)
        self.assertEqual(chosen, reverse)
        self.assertEqual(len(chosen), 12)
        self.assertEqual(len({r['sample_id'] for r in chosen}), 12)
        self.assertEqual(coverage['swe']['selected_strata'], 4)
        self.assertEqual(coverage['graph']['selected_strata'], 4)
        self.assertEqual(len(rows), 40)

    def test_changed_seed_selects_different_examples(self):
        a, _ = select_training_subset(self.population(), {'swe': 4, 'graph': 4}, 42)
        b, _ = select_training_subset(self.population(), {'swe': 4, 'graph': 4}, 43)
        self.assertNotEqual({r['sample_id'] for r in a}, {r['sample_id'] for r in b})

    def test_insufficient_population_and_missing_source_fail(self):
        for limits in ({'swe': 21, 'graph': 4}, {'swe': 4}, {'swe': 0, 'graph': 4}):
            with self.assertRaises(ValueError):
                select_training_subset(self.population(), limits, 42)

    def test_duplicate_sample_identity_fails(self):
        rows = self.population()
        with self.assertRaisesRegex(ValueError, 'Deduplicate'):
            select_training_subset(rows + [rows[0]], {'swe': 4, 'graph': 4}, 42)


if __name__ == '__main__':
    unittest.main()
