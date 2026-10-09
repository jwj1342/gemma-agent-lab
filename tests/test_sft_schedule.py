"""Check full coverage, DDP padding, deterministic resume order, and LR boundaries."""
import unittest
from src.sft_schedule import epoch_batches, lr_factor


class ScheduleTests(unittest.TestCase):
    def test_partial_batch_never_duplicates_supervision(self):
        batches = list(epoch_batches(19, 4, 2, 42, 0))
        self.assertTrue(all(len(batch) == 8 for batch in batches))
        indices = [i for batch in batches for i, weight in batch if weight]
        self.assertEqual(sorted(indices), list(range(19)))
        self.assertEqual(sum(1 - w for b in batches for _, w in b), 5)

    def test_rank_partition_and_resume_order(self):
        batches = list(epoch_batches(19, 4, 2, 42, 0))
        for batch in batches:
            active = [i for rank in range(4) for i, w in batch[rank::4] if w]
            self.assertEqual(sorted(active), sorted(i for i, w in batch if w))
        self.assertEqual(batches[1:], list(epoch_batches(19, 4, 2, 42, 0))[1:])
        self.assertNotEqual(batches, list(epoch_batches(19, 4, 2, 42, 1)))

    def test_warmup_and_decay(self):
        rates = [lr_factor(i, 100, .05, .1) for i in range(100)]
        self.assertAlmostEqual(rates[0], .2)
        self.assertAlmostEqual(rates[4], 1)
        self.assertAlmostEqual(rates[-1], .1)
        self.assertTrue(all(.1 <= r <= 1 for r in rates))


if __name__ == '__main__':
    unittest.main()
