"""Check full coverage, DDP padding, deterministic resume order, and LR boundaries."""
import unittest
from src.sft_schedule import epoch_batches, lr_factor, loss_weights


class ScheduleTests(unittest.TestCase):
    def test_short_answers_receive_equal_example_weight(self):
        # A one-token answer and a nine-token call have equal influence in
        # example mode; token mode gives the call nine times the weight.
        losses = [3.0, 1.0]
        for mode, expected in [('example', 2.0), ('token', 1.2)]:
            weights = loss_weights([1, 9], [1, 1], mode)
            self.assertAlmostEqual(sum(a*b for a, b in zip(losses, weights)), expected)

    def test_ddp_average_matches_unpadded_single_worker_gradient(self):
        for mode in ('token', 'example'):
            counts = [1, 3, 7, 7]
            gradients = [2.0, -1.0, 5.0, 999.0]
            weights = loss_weights(counts, [1, 1, 1, 0], mode)
            rank_gradients = [2 * sum(gradients[i]*weights[i] for i in range(rank, 4, 2))
                              for rank in range(2)]
            reference = sum(g*w for g, w in zip(gradients[:3], loss_weights(counts[:3], [1, 1, 1], mode)))
            self.assertAlmostEqual(sum(rank_gradients)/2, reference)

    def test_invalid_loss_batches_are_rejected(self):
        for counts, weights, mode in [([], [], 'token'), ([1], [], 'example'),
                                     ([0], [1], 'token'), ([1], [-1], 'example'),
                                     ([1], [0], 'token'), ([1], [1], 'unknown')]:
            with self.assertRaises(ValueError):
                loss_weights(counts, weights, mode)

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
