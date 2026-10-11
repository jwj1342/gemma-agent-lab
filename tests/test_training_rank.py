"""Validate GPU allocation mapping without importing Torch or initializing CUDA."""
import unittest

from scripts.launch_training_rank import isolated_environment


class TrainingRankTests(unittest.TestCase):
    def test_all_workers_get_distinct_allocated_gpus_and_keep_global_ranks(self):
        selected = []
        for rank in range(4):
            env = {'CUDA_VISIBLE_DEVICES': '2,4,6,7', 'LOCAL_RANK': str(rank),
                   'RANK': str(rank + 4), 'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '4'}
            out = isolated_environment(env)
            selected.append(out['CUDA_VISIBLE_DEVICES'])
            self.assertEqual(out['LOCAL_RANK'], '0')
            self.assertEqual(out['GEMMA_WORKER_LOCAL_RANK'], str(rank))
            self.assertEqual(out['RANK'], env['RANK'])
            self.assertEqual(out['WORLD_SIZE'], '8')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,4,6,7')
        self.assertEqual(selected, ['2', '4', '6', '7'])

    def test_uuid_and_single_gpu_allocations(self):
        for devices, rank, expected in [('GPU-first,GPU-second', '1', 'GPU-second'),
                                         ('3', '0', '3')]:
            out = isolated_environment({'CUDA_VISIBLE_DEVICES': devices, 'LOCAL_RANK': rank})
            self.assertEqual(out['CUDA_VISIBLE_DEVICES'], expected)

    def test_rejects_missing_duplicate_and_out_of_range_allocations(self):
        for env in [{}, {'CUDA_VISIBLE_DEVICES': '0'},
                    {'CUDA_VISIBLE_DEVICES': '', 'LOCAL_RANK': '0'},
                    {'CUDA_VISIBLE_DEVICES': '-1', 'LOCAL_RANK': '0'},
                    {'CUDA_VISIBLE_DEVICES': '0,0', 'LOCAL_RANK': '0'},
                    {'CUDA_VISIBLE_DEVICES': '0,,1', 'LOCAL_RANK': '0'},
                    {'CUDA_VISIBLE_DEVICES': '0,1', 'LOCAL_RANK': '-1'},
                    {'CUDA_VISIBLE_DEVICES': '0,1', 'LOCAL_RANK': '2'},
                    {'CUDA_VISIBLE_DEVICES': '0,1', 'LOCAL_RANK': 'invalid'}]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                isolated_environment(env)


if __name__ == '__main__':
    unittest.main()
