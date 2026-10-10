"""Guard probability probes against tokenizer/SDK coercion and kernel-path artifacts."""
import unittest
from unittest.mock import Mock

from scripts.check_lora_serving import adapter_effect, completion_token_ids, terminal_behavior


class ServingProbeTests(unittest.TestCase):
    def test_trained_adapter_must_finish_each_terminal_probe(self):
        rows = [{'kind': 'terminal', 'model': 'first_lora',
                 'clean_final_answer': True, 'output_limit_reached': False} for _ in range(4)]
        self.assertTrue(terminal_behavior(rows))
        self.assertFalse(terminal_behavior(rows[:3]))
        rows[0]['clean_final_answer'] = False
        self.assertFalse(terminal_behavior(rows))
        # A healthy base cannot compensate for a regressed adapter.
        rows.extend({'kind': 'terminal', 'model': '/model',
                     'clean_final_answer': True, 'output_limit_reached': False} for _ in range(4))
        self.assertFalse(terminal_behavior(rows))
        rows[0]['clean_final_answer'] = True
        rows[1]['output_limit_reached'] = True
        self.assertFalse(terminal_behavior(rows))

    def test_explicit_integer_prompt(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [2, 55, 106]
        example = {'messages': [{'role': 'user', 'content': 'Inspect source'},
                                {'role': 'assistant', 'content': 'Done'}], 'tools': []}
        self.assertEqual(completion_token_ids(tokenizer, example), [2, 55, 106])
        self.assertIs(tokenizer.apply_chat_template.call_args.kwargs['return_dict'], False)

    def test_reject_mapping_and_batched_prompts(self):
        tokenizer = Mock()
        for invalid in ({'input_ids': [2, 55], 'attention_mask': [1, 1]},
                        [[2, 55]], ['input_ids'], [], [True], [-1]):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                tokenizer.apply_chat_template.return_value = invalid
                completion_token_ids(tokenizer, {'messages': [], 'tools': []})

    def test_zero_update_and_repeat_noise_are_not_weight_effects(self):
        zero = {'a': -.1, 'b': -2.0}
        self.assertFalse(adapter_effect(zero, zero, zero)['measurable_effect'])
        self.assertFalse(adapter_effect(zero, {'a': -.1, 'b': -2.01},
                                        {'a': -.1, 'b': -2.02})['measurable_effect'])
        self.assertTrue(adapter_effect(zero, zero, {'a': -.1, 'b': -2.02})['measurable_effect'])
        self.assertTrue(adapter_effect(zero, zero, {'a': -.1, 'c': -2.0})['measurable_effect'])
        self.assertFalse(adapter_effect(zero, {'a': -.1, 'c': -2.0},
                                        {'a': -.1, 'd': -2.0})['measurable_effect'])


if __name__ == '__main__':
    unittest.main()
