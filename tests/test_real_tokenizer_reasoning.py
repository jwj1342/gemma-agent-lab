"""Opt-in CPU integration tests against the unmodified, hash-pinned Kaggle v2 tokenizer.

PR29_TOKENIZER_DIR=/path/to/files python -m unittest discover -s tests \
    -p test_real_tokenizer_reasoning.py -v
Requires transformers 5.13.1, tokenizers 0.22.2 and Jinja2. Never loads weights.
"""
import hashlib
import json
import os
from pathlib import Path
import unittest

from src.sft_data import action_examples, convert_trajectory, encode_example

HASHES = {
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': 'a284d1243b62be31faa9c13e1c28cece940c4abaa7bd9ad87b94f61b40687200',
    'chat_template.jinja': '94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413',
}


def call():
    return {'id': 'c1', 'function': {'name': 'submit', 'arguments': '{}'}}


class RealTokenizerReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('PR29_TOKENIZER_DIR')
        if not directory:
            raise unittest.SkipTest('Set PR29_TOKENIZER_DIR for pinned real-tokenizer checks')
        root = Path(directory)
        for name, digest in HASHES.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise AssertionError('Tokenizer file differs from Kaggle v2: ' + name)
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)

    def render(self, assistant):
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Synthetic check'}, assistant],
            tokenize=False, add_generation_prompt=False, enable_thinking=True)

    def test_official_template_reproduces_terminal_reasoning_omission(self):
        terminal = {'role': 'assistant', 'content': 'Done', 'reasoning': 'REASON_MARKER'}
        self.assertNotIn('REASON_MARKER', self.render(terminal))
        self.assertIn('REASON_MARKER', self.render({**terminal, 'tool_calls': [call()]}))

    def test_terminal_policy_round_trip_and_loss_boundary(self):
        for content in ('Done', [{'type': 'text', 'text': 'Done'}]):
            for reasoning in (None, '', 'REASON_MARKER'):
                for enabled in (False, True):
                    with self.subTest(content=content, reasoning=reasoning, enabled=enabled):
                        terminal = {'role': 'assistant', 'content': content}
                        if reasoning is not None:
                            terminal['reasoning'] = reasoning
                        row = {'resolved': True, 'messages': json.dumps([
                            {'role': 'user', 'content': 'Synthetic check'},
                            {'role': 'assistant', 'tool_calls': [call()]},
                            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'captured'}, terminal])}
                        converted = convert_trajectory(row, preserve_reasoning=enabled)
                        self.assertNotIn('reasoning', converted[-1])
                        self.assertEqual(converted[-1]['content'], 'Done')
                        example = list(action_examples(converted, {}))[-1]
                        encoded = encode_example(example, self.tokenizer, 8192)
                        self.assertIsNotNone(encoded)
                        boundary = next(i for i, token in enumerate(encoded['labels']) if token != -100)
                        self.assertEqual(encoded['labels'][:boundary], [-100] * boundary)
                        self.assertEqual(encoded['labels'][boundary:], encoded['input_ids'][boundary:])
                        target = self.tokenizer.decode(encoded['input_ids'][boundary:])
                        self.assertIn('Done', target)
                        self.assertNotIn('REASON_MARKER', target)

    def test_tool_reasoning_stays_in_supervised_thought_channel(self):
        for content in ('REASON_MARKER', [{'type': 'text', 'text': 'REASON_MARKER'}]):
            for enabled in (False, True):
                row = {'resolved': True, 'messages': json.dumps([
                    {'role': 'user', 'content': 'Synthetic check'},
                    {'role': 'assistant', 'content': content, 'tool_calls': [call()]}])}
                messages = convert_trajectory(row, preserve_reasoning=enabled)
                example = list(action_examples(messages, {}))[0]
                encoded = encode_example(example, self.tokenizer, 8192)
                self.assertIsNotNone(encoded)
                target = self.tokenizer.decode([v for v in encoded['labels'] if v != -100])
                self.assertEqual('REASON_MARKER' in target, enabled)
                if enabled:
                    self.assertIn('<|channel>thought', target)
                self.assertIn('call:submit_patch', target)
                self.assertFalse(target.endswith('<|tool_response>'))
