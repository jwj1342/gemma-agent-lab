"""Synthetic row shapes matching SWE-smith; no upstream task text or identifiers."""
import json
import unittest

from scripts.audit_sft_coverage import coverage


def row(messages):
    return {'resolved': True, 'messages': json.dumps(messages)}


class SFTCoverageTests(unittest.TestCase):
    def test_native_and_xml_shapes_have_separate_completeness(self):
        native = row([
            {'role': 'user', 'content': 'Synthetic task'},
            {'role': 'assistant', 'content': 'Synthetic thought', 'tool_calls': [
                {'id': 'c1', 'function': {'name': 'submit', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'captured'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Done'}], 'reasoning': 'Terminal thought'}])
        xml = row([{'role': 'user', 'content': 'Synthetic task'},
                   {'role': 'assistant', 'content': 'Synthetic thought\n<function=bash><parameter=command>printf synthetic</parameter></function>'}])
        result = coverage([native, xml])
        self.assertEqual(result['counts']['resolved_assistant_messages'], 3)
        self.assertEqual(result['counts']['xml_only_messages'], 1)
        self.assertEqual(result['counts']['fully_convertible_trajectories'], 1)
        self.assertEqual(result['fully_convertible_fraction_of_resolved'], 0.5)
        self.assertEqual(result['source_terminal_explicit_reasoning_fraction'], 1)
        self.assertEqual(result['counts']['terminal_reasoning_dropped_in_accepted'], 1)
