"""Check conversion semantics, provenance splitting, and assistant-only loss boundaries."""
import json
import unittest
from pathlib import Path
import tempfile

from src.sft_data import UnsupportedAction, action_examples, convert_call, convert_trajectory, encode_example, group_key, split_for_group, read_jsonl, is_setup_command


def upstream_call(name, arguments, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


class SFTDataTests(unittest.TestCase):
    def test_reasoning_is_opt_in_and_does_not_change_calls_or_observations(self):
        messages = [{"role": "user", "content": "Repair"},
                    {"role": "assistant", "content": "Do not infer thought from prose",
                     "reasoning_content": "Inspect /testbed/pkg.py before changing it.",
                     "tool_calls": [upstream_call("bash", {"command": "cat /testbed/pkg.py"})]},
                    {"role": "tool", "tool_call_id": "c1", "content": "actual source"}]
        row = {"resolved": True, "messages": json.dumps(messages)}
        old = convert_trajectory(row)
        kept = convert_trajectory(row, preserve_reasoning=True)
        self.assertNotIn("reasoning", old[2])
        self.assertEqual(kept[2].pop("reasoning"), "Inspect /workspace/pkg.py before changing it.")
        self.assertEqual(old, kept)
        self.assertEqual(kept[2]["content"], "")

    def test_reasoning_is_not_invented_and_ambiguous_formats_are_rejected(self):
        assistant = {"role": "assistant", "content": "ordinary text",
                     "tool_calls": [upstream_call("submit", {})]}
        def convert(value):
            return convert_trajectory({"resolved": True, "messages": json.dumps([
                {"role": "user", "content": "Repair"}, value])}, preserve_reasoning=True)
        self.assertNotIn("reasoning", convert(assistant)[-1])
        for fields in ({"reasoning": ["not text"]},
                       {"reasoning": "first", "reasoning_content": "different"}):
            with self.assertRaises(UnsupportedAction):
                convert({**assistant, **fields})
        self.assertEqual(convert({**assistant, "reasoning": "same", "reasoning_content": "same"})[-1]["reasoning"], "same")

    def test_terminal_reasoning_remains_after_real_submission(self):
        messages = [{"role": "user", "content": "Repair"},
                    {"role": "assistant", "tool_calls": [upstream_call("submit", {})]},
                    {"role": "tool", "tool_call_id": "c1", "content": "captured"},
                    {"role": "assistant", "content": "Done", "reasoning": "The tool confirmed submission."}]
        result = convert_trajectory({"resolved": True, "messages": json.dumps(messages)}, preserve_reasoning=True)
        self.assertEqual(result[-1]["reasoning"], "The tool confirmed submission.")
        self.assertEqual(list(action_examples(result, {}))[-1]["metadata"]["target_kind"], "terminal")

    def test_reasoning_tokens_are_supervised_but_prompt_is_masked(self):
        # Contract fixture only; the pinned real tokenizer remains a manual preflight.
        class ReasoningTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
                prefix = messages[0]["content"] + "ASSISTANT:"
                if add_generation_prompt:
                    return prefix
                return prefix + "<thought>" + messages[-1]["reasoning"] + "</thought>CALL<|tool_response>"

            def encode(self, text, **kwargs):
                return list(text.encode())

        example = {"messages": [{"role": "user", "content": "prompt"},
                   {"role": "assistant", "content": "", "reasoning": "source supports this",
                    "tool_calls": [upstream_call("submit", {})]}], "tools": [], "metadata": {}}
        result = encode_example(example, ReasoningTokenizer(), 256)
        target = bytes(t for t in result["labels"] if t != -100).decode()
        self.assertEqual(target, "<thought>source supports this</thought>CALL")
        prefix_length = len("promptASSISTANT:")
        self.assertEqual(result["labels"][:prefix_length], [-100] * prefix_length)

    def test_setup_filter_is_explicit_and_preserves_observation_integrity(self):
        for command in ("pip install -e .", "cd /workspace && python3 -m pip install missing",
                        "sudo apt-get install package", "git clone https://example.invalid/repo", "curl https://example.invalid"):
            self.assertTrue(is_setup_command(command))
        for command in ("pip list", "git status", "printf 'pip install documentation'", "pytest tests/test_target.py"):
            self.assertFalse(is_setup_command(command))
        row = {"resolved": True, "messages": json.dumps([{"role": "user", "content": "Repair"},
            {"role": "assistant", "tool_calls": [upstream_call("bash", {"command": "pip install missing"})]},
            {"role": "tool", "tool_call_id": "c1", "content": "Upstream install output"}])}
        self.assertEqual(convert_trajectory(row)[-1]["content"], "Upstream install output")
        with self.assertRaises(UnsupportedAction):
            convert_trajectory(row, reject_setup_actions=True)

    def test_unicode_separators_inside_json_strings(self):
        row = {"text": "source\u2028continued\u0085text"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(list(read_jsonl(path)), [row])

    def test_exact_edit_argument_names_and_content(self):
        call = convert_call(upstream_call("str_replace_editor", {"command": "str_replace", "path": "/testbed/pkg/mod.py",
                            "old_str": "if x:\n    return x", "new_str": "if x is not None:\n    return x"}))
        self.assertEqual(call["function"], {"name": "edit_file", "arguments": {"filepath": "pkg/mod.py",
                         "old_string": "if x:\n    return x", "new_string": "if x is not None:\n    return x"}})

    def test_open_ended_line_range(self):
        call = convert_call(upstream_call("str_replace_editor", {"command": "view", "path": "/testbed/pkg/mod.py", "view_range": [10, -1]}))
        self.assertEqual(call["function"]["arguments"], {"filepath": "pkg/mod.py", "start_line": 10})

    def test_unsupported_actions_are_rejected(self):
        for command in ("insert", "undo_edit"):
            with self.assertRaises(UnsupportedAction):
                convert_call(upstream_call("str_replace_editor", {"command": command, "path": "pkg/mod.py"}))
        for path in ("/etc/passwd", "/testbed/../secret", "/testbed/"):
            with self.assertRaises(UnsupportedAction):
                convert_call(upstream_call("str_replace_editor", {"command": "view", "path": path}))

    def test_only_upstream_success_and_real_observations(self):
        messages = [{"role": "user", "content": "Repair an issue"},
                    {"role": "assistant", "tool_calls": [upstream_call("bash", {"command": "cat /testbed/pkg.py"})]},
                    {"role": "tool", "content": [{"type": "text", "text": "OBSERVATION:\noriginal output"}], "tool_call_ids": ["c1"]}]
        row = {"resolved": True, "messages": json.dumps(messages)}
        converted = convert_trajectory(row)
        self.assertEqual(converted[-1]["content"], "OBSERVATION:\noriginal output")
        self.assertEqual(converted[-1]["name"], "run_command")
        self.assertEqual(converted[-1]["tool_call_id"], "c1")
        row["resolved"] = False
        with self.assertRaises(UnsupportedAction):
            convert_trajectory(row)

    def test_observation_requires_valid_call_link(self):
        with self.assertRaises(UnsupportedAction):
            convert_trajectory({"resolved": True, "messages": json.dumps([{"role": "tool", "tool_call_ids": ["unknown"], "content": "x"}])})

    def test_observed_submission_preserves_source_final_answer(self):
        messages = [{"role": "user", "content": "Repair an issue"},
            {"role": "assistant", "tool_calls": [upstream_call("submit", {})]},
            {"role": "tool", "tool_call_id": "c1", "content": "Submission captured"},
            {"role": "assistant", "content": "The patch is submitted."}]
        converted = convert_trajectory({"resolved": True, "messages": json.dumps(messages)})
        self.assertEqual(converted[-1], {"role": "assistant", "content": "The patch is submitted."})
        examples = list(action_examples(converted, {"group": "repo.commit"}))
        self.assertEqual([x["metadata"]["target_kind"] for x in examples], ["tool", "terminal"])

    def test_intermediate_submit_comment_does_not_truncate_later_actions(self):
        messages = [{"role": "user", "content": "Repair an issue"},
            {"role": "assistant", "tool_calls": [upstream_call("submit", {})]},
            {"role": "tool", "tool_call_id": "c1", "content": "Please revise this patch"},
            {"role": "assistant", "content": "I need another edit."},
            {"role": "assistant", "tool_calls": [upstream_call("bash", {"command": "git diff"}, "c2")]}]
        converted = convert_trajectory({"resolved": True, "messages": json.dumps(messages)})
        self.assertEqual(converted[-1]["tool_calls"][0]["function"]["name"], "run_command")
        self.assertEqual([x["metadata"]["target_kind"] for x in action_examples(converted, {})], ["tool", "tool"])

    def test_unobserved_submission_does_not_invent_final_message(self):
        messages = [{"role": "user", "content": "Repair an issue"},
                    {"role": "assistant", "tool_calls": [upstream_call("submit", {})]}]
        converted = convert_trajectory({"resolved": True, "messages": json.dumps(messages)})
        self.assertEqual(converted[-1]["tool_calls"][0]["function"]["name"], "submit_patch")
        self.assertEqual(len(list(action_examples(converted, {}))), 1)

    def test_target_keeps_final_stop_but_excludes_tool_response_prompt(self):
        class ProtocolTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
                text = "".join(m.get("content", "") for m in messages)
                if add_generation_prompt:
                    return text + "ASSISTANT:"
                if messages[-1].get("tool_calls"):
                    return messages[0]["content"] + "ASSISTANT:CALL<|tool_response>"
                return messages[0]["content"] + "ASSISTANT:" + messages[-1]["content"] + "<turn|>"

            def encode(self, text, **kwargs):
                return [ord(c) for c in text]

        tokenizer = ProtocolTokenizer()
        for assistant, expected in [({"role": "assistant", "content": "", "tool_calls": [upstream_call("submit", {})]}, "CALL"),
                                    ({"role": "assistant", "content": "Done."}, "Done.<turn|>")]:
            encoded = encode_example({"messages": [{"role": "user", "content": "prompt"}, assistant],
                                      "tools": [], "metadata": {}}, tokenizer, 100)
            supervised = "".join(chr(t) for t in encoded["labels"] if t != -100)
            self.assertEqual(supervised, expected)

    def test_snapshot_group_prevents_mutation_leakage(self):
        first = group_key("owner__repo.abcdef.mutation_a__x")
        second = group_key("owner__repo.abcdef.mutation_b__y")
        self.assertEqual(first, second)
        self.assertEqual(split_for_group(first), split_for_group(second))

    def test_prompt_tokens_have_no_loss(self):
        class TinyTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                text = "".join(m.get("content", "") for m in messages)
                return text

            def encode(self, text, **kwargs):
                return [ord(c) for c in text]

        example = {"messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "action"}],
                   "tools": [], "metadata": {"group": "repo.commit"}}
        result = encode_example(example, TinyTokenizer(), 32)
        self.assertEqual(result["labels"][:6], [-100] * 6)
        self.assertEqual(result["labels"][6:], [ord(c) for c in "action"])
        self.assertIsNone(encode_example(example, TinyTokenizer(), 10))


if __name__ == "__main__":
    unittest.main()
