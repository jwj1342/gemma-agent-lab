"""Build termination labels after executing official tools on controlled Git fixtures.

These fixtures test submission protocol, not repository repair ability. Run inside the
evaluation container in a Slurm allocation; no third-party patches are replayed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace

from src.sft_data import SYSTEM, action_examples, split_for_group


class LocalFixtureProcess:
    """Translate the official sandbox working directory to a controlled local fixture."""

    def __init__(self, workspace):
        self.workspace = workspace

    def exec(self, container_id, command, timeout=None):
        command = command.replace("/workspace", shlex.quote(str(self.workspace)))
        result = subprocess.run(command, shell=True, cwd=self.workspace, capture_output=True,
                                text=True, timeout=timeout or 30)
        return SimpleNamespace(stdout=result.stdout, stderr=result.stderr,
                               exit_code=result.returncode, timed_out=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=int, default=128)
    args = parser.parse_args()
    from swegemma.tools.execution import run_command, submit_patch, get_status
    if args.output.exists():
        raise ValueError("Terminal dataset already exists")
    args.output.mkdir(parents=True)
    records = {"train": [], "validation": []}
    evidence = []
    for number in range(args.fixtures):
        group = f"controlled-submission.fixture-{number:03d}"
        split = split_for_group(group)
        with tempfile.TemporaryDirectory(prefix="fixture-", dir=args.output) as directory:
            workspace = Path(directory)
            file = workspace / "example.py"
            file.write_text("def answer():\n    return 0\n")
            for command in (["git", "init", "-q"], ["git", "add", "example.py"],
                            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                             "commit", "-qm", "Baseline"], ["git", "tag", "_swegemma_baseline"]):
                subprocess.run(command, cwd=workspace, check=True, capture_output=True)
            file.write_text(f"def answer():\n    return {number + 1}\n")
            ctx = SimpleNamespace(docker=LocalFixtureProcess(workspace), container_id="controlled-fixture",
                harness=SimpleNamespace(command_timeout_seconds=30, max_stdout_chars=5000),
                budget=SimpleNamespace(time_minutes=None, tool_calls=None, turns=None),
                tool_calls_used=0, submitted_patch=None, patch_submitted=False)

            def check_budget(count_tool_call=True):
                if count_tool_call:
                    ctx.tool_calls_used += 1
                return None

            ctx.check_budget = check_budget
            command = f'python3 -c "from example import answer; assert answer() == {number + 1}; print(\'check passed\')"'
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                f"The requested edit to example.answer is complete: it should return {number + 1}. "
                "Validate this value, capture the patch, and provide a brief completion message."}]
            for name, arguments, operation in [("run_command", {"command": command}, run_command),
                                               ("submit_patch", {}, submit_patch)]:
                call_id = f"{number}-{name}"
                result = operation(ctx, **arguments)
                if json.loads(result).get("status") != "ok":
                    raise ValueError("Controlled official tool call failed")
                messages.extend([{"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
                    "type": "function", "function": {"name": name, "arguments": arguments}}]},
                    {"role": "tool", "tool_call_id": call_id, "name": name, "content": result}])
            if number % 2:
                call_id = f"{number}-get_status"
                result = get_status(ctx)
                if not json.loads(result)["patch_submitted"]:
                    raise ValueError("Submission state was lost")
                messages.extend([{"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
                    "type": "function", "function": {"name": "get_status", "arguments": {}}}]},
                    {"role": "tool", "tool_call_id": call_id, "name": "get_status", "content": result}])
            messages.append({"role": "assistant", "content": "The patch is submitted."})
            metadata = {"source": "controlled submission protocol fixture", "group": group, "split": split,
                "label_origin": "team-authored terminal answer after an observed successful submission",
                "verification": "official swegemma 0.2.7 tools executed against a real temporary Git working tree",
                "scope": "submission control flow; no SWE repair label"}
            terminal = list(action_examples(messages, metadata))[-1]
            if terminal["metadata"]["target_kind"] != "terminal":
                raise ValueError("Missing terminal assistant target")
            records[split].append(terminal)
            evidence.append({"group": group, "split": split,
                "patch_sha256": hashlib.sha256(ctx.submitted_patch.encode()).hexdigest(),
                "observations": [m for m in messages if m["role"] == "tool"]})
    for split, values in records.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in values))
    manifest = {"harness_version": "0.2.7", "fixtures": args.fixtures,
        "examples": {s: len(v) for s, v in records.items()}, "executed_fixtures": evidence,
        "scope": "Real test/submission/status observations on controlled fixtures; synthetic final text, no repair claim"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "executed_fixtures"}, indent=2))


if __name__ == "__main__":
    main()
