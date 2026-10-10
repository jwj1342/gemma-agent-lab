"""Build termination labels after executing official tools on controlled Git fixtures.

These fixtures test submission protocol, not repository repair ability. Run inside the
evaluation container in a Slurm allocation; no third-party patches are replayed here.
"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace

from src.sft_data import SYSTEM, action_examples, read_jsonl, split_for_group


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
    parser.add_argument("--validation-from", type=Path,
                        help="Retain an existing held-out JSONL file byte for byte")
    args = parser.parse_args()
    from swegemma.tools.execution import run_command, submit_patch, get_status
    harness_version = version('swegemma')
    if harness_version != '0.2.10':
        raise RuntimeError('This generation run requires the reviewed version 29 harness')
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
            files = [workspace / "example.py"] + [workspace / f"helper_{i}.py"
                                                   for i in range(number % 3)]
            for file in files:
                file.write_text("def answer():\n    return 0\n")
            (workspace / '.gitignore').write_text('__pycache__/\n*.py[cod]\n')
            for command in (["git", "init", "-q"], ["git", "add", "."],
                            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                             "commit", "-qm", "Baseline"], ["git", "tag", "_swegemma_baseline"]):
                subprocess.run(command, cwd=workspace, check=True, capture_output=True)
            for file in files:
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
            assertions = '; '.join(f'import {f.stem}; assert {f.stem}.answer() == {number + 1}' for f in files)
            command = f'python3 -B -c "{assertions}; print(\'check passed\')"'
            modules = ', '.join(f.stem for f in files)
            requests = [
                f"Update answer() in {modules} to return {number + 1}. Check the result and submit the change.",
                f"The return value in {modules} should be {number + 1}, rather than zero. Make this focused correction.",
                f"Correct answer() in {modules} to return {number + 1}; verify the affected functions.",
                f"Change the result of answer() in {modules} to {number + 1}. Keep unrelated code unchanged."]
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": requests[number % len(requests)]}]
            for name, arguments, operation in [("run_command", {"command": command}, run_command),
                                               ("submit_patch", {}, submit_patch)]:
                call_id = f"{number}-{name}"
                result = operation(ctx, **arguments)
                if json.loads(result).get("status") != "ok":
                    raise ValueError("Controlled official tool call failed")
                if name == "submit_patch" and json.loads(result)["files_changed"] != len(files):
                    raise ValueError("Captured fixture patch includes unexpected files")
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
            endings = [
                f"Updated example.answer to return {number + 1}. The assertion passed, and the patch is submitted.",
                "The requested change is complete. The regression check passed and the patch has been captured.",
                f"Validation confirmed the expected result ({number + 1}). Submitted the patch.",
                "The patch is submitted. The focused test passed."]
            messages.append({"role": "assistant", "content": endings[number % len(endings)]})
            metadata = {"source": "controlled submission protocol fixture", "group": group, "split": split,
                "changed_source_files": len(files),
                "label_origin": "team-authored terminal answer after an observed successful submission",
                "verification": f"official swegemma {harness_version} tools executed against a real temporary Git working tree",
                "scope": "submission control flow; no SWE repair label"}
            terminal = list(action_examples(messages, metadata))[-1]
            if terminal["metadata"]["target_kind"] != "terminal":
                raise ValueError("Missing terminal assistant target")
            records[split].append(terminal)
            evidence.append({"group": group, "split": split,
                "patch_sha256": hashlib.sha256(ctx.submitted_patch.encode()).hexdigest(),
                "observations": [m for m in messages if m["role"] == "tool"]})
    validation_origin = None
    if args.validation_from:
        records['validation'] = list(read_jsonl(args.validation_from))
        if not records['validation']:
            raise ValueError('The retained validation population must be nonempty')
        if ({e['metadata']['group'] for e in records['train']} &
                {e['metadata']['group'] for e in records['validation']}):
            raise ValueError('Retained validation groups overlap training fixtures')
        validation_origin = {'path': str(args.validation_from),
                             'sha256': hashlib.sha256(args.validation_from.read_bytes()).hexdigest()}
    for split, values in records.items():
        target = args.output / f"{split}.jsonl"
        if split == 'validation' and args.validation_from:
            target.write_bytes(args.validation_from.read_bytes())
        else:
            target.write_text("".join(json.dumps(x) + "\n" for x in values))
    manifest = {"harness_version": harness_version, "fixtures": args.fixtures,
        "examples": {s: len(v) for s, v in records.items()}, "executed_fixtures": evidence,
        "retained_validation": validation_origin,
        "generated_validation_used": args.validation_from is None,
        "training_changed_file_counts": sorted({e['metadata']['changed_source_files'] for e in records['train']}),
        "scope": "Real test/submission/status observations on controlled fixtures; synthetic final text, no repair claim"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "executed_fixtures"}, indent=2))


if __name__ == "__main__":
    main()
