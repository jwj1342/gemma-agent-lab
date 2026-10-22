"""Inspect a candidate repair without changing or executing repository code."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tokenize


PROTECTED_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
DEBUGGER_PATTERN = re.compile(r"(?:\bbreakpoint\s*\(|\b(?:i?pdb)\.set_trace\s*\()")
CONFLICT_PATTERN = re.compile(r"^(?:<<<<<<<|=======|>>>>>>>)")


def git(root: Path, *args: str) -> bytes:
    """Run one fixed Git query and return its bytes."""
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
        timeout=20,
    )
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return process.stdout


def repository_root(explicit: str | None) -> Path:
    """Resolve the explicit root, the evaluation mount, or the current repository."""
    candidates = [Path(explicit)] if explicit else [Path("/workspace"), Path.cwd()]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            resolved = git(candidate.resolve(), "rev-parse", "--show-toplevel")
        except RuntimeError:
            continue
        return Path(resolved.decode("utf-8", errors="strict").strip()).resolve()
    raise RuntimeError("no Git repository found at --root, /workspace, or the current directory")


def decode_paths(payload: bytes) -> list[str]:
    return [item.decode("utf-8", errors="surrogateescape") for item in payload.split(b"\0") if item]


def tracked_changes(root: Path) -> dict[str, str]:
    """Return tracked paths changed from HEAD with rename detection disabled."""
    fields = decode_paths(git(root, "diff", "--name-status", "--no-renames", "-z", "HEAD", "--"))
    if len(fields) % 2:
        raise RuntimeError("unexpected git diff --name-status output")
    return {fields[index + 1]: fields[index] for index in range(0, len(fields), 2)}


def numstat(root: Path) -> dict[str, tuple[int | None, int | None]]:
    result = {}
    for record in decode_paths(git(root, "diff", "--numstat", "--no-renames", "-z", "HEAD", "--")):
        added, deleted, path = record.split("\t", 2)
        result[path] = (
            None if added == "-" else int(added),
            None if deleted == "-" else int(deleted),
        )
    return result


def is_protected(path: str) -> bool:
    pure = PurePosixPath(path)
    lower_parts = tuple(part.lower() for part in pure.parts)
    name = pure.name.lower()
    return (
        "tests" in lower_parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name in PROTECTED_NAMES
    )


def safe_path(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    if not candidate.is_relative_to(root):
        raise RuntimeError(f"changed path escapes repository: {relative}")
    return candidate


def added_lines(root: Path, untracked: list[str]) -> list[tuple[str, str]]:
    """Return added tracked and untracked text lines for deterministic residue checks."""
    patch = git(
        root,
        "diff",
        "--no-ext-diff",
        "--unified=0",
        "--no-renames",
        "HEAD",
        "--",
    ).decode("utf-8", errors="replace")
    result: list[tuple[str, str]] = []
    current = ""
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif current and line.startswith("+") and not line.startswith("+++"):
            result.append((current, line[1:]))
    for relative in untracked:
        path = safe_path(root, relative)
        if path.is_file() and path.stat().st_size <= 1_000_000:
            text = path.read_text(encoding="utf-8", errors="replace")
            result.extend((relative, line) for line in text.splitlines())
    return result


def syntax_errors(root: Path, changes: dict[str, str]) -> list[dict[str, object]]:
    errors = []
    for relative, status in sorted(changes.items()):
        path = safe_path(root, relative)
        if status == "D" or path.suffix.lower() != ".py" or not path.is_file():
            continue
        try:
            with tokenize.open(path) as stream:
                ast.parse(stream.read(), filename=relative)
        except (OSError, SyntaxError, UnicodeError) as error:
            errors.append(
                {
                    "path": relative,
                    "line": getattr(error, "lineno", None),
                    "message": getattr(error, "msg", str(error)),
                }
            )
    return errors


def analyze(root: Path) -> dict[str, object]:
    tracked = tracked_changes(root)
    untracked = sorted(decode_paths(git(root, "ls-files", "--others", "--exclude-standard", "-z")))
    changes = dict(tracked)
    changes.update({path: "?" for path in untracked})
    stats = numstat(root)
    files = [
        {
            "path": path,
            "status": status,
            "added": stats.get(path, (None, None))[0],
            "deleted": stats.get(path, (None, None))[1],
        }
        for path, status in sorted(changes.items())
    ]
    protected = sorted(path for path in changes if is_protected(path))
    candidate_files = sorted(path for path in changes if not is_protected(path))
    syntax = syntax_errors(root, changes)
    conflicts = []
    debuggers = []
    for path, line in added_lines(root, untracked):
        if CONFLICT_PATTERN.match(line):
            conflicts.append({"path": path, "line": line[:160]})
        if DEBUGGER_PATTERN.search(line):
            debuggers.append({"path": path, "line": line[:160]})
    reasons = []
    if not changes:
        reasons.append("empty_patch")
    if not candidate_files and changes:
        reasons.append("no_candidate_source_change")
    if protected:
        reasons.append("protected_edits")
    if syntax:
        reasons.append("python_syntax_errors")
    if conflicts:
        reasons.append("conflict_markers")
    if debuggers:
        reasons.append("debugger_residue")
    return {
        "status": "repair" if reasons else "ready",
        "reasons": reasons,
        "changed_files": files,
        "candidate_files": candidate_files,
        "protected_edits": protected,
        "syntax_errors": syntax,
        "conflict_markers": conflicts,
        "debugger_residue": debuggers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Git worktree to inspect; defaults to /workspace")
    args = parser.parse_args(argv)
    try:
        result = analyze(repository_root(args.root))
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        result = {"status": "repair", "reasons": ["checkpoint_error"], "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
