"""Task data models and parsers for SWE-bench evaluation."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adk_eval_core.models.base import BaseTaskResult
from adk_eval_core.runner import EvaluationResult as _BaseEvaluationResult
from pydantic import ConfigDict, Field


def _unquote_git_path(raw: str) -> str:
    """Unquotes a git diff file path and strips leading a/ or b/ prefixes."""
    s = raw.strip()
    if not s:
        return ''
    if s.startswith('"'):
        end = 1
        while end < len(s):
            if s[end] == '"' and s[end - 1] != '\\':
                break
            end += 1
        s = s[1:end].encode('utf-8').decode('unicode_escape')
    else:
        s = s.split('\t', 1)[0].strip()
    if s.startswith(('a/', 'b/')):
        s = s[2:]
    return s.strip()


def _normalize_test_nodes(val: Any) -> tuple[str, ...]:
    """Normalizes FAIL_TO_PASS / PASS_TO_PASS values (JSON string or sequence) into a tuple of strings."""
    if not val:
        return ()
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return ()
        if stripped.startswith('['):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return tuple(str(x).strip() for x in parsed if str(x).strip())
            except Exception:
                pass
        return tuple(x.strip() for x in stripped.split(',') if x.strip())
    if isinstance(val, (list, tuple, set)):
        return tuple(str(x).strip() for x in val if str(x).strip())
    return ()


def extract_test_files_from_patch(patch: str) -> list[str]:
    """Extracts test destination file paths from a unified diff / patch string."""
    if not patch or not patch.strip():
        return []
    import re

    files: list[str] = []
    seen: set[str] = set()
    in_hunk = False
    rem_old = 0
    rem_new = 0
    deleted_current_file = False

    def _add(p: str) -> None:
        if p and p != '/dev/null' and p not in seen:
            seen.add(p)
            files.append(p)

    for line in patch.splitlines():
        if line.startswith('diff --git '):
            in_hunk = False
            rem_old = 0
            rem_new = 0
            deleted_current_file = False
            continue
        if line.startswith('@@ '):
            in_hunk = True
            m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if m:
                rem_old = int(m.group(2)) if m.group(2) is not None else 1
                rem_new = int(m.group(4)) if m.group(4) is not None else 1
            continue
        if line.startswith(('--- a/', '--- /dev/null')):
            in_hunk = False
            continue
        if in_hunk:
            if line.startswith(' '):
                rem_old -= 1
                rem_new -= 1
            elif line.startswith('-'):
                rem_old -= 1
            elif line.startswith('+'):
                rem_new -= 1
            if rem_old <= 0 and rem_new <= 0:
                in_hunk = False
            continue
        if line.startswith('diff --git '):
            deleted_current_file = False
            continue
        if line.startswith('--- '):
            continue
        if line.startswith('deleted file mode'):
            deleted_current_file = True
        elif line.startswith('rename to '):
            _add(_unquote_git_path(line[len('rename to ') :]))
        elif line.startswith('+++ ') and not deleted_current_file:
            _add(_unquote_git_path(line[4:]))
    return files


@dataclass(frozen=True)
class Task:
    """A single SWE-bench task instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ''  # gold solution (never shown to agent)
    test_patch: str = ''  # verification tests (applied in eval container)
    hints_text: str = ''
    created_at: str = ''
    FAIL_TO_PASS: tuple[str, ...] = ()
    PASS_TO_PASS: tuple[str, ...] = ()
    environment_setup_commit: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'FAIL_TO_PASS', _normalize_test_nodes(self.FAIL_TO_PASS))
        object.__setattr__(self, 'PASS_TO_PASS', _normalize_test_nodes(self.PASS_TO_PASS))

    @property
    def repo_org(self) -> str:
        """Repository organization (e.g., 'fastapi')."""
        parts = self.repo.split('/', 1)
        return parts[0]

    @property
    def repo_name(self) -> str:
        """Repository name (e.g., 'fastapi')."""
        parts = self.repo.split('/', 1)
        return parts[1] if len(parts) > 1 else parts[0]

    @property
    def test_files(self) -> list[str]:
        """File paths modified by the test_patch."""
        return extract_test_files_from_patch(self.test_patch)


class TaskResult(BaseTaskResult):
    """Result of evaluating a single task."""

    model_config = ConfigDict(populate_by_name=True, extra='allow')

    repo: str = Field(
        default='', description="Repository identifier (e.g., 'fastapi/fastapi')."
    )
    agent_patch: str = Field(
        default='', description='Diff / patch produced by the agent.'
    )
    test_output: str = Field(default='', description='Pytest stdout/stderr output.')
    test_exit_code: int = Field(
        default=0, description='Process exit code from running pytest.'
    )
    trace: Any = Field(default=None, description='SessionTrace instance if recorded.')
    tool_calls: int = Field(
        default=0, description='Total number of tool calls executed.'
    )
    task_index: int | None = Field(
        default=None, description='Original 1-based task index.'
    )

    def __init__(
        self,
        task_id: str | None = None,
        instance_id: str | None = None,
        error: str | None = None,
        trace_json_path: str | None = None,
        **data: Any,
    ) -> None:
        tid = (
            task_id
            or instance_id
            or data.pop('task_id', None)
            or data.pop('instance_id', None)
        )
        if error is not None and 'error_message' not in data:
            data['error_message'] = error
        if trace_json_path is not None and 'trace_path' not in data:
            data['trace_path'] = trace_json_path
        super().__init__(task_id=tid, **data)


@dataclass
class EvaluationResult(_BaseEvaluationResult):
    """Aggregate result of evaluating all tasks."""

    task_results: list[TaskResult] = field(default_factory=list)

    def group_by(
        self,
        key: str | Any = 'repo',
        default: str = 'default',
    ) -> dict[str, dict[str, Any]]:
        """Group task results and compute summary statistics per group."""
        return super().group_by(key=key, default=default)

    def by_repo(self) -> dict[str, dict[str, int | float]]:
        """Breakdown by repository."""
        return self.group_by(key='repo', default='')



def load_tasks(path: Path) -> list[Task]:
    """Load and deduplicate tasks from a JSONL file."""
    task_field_names = {f.name for f in dataclasses.fields(Task)}
    seen: set[str] = set()
    tasks: list[Task] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            instance_id = data.get('instance_id')
            if instance_id and instance_id not in seen:
                seen.add(instance_id)
                filtered_data = {k: v for k, v in data.items() if k in task_field_names}
                tasks.append(Task(**filtered_data))
    return tasks
