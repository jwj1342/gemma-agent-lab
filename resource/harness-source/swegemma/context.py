"""Context managing tools, budgets, and operational limits for SWE-gemma evaluation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from swegemma.budget import EvaluationBudget, HarnessLimits
from swegemma.sandbox import BaseSandboxManager
from swegemma.tools import (
    create_tools as create_context_tools,
)

logger = logging.getLogger(__name__)

__all__ = [
    'DisplaySnapshot',
    'SwegemmaContext',
]


@dataclass(frozen=True)
class DisplaySnapshot:
    """Immutable snapshot of context display state for thread-safe UI rendering."""

    task_id: str
    repo: str
    phase: str
    tools_used: int
    max_tools: int | None
    elapsed_seconds: float
    activity: str
    agent_started: bool = False


class SwegemmaContext:
    """Context managing the sandbox, tool state, budgets, and execution limits."""

    def __init__(
        self,
        *,
        docker_manager: BaseSandboxManager | Any = None,
        container_id: str = '',
        problem_statement: str = '',
        hints_text: str = '',
        task: Any | None = None,
        task_id: str | None = None,
        repo: str | None = None,
        graph_dir: str = 'data/graphs',
        embeddings_dir: str = 'data/embeddings',
        budget: EvaluationBudget | None = None,
        harness: HarnessLimits | None = None,
        # Direct convenience kwargs for backward compatibility
        max_tool_calls: int | None = None,
        max_time_minutes: int | float | None = None,
        max_exec_seconds: int | None = None,
        max_stdout_chars: int | None = None,
    ) -> None:
        self.docker = docker_manager
        self._lock = threading.RLock()
        self.tool_lock = threading.RLock()
        self._container_id = container_id
        self._sandbox: Any | None = None
        self._update_sandbox()

        self.problem_statement = problem_statement or (
            task.problem_statement
            if task and hasattr(task, 'problem_statement')
            else ''
        )
        self.hints_text = hints_text or (
            task.hints_text if task and hasattr(task, 'hints_text') else ''
        )
        self.task = task
        self._task_id = (
            task_id
            if task_id is not None
            else (task.instance_id if task and hasattr(task, 'instance_id') else None)
        )
        self.repo = repo or (task.repo if task and hasattr(task, 'repo') else None)
        self.graph_dir = graph_dir
        self.embeddings_dir = embeddings_dir

        # Construct budget (resolving direct kwargs if provided)
        base_budget = budget or EvaluationBudget()
        final_tool_calls = (
            max_tool_calls if max_tool_calls is not None else base_budget.tool_calls
        )
        final_time_minutes = (
            max_time_minutes
            if max_time_minutes is not None
            else base_budget.time_minutes
        )
        self.budget = EvaluationBudget(
            time_minutes=final_time_minutes,
            tool_calls=final_tool_calls,
            turns=base_budget.turns,
            cost_usd=base_budget.cost_usd,
            total_tokens=base_budget.total_tokens,
        )

        # Construct harness limits (resolving direct kwargs if provided)
        base_harness = harness or HarnessLimits()
        final_cmd_timeout = (
            max_exec_seconds
            if max_exec_seconds is not None
            else base_harness.command_timeout_seconds
        )
        final_stdout_chars = (
            max_stdout_chars
            if max_stdout_chars is not None
            else base_harness.max_stdout_chars
        )
        self.harness = HarnessLimits(
            command_timeout_seconds=final_cmd_timeout,
            max_stdout_chars=final_stdout_chars,
            max_file_lines=base_harness.max_file_lines,
            max_file_chars=base_harness.max_file_chars,
        )

        # Legacy convenience properties for backward compatibility
        self.max_time_minutes = self.budget.time_minutes
        self.max_tool_calls = self.budget.tool_calls
        self.max_exec_seconds = self.harness.command_timeout_seconds
        self.max_stdout_chars = self.harness.max_stdout_chars

        self.task_start_time: float = time.perf_counter()
        self.agent_start_time: float | None = None
        self.agent_end_time: float | None = None
        self.start_time: float = self.task_start_time  # Backward compatibility alias
        self.tool_calls_used = 0
        self.llm_calls_used = 0
        self.current_phase = 'Starting'
        self.current_activity = ''
        self._handled_event_ids: set[Any] = set()
        self.submitted_patch: str | None = None
        self.patch_submitted: bool = False
        self._token_budget = None
        self._total_cost_usd: float = 0.0
        self._total_tokens: int = 0

    @property
    def container_id(self) -> str:
        with self._lock:
            return self._container_id

    @container_id.setter
    def container_id(self, value: str) -> None:
        with self._lock:
            self._container_id = value
            self._update_sandbox()

    def _update_sandbox(self) -> None:
        if self.docker and self._container_id:
            if hasattr(self.docker, 'get_sandbox'):
                self._sandbox = self.docker.get_sandbox(self._container_id)
            else:
                from swegemma.sandbox.base import ManagedSandbox

                self._sandbox = ManagedSandbox(
                    manager=self.docker, sandbox_id=self._container_id
                )
        else:
            self._sandbox = None

    @property
    def sandbox(self) -> Any | None:
        with self._lock:
            return self._sandbox

    @sandbox.setter
    def sandbox(self, value: Any) -> None:
        with self._lock:
            self._sandbox = value

    def start_agent_session(self) -> None:
        """Mark the start of the active agent problem-solving session."""
        with self._lock:
            self.agent_start_time = time.perf_counter()
            self.agent_end_time = None

    def stop_agent_session(self) -> None:
        """Mark the end of the active agent session to freeze the agent elapsed timer."""
        with self._lock:
            if self.agent_start_time is not None and self.agent_end_time is None:
                self.agent_end_time = time.perf_counter()

    def record_llm_call(self) -> None:
        """Record that an LLM call was executed."""
        with self._lock:
            self.llm_calls_used += 1

    def set_activity(self, activity: str) -> None:
        """Set current runtime activity string."""
        with self._lock:
            self.current_activity = activity

    def set_phase(self, phase: str) -> None:
        """Set current evaluation phase."""
        with self._lock:
            self.current_phase = phase

    def handle_adk_event(self, event: Any, agent_name: str | None = None) -> None:
        """Synchronously extract tool/thought activity preview and count model calls."""
        if event is None:
            return

        with self._lock:
            event_id = getattr(event, 'id', None) or id(event)
            is_partial = getattr(event, 'partial', False)
            event_key = (event_id, is_partial)
            if event_key in self._handled_event_ids:
                return
            self._handled_event_ids.add(event_key)

        try:
            content = getattr(event, 'content', None)
            parts = getattr(content, 'parts', []) if content else []

            # 1. Activity preview extraction
            activity_found = False
            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    name = getattr(fc, 'name', '') or 'tool'
                    args = dict(fc.args) if getattr(fc, 'args', None) else {}
                    preview_arg = ''
                    for key in (
                        'command',
                        'path',
                        'file_path',
                        'query',
                        'pattern',
                        'filename',
                        'prompt',
                    ):
                        val = args.get(key)
                        if val:
                            preview_arg = str(val)
                            break
                    if not preview_arg and args:
                        preview_arg = str(next(iter(args.values())))
                    if preview_arg and preview_arg.strip():
                        lines = preview_arg.strip().splitlines()
                        first_line = lines[0] if lines else ''
                        if len(first_line) > 30:
                            first_line = first_line[:27] + '...'
                        activity = (
                            f'{name}({first_line})' if first_line else f'{name}()'
                        )
                    else:
                        activity = f'{name}()'
                    self.set_activity(activity)
                    activity_found = True
                    break
                elif hasattr(part, 'thought') and part.thought:
                    self.set_activity('Thinking...')
                    activity_found = True
                    break

            if not activity_found:
                event_type = getattr(event, 'type', None)
                if event_type == 'call':
                    name = getattr(event, 'tool_name', 'tool')
                    self.set_activity(f'{name}()')
                elif event_type == 'thought':
                    self.set_activity('Thinking...')

            # 2. LLM call counting (all non-partial model events from root or sub-agents)
            is_partial = getattr(event, 'partial', False)
            if not is_partial:
                author = getattr(event, 'author', None)
                role = getattr(content, 'role', None) if content else None
                non_model_authors = {'user', 'harness', 'system', 'tool'}

                is_non_model = (author is not None and author in non_model_authors) or (
                    role is not None and role in ('user', 'tool')
                )
                is_agent = not is_non_model and author is not None

                if is_agent:
                    has_content = content is not None and (
                        bool(parts)
                        or bool(getattr(content, 'text', None))
                        or isinstance(content, str)
                    )
                    is_pure_tool_response = bool(parts) and all(
                        getattr(p, 'function_response', None) for p in parts
                    )
                    if has_content and not is_pure_tool_response:
                        self.record_llm_call()
                        usage = getattr(event, 'usage_metadata', None)
                        if usage is not None:
                            in_toks = int(getattr(usage, 'prompt_token_count', 0) or 0)
                            out_toks = int(
                                getattr(usage, 'candidates_token_count', 0) or 0
                            )
                            toks = getattr(usage, 'total_token_count', None)
                            if toks is None:
                                toks = in_toks + out_toks
                            if isinstance(toks, (int, float)) and toks > 0:
                                with self._lock:
                                    self._total_tokens += int(toks)
                            if self._token_budget is not None and hasattr(
                                self._token_budget, 'record_usage'
                            ):
                                with contextlib.suppress(Exception):
                                    self._token_budget.record_usage(
                                        input_tokens=in_toks,
                                        output_tokens=out_toks,
                                    )
        except Exception as e:
            logger.debug('Failed to handle ADK event: %s', e)

    @property
    def task_id(self) -> str:
        """Identifier of task/instance being evaluated."""
        if hasattr(self, '_task_id') and self._task_id:
            return self._task_id
        if hasattr(self, 'task') and self.task is not None:
            if hasattr(self.task, 'instance_id'):
                return self.task.instance_id
            return str(self.task)
        return 'uninitialized'

    @property
    def graph_repo_key(self) -> str:
        """Repository key for graph/embedding resolution, preferring snapshot-specific key."""
        if hasattr(self, 'task') and self.task:
            if isinstance(self.task, dict):
                base_commit = self.task.get('base_commit') or ''
                repo = self.task.get('repo') or self.repo or ''
            else:
                base_commit = getattr(self.task, 'base_commit', '') or ''
                repo = getattr(self.task, 'repo', '') or self.repo or ''
            repo_short = repo.split('/')[-1] if repo else ''
            if repo_short and base_commit:
                return f'{repo_short}_{base_commit}'
        return self.repo or ''

    @property
    def candidate_id(self) -> str | None:
        """Identifier of candidate under evaluation, if set."""
        return None

    @property
    def status(self) -> str:
        """Current evaluation status."""
        return 'completed' if self.patch_submitted else 'running'

    @property
    def tool_calls(self) -> int:
        """Number of tool calls executed."""
        return self.tool_calls_used

    @property
    def token_budget(self) -> Any | None:
        """Token budget tracker if attached."""
        return self._token_budget

    @token_budget.setter
    def token_budget(self, value: Any | None) -> None:
        self._token_budget = value

    @property
    def total_cost_usd(self) -> float:
        """Total monetary spend in USD across the session."""
        tb_cost = (
            getattr(self._token_budget, 'total_cost_usd', 0.0)
            if self._token_budget is not None
            else 0.0
        )
        return max(self._total_cost_usd, float(tb_cost or 0.0))

    @total_cost_usd.setter
    def total_cost_usd(self, value: float) -> None:
        self._total_cost_usd = float(value)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across the session."""
        tb_tokens = (
            getattr(self._token_budget, 'total_tokens', 0)
            if self._token_budget is not None
            else 0
        )
        return max(self._total_tokens, int(tb_tokens or 0))

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        self._total_tokens = int(value)

    @property
    def metrics(self) -> dict[str, Any]:
        """Domain metrics dictionary."""
        return {
            'patch_submitted': self.patch_submitted,
            'repo': self.repo or '',
            'container_id': self.container_id,
        }

    def to_status_dict(self) -> dict[str, Any]:
        """Convert context status to a dictionary for UI HUDs."""
        return {
            'task_id': self.task_id,
            'candidate_id': self.candidate_id,
            'status': self.status,
            'phase': self.current_phase,
            'activity': self.current_activity,
            'elapsed_seconds': self.elapsed_seconds,
            'task_elapsed_seconds': self.task_elapsed_seconds,
            'agent_elapsed_seconds': self.agent_elapsed_seconds,
            'max_time_minutes': self.max_time_minutes,
            'tool_calls': self.tool_calls,
            'max_tool_calls': self.max_tool_calls,
            'llm_calls': self.llm_calls_used,
            'turns': self.llm_calls_used,
            'max_turns': self.budget.turns,
            'max_budget_usd': (
                self.token_budget.max_budget_usd
                if self.token_budget and self.token_budget.max_budget_usd is not None
                else self.budget.cost_usd
            ),
            'metrics': self.metrics,
        }

    @property
    def agent_elapsed_seconds(self) -> float:
        """Elapsed time in seconds of the active agent session, or 0.0 if not started."""
        if self.agent_start_time is None:
            return 0.0
        end_time = (
            self.agent_end_time
            if self.agent_end_time is not None
            else time.perf_counter()
        )
        return max(0.0, end_time - self.agent_start_time)

    @property
    def task_elapsed_seconds(self) -> float:
        """Total elapsed wall time in seconds since task setup began."""
        return time.perf_counter() - self.task_start_time

    @property
    def elapsed_seconds(self) -> float:
        """Active session elapsed time for budget checks; falls back to task elapsed."""
        if self.agent_start_time is not None:
            return self.agent_elapsed_seconds
        return self.task_elapsed_seconds

    @property
    def remaining_time_seconds(self) -> float | None:
        """Remaining agent session time in seconds, or None if session time is unconstrained."""
        if self.budget.time_minutes is None:
            return None
        if self.agent_start_time is None:
            return self.budget.time_minutes * 60.0
        return max(0.0, (self.budget.time_minutes * 60.0) - self.agent_elapsed_seconds)

    def check_budget(self, *, count_tool_call: bool = True) -> str | None:
        """Check consumable budget limits and increment call count. Returns error JSON if exceeded."""
        with self._lock:
            if (
                count_tool_call
                and self.budget.tool_calls is not None
                and self.tool_calls_used >= self.budget.tool_calls
            ):
                return json.dumps(
                    {
                        'status': 'error',
                        'error_type': 'BudgetExceeded',
                        'error_message': f'Tool call budget exhausted ({self.budget.tool_calls} calls)',
                    },
                    ensure_ascii=False,
                )

            if (
                count_tool_call
                and self.budget.turns is not None
                and self.llm_calls_used >= self.budget.turns
            ):
                return json.dumps(
                    {
                        'status': 'error',
                        'error_type': 'BudgetExceeded',
                        'error_message': f'Turn budget exhausted ({self.budget.turns} turns)',
                    },
                    ensure_ascii=False,
                )

            if (
                count_tool_call
                and self.budget.cost_usd is not None
                and self.total_cost_usd >= self.budget.cost_usd
            ):
                return json.dumps(
                    {
                        'status': 'error',
                        'error_type': 'BudgetExceeded',
                        'error_message': f'Cost budget exhausted (${self.budget.cost_usd:.4f} USD)',
                    },
                    ensure_ascii=False,
                )

            if (
                count_tool_call
                and self.budget.total_tokens is not None
                and self.total_tokens >= self.budget.total_tokens
            ):
                return json.dumps(
                    {
                        'status': 'error',
                        'error_type': 'BudgetExceeded',
                        'error_message': f'Token budget exhausted ({self.budget.total_tokens} tokens)',
                    },
                    ensure_ascii=False,
                )

            if self.budget.time_minutes is not None:
                if self.budget.time_minutes <= 0:
                    return json.dumps(
                        {
                            'status': 'error',
                            'error_type': 'TimeoutExceeded',
                            'error_message': f'Session time budget exhausted ({self.budget.time_minutes} minutes)',
                        },
                        ensure_ascii=False,
                    )
                if self.agent_start_time is not None:
                    elapsed_minutes = self.agent_elapsed_seconds / 60.0
                    if elapsed_minutes > self.budget.time_minutes:
                        return json.dumps(
                            {
                                'status': 'error',
                                'error_type': 'TimeoutExceeded',
                                'error_message': f'Session time budget exhausted ({self.budget.time_minutes} minutes)',
                            },
                            ensure_ascii=False,
                        )

            if count_tool_call:
                self.tool_calls_used += 1
            return None

    def get_display_snapshot(self) -> DisplaySnapshot:
        """Capture an immutable snapshot of context state for thread-safe UI rendering."""
        with self._lock:
            eff_tools = getattr(self, 'tool_calls_used', 0) or 0
            eff_max_tools = (
                self.budget.tool_calls
                if (self.budget and self.budget.tool_calls is not None)
                else getattr(self, 'max_tool_calls', None)
            )
            agent_started = self.agent_start_time is not None
            eff_elapsed = self.agent_elapsed_seconds if agent_started else 0.0
            return DisplaySnapshot(
                task_id=self.task_id
                if (self.task_id and self.task_id != 'uninitialized')
                else '',
                repo=self.repo or '',
                phase=self.current_phase or 'Starting',
                tools_used=eff_tools,
                max_tools=eff_max_tools,
                elapsed_seconds=eff_elapsed,
                activity=self.current_activity or '',
                agent_started=agent_started,
            )

    def exec(self, command: str, *, timeout: int | None = None) -> Any:
        """Execute a command in this context's sandbox."""
        if self.sandbox is not None and hasattr(self.sandbox, 'exec'):
            return self.sandbox.exec(command, timeout=timeout)
        if self.docker and self.container_id:
            return self.docker.exec(self.container_id, command, timeout=timeout)
        raise RuntimeError('Sandbox container is not initialized in this context')

    async def exec_async(self, command: str, *, timeout: int | None = None) -> Any:
        """Execute a command in this context's sandbox asynchronously without blocking the event loop."""
        if self.sandbox is not None and hasattr(self.sandbox, 'exec_async'):
            return await self.sandbox.exec_async(command, timeout=timeout)
        if self.docker and self.container_id:
            if hasattr(self.docker, 'exec_async'):
                return await self.docker.exec_async(
                    self.container_id, command, timeout=timeout
                )
            return await asyncio.to_thread(
                self.docker.exec, self.container_id, command, timeout=timeout
            )
        raise RuntimeError('Sandbox container is not initialized in this context')

    def create_tools(self) -> dict[str, Callable]:
        """Create tool functions bound to this context.

        Returns a dict mapping tool names to callables.
        """
        return create_context_tools(self)
