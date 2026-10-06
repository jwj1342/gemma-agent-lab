"""ADK plugin and rich display utilities for real-time evaluation progress.

Re-exports and wraps display components from adk_eval_core.ui and adk_eval_core.plugins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adk_eval_core.plugins import EventDisplayPlugin
from adk_eval_core.ui import (
    EvaluationDashboard as _CoreEvaluationDashboard,
)
from adk_eval_core.ui import (
    EventDisplay as _CoreEventDisplay,
)
from adk_eval_core.ui import (
    SlotState,
    print_evaluation_summary,
)
from adk_eval_core.ui import (
    StatusPanel as _CoreStatusPanel,
)
from adk_eval_core.viewer import TraceViewer
from rich.console import Console


class EvaluationDashboard(_CoreEvaluationDashboard):
    """Central terminal dashboard managing live progress across concurrent workers."""

    def __init__(
        self,
        total_tasks: int = 0,
        concurrency: int = 1,
        results_dir: Path | str | None = None,
        console: Console | None = None,
        title: str = '[bold green]SWE-gemma Concurrent Evaluation[/bold green]',
    ) -> None:
        super().__init__(
            total_tasks=total_tasks,
            concurrency=concurrency,
            results_dir=results_dir,
            console=console,
            title=title,
        )


class StatusPanel(_CoreStatusPanel):
    """Renders the persistent status HUD for SWE-gemma evaluations."""

    def __init__(
        self,
        *,
        ctx: Any | None = None,
        instance_id: str = '',
        repo: str = '',
        task_index: int = 1,
        total_tasks: int = 1,
        **kwargs: Any,
    ) -> None:
        title = '[bold green]SWE-gemma Agent Evaluator[/bold green]'
        task_prefix = f'[{task_index}/{total_tasks}] ' if total_tasks > 1 else ''
        sub_id = f'{instance_id} ({repo})' if repo else instance_id
        subtitle = f'{task_prefix}{sub_id}' if sub_id else ''
        super().__init__(
            ctx=ctx,
            title=title,
            subtitle=subtitle,
            **kwargs,
        )
        self._instance_id = instance_id
        self._repo = repo
        self._task_index = task_index
        self._total_tasks = total_tasks

    def set_context(self, ctx: Any, instance_id: str, repo: str) -> None:
        """Update live context reference for a new task."""
        self.ctx = ctx
        self.budget_ctx = ctx
        self._instance_id = instance_id
        self._repo = repo
        task_prefix = (
            f'[{self._task_index}/{self._total_tasks}] '
            if self._total_tasks > 1
            else ''
        )
        sub_id = f'{instance_id} ({repo})' if repo else instance_id
        self.subtitle = f'{task_prefix}{sub_id}' if sub_id else ''
        self.reset()


class EventDisplayManager(_CoreEventDisplay):
    """Manages rich live display HUD and scrolling event log in terminal and Jupyter."""

    def __init__(
        self,
        *,
        ctx: Any | None = None,
        instance_id: str = '',
        repo: str = '',
        task_index: int = 1,
        total_tasks: int = 1,
        console: Console | None = None,
        enable_live: bool = True,
        **kwargs: Any,
    ) -> None:
        title = '[bold green]SWE-gemma Agent Evaluator[/bold green]'
        super().__init__(
            console=console,
            title=title,
            ctx=ctx,
            enable_live=enable_live,
            **kwargs,
        )
        self.hud = StatusPanel(
            ctx=ctx,
            instance_id=instance_id,
            repo=repo,
            task_index=task_index,
            total_tasks=total_tasks,
        )
        self.panel = self.hud
        self._panel = self.hud


# Alias for backward compatibility
EventDisplay = EventDisplayManager

__all__ = [
    'EvaluationDashboard',
    'EventDisplay',
    'EventDisplayManager',
    'EventDisplayPlugin',
    'SlotState',
    'StatusPanel',
    'TraceViewer',
    'print_evaluation_summary',
]
