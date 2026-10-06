"""Central terminal dashboard managing live progress across concurrent workers."""

from __future__ import annotations

import contextlib
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class SlotState:
    """State of an evaluation worker slot."""

    slot_id: int
    context: Any | None = None
    start_time: float = 0.0


class EvaluationDashboard:
    """Central terminal dashboard managing live progress across concurrent workers."""

    def __init__(
        self,
        total_tasks: int = 0,
        concurrency: int = 1,
        results_dir: Path | str | None = None,
        console: Console | None = None,
        title: str = "[bold green]Concurrent Evaluation[/bold green]",
    ) -> None:
        self.total_tasks = total_tasks
        self.concurrency = max(1, concurrency)
        self.results_dir = Path(results_dir) if results_dir else None
        self._console = console or Console()
        self.title = title
        self.slots: dict[int, SlotState] = {
            i: SlotState(slot_id=i) for i in range(self.concurrency)
        }
        self.completed_tasks = 0
        self.resolved_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self._live: Live | None = None
        self._lock = threading.RLock()
        self._completed_task_ids: set[str] = set()
        self._anon_task_counter: int = 0

    @property
    def live(self) -> Live | None:
        return self._live

    def attach_context(self, slot_id: int, context: Any) -> None:
        """Attach a task execution context to the slot for pull-based metrics."""
        with self._lock:
            slot = self.slots.get(slot_id)
            if slot is None:
                slot = SlotState(slot_id=slot_id)
                self.slots[slot_id] = slot
            slot.context = context
            slot.start_time = time.time()
        self._refresh()

    def task_started(self, slot_id: int, context: Any) -> None:
        """Attach a task execution context to the slot when task starts."""
        self.attach_context(slot_id, context)

    def task_completed(
        self,
        slot_id: int,
        resolved: bool,
        duration: float = 0.0,
        error: str | None = None,
        agent_duration: float | None = None,
    ) -> None:
        with self._lock:
            slot = self.slots.get(slot_id)
            if slot is None or slot.context is None:
                return

            ctx = slot.context
            task_id_attr = getattr(ctx, "task_id", None)
            if task_id_attr and task_id_attr != "uninitialized":
                eff_task_id = str(task_id_attr)
            else:
                self._anon_task_counter += 1
                eff_task_id = f"ctx-{id(ctx)}-{self._anon_task_counter}"
            if eff_task_id in self._completed_task_ids:
                slot.context = None
                slot.start_time = 0.0
                return
            self._completed_task_ids.add(eff_task_id)

            self.completed_tasks += 1
            if resolved:
                self.resolved_tasks += 1
            else:
                self.failed_tasks += 1

            # Fallback duration if 0.0 was passed
            eff_duration = getattr(
                ctx, "task_elapsed_seconds", getattr(ctx, "elapsed_seconds", 0.0)
            )
            if duration <= 0.0 and eff_duration > 0.0:
                duration = eff_duration
            elif duration <= 0.0 and slot.start_time > 0.0:
                duration = max(0.0, time.time() - slot.start_time)

            # Format total duration
            m, s = divmod(int(duration), 60)
            dur_str = f"{m}m {s:02d}s" if m > 0 else f"{duration:.1f}s"
            ts_str = time.strftime("%H:%M:%S")

            # Format agent session duration
            eff_agent_dur = agent_duration
            if eff_agent_dur is None:
                eff_agent_dur = getattr(ctx, "agent_elapsed_seconds", None)

            count = getattr(ctx, "tool_calls_used", None)
            if count is None:
                count = getattr(ctx, "tool_calls", 0)
            if count is None:
                count = 0

            details: list[str] = []
            if eff_agent_dur is not None and eff_agent_dur > 0.0:
                am, as_ = divmod(int(eff_agent_dur), 60)
                agent_dur_str = (
                    f"{am}m {as_:02d}s" if am > 0 else f"{eff_agent_dur:.1f}s"
                )
                details.append(f"agent: {agent_dur_str}")

            if count > 0:
                tools_label = "tool call" if count == 1 else "tool calls"
                details.append(f"{count} {tools_label}")

            details_str = f" ({', '.join(details)})" if details else ""

            # Reset slot
            slot.context = None
            slot.start_time = 0.0

            completed_tasks_snap = self.completed_tasks
            total_tasks_snap = self.total_tasks

        # Completion scrollback line rendered and printed outside self._lock
        total_str = (
            f"{completed_tasks_snap}/{total_tasks_snap}"
            if total_tasks_snap > 0
            else str(completed_tasks_snap)
        )
        safe_task_id = escape(eff_task_id)

        if resolved:
            line = (
                f"[dim][{ts_str}][/dim] [bold green]✓[/bold green] "
                f"[{total_str}] [bold]{safe_task_id}[/bold]: "
                f"[bold green]RESOLVED[/bold green] in {dur_str}{details_str}"
            )
        elif error and error.strip():
            err_lines = error.strip().splitlines()
            err_preview = err_lines[0] if err_lines else ""
            if len(err_preview) > 50:
                err_preview = err_preview[:47] + "..."
            safe_err = escape(err_preview)
            line = (
                f"[dim][{ts_str}][/dim] [bold red]✗[/bold red] "
                f"[{total_str}] [bold]{safe_task_id}[/bold]: "
                f"[bold red]FAILED[/bold red] ([italic]{safe_err}[/italic]) in {dur_str}{details_str}"
            )
        else:
            line = (
                f"[dim][{ts_str}][/dim] [bold red]✗[/bold red] "
                f"[{total_str}] [bold]{safe_task_id}[/bold]: "
                f"[bold red]UNRESOLVED[/bold red] in {dur_str}{details_str}"
            )

        target_console = self._live.console if self._live else self._console
        target_console.print(line)
        self._refresh()

    def _build_renderable(self) -> Group:
        # 1. Take a snapshot of state under self._lock
        with self._lock:
            elapsed = time.time() - self.start_time
            completed_tasks = self.completed_tasks
            total_tasks = self.total_tasks
            resolved_tasks = self.resolved_tasks
            failed_tasks = self.failed_tasks
            concurrency = self.concurrency

            all_slot_ids = sorted(set(range(concurrency)) | set(self.slots.keys()))
            slots_snapshot: list[tuple[int, Any | None]] = [
                (
                    slot_id,
                    self.slots[slot_id].context if slot_id in self.slots else None,
                )
                for slot_id in all_slot_ids
            ]

        # 2. Extract slot and context metrics outside self._lock to prevent lock-order inversion
        slot_data: list[
            tuple[int, bool, str, str, str, int, int | None, float, str, bool]
        ] = []
        active_count = 0
        for slot_id, ctx in slots_snapshot:
            if ctx is None or getattr(ctx, "current_phase", "Idle") == "Idle":
                slot_data.append(
                    (
                        slot_id,
                        True,
                        "",
                        "",
                        "Idle",
                        0,
                        None,
                        0.0,
                        "Waiting for task...",
                        False,
                    )
                )
            else:
                active_count += 1
                if hasattr(ctx, "get_display_snapshot"):
                    snap = ctx.get_display_snapshot()
                    agent_started = getattr(
                        snap,
                        "agent_started",
                        ("Agent" in snap.phase or "Verif" in snap.phase),
                    )
                    slot_data.append(
                        (
                            slot_id,
                            False,
                            snap.task_id,
                            snap.repo,
                            snap.phase,
                            snap.tools_used,
                            snap.max_tools,
                            snap.elapsed_seconds,
                            snap.activity,
                            agent_started,
                        )
                    )
                else:
                    task_id_val = getattr(ctx, "task_id", "")
                    eff_task_id = (
                        str(task_id_val)
                        if (task_id_val and task_id_val != "uninitialized")
                        else ""
                    )
                    eff_repo = str(getattr(ctx, "repo", "") or "")
                    eff_phase = getattr(ctx, "current_phase", "Starting") or "Starting"
                    eff_tools = getattr(ctx, "tool_calls_used", None)
                    if eff_tools is None:
                        eff_tools = getattr(ctx, "tool_calls", 0)
                    eff_tools = eff_tools or 0
                    budget_obj = getattr(ctx, "budget", None)
                    eff_max_tools = (
                        budget_obj.tool_calls
                        if (
                            budget_obj
                            and getattr(budget_obj, "tool_calls", None) is not None
                        )
                        else getattr(ctx, "max_tool_calls", None)
                    )
                    agent_started = getattr(ctx, "agent_start_time", None) is not None
                    eff_elapsed = (
                        getattr(ctx, "agent_elapsed_seconds", 0.0)
                        if agent_started
                        else 0.0
                    )
                    eff_activity = getattr(ctx, "current_activity", "") or ""
                    slot_data.append(
                        (
                            slot_id,
                            False,
                            eff_task_id,
                            eff_repo,
                            eff_phase,
                            eff_tools,
                            eff_max_tools,
                            eff_elapsed,
                            eff_activity,
                            agent_started,
                        )
                    )

        # 3. Build Rich objects outside self._lock to prevent lock-order inversion with Live._lock
        em, es = divmod(int(elapsed), 60)
        elapsed_str = f"{em}m {es:02d}s"

        progress_pct = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
        pass_rate_pct = (
            (resolved_tasks / completed_tasks * 100) if completed_tasks > 0 else 0.0
        )

        # Header Panel
        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left", no_wrap=True)
        header_table.add_column(justify="right", no_wrap=True)

        line1_left = Text.from_markup(
            f"Progress: [bold cyan]{completed_tasks}/{total_tasks}[/bold cyan] ({progress_pct:.1f}%)"
        )
        line1_right = Text.from_markup(f"Elapsed: [bold]{elapsed_str}[/bold]")
        header_table.add_row(line1_left, line1_right)

        line2_left = Text.from_markup(
            f"Resolved: [bold green]{resolved_tasks}[/bold green] │ "
            f"Failed: [bold red]{failed_tasks}[/bold red] │ "
            f"Pass Rate: [bold]{pass_rate_pct:.1f}%[/bold]"
        )
        line2_right = Text.from_markup(
            f"Active Workers: [bold cyan]{active_count}[/bold cyan]/{concurrency}"
        )
        header_table.add_row(line2_left, line2_right)

        header_panel = Panel(
            header_table,
            title=self.title,
            border_style="green",
            padding=(0, 1),
        )

        # Active Workers Table
        console_width = getattr(self._console, "width", 80) or 80
        worker_table = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
        )
        if console_width < 95:
            worker_table.add_column("Slot", justify="center", width=4, no_wrap=True)
            worker_table.add_column(
                "Task ID", style="bold", max_width=16, overflow="ellipsis", no_wrap=True
            )
            worker_table.add_column(
                "Repo", style="dim", max_width=8, overflow="ellipsis", no_wrap=True
            )
            worker_table.add_column("Phase", width=8, no_wrap=True)
            worker_table.add_column("Tools", justify="center", width=5, no_wrap=True)
            worker_table.add_column("Elapsed", justify="right", width=7, no_wrap=True)
            worker_table.add_column(
                "Activity", ratio=1, overflow="ellipsis", no_wrap=True
            )
        else:
            worker_table.add_column("Slot", justify="center", width=5, no_wrap=True)
            worker_table.add_column(
                "Task ID", style="bold", max_width=28, overflow="ellipsis", no_wrap=True
            )
            worker_table.add_column(
                "Repo", style="dim", max_width=16, overflow="ellipsis", no_wrap=True
            )
            worker_table.add_column("Phase", width=11, no_wrap=True)
            worker_table.add_column("Tools", justify="center", width=7, no_wrap=True)
            worker_table.add_column("Elapsed", justify="right", width=7, no_wrap=True)
            worker_table.add_column(
                "Activity", ratio=1, overflow="ellipsis", no_wrap=True
            )

        for (
            slot_id,
            is_idle,
            eff_task_id,
            eff_repo,
            eff_phase,
            eff_tools,
            eff_max_tools,
            eff_elapsed,
            eff_activity,
            agent_started,
        ) in slot_data:
            if is_idle:
                worker_table.add_row(
                    f"#{slot_id + 1}",
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("Idle", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("Waiting for task...", style="dim"),
                )
            else:
                phase_color = "cyan"
                if "Agent" in eff_phase:
                    phase_color = "bold blue"
                elif "Verif" in eff_phase:
                    phase_color = "bold magenta"
                elif "Start" in eff_phase:
                    phase_color = "yellow"

                if not agent_started or "Start" in eff_phase:
                    slot_elapsed = Text("—", style="dim")
                else:
                    sem, ses = divmod(int(eff_elapsed), 60)
                    slot_elapsed = f"{sem}:{ses:02d}"

                tools_str = (
                    f"{eff_tools}/{eff_max_tools}"
                    if eff_max_tools is not None
                    else str(eff_tools)
                )

                worker_table.add_row(
                    f"#{slot_id + 1}",
                    Text(eff_task_id, style="bold"),
                    Text(eff_repo, style="dim"),
                    Text(eff_phase, style=phase_color),
                    tools_str,
                    slot_elapsed,
                    Text(eff_activity),
                )

        # Footer
        if self.results_dir:
            logs_path = self.results_dir / "logs"
            footer_text = Text.from_markup(
                f"[dim]Logs: [underline]{escape(str(logs_path))}[/underline]  │  tail -f {escape(str(logs_path))}/<task_id>.log[/dim]"
            )
        else:
            footer_text = Text.from_markup(
                "[dim]Inspect logs: tail -f <results_dir>/logs/<task_id>.log[/dim]"
            )

        return Group(header_panel, worker_table, footer_text)

    def render(self) -> Group:
        return self._build_renderable()

    def __rich__(self) -> Group:
        return self._build_renderable()

    def _refresh(self) -> None:
        if self._live is not None:
            with contextlib.suppress(Exception):
                if hasattr(self._live, "refresh"):
                    self._live.refresh()
                elif hasattr(self._live, "update"):
                    self._live.update(self._build_renderable(), refresh=True)

    def start(self) -> None:
        with self._lock:
            self.start_time = time.time()
            if self._live is None:
                self._live = Live(
                    get_renderable=self._build_renderable,
                    console=self._console,
                    refresh_per_second=4,
                )
                live = self._live
            else:
                live = None
        if live is not None:
            live.start()

    def stop(self) -> None:
        with self._lock:
            live = self._live
            self._live = None
        if live is not None:
            live.stop()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.stop()


__all__ = ["EvaluationDashboard", "SlotState"]
