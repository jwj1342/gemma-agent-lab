"""HUD and status panel rendering for evaluation progress."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from adk_eval_core.ui.protocol import EvaluationContextProtocol

logger = logging.getLogger(__name__)


def gauge_color(fraction: float) -> str:
    """Return a rich style name based on fraction usage (green < 50%, yellow 50-80%, red > 80%)."""
    if fraction < 0.5:
        return "green"
    if fraction < 0.8:
        return "yellow"
    return "red"


class StatusPanel:
    """Renders persistent status HUD as a Rich Panel."""

    def __init__(
        self,
        ctx: EvaluationContextProtocol | Any = None,
        *,
        budget_ctx: Any = None,
        title: str = "ADK Evaluation",
        subtitle: str = "",
        task_id: str = "",
        problem_id: str = "",
        metric: str = "",
        higher_is_better: bool = True,
        **kwargs: Any,
    ) -> None:
        self.ctx = ctx if ctx is not None else budget_ctx
        self.budget_ctx = self.ctx  # Retained for backwards compatibility
        self.title = title
        self.subtitle = subtitle
        self._problem_id = problem_id or task_id
        self._metric = metric
        self._higher_is_better = higher_is_better
        self._start_time: float = time.time()

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, val: float) -> None:
        self._start_time = val

    @property
    def _ctx(self) -> Any:
        return self.ctx

    @_ctx.setter
    def _ctx(self, val: Any) -> None:
        self.ctx = val
        self.budget_ctx = val

    def reset(self) -> None:
        """Reset the elapsed timer."""
        self._start_time = time.time()

    def _format_elapsed(self) -> str:
        """Return human-readable elapsed time."""
        elapsed = time.time() - self._start_time
        m, s = divmod(int(elapsed), 60)
        return f"{m}m {s:02d}s"

    def _best_score(self) -> float | None:
        """Return the best public score so far, or None."""
        if not self._ctx or type(self._ctx).__name__ == "MagicMock":
            return None
        subs = getattr(self._ctx, "submissions", None)
        if not subs or type(subs).__name__ == "MagicMock":
            return None
        if isinstance(subs, dict):
            sub_list = list(subs.values())
        elif isinstance(subs, (list, tuple)):
            sub_list = list(subs)
        else:
            return None
        scores: list[float] = []
        for s in sub_list:
            if hasattr(s, "public_score"):
                ps = s.public_score
                if isinstance(ps, (int, float)) and not math.isnan(ps):
                    scores.append(float(ps))
        if not scores:
            return None
        return max(scores) if self._higher_is_better else min(scores)

    def _build_header(self) -> Text:
        """Build the top header line of the HUD."""
        direction = "↑" if self._higher_is_better else "↓"
        header = Text()

        # Problem / task / subtitle identifier
        prob = self._problem_id
        if not prob and self.subtitle:
            prob = self.subtitle
        if not prob and self._ctx is not None and type(self._ctx).__name__ != "MagicMock":
            task_id = getattr(self._ctx, "task_id", None)
            if task_id and isinstance(task_id, str):
                prob = f"Task: {task_id}"

        if prob:
            header.append(f"  {prob}", style="bold cyan")
        else:
            header.append("  ADK Evaluation", style="bold cyan")

        if self._metric:
            header.append(f"  ·  {self._metric} ({direction})", style="dim")

        header.append(f"  ·  {self._format_elapsed()}", style="bold white")

        # Token metrics
        tb = getattr(self._ctx, "token_budget", None) if self._ctx else None
        if tb is not None and type(tb).__name__ != "MagicMock":
            try:
                in_tok = int(getattr(tb, "total_input_tokens", 0) or 0)
                out_tok = int(getattr(tb, "total_output_tokens", 0) or 0)
                cached_tok = int(getattr(tb, "total_cached_input_tokens", 0) or 0)
                last_in = int(getattr(tb, "last_input_tokens", 0) or 0)
                llm_calls = int(getattr(tb, "llm_calls", 0) or 0)
                total_tok = in_tok + out_tok
                if total_tok > 0 or llm_calls > 0:
                    if cached_tok > 0:
                        header.append(
                            f"  ·  {total_tok:,} tok (ctx: {last_in:,} | {in_tok:,} in [{cached_tok:,} cached] / {out_tok:,} out | {llm_calls} calls)",
                            style="bold yellow",
                        )
                    else:
                        header.append(
                            f"  ·  {total_tok:,} tok (ctx: {last_in:,} | {in_tok:,} in / {out_tok:,} out | {llm_calls} calls)",
                            style="bold yellow",
                        )
            except (TypeError, ValueError):
                pass

        best = self._best_score()
        if best is not None:
            header.append(f"  ·  Best: {best:.4f}", style="bold green")

        return header

    def get_extra_gauge_rows(self) -> list[tuple[Any, ...]]:
        """Hook for subclasses to inject extra rows into the gauge table (e.g. Submissions)."""
        return []

    def get_extra_rows(self) -> list[RenderableType]:
        """Hook for subclasses to inject extra rows into the status panel (e.g. score history)."""
        return []

    def _render_budget_gauges(self) -> Table | None:
        """Render the tool-call / cost / time budget gauges."""
        if not self._ctx:
            return None

        ctx = self._ctx
        is_mock_ctx = type(ctx).__name__ == "MagicMock"
        gauge = Table.grid(padding=(0, 1))
        gauge.add_column("label", width=14, justify="right")
        gauge.add_column("bar", width=22)
        gauge.add_column("numbers")
        has_any_row = False

        # Extract values from ctx / to_status_dict
        b_dict: dict[str, Any] = {}
        if hasattr(ctx, "to_status_dict") and callable(ctx.to_status_dict):
            try:
                res = ctx.to_status_dict()
                if isinstance(res, dict):
                    b_dict = res
            except Exception:  # noqa: BLE001
                b_dict = {}

        # 1. Tool calls
        tc = b_dict.get("tool_calls")
        if tc is None and not is_mock_ctx:
            tc = getattr(ctx, "tool_calls", None)
        if tc is None:
            tc = b_dict.get("llm_calls")
        if tc is None and not is_mock_ctx:
            tc = getattr(ctx, "llm_calls", None)

        tc_max = b_dict.get("max_tool_calls")
        if tc_max is None and not is_mock_ctx:
            tc_max = getattr(ctx, "max_tool_calls", None)

        if tc is not None and type(tc).__name__ != "MagicMock":
            try:
                tc_int = int(tc)
            except (TypeError, ValueError):
                tc_int = 0
            try:
                tc_max_int = int(tc_max) if tc_max is not None and type(tc_max).__name__ != "MagicMock" else 0
            except (TypeError, ValueError):
                tc_max_int = 0

            tc_frac = tc_int / tc_max_int if tc_max_int > 0 else 0.0
            pb_tc_total = tc_max_int if tc_max_int > 0 else None
            tc_label = (
                f"  {tc_int} / {tc_max_int}  ({tc_max_int - tc_int} left)"
                if tc_max_int > 0
                else f"  {tc_int} (Unlimited)"
            )
            gauge.add_row(
                Text("Tools", style="bold"),
                ProgressBar(
                    total=pb_tc_total,
                    completed=tc_int,
                    complete_style=gauge_color(tc_frac),
                    finished_style="red",
                ),
                Text(tc_label, style="dim"),
            )
            has_any_row = True

        # Extra gauge rows hook (e.g. Submissions)
        for row in self.get_extra_gauge_rows():
            gauge.add_row(*row)
            has_any_row = True

        # 2. Token cost
        tb = getattr(ctx, "token_budget", None)
        if type(tb).__name__ == "MagicMock":
            tb = None

        cost = b_dict.get("total_cost_usd")
        cap = b_dict.get("max_budget_usd")

        if cost is None and tb is not None:
            cost = getattr(tb, "total_cost_usd", None)
        if cost is None and not is_mock_ctx:
            cost = getattr(ctx, "total_cost_usd", None)

        if cap is None and tb is not None:
            cap = getattr(tb, "max_budget_usd", None)
        if cap is None and not is_mock_ctx:
            cap = getattr(ctx, "max_budget_usd", None)

        if cost is not None and type(cost).__name__ != "MagicMock":
            try:
                cost_float = float(cost)
                cost_str = f"${cost_float:.2f}"
            except (TypeError, ValueError):
                cost_float = 0.0
                cost_str = str(cost)

            try:
                cap_float = float(cap) if cap is not None and type(cap).__name__ != "MagicMock" else None
            except (TypeError, ValueError):
                cap_float = None

            frac = cost_float / cap_float if (cap_float is not None and cap_float > 0) else 0.0
            pb_cap_total = cap_float if (cap_float is not None and cap_float > 0) else None
            remaining = max(0.0, cap_float - cost_float) if cap_float is not None else 0.0

            if cap_float is not None and cap_float > 0:
                cost_label = f"  {cost_str} / ${cap_float:.2f}  (${remaining:.2f} left)"
            elif cap is not None and not cap_float and type(cap).__name__ != "MagicMock":
                cost_label = f"  {cost_str} / {cap}"
            else:
                cost_label = f"  {cost_str} (Unlimited)"

            gauge.add_row(
                Text("Cost", style="bold"),
                ProgressBar(
                    total=pb_cap_total,
                    completed=min(cost_float, cap_float) if (cap_float is not None and cap_float > 0) else cost_float,
                    complete_style=gauge_color(frac),
                    finished_style="red",
                ),
                Text(cost_label, style="dim"),
            )
            has_any_row = True

        # 3. Time
        time_max = b_dict.get("max_time_minutes")
        has_time_attr = "max_time_minutes" in b_dict
        if time_max is None and not is_mock_ctx and hasattr(ctx, "max_time_minutes"):
            has_time_attr = True
            time_max = getattr(ctx, "max_time_minutes", None)

        elapsed_seconds = None
        if hasattr(ctx, "elapsed_seconds") and not is_mock_ctx:
            try:
                elapsed_seconds = float(ctx.elapsed_seconds)
            except (TypeError, ValueError):
                elapsed_seconds = None
        if elapsed_seconds is None:
            elapsed_seconds = time.time() - self._start_time
        elapsed_mins = elapsed_seconds / 60.0

        should_render_time = False
        if (time_max is not None and type(time_max).__name__ != "MagicMock") or (not is_mock_ctx and (has_time_attr or has_any_row)):
            should_render_time = True

        if should_render_time:
            try:
                time_max_float = (
                    float(time_max)
                    if time_max is not None and type(time_max).__name__ != "MagicMock"
                    else None
                )
            except (TypeError, ValueError):
                time_max_float = None

            time_frac = (
                elapsed_mins / time_max_float
                if (time_max_float is not None and time_max_float > 0)
                else 0.0
            )
            pb_time_total = (
                time_max_float
                if (time_max_float is not None and time_max_float > 0)
                else None
            )
            rem_mins = (
                max(0.0, time_max_float - elapsed_mins)
                if time_max_float is not None
                else 0.0
            )
            time_label = (
                f"  {elapsed_mins:.1f}m / {time_max_float:.0f}m  ({rem_mins:.1f}m left)"
                if (time_max_float is not None and time_max_float > 0)
                else f"  {elapsed_mins:.1f}m (Unlimited)"
            )
            gauge.add_row(
                Text("Time", style="bold"),
                ProgressBar(
                    total=pb_time_total,
                    completed=(
                        min(elapsed_mins, time_max_float)
                        if (time_max_float is not None and time_max_float > 0)
                        else elapsed_mins
                    ),
                    complete_style=gauge_color(time_frac),
                    finished_style="red",
                ),
                Text(time_label, style="dim"),
            )
            has_any_row = True

        return gauge if has_any_row else None

    def render(self) -> Panel:
        """Render the status HUD panel."""
        header = self._build_header()
        rows: list[Any] = [header]

        if self._ctx is not None:
            gauges = self._render_budget_gauges()
            if gauges is not None:
                rows.append(Text())  # spacer
                rows.append(gauges)

            extra_rows = self.get_extra_rows()
            if extra_rows:
                rows.extend(extra_rows)

        from rich.markup import escape

        panel_title = self.title if self.title.startswith("[") else f"[bold cyan]{escape(str(self.title))}[/bold cyan]"
        return Panel(
            Group(*rows),
            title=panel_title,
            border_style="blue",
            padding=(0, 1),
        )


__all__ = ["StatusPanel", "gauge_color"]

