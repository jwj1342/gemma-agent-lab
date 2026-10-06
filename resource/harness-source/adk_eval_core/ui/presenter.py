"""Evaluation results presentation protocols and standard presenters."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from adk_eval_core.runner import EvaluationResult, TaskResult
from adk_eval_core.ui.protocol import ResultPresenter


class DefaultSummaryPresenter:
    """Standard evaluation summary presenter supporting Rich tables, Plaintext, Markdown, and JSON."""

    def __init__(self, title: str = "Evaluation Results Summary") -> None:
        self.title = title

    def present(
        self,
        result: EvaluationResult | TaskResult | list[TaskResult] | Any,
        console: Console | None = None,
        *,
        output_format: str = "rich",
        group_by: str | None = None,
        show_metadata: bool = False,
        **kwargs: Any,
    ) -> str | Any:
        """Present evaluation results in the specified format."""
        task_results: list[TaskResult] = []
        if isinstance(result, EvaluationResult):
            task_results = result.task_results
        elif isinstance(result, TaskResult):
            task_results = [result]
        elif isinstance(result, list):
            task_results = [r for r in result if isinstance(r, TaskResult) or hasattr(r, "resolved")]
        elif hasattr(result, "task_results"):
            task_results = getattr(result, "task_results", [])
        elif hasattr(result, "results"):
            task_results = getattr(result, "results", [])
        else:
            task_results = []

        total = len(task_results)
        resolved_count = sum(1 for tr in task_results if getattr(tr, "resolved", False))
        rate = (resolved_count / total * 100) if total > 0 else 0.0

        if output_format == "json":
            tasks_data = []
            for tr in task_results:
                tid = getattr(tr, "instance_id", getattr(tr, "task_id", str(tr)))
                dur_raw = getattr(tr, "duration_seconds", None)
                try:
                    dur = float(dur_raw) if dur_raw is not None else 0.0
                except (TypeError, ValueError):
                    dur = 0.0
                tasks_data.append({
                    "instance_id": tid,
                    "resolved": getattr(tr, "resolved", False),
                    "duration_seconds": dur,
                    "error": getattr(tr, "error", None),
                    "metadata": getattr(tr, "metadata", {}),
                })
            data: dict[str, Any] = {
                "title": self.title,
                "total": total,
                "resolved": resolved_count,
                "resolution_rate": rate,
                "tasks": tasks_data,
            }
            if group_by:
                group_by_fn = getattr(result, "group_by", None)
                by_repo_fn = getattr(result, "by_repo", None)
                if callable(group_by_fn):
                    data["groups"] = group_by_fn(group_by)
                elif group_by == "repo" and callable(by_repo_fn):
                    data["groups"] = by_repo_fn()
                elif task_results:
                    data["groups"] = EvaluationResult(task_results=task_results).group_by(group_by)
            out_str = json.dumps(data, indent=2)
            if console:
                console.print(out_str)
            return out_str

        if output_format == "markdown":
            lines = [
                f"# {self.title}",
                "",
                f"**Total Tasks**: {total} | **Resolved**: {resolved_count} ({rate:.2f}%)",
                "",
                "| Task ID | Resolved | Duration | Error |",
                "| :--- | :---: | :---: | :--- |",
            ]
            for tr in task_results:
                tid = str(getattr(tr, "instance_id", getattr(tr, "task_id", str(tr)))).replace("|", "\\|").replace("`", "\\`").replace("\n", " ").replace("\r", "")
                res = "✓ YES" if getattr(tr, "resolved", False) else "✗ NO"
                dur_raw = getattr(tr, "duration_seconds", None)
                try:
                    dur = float(dur_raw) if dur_raw is not None else 0.0
                except (TypeError, ValueError):
                    dur = 0.0
                err_cell = str(getattr(tr, "error", "") or "").replace("|", "\\|").replace("\n", " ").replace("\r", "")
                lines.append(f"| `{tid}` | {res} | {dur:.1f}s | {err_cell} |")
            out_str = "\n".join(lines)
            if console:
                console.print(out_str)
            return out_str

        if output_format in ("text", "plain"):
            lines = [
                f"=== {self.title} ===",
                f"Resolved: {resolved_count}/{total} ({rate:.2f}%)",
                "-" * 60,
                f"{'Task ID':<30} {'Resolved':<10} {'Duration':<10}",
                "-" * 60,
            ]
            for tr in task_results:
                tid = str(getattr(tr, "instance_id", getattr(tr, "task_id", str(tr))))[:28]
                res = "YES" if getattr(tr, "resolved", False) else "NO"
                dur_raw = getattr(tr, "duration_seconds", None)
                try:
                    dur = float(dur_raw) if dur_raw is not None else 0.0
                except (TypeError, ValueError):
                    dur = 0.0
                lines.append(f"{tid:<30} {res:<10} {dur:.1f}s")
            lines.append("-" * 60)
            out_str = "\n".join(lines)
            if console:
                console.print(out_str)
            return out_str

        from rich.markup import escape

        # Default rich rendering
        table = Table(
            title=f"[bold green]{escape(str(self.title))}[/bold green]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Task ID / Instance", style="bold white")
        table.add_column("Resolved", justify="center")
        table.add_column("Duration", justify="right")
        if any(getattr(tr, "error", None) for tr in task_results):
            table.add_column("Error", style="bold red")

        for tr in task_results:
            tid = escape(str(getattr(tr, "instance_id", getattr(tr, "task_id", str(tr)))))
            res = getattr(tr, "resolved", False)
            dur_raw = getattr(tr, "duration_seconds", None)
            try:
                dur = float(dur_raw) if dur_raw is not None else 0.0
            except (TypeError, ValueError):
                dur = 0.0
            res_str = "[bold green]✓ YES[/bold green]" if res else "[bold red]✗ NO[/bold red]"
            row = [tid, res_str, f"{dur:.1f}s"]
            if any(getattr(t, "error", None) for t in task_results):
                row.append(escape(str(getattr(tr, "error", "") or "")))
            table.add_row(*row)

        if console:
            console.print()
            console.print(table)
            console.print(
                f"[bold]Total Resolved:[/bold] [bold yellow]{resolved_count}/{total}[/bold yellow] "
                f"([bold cyan]{rate:.2f}%[/bold cyan])"
            )
            console.print()

        return table


__all__ = [
    "DefaultSummaryPresenter",
    "ResultPresenter",
]
