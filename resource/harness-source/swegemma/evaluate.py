"""SWE-bench evaluation orchestrator."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any

from adk_eval_core.tracing import SessionTrace

from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.display import EvaluationDashboard
from swegemma.harness.agent_runner import run_agent_sandbox
from swegemma.harness.container_setup import (
    install_test_dependencies,
    resolve_wheels_dir,
    setup_container_wheels,
)
from swegemma.harness.verification import save_trace_artifact, verify_task
from swegemma.models import (
    EvaluationResult,
    Task,
    TaskResult,
    extract_test_files_from_patch,
    load_tasks,
)
from swegemma.results import append_task_result, save_results
from swegemma.sandbox import (
    ContainerConfig,
    ContainerManager,
    SubprocessManager,
)
from swegemma.warnings import _suppress_warnings

logger = logging.getLogger(__name__)

# Suppress OpenTelemetry detach errors caused by async generator context isolation
logging.getLogger('opentelemetry.context').setLevel(logging.CRITICAL)
logging.getLogger('opentelemetry').setLevel(logging.CRITICAL)

# Re-export for public API and backwards compatibility
__all__ = [
    'EvalConfig',
    'Evaluator',
    'extract_test_files_from_patch',
]


class Evaluator:
    """Orchestrates evaluation of an agent submission on SWE-bench tasks."""

    def __init__(self, config: EvalConfig) -> None:
        _suppress_warnings()
        self.config = config
        self._secret_hydration_cache: tuple[dict[str, str], dict[str, str], Path | None] | None = None
        cmd_timeout = (
            config.harness.command_timeout_seconds
            if config.harness.command_timeout_seconds is not None
            else 300
        )
        if config.sandbox == 'docker':
            container_config = ContainerConfig(
                image=config.image,
                timeout_seconds=cmd_timeout,
            )
            self.sandbox: Any = ContainerManager(container_config)
        else:
            self.sandbox = SubprocessManager(
                timeout_seconds=cmd_timeout,
            )
        self.docker = self.sandbox

    def _get_secret_bundle_data(
        self,
    ) -> tuple[dict[str, str], dict[str, str], Path | None]:
        """Loads and caches (test_patch_map, patch_map, secret_dir) from the sibling secret data bundle."""
        if self._secret_hydration_cache is not None:
            return self._secret_hydration_cache

        secret_dir: Path | None = None
        roots = [
            p for p in (self.config.tasks_path, self.config.snapshots_dir) if p is not None
        ]
        for root in roots:
            try:
                resolved = Path(root).resolve()
            except Exception:
                continue
            for anc in [resolved, *resolved.parents]:
                cand = anc / 'secret'
                if (cand / 'solution.parquet').is_file() or (cand / 'solution.csv').is_file():
                    secret_dir = cand
                    break
            if secret_dir is not None:
                break

        test_patch_map: dict[str, str] = {}
        patch_map: dict[str, str] = {}
        if secret_dir is not None:
            sol_parquet = secret_dir / 'solution.parquet'
            sol_csv = secret_dir / 'solution.csv'
            if sol_parquet.is_file():
                try:
                    import pandas as pd

                    sol_df = pd.read_parquet(sol_parquet)
                    for row in sol_df.to_dict(orient='records'):
                        iid = str(row.get('id') or row.get('instance_id') or '').strip()
                        tp = str(row.get('test_patch') or '')
                        if iid and tp:
                            test_patch_map[iid] = tp
                except Exception as e:
                    logger.warning('Failed reading %s: %s', sol_parquet, e)
            elif sol_csv.is_file():
                try:
                    csv.field_size_limit(sys.maxsize)
                except OverflowError:
                    csv.field_size_limit(2**31 - 1)
                try:
                    with open(sol_csv, encoding='utf-8') as f:
                        for row in csv.DictReader(f):
                            iid = (row.get('id') or row.get('instance_id') or '').strip()
                            tp = row.get('test_patch') or ''
                            if iid and tp:
                                test_patch_map[iid] = tp
                except Exception as e:
                    logger.warning('Failed reading %s: %s', sol_csv, e)

            perf_parquet = secret_dir / 'perfect_submission.parquet'
            perf_csv = secret_dir / 'perfect_submission.csv'
            if perf_parquet.is_file():
                try:
                    import pandas as pd

                    perf_df = pd.read_parquet(perf_parquet)
                    for row in perf_df.to_dict(orient='records'):
                        iid = str(row.get('id') or row.get('instance_id') or '').strip()
                        pred = str(row.get('prediction') or row.get('patch') or '')
                        if iid and pred:
                            patch_map[iid] = pred
                except Exception as e:
                    logger.warning('Failed reading %s: %s', perf_parquet, e)
            elif perf_csv.is_file():
                try:
                    with open(perf_csv, encoding='utf-8') as f:
                        for row in csv.DictReader(f):
                            iid = (row.get('id') or row.get('instance_id') or '').strip()
                            pred = row.get('prediction') or row.get('patch') or ''
                            if iid and pred:
                                patch_map[iid] = pred
                except Exception as e:
                    logger.warning('Failed reading %s: %s', perf_csv, e)

        self._secret_hydration_cache = (test_patch_map, patch_map, secret_dir)
        return self._secret_hydration_cache

    def _hydrate_task_from_secret(self, task: Task) -> Task:
        """Hydrates stripped test_patch and patch fields from the secret data bundle if needed."""
        if task.test_patch and task.patch:
            return task
        test_patch_map, patch_map, _ = self._get_secret_bundle_data()
        new_test_patch = task.test_patch or test_patch_map.get(task.instance_id, '')
        new_patch = task.patch or patch_map.get(task.instance_id, '')
        if new_test_patch != task.test_patch or new_patch != task.patch:
            return dataclasses.replace(
                task,
                test_patch=new_test_patch,
                patch=new_patch,
            )
        return task

    def _resolve_wheels_dir(self) -> Path | None:
        """Finds the local wheels directory containing task dependencies."""
        return resolve_wheels_dir(self.config)

    def _setup_container_wheels(self, container_id: str) -> None:
        """Copies task wheels into the sandbox container if available on host and not already present."""
        setup_container_wheels(self.docker, container_id, self.config)

    def _install_test_dependencies(self, container_id: str, repo: str = '') -> None:
        """Installs test dependencies discovered from pyproject.toml and requirements files."""
        install_test_dependencies(
            self.docker, container_id, repo, config=self.config
        )

    async def _run_agent_sandbox(
        self,
        task: Task,
        snapshot_path: Path,
        task_index: int,
        total_tasks: int,
        slot_id: int | None = None,
        dashboard: EvaluationDashboard | None = None,
        context: SwegemmaContext | None = None,
        base_snapshot_path: Path | None = None,
        patch_path: Path | None = None,
    ) -> tuple[str, str | None, SessionTrace]:
        """Runs the agent in the sandbox container to generate a patch."""
        return await run_agent_sandbox(
            self.docker,
            self.config,
            task,
            snapshot_path,
            base_snapshot_path=base_snapshot_path,
            patch_path=patch_path,
            task_index=task_index,
            total_tasks=total_tasks,
            slot_id=slot_id,
            dashboard=dashboard,
            context=context,
        )

    async def evaluate_task(
        self,
        task: Task,
        task_index: int = 1,
        total_tasks: int = 1,
        slot_id: int | None = None,
        dashboard: EvaluationDashboard | None = None,
        context: SwegemmaContext | None = None,
    ) -> TaskResult:
        """Evaluate a single task using the two-container pipeline."""
        task = self._hydrate_task_from_secret(task)
        logger.debug('Evaluating task %s (%s)', task.instance_id, task.repo)
        start_time = time.perf_counter()

        slots_map = getattr(dashboard, 'slots', None) if dashboard is not None else None
        slot = (
            slots_map.get(slot_id)
            if isinstance(slots_map, dict) and slot_id is not None
            else None
        )
        if context is None and slot is not None and slot.context is not None:
            context = slot.context

        if context is None:
            context = SwegemmaContext(
                docker_manager=self.docker,
                container_id='',
                problem_statement=task.problem_statement,
                hints_text=getattr(task, 'hints_text', '') or '',
                task=task,
                repo=task.repo,
                graph_dir=self.config.graph_dir,
                embeddings_dir=self.config.embeddings_dir,
                budget=self.config.budget,
                harness=self.config.harness,
            )
            if (
                dashboard is not None
                and slot_id is not None
                and hasattr(dashboard, 'attach_context')
            ):
                dashboard.attach_context(slot_id, context)
        elif (
            dashboard is not None
            and slot_id is not None
            and hasattr(dashboard, 'attach_context')
            and (slot is None or slot.context is not context)
        ):
            dashboard.attach_context(slot_id, context)

        snapshot_path, base_snapshot_path, patch_path = resolve_task_snapshot_paths(
            self.config.snapshots_dir, task.instance_id, task.repo
        )
        if not snapshot_path.exists():
            _, _, secret_dir = self._get_secret_bundle_data()
            if secret_dir is not None and (secret_dir / 'sandbox' / 'snapshots').is_dir():
                snapshot_path, base_snapshot_path, patch_path = resolve_task_snapshot_paths(
                    secret_dir / 'sandbox' / 'snapshots', task.instance_id, task.repo
                )
        if not snapshot_path.exists():
            err_msg = f'Snapshot file not found: {self.config.snapshots_dir / f"{task.instance_id}.tar.gz"} or .tgz'
            duration = time.perf_counter() - start_time
            if dashboard is not None and slot_id is not None:
                dashboard.task_completed(slot_id, False, duration, err_msg)
            return TaskResult(
                instance_id=task.instance_id,
                repo=task.repo,
                resolved=False,
                agent_patch='',
                test_output='',
                test_exit_code=-1,
                duration_seconds=duration,
                error=err_msg,
                task_index=task_index,
            )

        # -------------------------------------------------------------------
        # Phase 1: Sandbox Execution (Container A)
        # -------------------------------------------------------------------
        if self.config.skip_agent_patch:
            agent_patch = ''
            agent_error = None
            trace: SessionTrace | None = SessionTrace()
        else:
            run_kwargs: dict[str, Any] = {
                'slot_id': slot_id,
                'dashboard': dashboard,
                'context': context,
            }
            sig = inspect.signature(self._run_agent_sandbox)
            if 'base_snapshot_path' in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                run_kwargs['base_snapshot_path'] = base_snapshot_path
                run_kwargs['patch_path'] = patch_path
            agent_patch, agent_error, trace = await self._run_agent_sandbox(
                task,
                snapshot_path,
                task_index,
                total_tasks,
                **run_kwargs,
            )

        if agent_error and not agent_patch:
            json_path = save_trace_artifact(
                trace, self.config.results_dir, task.instance_id
            )
            duration = time.perf_counter() - start_time
            llm_calls = context.llm_calls_used
            if dashboard is not None and slot_id is not None:
                dashboard.task_completed(
                    slot_id,
                    False,
                    duration,
                    agent_error,
                    agent_duration=context.agent_elapsed_seconds if context else None,
                )
            return TaskResult(
                instance_id=task.instance_id,
                repo=task.repo,
                resolved=False,
                agent_patch='',
                test_output='',
                test_exit_code=-1,
                duration_seconds=duration,
                error=agent_error,
                trace_json_path=json_path,
                total_llm_calls=llm_calls,
                tool_calls=context.tool_calls_used,
                task_index=task_index,
            )

        # -------------------------------------------------------------------
        # Phase 2: Evaluation Container (Container B)
        # -------------------------------------------------------------------
        context.set_phase('Verifying')
        context.set_activity('Running test suite...')

        verify_kwargs: dict[str, Any] = {
            'agent_patch': agent_patch,
            'agent_error': agent_error,
            'trace': trace,
            'start_time': start_time,
        }
        verify_sig = inspect.signature(verify_task)
        if 'base_snapshot_path' in verify_sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in verify_sig.parameters.values()
        ):
            verify_kwargs['base_snapshot_path'] = base_snapshot_path
            verify_kwargs['patch_path'] = patch_path

        res = await verify_task(
            self.docker,
            self.config,
            task,
            snapshot_path,
            **verify_kwargs,
        )
        llm_calls = context.llm_calls_used
        res.total_llm_calls = llm_calls
        res.tool_calls = context.tool_calls_used
        res.task_index = task_index
        if dashboard is not None and slot_id is not None:
            dashboard.task_completed(
                slot_id,
                res.resolved,
                res.duration_seconds,
                res.error,
                agent_duration=context.agent_elapsed_seconds if context else None,
            )
        return res

    async def run(self) -> EvaluationResult:
        """Run evaluation across all configured tasks."""
        tasks = [
            self._hydrate_task_from_secret(t)
            for t in load_tasks(self.config.tasks_path)
        ]
        if self.config.task_ids:
            task_set = set(self.config.task_ids)
            tasks = [t for t in tasks if t.instance_id in task_set]

        if self.config.num_shards and self.config.num_shards > 0:
            shard_idx = self.config.shard_index or 0
            tasks = [
                t
                for idx, t in enumerate(tasks)
                if idx % self.config.num_shards == shard_idx
            ]
            logger.info(
                'Running shard %d/%d (%d tasks)',
                shard_idx + 1,
                self.config.num_shards,
                len(tasks),
            )

        logger.debug('Starting evaluation on %d tasks...', len(tasks))
        results: list[TaskResult] = []
        concurrency = max(1, self.config.concurrency or 1)

        display_mode = getattr(self.config, 'display_mode', 'auto')
        if display_mode == 'single' and concurrency > 1:
            logger.warning(
                '--display single is not compatible with --concurrency %d; '
                'falling back to dashboard mode.',
                concurrency,
            )
            display_mode = 'dashboard'
            self.config.display_mode = 'dashboard'

        use_dashboard = display_mode == 'dashboard' or (
            display_mode == 'auto' and concurrency > 1
        )

        async def _run_task_worker(
            i: int,
            task: Task,
            lock: asyncio.Lock,
            *,
            slot: int | None = None,
            active_dashboard: Any | None = None,
        ) -> TaskResult:
            worker_start = time.perf_counter()
            context: SwegemmaContext | None = None
            res: TaskResult | None = None
            try:
                context = SwegemmaContext(
                    docker_manager=self.docker,
                    container_id='',
                    problem_statement=task.problem_statement,
                    hints_text=getattr(task, 'hints_text', '') or '',
                    task=task,
                    repo=task.repo,
                    graph_dir=self.config.graph_dir,
                    embeddings_dir=self.config.embeddings_dir,
                    budget=self.config.budget,
                    harness=self.config.harness,
                )
                eval_kwargs: dict[str, Any] = {
                    'task_index': i,
                    'total_tasks': len(tasks),
                }
                if active_dashboard is not None and slot is not None:
                    active_dashboard.attach_context(slot, context)
                    eval_kwargs['slot_id'] = slot
                    eval_kwargs['dashboard'] = active_dashboard

                try:
                    target_fn = (
                        getattr(self.evaluate_task, 'side_effect', None)
                        or getattr(self.evaluate_task, '__wrapped__', None)
                        or self.evaluate_task
                    )
                    sig = inspect.signature(target_fn)
                    if 'context' in sig.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    ):
                        eval_kwargs['context'] = context
                except (ValueError, TypeError):
                    eval_kwargs['context'] = context

                try:
                    res = await self.evaluate_task(task, **eval_kwargs)
                except TypeError as te:
                    if (
                        'context' in eval_kwargs
                        and 'unexpected keyword argument' in str(te)
                    ):
                        eval_kwargs.pop('context', None)
                        res = await self.evaluate_task(task, **eval_kwargs)
                    else:
                        raise
                res.task_index = i
            except Exception as e:
                dur = time.perf_counter() - worker_start
                slots_map = (
                    getattr(active_dashboard, 'slots', None)
                    if active_dashboard is not None
                    else None
                )
                slot_obj = (
                    slots_map.get(slot)
                    if (isinstance(slots_map, dict) and slot is not None)
                    else None
                )
                err_turns = (
                    slot_obj.context.llm_calls_used
                    if (slot_obj and slot_obj.context)
                    else (
                        getattr(context, 'llm_calls_used', 0)
                        if context is not None
                        else 0
                    )
                )
                err_tools = (
                    slot_obj.context.tool_calls_used
                    if (slot_obj and slot_obj.context)
                    else (
                        getattr(context, 'tool_calls_used', 0)
                        if context is not None
                        else 0
                    )
                )
                if active_dashboard is not None and slot is not None:
                    active_dashboard.task_completed(slot, False, dur, str(e))
                logger.exception(
                    'Task %s worker failed with unexpected error',
                    task.instance_id,
                )
                res = TaskResult(
                    instance_id=task.instance_id,
                    repo=task.repo,
                    resolved=False,
                    agent_patch='',
                    test_output='',
                    test_exit_code=-1,
                    duration_seconds=dur,
                    error=f'Unexpected evaluation worker error: {e}',
                    total_llm_calls=err_turns,
                    tool_calls=err_tools,
                    task_index=i,
                )

            async with lock:
                results.append(res)
                try:
                    await asyncio.to_thread(
                        append_task_result,
                        res,
                        self.config.results_dir,
                        list(results),
                    )
                except Exception as persist_err:
                    logger.error(
                        'Failed to append task result for %s: %s',
                        task.instance_id,
                        persist_err,
                    )
            return res

        try:
            if use_dashboard:
                dashboard = EvaluationDashboard(
                    total_tasks=len(tasks),
                    concurrency=concurrency,
                    results_dir=self.config.results_dir,
                )
                slot_queue: asyncio.Queue[int] = asyncio.Queue()
                for s in range(concurrency):
                    slot_queue.put_nowait(s)
                lock = asyncio.Lock()

                async def _eval_worker(i: int, task: Task) -> TaskResult:
                    slot = await slot_queue.get()
                    try:
                        return await _run_task_worker(
                            i, task, lock, slot=slot, active_dashboard=dashboard
                        )
                    finally:
                        slot_queue.put_nowait(slot)

                with dashboard:
                    tasks_coros = [
                        _eval_worker(i, task) for i, task in enumerate(tasks, start=1)
                    ]
                    await asyncio.gather(*tasks_coros)

            elif concurrency == 1:
                lock = asyncio.Lock()
                for i, task in enumerate(tasks, start=1):
                    logger.debug(
                        '[%d/%d] Processing task %s', i, len(tasks), task.instance_id
                    )
                    await _run_task_worker(i, task, lock)
            else:
                sem = asyncio.Semaphore(concurrency)
                lock = asyncio.Lock()

                async def _eval_worker(i: int, task: Task) -> TaskResult:
                    async with sem:
                        logger.debug(
                            '[%d/%d] Processing task %s (worker)',
                            i,
                            len(tasks),
                            task.instance_id,
                        )
                        return await _run_task_worker(i, task, lock)

                tasks_coros = [
                    _eval_worker(i, task) for i, task in enumerate(tasks, start=1)
                ]
                await asyncio.gather(*tasks_coros)
        finally:
            if hasattr(self.docker, 'cleanup_all'):
                try:
                    self.docker.cleanup_all()
                except Exception as cleanup_err:
                    logger.debug('Sandbox cleanup_all notice: %s', cleanup_err)

        results.sort(
            key=lambda r: (
                getattr(r, 'task_index', None) is None,
                getattr(r, 'task_index', 0),
                r.instance_id,
            )
        )
        eval_res = EvaluationResult(task_results=results)
        try:
            save_results(eval_res, self.config.results_dir)
        except Exception as save_err:
            logger.error('Failed to save final evaluation results: %s', save_err)

        from swegemma.display import print_evaluation_summary

        if display_mode != 'quiet':
            print_evaluation_summary(eval_res.task_results)
        return eval_res
