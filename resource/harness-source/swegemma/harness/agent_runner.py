"""Phase 1 agent execution runner inside the SWE-bench sandbox environment."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from adk_eval_core.plugins import ModelRetryPlugin
from adk_eval_core.tracing import SessionTrace
from adk_submission import compile_submission
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from rich.console import Console

from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.display import (
    EvaluationDashboard,
    EventDisplayManager,
    EventDisplayPlugin,
)
from swegemma.harness.container_setup import (
    ContainerSetupError,
    disable_test_runners_in_sandbox,
    extract_snapshot,
    install_editable_package,
    install_test_dependencies,
    setup_baseline_commit,
    setup_container_wheels,
    setup_git_exclude,
    setup_workspace_test_config,
)
from swegemma.models import Task, resolve_swegemma_adapter
from swegemma.sandbox import (
    AdkSandboxCodeExecutor,
    BaseSandboxManager,
    sandbox_exec,
    sandbox_start,
    sandbox_stop,
)

logger = logging.getLogger(__name__)

__all__ = ['build_agent_prompt', 'run_agent_sandbox']


def build_agent_prompt(
    task: Task | Any,
    config: EvalConfig | Any,
    workspace_tree: str = '',
    *,
    enable_sandbox_testing: bool | None = None,
) -> str:
    """Build the initial prompt for the software engineering agent."""
    if enable_sandbox_testing is None:
        cfg_testing = getattr(config, 'enable_sandbox_testing', None)
        enable_sandbox_testing = (
            bool(cfg_testing) if isinstance(cfg_testing, bool) else True
        )

    budget_lines = []
    if getattr(config, 'budget', None):
        if config.budget.time_minutes is not None:
            budget_lines.append(
                f'- Time allowance: {config.budget.time_minutes} minutes'
            )
        if config.budget.tool_calls is not None:
            budget_lines.append(
                f'- Tool calls allowance: {config.budget.tool_calls} calls'
            )
        if config.budget.turns is not None:
            budget_lines.append(f'- Max loop iterations: {config.budget.turns} turns')
        if config.budget.cost_usd is not None:
            budget_lines.append(f'- Cost budget: ${config.budget.cost_usd:.2f} USD')

    harness_lines = []
    if getattr(config, 'harness', None):
        if config.harness.command_timeout_seconds is not None:
            harness_lines.append(
                f'- Single command timeout: {config.harness.command_timeout_seconds} seconds (commands exceeding this fail without ending the session)'
            )
        if config.harness.max_stdout_chars is not None:
            harness_lines.append(
                f'- Command output limit: {config.harness.max_stdout_chars} characters'
            )
        if config.harness.max_file_lines is not None:
            harness_lines.append(
                f'- File view limit: {config.harness.max_file_lines} lines per read_file call'
            )
        if getattr(config.harness, 'max_file_chars', None) is not None:
            harness_lines.append(
                f'- File character limit: {config.harness.max_file_chars} characters per read_file call'
            )
    harness_lines.append(
        '- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY pre-installed. Do NOT attempt to run pip install or download packages.'
    )

    prompt_parts = [
        f'You are evaluating a software engineering task for repository {task.repo}.\n\nProblem Statement:\n{task.problem_statement}\n'
    ]
    hints = getattr(task, 'hints_text', '') or ''
    if isinstance(hints, str) and hints.strip():
        prompt_parts.append(f'## Hints:\n{hints.strip()}\n')

    if budget_lines:
        prompt_parts.append(
            '## Task Budget (Session terminates when any budget is exhausted)\n'
            + '\n'.join(budget_lines)
            + '\n'
        )
    if harness_lines:
        prompt_parts.append(
            '## Execution Environment Rules\n' + '\n'.join(harness_lines) + '\n'
        )

    instruction_lines = [
        '## Instructions:',
        '0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, system site-packages). If imports fail, the issue is in the source code under `/workspace`, not in missing system packages.',
        '1. Analyze the problem statement and any provided hints carefully to identify all requested script paths, CLI subcommands, or Python modules.',
        '2. Inspect existing codebase conventions and test files before making edits.',
    ]
    if not enable_sandbox_testing:
        instruction_lines.append(
            '3. The `pytest` and `unittest` test frameworks are intentionally disabled in this sandbox to avoid test sweeps and pre-existing environment failure loops. Do NOT import `unittest` or run test discovery. To verify your implementation, run targeted inline assertions via `python3 -c "..."` without unittest.'
        )
    else:
        instruction_lines.append(
            '3. Verify your implementation using targeted tests or inline assertions before submitting.'
        )
    instruction_lines.extend([
        '4. Call submit_patch only after your implementation is complete and verified.',
        '5. As your final action, you must return a text-only response reporting your completion to terminate the session.',
    ])
    prompt_parts.append('\n'.join(instruction_lines))

    # Check graph tool availability
    graph_available = False
    if getattr(config, 'graph_dir', None) and getattr(config, 'embeddings_dir', None):
        repo_str = getattr(task, 'repo', '') or ''
        repo_short = repo_str.split('/')[-1] if repo_str else ''
        repo_slug = repo_str.replace('/', '_') if repo_str else ''
        base_commit = getattr(task, 'base_commit', None)
        g_candidates = []
        e_candidates = []
        if base_commit:
            g_candidates.extend([
                Path(config.graph_dir) / f'{repo_short}_{base_commit}.json',
                Path(config.graph_dir) / f'{repo_slug}_{base_commit}.json',
            ])
            e_candidates.extend([
                Path(config.embeddings_dir) / f'{repo_short}_{base_commit}.npz',
                Path(config.embeddings_dir) / f'{repo_slug}_{base_commit}.npz',
            ])
        g_candidates.extend([
            Path(config.graph_dir) / f'{repo_slug}.json',
            Path(config.graph_dir) / f'{repo_short}.json',
        ])
        e_candidates.extend([
            Path(config.embeddings_dir) / f'{repo_slug}.npz',
            Path(config.embeddings_dir) / f'{repo_short}.npz',
        ])
        has_g = any(p.exists() and p.stat().st_size > 100 for p in g_candidates)
        has_e = any(p.exists() and p.stat().st_size > 100 for p in e_candidates)
        if has_g and has_e:
            graph_available = True
    if graph_available:
        prompt_parts.append(
            '## Code Intelligence Tools\n'
            'This repository has pre-built code graph and embedding data. Use these tools for fast, targeted navigation:\n'
            '- `search_similar_code(query)`: Find semantically similar functions/classes by keyword.\n'
            '- `get_code_neighbors(node)`: Find callers, callees, and definitions related to a symbol.\n'
            '- `get_code_subgraph(nodes)`: Get the induced subgraph for a set of symbols.\n'
        )

    if workspace_tree:
        prompt_parts.append(
            '## Workspace Layout\n'
            'The repository is located at `/workspace`. Here is the directory tree (up to 3 levels):\n'
            '```\n' + workspace_tree + '\n```\n'
        )

    return '\n'.join(prompt_parts)


async def run_agent_sandbox(
    docker: BaseSandboxManager,
    config: EvalConfig,
    task: Task,
    snapshot_path: Path,
    *,
    base_snapshot_path: Path | None = None,
    patch_path: Path | None = None,
    task_index: int = 1,
    total_tasks: int = 1,
    slot_id: int | None = None,
    dashboard: EvaluationDashboard | None = None,
    context: SwegemmaContext | None = None,
) -> tuple[str, str | None, SessionTrace]:
    """Runs the agent in the sandbox container to generate a patch."""
    sandbox_id = await sandbox_start(docker)
    agent_patch = ''
    agent_error: str | None = None
    log_file = None
    file_display_mgr: EventDisplayManager | None = None

    trace = SessionTrace()
    trace.start()

    try:
        # Set up per-task log file in <results_dir>/logs/<instance_id>.log
        logs_dir = config.results_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_instance_id = task.instance_id.replace('/', '_')
        log_path = logs_dir / f'{safe_instance_id}.log'
        log_file = open(log_path, 'w', buffering=1, encoding='utf-8')  # noqa: SIM115
        file_console = Console(file=log_file, force_terminal=True, width=120)

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
                docker_manager=docker,
                container_id=sandbox_id,
                problem_statement=task.problem_statement,
                hints_text=getattr(task, 'hints_text', '') or '',
                task=task,
                repo=task.repo,
                graph_dir=config.graph_dir,
                embeddings_dir=config.embeddings_dir,
                budget=config.budget,
                harness=config.harness,
            )
            if (
                dashboard is not None
                and slot_id is not None
                and hasattr(dashboard, 'attach_context')
            ):
                dashboard.attach_context(slot_id, context)
        else:
            context.docker = docker
            context.container_id = sandbox_id
            if (
                dashboard is not None
                and slot_id is not None
                and hasattr(dashboard, 'attach_context')
                and (slot is None or slot.context is not context)
            ):
                dashboard.attach_context(slot_id, context)

        context.set_phase('Starting')
        context.set_activity('Setting up container...')

        # Setup wheels in container
        await asyncio.to_thread(setup_container_wheels, docker, sandbox_id, config)

        context.set_activity('Extracting snapshot...')

        # Copy snapshot into sandbox container and extract (supports base + patch)
        await asyncio.to_thread(
            extract_snapshot,
            docker,
            sandbox_id,
            snapshot_path,
            patch_path=patch_path,
            base_snapshot_path=base_snapshot_path,
        )

        # Configure git exclude to ignore bytecode and test cache artifacts
        await asyncio.to_thread(setup_git_exclude, docker, sandbox_id)

        # Install editable package using baked-in wheels
        await asyncio.to_thread(install_editable_package, docker, sandbox_id)

        # Install test dependencies
        await asyncio.to_thread(
            install_test_dependencies, docker, sandbox_id, task.repo, config=config
        )

        # Setup hermetic test configuration in workspace
        await asyncio.to_thread(
            setup_workspace_test_config, docker, sandbox_id, repo=task.repo
        )

        # Create clean baseline commit for git diff calculation
        await asyncio.to_thread(setup_baseline_commit, docker, sandbox_id, 'baseline')

        # Test runners are re-enabled in sandbox container for experiment
        # Set enable_sandbox_testing to False to mask pytest and unittest
        cfg_testing = getattr(config, 'enable_sandbox_testing', None)
        enable_sandbox_testing = (
            bool(cfg_testing) if isinstance(cfg_testing, bool) else True
        )
        if not enable_sandbox_testing:
            await asyncio.to_thread(disable_test_runners_in_sandbox, docker, sandbox_id)

        # Capture workspace layout for prompt injection
        tree_cmd = (
            'cd /workspace && find . -maxdepth 3 -not -path "./.git/*" '
            '-not -name "*.pyc" -not -name "__pycache__" | sort | head -150'
        )
        tree_res = await sandbox_exec(docker, sandbox_id, tree_cmd)
        workspace_tree = tree_res.stdout.strip() if tree_res.exit_code == 0 else ''

        tools = context.create_tools()

        # Compile agent submission
        script_timeout = (
            config.harness.command_timeout_seconds
            if config.harness.command_timeout_seconds is not None
            else 300
        )
        code_executor = AdkSandboxCodeExecutor(
            sandbox=context.sandbox,
            timeout_seconds=script_timeout,
            start_time=context.task_start_time,
            max_time_minutes=config.budget.time_minutes,
            budget_check_fn=context.check_budget,
        )
        agent: BaseAgent = compile_submission(
            submission_dir=config.submission_dir,
            tool_registry=tools,
            model_registry=config.models,
            code_executor=code_executor,
            script_timeout=script_timeout,
            limits=config.limits,
            generation_constraints=config.generation_constraints,
            adapter_manifest=config.adapter_manifest,
            adapter_resolver_fn=lambda base, info: resolve_swegemma_adapter(
                base, info, config.models
            ),
        )

        display_mode = getattr(config, 'display_mode', 'auto')
        if dashboard is not None or display_mode == 'quiet':
            display_mgr = EventDisplayManager(
                ctx=context,
                instance_id=task.instance_id,
                repo=task.repo,
                task_index=task_index,
                total_tasks=total_tasks,
                console=file_console,
                enable_live=False,
            )
        else:
            display_mgr = EventDisplayManager(
                ctx=context,
                instance_id=task.instance_id,
                repo=task.repo,
                task_index=task_index,
                total_tasks=total_tasks,
                enable_live=True,
            )
            file_display_mgr = EventDisplayManager(
                ctx=context,
                instance_id=task.instance_id,
                repo=task.repo,
                task_index=task_index,
                total_tasks=total_tasks,
                console=file_console,
                enable_live=False,
            )

        def _on_event(event: Any) -> None:
            trace.record_event(event)
            if file_display_mgr is not None:
                file_display_mgr.on_event(event)
            context.handle_adk_event(event, agent_name=agent.name)

        display_plugin = EventDisplayPlugin(
            trace_callback=_on_event,
            display_manager=display_mgr,
        )
        retry_plugin = ModelRetryPlugin()

        # Setup ADK Runner and App
        session_service = InMemorySessionService()
        session_state = {
            'problem_description': task.problem_statement,
        }
        hints = getattr(task, 'hints_text', '') or ''
        if isinstance(hints, str) and hints.strip():
            session_state['hints'] = hints.strip()
        session = await session_service.create_session(
            app_name='swegemma_eval',
            user_id='eval_user',
            state=session_state,
        )
        app_kwargs: dict[str, Any] = {
            'name': 'swegemma_eval',
            'root_agent': agent,
            'plugins': [display_plugin, retry_plugin],
        }
        if config.context_cache_config is not None:
            app_kwargs['context_cache_config'] = config.context_cache_config
        if config.events_compaction_config is not None:
            app_kwargs['events_compaction_config'] = config.events_compaction_config

        app = App(**app_kwargs)
        runner = Runner(app=app, session_service=session_service)

        initial_prompt = build_agent_prompt(
            task=task,
            config=config,
            workspace_tree=workspace_tree,
            enable_sandbox_testing=enable_sandbox_testing,
        )

        user_message = genai_types.Content(
            role='user',
            parts=[genai_types.Part(text=initial_prompt)],
        )

        # Execute agent within time and turn budgets
        max_llm_calls = (
            config.budget.turns if config.budget.turns is not None else 500
        )

        timeout_sec = (
            config.budget.time_minutes * 60
            if config.budget.time_minutes is not None
            else None
        )
        disp_context = display_mgr if display_mgr else contextlib.nullcontext()
        timeout_context = (
            asyncio.timeout(timeout_sec)
            if timeout_sec is not None
            else contextlib.nullcontext()
        )

        # Record prompts into trace for TraceViewer and offline analysis
        system_instruction = (
            getattr(agent, 'instruction', None)
            or getattr(agent, 'system_instruction', None)
            or ''
        )
        if callable(system_instruction):
            try:
                system_instruction = str(system_instruction())
            except Exception:
                system_instruction = ''

        if isinstance(system_instruction, str) and system_instruction.strip():
            trace.record_custom(
                'system_instruction',
                system_instruction.strip(),
                author='harness',
            )
        trace.record_custom(
            'task_prompt',
            initial_prompt,
            author='harness',
        )

        if file_display_mgr is not None:
            file_display_mgr.start()

        try:
            with disp_context:
                if display_mgr:
                    if (
                        isinstance(system_instruction, str)
                        and system_instruction.strip()
                    ):
                        display_mgr.log_system_instruction(
                            system_instruction.strip(), author='system'
                        )
                    display_mgr.log_task_prompt(initial_prompt, author='user')
                if file_display_mgr is not None:
                    if (
                        isinstance(system_instruction, str)
                        and system_instruction.strip()
                    ):
                        file_display_mgr.log_system_instruction(
                            system_instruction.strip(), author='system'
                        )
                    file_display_mgr.log_task_prompt(initial_prompt, author='user')

                context.set_phase('Agent Loop')
                context.set_activity('Starting agent loop...')
                context.start_agent_session()
                code_executor.start_time = context.agent_start_time

                async with timeout_context:
                    max_nudges = 3
                    consecutive_nudges = 0
                    current_message = user_message

                    while True:
                        remaining_turns = max_llm_calls - context.llm_calls_used
                        if remaining_turns <= 0:
                            agent_error = (
                                f'Agent exceeded turns budget ({config.budget.turns} turns)'
                                if config.budget.turns is not None
                                else 'Agent exceeded maximum allowed LLM turns'
                            )
                            logger.warning(
                                'Task %s: LLM turn budget exhausted.', task.instance_id
                            )
                            break

                        run_config = RunConfig(max_llm_calls=remaining_turns)
                        last_event_finish_reason = None
                        last_assistant_text = ''
                        turn_has_tool_call = False
                        prev_calls = context.llm_calls_used
                        async for _event in runner.run_async(
                            user_id='eval_user',
                            session_id=session.id,
                            new_message=current_message,
                            run_config=run_config,
                        ):
                            # Process event in case mock runner bypassed ADK plugins
                            context.handle_adk_event(_event, agent_name=agent.name)

                            if (
                                hasattr(_event, 'finish_reason')
                                and _event.finish_reason is not None
                            ):
                                last_event_finish_reason = _event.finish_reason

                            content = getattr(_event, 'content', None)
                            parts = getattr(content, 'parts', []) if content else []
                            event_author = getattr(_event, 'author', None)

                            if event_author not in {'user', 'harness', 'system', 'tool'}:
                                for p in parts:
                                    if getattr(p, 'text', None) and not getattr(
                                        p, 'thought', False
                                    ):
                                        last_assistant_text += str(p.text)

                            has_function_call = any(
                                getattr(p, 'function_call', None) for p in parts
                            )
                            if has_function_call and (
                                context.budget.tool_calls is None
                                or context.tool_calls_used < context.budget.tool_calls
                            ):
                                turn_has_tool_call = True

                            if (
                                hasattr(_event, 'is_final_response')
                                and _event.is_final_response()
                            ):
                                has_text = any(
                                    getattr(p, 'text', None)
                                    and not getattr(p, 'thought', False)
                                    for p in parts
                                )
                                has_function_response = any(
                                    getattr(p, 'function_response', None) for p in parts
                                )

                                if (
                                    event_author not in {'user', 'harness', 'system', 'tool'}
                                    and has_text
                                    and not has_function_response
                                    and not has_function_call
                                    and context.patch_submitted
                                ):
                                    logger.debug(
                                        'Agent submitted patch and returned final completion message; terminating.'
                                    )
                                    break

                            if (
                                timeout_sec is not None
                                and context.elapsed_seconds > timeout_sec
                            ):
                                agent_error = f'Agent exceeded session timeout ({config.budget.time_minutes} min)'
                                logger.warning(
                                    'Task %s agent timed out.', task.instance_id
                                )
                                break

                        # Count at least 1 call if mock runner yielded no events
                        if context.llm_calls_used == prev_calls:
                            context.record_llm_call()

                        try:
                            s = await session_service.get_session(
                                app_name='swegemma_eval',
                                user_id='eval_user',
                                session_id=session.id,
                            )
                            if s:
                                sess_calls = 0
                                for e in s.events:
                                    if getattr(e, 'partial', False):
                                        continue
                                    if getattr(e, 'author', None) in {
                                        None,
                                        'user',
                                        'harness',
                                        'system',
                                        'tool',
                                    }:
                                        continue
                                    e_content = getattr(e, 'content', None)
                                    if getattr(e_content, 'role', None) in (
                                        'user',
                                        'tool',
                                    ):
                                        continue
                                    e_parts = getattr(e_content, 'parts', []) if e_content else []
                                    if e_parts and all(
                                        getattr(p, 'function_response', None)
                                        for p in e_parts
                                    ):
                                        continue
                                    sess_calls += 1
                                context.llm_calls_used = max(
                                    context.llm_calls_used, sess_calls
                                )
                        except Exception as sess_err:
                            logger.debug(
                                'Failed to inspect session events: %s', sess_err
                            )

                        if context.patch_submitted:
                            context.set_activity('Patch submitted')
                            break

                        if (
                            timeout_sec is not None
                            and context.elapsed_seconds > timeout_sec
                        ):
                            if not agent_error:
                                agent_error = f'Agent exceeded session timeout ({config.budget.time_minutes} min)'
                            break

                        if (
                            context.budget.tool_calls is not None
                            and context.tool_calls_used >= context.budget.tool_calls
                        ):
                            if not agent_error:
                                agent_error = (
                                    f'Agent exceeded tool call budget ({context.budget.tool_calls} calls)'
                                )
                            logger.warning(
                                'Task %s: Tool call budget (%d) exhausted; terminating agent loop.',
                                task.instance_id,
                                context.budget.tool_calls,
                            )
                            break

                        if context.llm_calls_used >= max_llm_calls:
                            if not agent_error:
                                agent_error = (
                                    f'Agent exceeded turns budget ({config.budget.turns} turns)'
                                    if config.budget.turns is not None
                                    else 'Agent exceeded maximum allowed LLM turns'
                                )
                            logger.warning(
                                'Task %s: LLM turn budget exhausted.', task.instance_id
                            )
                            break

                        if agent_error:
                            break

                        if turn_has_tool_call:
                            consecutive_nudges = 0

                        if consecutive_nudges >= max_nudges:
                            logger.warning(
                                'Task %s: Reached maximum continuation nudges (%d) without patch submission.',
                                task.instance_id,
                                max_nudges,
                            )
                            break

                        consecutive_nudges += 1
                        is_length_truncation = False
                        if last_event_finish_reason is not None:
                            fr_str = str(last_event_finish_reason).upper()
                            if 'MAX_TOKENS' in fr_str or 'LENGTH' in fr_str:
                                is_length_truncation = True

                        has_truncated_tool_call = '<|tool_call>' in last_assistant_text

                        if has_truncated_tool_call:
                            nudge_prompt = (
                                'Your previous response reached the token limit before the tool call finished closing '
                                '(<|tool_call|> was cut off). Do NOT repeat your prior reasoning in thought—emit your '
                                'next tool call immediately, and if calling edit_file or write_file, split the change '
                                'into smaller incremental edits.'
                            )
                        elif is_length_truncation:
                            nudge_prompt = (
                                'Your previous response reached the token limit while thinking before a tool call was '
                                'completed. Do NOT repeat your analysis in thought—keep reasoning under a few sentences '
                                'and emit your next tool call immediately, or call submit_patch when you have completed '
                                'and verified your changes.'
                            )
                        else:
                            nudge_prompt = (
                                'Please continue your work using the available tools, or call submit_patch when '
                                'you have completed and verified your changes.'
                            )

                        logger.info(
                            'Task %s: Agent ended turn without calling submit_patch (nudge %d/%d). Prompting continuation.',
                            task.instance_id,
                            consecutive_nudges,
                            max_nudges,
                        )
                        trace.record_custom(
                            'continuation_nudge', nudge_prompt, author='harness'
                        )
                        context.set_activity(
                            f'Nudge ({consecutive_nudges}/{max_nudges})'
                        )
                        if display_mgr:
                            display_mgr.log_task_prompt(nudge_prompt, author='user')
                        if file_display_mgr is not None:
                            file_display_mgr.log_task_prompt(
                                nudge_prompt, author='user'
                            )

                        current_message = genai_types.Content(
                            role='user',
                            parts=[genai_types.Part(text=nudge_prompt)],
                        )

        except (TimeoutError, LlmCallsLimitExceededError) as e:
            if isinstance(e, TimeoutError):
                agent_error = (
                    f'Agent exceeded session timeout ({config.budget.time_minutes} min)'
                )
                logger.warning('Task %s agent timed out.', task.instance_id)
            else:
                agent_error = (
                    f'Agent exceeded turns budget ({config.budget.turns} turns)'
                    if config.budget.turns is not None
                    else 'Agent exceeded maximum allowed LLM turns'
                )
                logger.warning('Task %s agent exceeded turns budget.', task.instance_id)

        if context is not None and hasattr(context, 'stop_agent_session'):
            context.stop_agent_session()

        agent_patch = (context.submitted_patch if context is not None else '') or ''

        if not (context is not None and context.patch_submitted):
            # Fallback: check if the agent made unsubmitted modifications on disk
            try:
                await sandbox_exec(docker, sandbox_id, 'cd /workspace && git add -N .')
                diff_res = await sandbox_exec(
                    docker,
                    sandbox_id,
                    'cd /workspace && (git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD)',
                )
                unsubmitted_diff = diff_res.stdout or ''
                if unsubmitted_diff.strip():
                    logger.info(
                        'Task %s: Captured unsubmitted working tree modifications as fallback patch (%d bytes)',
                        task.instance_id,
                        len(unsubmitted_diff),
                    )
                    agent_patch = unsubmitted_diff
                else:
                    if not agent_error:
                        agent_error = (
                            'Agent completed execution without calling submit_patch.'
                        )
                    logger.warning(
                        'Task %s completed without explicit submit_patch call and no working tree modifications.',
                        task.instance_id,
                    )
            except Exception as fallback_err:
                logger.warning(
                    'Failed to check fallback working tree diff for task %s: %s',
                    task.instance_id,
                    fallback_err,
                )
                if not agent_error:
                    agent_error = (
                        'Agent completed execution without calling submit_patch.'
                    )

    except ContainerSetupError:
        raise
    except Exception as e:
        logger.exception('Error during sandbox execution for task %s', task.instance_id)
        agent_error = f'Sandbox execution error: {e}'
    finally:
        if context is not None and hasattr(context, 'stop_agent_session'):
            context.stop_agent_session()
        if file_display_mgr is not None:
            file_display_mgr.stop()
        if log_file is not None:
            log_file.flush()
            log_file.close()
        await sandbox_stop(docker, sandbox_id)

    return agent_patch, agent_error, trace
