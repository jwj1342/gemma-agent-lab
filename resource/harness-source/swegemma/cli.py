"""CLI interface for SWE-gemma."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from swegemma.evaluate import EvalConfig, Evaluator
from swegemma.models import setup_gemma_model_registry
from swegemma.warnings import _suppress_warnings


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point for SWE-gemma."""
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        prog='swegemma',
        description='SWE-bench evaluation system for Gemma 4 agents.',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subcommand: eval
    eval_parser = subparsers.add_parser(
        'eval', help='Evaluate an agent on SWE-bench tasks'
    )
    eval_parser.add_argument(
        '--tasks', required=True, type=Path, help='Path to tasks.jsonl'
    )
    eval_parser.add_argument(
        '--snapshots-dir',
        required=True,
        type=Path,
        help='Directory with repository snapshots',
    )
    eval_parser.add_argument(
        '--results-dir',
        required=True,
        type=Path,
        help='Output directory for evaluation results',
    )
    eval_parser.add_argument(
        '--submission-dir',
        required=True,
        type=Path,
        help='Directory containing agent package or notebook',
    )
    eval_parser.add_argument(
        '--image', default='swebench-sandbox:latest', help='Sandbox docker image'
    )
    eval_parser.add_argument(
        '--sandbox',
        choices=['docker', 'subprocess'],
        default='docker',
        help='Sandbox manager type',
    )
    eval_parser.add_argument(
        '--max-tool-calls',
        type=int,
        default=None,
        help='Maximum allowed tool calls per task',
    )
    eval_parser.add_argument(
        '--max-turns',
        type=int,
        default=None,
        help='Maximum allowed LLM turns per task',
    )
    eval_parser.add_argument(
        '--timeout-seconds',
        type=int,
        default=None,
        help='Single command execution timeout in seconds',
    )
    eval_parser.add_argument(
        '--max-time-minutes',
        type=float,
        default=60.0,
        help='Maximum session time per task in minutes',
    )
    eval_parser.add_argument(
        '--models-yaml',
        type=Path,
        default=None,
        help='Path to models.yaml registry config',
    )
    eval_parser.add_argument(
        '--task-id',
        type=str,
        action='append',
        dest='task_ids',
        help='Specific task instance_id(s) to evaluate (repeatable)',
    )
    eval_parser.add_argument(
        '--task-ids',
        nargs='+',
        dest='task_ids',
        help='Specific task instance_id(s) to evaluate',
    )
    eval_parser.add_argument(
        '--skip-agent-patch',
        action='store_true',
        help='Skip applying agent patch to evaluate baseline behavior',
    )
    eval_parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='Maximum number of parallel task evaluations (default: 1)',
    )
    eval_parser.add_argument(
        '--shard-index',
        type=int,
        default=None,
        help='0-based shard index for distributed evaluation',
    )
    eval_parser.add_argument(
        '--num-shards',
        type=int,
        default=None,
        help='Total number of shards for distributed evaluation',
    )
    eval_parser.add_argument(
        '--verbose', action='store_true', help='Enable verbose output'
    )
    eval_parser.add_argument(
        '--display',
        choices=['auto', 'dashboard', 'single', 'quiet'],
        default='auto',
        help='Display mode for evaluation: auto (dashboard if concurrency > 1 else single HUD), dashboard, single, or quiet (default: auto)',
    )

    args = parser.parse_args(argv)

    if args.command == 'eval':
        _suppress_warnings()
        models = setup_gemma_model_registry(models_yaml_path=args.models_yaml)
        config = EvalConfig(
            tasks_path=args.tasks,
            snapshots_dir=args.snapshots_dir,
            results_dir=args.results_dir,
            submission_dir=args.submission_dir,
            models=models,
            image=args.image,
            sandbox=args.sandbox,
            timeout_seconds=args.timeout_seconds,
            max_tool_calls=args.max_tool_calls,
            max_turns=args.max_turns,
            max_time_minutes=args.max_time_minutes,
            task_ids=args.task_ids,
            skip_agent_patch=args.skip_agent_patch,
            concurrency=args.concurrency,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            verbose=args.verbose,
            display_mode=args.display,
        )
        evaluator = Evaluator(config)
        result = asyncio.run(evaluator.run())
        if args.display != 'quiet':
            print(
                f'Evaluation complete: {result.resolved}/{result.total} resolved ({result.resolution_rate:.2%})'
            )
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
