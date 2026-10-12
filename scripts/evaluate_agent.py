"""Run official harness phases separately; batch inference shares one model server."""
import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['infer', 'verify'])
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--task-id')
    selection.add_argument('--task-ids', nargs='+', help='Batch mode: one server, fresh sandbox per task')
    p.add_argument('--tasks', type=Path, default=Path('/data/tasks.jsonl'))
    p.add_argument('--data-dir', type=Path, default=Path('/data'))
    p.add_argument('--agent-dir', type=Path, default=Path('/agent'))
    p.add_argument('--results', type=Path, default=Path('/work/run/results'))
    p.add_argument('--model', type=Path, default=Path('/model'))
    p.add_argument('--tensor-parallel-size', type=int, choices=[1, 4], default=1)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.80)
    p.add_argument('--prediction', type=Path, default=Path('/predictions/prediction.json'))
    p.add_argument('--patch-source', choices=['agent', 'gold', 'empty'], default='agent')
    return p.parse_args()


def save(path, value, receipt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.partial')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)
    if receipt:
        path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '\n')


async def run(args):
    import litellm
    import yaml
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import EvalConfig, ALLOWED_ADAPTER_EXTENSIONS
    from swegemma.context import SwegemmaContext
    from swegemma.evaluate import Evaluator
    from swegemma.harness.agent_runner import run_agent_sandbox
    from swegemma.harness.verification import verify_task, save_trace_artifact
    from swegemma.models import load_tasks
    from swegemma.models.registry import setup_gemma_model_registry
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig

    litellm.drop_params = True
    batch = bool(args.task_ids)
    ids = args.task_ids or [args.task_id]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs')
    lookup = {t.instance_id: t for t in load_tasks(args.tasks)}
    tasks = [lookup[i] for i in ids]
    if args.mode == 'infer' and any(t.patch or t.test_patch or t.FAIL_TO_PASS or t.PASS_TO_PASS for t in tasks):
        raise ValueError('Inference input contains validation labels')
    args.results.mkdir(parents=True, exist_ok=True)
    budget = yaml.safe_load((args.agent_dir / 'eval_config.yaml').read_text())['evaluation']
    if batch and args.mode == 'verify':
        # Submission budgets govern inference. Independent grading uses the
        # official default command timeout, including longer repository tests.
        budget = {**budget, 'timeout_seconds': 300}
    adapters = discover_adapters(str(args.agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    versions = {n: version(n) for n in ['swegemma', 'adk-submission', 'google-adk', 'torch', 'vllm']}
    server = None
    session = {'mode': args.mode, 'patch_source': args.patch_source, 'task_ids': ids,
               'started_at': datetime.now(timezone.utc).isoformat(), 'model_startups': 0,
               'versions': versions, 'budget': budget, 'batch': batch}
    session_name = str(time.time_ns())
    total_start = time.perf_counter()
    phase = 'initialization'
    try:
        if args.mode == 'infer':
            phase = 'model_startup'
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != args.tensor_parallel_size:
                raise RuntimeError('A matching Slurm GPU allocation is required')
            hardware = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'tensor_parallel_size': args.tensor_parallel_size,
                        'devices': [{'index': i, 'name': torch.cuda.get_device_name(i),
                                     'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                                    for i in range(torch.cuda.device_count())]}
            save(args.results / 'hardware.json', hardware)
            session['hardware'] = hardware
            server = VllmServer(VllmConfig(
                model=str(args.model), host='127.0.0.1', port=int(os.environ.get('HPC_VLLM_PORT', '8000')),
                tensor_parallel_size=args.tensor_parallel_size, max_model_len=32768, dtype='bfloat16',
                gpu_memory_utilization=args.gpu_memory_utilization, tool_call_parser='gemma4', reasoning_parser='gemma4',
                enable_auto_tool_choice=True, enable_lora=bool(adapters.adapters), startup_timeout=1200,
                extra_args=['--seed', '42'] if batch else [],
            ), adapter_manifest=adapters)
            print('vLLM log:', server.log_path, flush=True)
            start = time.perf_counter()
            server.start()
            session['model_startups'] = 1
            session['model_startup_seconds'] = time.perf_counter() - start
            models = server.create_model_registry(aliases=['gemma-4-31b-it-qat-w4a16-ct'],
                                                   model_prefix='openai/', api_key='EMPTY')
        else:
            models = setup_gemma_model_registry(api_base='http://127.0.0.1:8000/v1', api_key='EMPTY')
        for task in tasks:
            results = ((args.results.parent / task.instance_id / 'results') if args.mode == 'infer'
                       else args.results / task.instance_id) if batch else args.results
            results.mkdir(parents=True, exist_ok=True)
            prediction_path = results.parent / 'prediction.json'
            if batch and (results / 'result.json').exists():
                raise ValueError('Refusing to overwrite a completed task result')
            config = EvalConfig(tasks_path=args.tasks, snapshots_dir=args.data_dir / 'snapshots',
                                results_dir=results, submission_dir=args.agent_dir, models=models,
                                sandbox='subprocess', wheels_dir=args.data_dir / 'wheels',
                                graph_dir=str(args.data_dir / 'graphs'), embeddings_dir=str(args.data_dir / 'embeddings'),
                                adapter_manifest=adapters, display_mode='quiet', verbose=False, **budget,
                                context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                                events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                                               token_threshold=14336, event_retention_size=5))
            evaluator = Evaluator(config)
            if batch:
                from local_eval_setup import configure
                configure(evaluator)
            snapshot = args.data_dir / 'snapshots' / (task.instance_id + '.tgz')
            start = time.perf_counter()
            try:
                if args.mode == 'infer':
                    phase = 'agent_inference'
                    ctx = SwegemmaContext(docker_manager=evaluator.docker, task=task, repo=task.repo,
                                          graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,
                                          budget=config.budget, harness=config.harness)
                    patch, error, trace = await run_agent_sandbox(evaluator.docker, config, task, snapshot, context=ctx)
                    trace_path = save_trace_artifact(trace, results, task.instance_id)
                    result = {'task_id': task.instance_id, 'agent_patch': patch, 'error': error,
                              'tool_calls': ctx.tool_calls_used, 'llm_calls': ctx.llm_calls_used,
                              'trace_path': str(trace_path), 'duration_seconds': time.perf_counter() - start,
                              'agent_seconds': ctx.agent_elapsed_seconds}
                    save(prediction_path, result, receipt=batch)
                    if not batch and (error or not patch):
                        raise RuntimeError('Inference did not produce a clean nonempty patch; inspect prediction.json')
                else:
                    phase = 'patch_verification'
                    prediction_file = (args.prediction.parent / task.instance_id / 'prediction.json') if batch else args.prediction
                    prediction = json.loads(prediction_file.read_text()) if args.patch_source == 'agent' else None
                    if prediction and prediction['task_id'] != task.instance_id:
                        raise ValueError('Prediction task mismatch')
                    patch = prediction['agent_patch'] if prediction else task.patch if args.patch_source == 'gold' else ''
                    result = await verify_task(evaluator.docker, config, task, snapshot, agent_patch=patch,
                                               start_time=start, fast_path=False)
                    result = result.model_dump(mode='json', exclude={'trace'})
                    result['patch_source'] = args.patch_source
                    (results / 'tests.log').write_text(result.get('test_output', ''))
                result['recorded_at'] = datetime.now(timezone.utc).isoformat()
                result['versions'] = versions
                save(results / 'result.json', result, receipt=batch)
                print(json.dumps({k: v for k, v in result.items() if k not in {'agent_patch', 'test_output'}}), flush=True)
            except Exception as error:
                failure = {'task_id': task.instance_id, 'mode': args.mode, 'phase': phase,
                           'error_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
                save(results / 'failure.json', failure)
                if not batch:
                    raise
                # Preserve failed attempts and continue. A new candidate/run is needed to retry;
                # resuming must not silently select the best of multiple stochastic attempts.
                if args.mode == 'infer':
                    if not prediction_path.exists():
                        save(prediction_path, {'task_id': task.instance_id, 'agent_patch': '', 'error': str(error),
                                              'llm_calls': 0, 'tool_calls': 0, 'duration_seconds': time.perf_counter() - start}, True)
                else:
                    save(results / 'result.json', {'task_id': task.instance_id, 'resolved': False,
                                                   'test_exit_code': -1, 'error_message': str(error),
                                                   'test_output': '', 'duration_seconds': time.perf_counter() - start}, True)
            finally:
                evaluator.sandbox.close()
        session['completed'] = True
    except Exception as error:
        session.update(completed=False, error=str(error), phase=phase)
        captured = getattr(error, 'output', None)
        if captured:
            (args.results / 'vllm-error-output.log').write_text(str(captured))
        save(args.results / 'failure.json', {'mode': args.mode, 'phase': phase, 'message': str(error),
                                            'traceback': traceback.format_exc()})
        raise
    finally:
        try:
            if server is not None:
                try:
                    if server.log_path and Path(server.log_path).is_file():
                        shutil.copyfile(server.log_path, args.results / (f'vllm-{session_name}.log' if batch else 'vllm.log'))
                finally:
                    server.stop()
        finally:
            session['total_seconds'] = time.perf_counter() - total_start
            save(args.results / 'sessions' / f'{session_name}.json', session)


if __name__ == '__main__':
    asyncio.run(run(parse_args()))
