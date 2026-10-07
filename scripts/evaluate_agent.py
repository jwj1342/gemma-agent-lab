"""Run real official harness phases separately inside the isolated HPC container."""
import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time
import traceback


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['infer','verify'])
    p.add_argument('--task-id',required=True)
    p.add_argument('--tasks',type=Path,default=Path('/data/tasks.jsonl'))
    p.add_argument('--data-dir',type=Path,default=Path('/data'))
    p.add_argument('--agent-dir',type=Path,default=Path('/agent'))
    p.add_argument('--results',type=Path,default=Path('/work/run/results'))
    p.add_argument('--model',type=Path,default=Path('/model'))
    p.add_argument('--tensor-parallel-size',type=int,choices=[1,4],default=1)
    p.add_argument('--gpu-memory-utilization',type=float,default=0.80)
    p.add_argument('--prediction',type=Path,default=Path('/predictions/prediction.json'))
    p.add_argument('--patch-source',choices=['agent','gold','empty'],default='agent')
    return p.parse_args()


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

    litellm.drop_params=True
    selected=[t for t in load_tasks(args.tasks) if t.instance_id==args.task_id]
    if len(selected)!=1:raise ValueError('Task not uniquely found')
    task=selected[0]
    if args.mode=='infer' and (task.patch or task.test_patch or task.FAIL_TO_PASS or task.PASS_TO_PASS):
        raise ValueError('Inference input contains validation labels')
    args.results.mkdir(parents=True,exist_ok=True)
    budget=yaml.safe_load((args.agent_dir/'eval_config.yaml').read_text())['evaluation']
    adapters=discover_adapters(str(args.agent_dir),adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    server=None
    evaluator=None
    phase='initialization'
    try:
        if args.mode=='infer':
            phase='model_startup'
            import torch
            if not torch.cuda.is_available():raise RuntimeError('A Slurm GPU allocation is required')
            if torch.cuda.device_count()!=args.tensor_parallel_size:
                raise RuntimeError('Visible GPU count does not match the requested tensor-parallel size')
            (args.results/'hardware.json').write_text(json.dumps({
                'torch':torch.__version__,'cuda':torch.version.cuda,
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'tensor_parallel_size':args.tensor_parallel_size,
                'devices':[{'index':i,'name':torch.cuda.get_device_name(i),
                            'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory}
                           for i in range(torch.cuda.device_count())]},indent=2)+'\n')
            server=VllmServer(VllmConfig(
                model=str(args.model),host='127.0.0.1',port=int(os.environ.get('HPC_VLLM_PORT', '8000')),
                tensor_parallel_size=args.tensor_parallel_size,max_model_len=32768,dtype='bfloat16',
                gpu_memory_utilization=args.gpu_memory_utilization,tool_call_parser='gemma4',reasoning_parser='gemma4',
                enable_auto_tool_choice=True,enable_lora=bool(adapters.adapters),startup_timeout=1200,
            ),adapter_manifest=adapters)
            print('vLLM log:', server.log_path, flush=True)
            server.start()
            models=server.create_model_registry(aliases=['gemma-4-31b-it-qat-w4a16-ct'],
                                                model_prefix='openai/',api_key='EMPTY')
        else:
            # Registry construction does not contact an endpoint; verification makes no LLM calls.
            models=setup_gemma_model_registry(api_base='http://127.0.0.1:8000/v1',api_key='EMPTY')
        config=EvalConfig(tasks_path=args.tasks,snapshots_dir=args.data_dir/'snapshots',
                          results_dir=args.results,submission_dir=args.agent_dir,models=models,
                          sandbox='subprocess',wheels_dir=args.data_dir/'wheels',
                          graph_dir=str(args.data_dir/'graphs'),embeddings_dir=str(args.data_dir/'embeddings'),
                          adapter_manifest=adapters,display_mode='quiet',verbose=False,**budget,
                          context_cache_config=ContextCacheConfig(min_tokens=2048,ttl_seconds=1800,cache_intervals=10),
                          events_compaction_config=EventsCompactionConfig(compaction_interval=15,overlap_size=2,
                                                                         token_threshold=14336,event_retention_size=5))
        evaluator=Evaluator(config)
        snapshot=args.data_dir/'snapshots'/(task.instance_id+'.tgz')
        start=time.perf_counter()
        if args.mode=='infer':
            phase='agent_inference'
            ctx=SwegemmaContext(docker_manager=evaluator.docker,task=task,repo=task.repo,
                               graph_dir=config.graph_dir,embeddings_dir=config.embeddings_dir,
                               budget=config.budget,harness=config.harness)
            patch,error,trace=await run_agent_sandbox(evaluator.docker,config,task,snapshot,context=ctx)
            trace_path=save_trace_artifact(trace,args.results,task.instance_id)
            result={'task_id':task.instance_id,'agent_patch':patch,'error':error,
                    'tool_calls':ctx.tool_calls_used,'llm_calls':ctx.llm_calls_used,
                    'trace_path':str(trace_path),'duration_seconds':time.perf_counter()-start}
            (args.results.parent/'prediction.json').write_text(json.dumps(result,indent=2)+'\n')
            if error or not patch:
                raise RuntimeError('Inference did not produce a clean nonempty patch; inspect prediction.json')
        else:
            phase='patch_verification'
            prediction=json.loads(args.prediction.read_text()) if args.patch_source=='agent' else None
            if prediction and prediction['task_id']!=task.instance_id:raise ValueError('Prediction task mismatch')
            patch=(prediction['agent_patch'] if prediction else task.patch if args.patch_source=='gold' else '')
            result=await verify_task(evaluator.docker,config,task,snapshot,agent_patch=patch,
                                     start_time=start,fast_path=False)
            result=result.model_dump(mode='json',exclude={'trace'})
            result['patch_source']=args.patch_source
            (args.results/'tests.log').write_text(result.get('test_output',''))
        result['recorded_at']=datetime.now(timezone.utc).isoformat()
        result['versions']={n:version(n) for n in ['swegemma','adk-submission','google-adk','torch','vllm']}
        (args.results/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in {'agent_patch','test_output'}},indent=2))
    except Exception as error:
        captured=getattr(error,'output',None)
        if captured:
            (args.results/'vllm-error-output.log').write_text(str(captured))
        failure={'recorded_at':datetime.now(timezone.utc).isoformat(),'task_id':args.task_id,
                 'mode':args.mode,'phase':phase,'error_type':type(error).__name__,
                 'message':str(error),'traceback':traceback.format_exc(),
                 'vllm_log':server.log_path if server is not None else None}
        (args.results/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
    finally:
        try:
            if evaluator is not None:evaluator.sandbox.close()
        finally:
            if server is not None:
                # The official server deletes its temporary log during stop().
                # Save it first, but always terminate the GPU process on exit.
                try:
                    if server.log_path and Path(server.log_path).is_file():
                        shutil.copyfile(server.log_path,args.results/'vllm.log')
                finally:
                    server.stop()


if __name__=='__main__':
    asyncio.run(run(parse_args()))
