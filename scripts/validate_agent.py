"""Compile an agent using the installed official harness, without starting a model."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')


def validate(agent_dir):
    import litellm
    from adk_submission import compile_submission, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.models.discovery import validate_single_declared_model
    from swegemma.models.registry import setup_gemma_model_registry, resolve_swegemma_adapter

    litellm.drop_params = True
    target = 'gemma-4-31b-it-qat-w4a16-ct'
    declared = validate_single_declared_model(agent_dir)
    if declared != target:
        raise ValueError(f'Competition model must be {target}; got {declared}')
    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    limits, generation = build_submission_limits()
    models = setup_gemma_model_registry(api_base='http://127.0.0.1:8000/v1', api_key='EMPTY',
                                       served_model=target, adapter_manifest=adapters)
    tools = SwegemmaContext().create_tools()
    root = compile_submission(agent_dir, tool_registry=tools, model_registry=models,
                              limits=limits, generation_constraints=generation,
                              adapter_manifest=adapters, adapter_resolver_fn=resolve_swegemma_adapter)
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'status':'compiled',
            'root_name':root.name, 'root_class':type(root).__name__, 'model':declared,
            'adapters':sorted(adapters.adapters), 'registered_tools':sorted(tools),
            'versions':{n:version(n) for n in ['swegemma','adk-submission','adk-eval-core','google-adk','google-genai']},
            'scope':'Official compiler, model rule and competition limits; model inference and sandbox tests not executed'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    result=validate(args.agent_dir.resolve())
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text)
