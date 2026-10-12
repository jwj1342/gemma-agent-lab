"""Reload a trained LoRA adapter in the official competition runtime and measure its effect.

The acceptance requirement is that the saved adapter is not merely a valid file but is *honoured* by
the stack the competition serves with. This worker therefore mounts the adapter into the same
wheelhouse vLLM configuration already measured for this base and compares greedy top-logprobs across
three generations:

1. the base model (first run),
2. the base model again (reproducibility control: a nonzero delta here invalidates the comparison),
3. the base model with the trained adapter mounted.

Verdicts are attributable: identical logprobs with a clean control means the adapter was dropped;
a measurable delta means the adapter reached the engine. It imports ``vllm`` only -- importing
``transformers`` or ``peft`` after the wheelhouse install raises the documented torchao conflict on
this image, and the serving check needs neither.

    python serving_l4_adapter.py --base <mounted model> --adapter /kaggle/working/l4-feasibility/adapter \
        --work /kaggle/working/l4-feasibility
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

PROMPT = ('def render_report(rows):\n    """Render rows into a fixed-width report."""\n')


def top_map(output) -> dict:
    """position -> {token_id: logprob} for the first generated token list of one request."""
    mapping = {}
    for index, entry in enumerate(output.outputs[0].logprobs or []):
        mapping[index] = {str(token): round(value.logprob, 6) for token, value in entry.items()}
    return mapping


def divergence(left: dict, right: dict) -> dict:
    shared = sorted(set(left) & set(right))
    deltas = [max(abs(left[p][t] - right[p][t]) for t in set(left[p]) & set(right[p]))
              for p in shared if set(left[p]) & set(right[p])]
    return {'shared_positions': len(shared), 'max_abs_logprob_delta': max(deltas) if deltas else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=40)
    parser.add_argument('--lora-rank', type=int, default=128)
    args = parser.parse_args()

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def progress(phase: str, **fields) -> None:
        record = {'phase': phase, 'elapsed_s': round(time.monotonic() - started, 3), **fields}
        with (work / 'serving-progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
        print(f'[l4-serving] {phase} {json.dumps(fields, default=str)}', flush=True)

    result = {'base': args.base, 'adapter': str(args.adapter), 'status': 'not_run'}
    weights = args.adapter / 'adapter_model.safetensors'
    if not weights.is_file():
        result['status'] = 'no_adapter'
        result['error'] = f'{weights} is absent, so there is nothing to reload'
        (work / 'serving-result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        progress('no_adapter', path=str(weights))
        return 1
    result['adapter_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    progress('import_done', seconds=round(time.monotonic() - started, 3))
    llm = LLM(model=args.base, tensor_parallel_size=4, max_model_len=4096,
              gpu_memory_utilization=0.90, enable_lora=True, max_loras=1,
              max_lora_rank=args.lora_rank, enforce_eager=True, quantization='compressed-tensors')
    progress('engine_ready', seconds=round(time.monotonic() - started, 3))
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, logprobs=3)
    request = LoRARequest('trained_l4', 1, str(args.adapter))

    try:
        base_first = llm.generate([PROMPT], sampling, use_tqdm=False)[0]
        progress('base_generated', seconds=round(time.monotonic() - started, 3))
        base_again = llm.generate([PROMPT], sampling, use_tqdm=False)[0]
        progress('base_control_generated', seconds=round(time.monotonic() - started, 3))
        adapted = llm.generate([PROMPT], sampling, lora_request=request, use_tqdm=False)[0]
        progress('adapter_generated', seconds=round(time.monotonic() - started, 3))
    except Exception as error:
        result.update({'status': 'engine_error', 'error': f'{type(error).__name__}: {error}'})
        (work / 'serving-result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        progress('engine_error', error=result['error'])
        return 1

    control = divergence(top_map(base_first), top_map(base_again))
    effect = divergence(top_map(base_first), top_map(adapted))
    result.update({
        'control_base_versus_base': control,
        'effect_base_versus_adapter': effect,
        'tokens_changed': base_first.outputs[0].token_ids != adapted.outputs[0].token_ids,
        'text_changed': base_first.outputs[0].text != adapted.outputs[0].text,
        'control_text': base_first.outputs[0].text[:200],
    })
    if control['max_abs_logprob_delta'] != 0.0:
        result['status'] = 'control_failed'
        result['verdict'] = ('the engine is not reproducible on identical input, so this session cannot '
                            'attribute any change to the adapter')
    elif effect['max_abs_logprob_delta'] > 0.0:
        result['status'] = 'applied'
        result['verdict'] = ('the trained adapter is honoured by the official serving runtime: the '
                             'engine reproduces its own logprobs exactly and the adapter moves them')
    else:
        result['status'] = 'inert'
        result['verdict'] = ('the trained adapter is silently dropped: logprobs are identical with and '
                             'without it while the repetition control is clean')
    result['seconds'] = round(time.monotonic() - started, 3)
    (work / 'serving-result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    progress('finished', status=result['status'])
    return 0 if result['status'] == 'applied' else 1


if __name__ == '__main__':
    raise SystemExit(main())
