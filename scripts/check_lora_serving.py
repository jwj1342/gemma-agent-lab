"""Load a trained adapter with official quantized vLLM and compare fixed held-out actions."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from src.sft_data import read_jsonl


def completion_token_ids(tokenizer, example):
    """Request token IDs explicitly; Transformers 5 defaults to a BatchEncoding."""
    prompt = tokenizer.apply_chat_template(example['messages'][:-1], tools=example['tools'],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
    if not isinstance(prompt, list) or not prompt or any(type(t) is not int or t < 0 for t in prompt):
        raise ValueError('Completion probe requires one nonempty list of integer token IDs')
    return prompt


def probability_difference(reference, candidate):
    shared = set(reference) & set(candidate)
    return {'max_logprob_difference': max((abs(reference[t] - candidate[t]) for t in shared), default=0),
            'top_tokens_changed': set(reference) != set(candidate)}


def adapter_effect(zero, repeated_zero, trained):
    jitter = probability_difference(zero, repeated_zero)
    change = probability_difference(zero, trained)
    return {'zero_repeat': jitter, 'trained_vs_zero': change,
            'measurable_effect': (not jitter['top_tokens_changed']) and
            (change['top_tokens_changed'] or
             change['max_logprob_difference'] > max(1e-5, 10 * jitter['max_logprob_difference']))}


def create_zero_adapter(source, destination):
    """Keep the adapter path active with a mathematically zero LoRA update."""
    from safetensors.torch import load_file, save_file
    weights = load_file(source / 'adapter_model.safetensors')
    if not weights or any('.lora_A.' not in k and '.lora_B.' not in k for k in weights):
        raise ValueError('The zero-update control requires ordinary LoRA A/B tensors only')
    matrices = [value for name, value in weights.items() if '.lora_B.' in name]
    if not matrices:
        raise ValueError('No LoRA B matrices found')
    for tensor in matrices:
        tensor.zero_()
    destination.mkdir()
    shutil.copyfile(source / 'adapter_config.json', destination / 'adapter_config.json')
    save_file(weights, destination / 'adapter_model.safetensors')
    return len(matrices)


def select_examples(path, kind):
    examples = list(read_jsonl(path / "validation.jsonl"))
    wanted = (["run_command", "read_file", "edit_file", "submit_patch"] if kind == "swe" else
              ["get_code_neighbors", "get_code_subgraph", "search_similar_code", "run_command"])
    selected, groups = [], set()
    for name in wanted:
        matches = [e for e in examples if e["metadata"]["group"] not in groups
                   and e["messages"][-1].get("tool_calls")
                   and e["messages"][-1]["tool_calls"][0]["function"]["name"] == name
                   and len(json.dumps(e["messages"])) < 24000]
        if not matches:
            raise ValueError("No short, distinct validation group for " + name)
        example = matches[0]
        selected.append(example)
        groups.add(example["metadata"]["group"])
    return selected


def validate_arguments(arguments, schema):
    if not isinstance(arguments, dict) or any(k not in arguments for k in schema["required"]):
        return False
    for key, value in arguments.items():
        if key not in schema["properties"]:
            return False
        expected = schema["properties"][key]["type"]
        if expected == "string" and not isinstance(value, str):
            return False
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return False
        if expected == "array" and (not isinstance(value, list) or not all(isinstance(x, str) for x in value)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--graph-data", type=Path, required=True)
    parser.add_argument("--terminal-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from adk_submission import VllmConfig, VllmServer
    from openai import OpenAI
    from transformers import AutoTokenizer
    if args.output.exists():
        raise ValueError("Output directory already exists")
    args.output.mkdir(parents=True)
    zero_adapter = args.output / 'zero-adapter'
    zeroed_matrices = create_zero_adapter(args.adapter, zero_adapter)
    sources = [("swe", select_examples(args.data, "swe")), ("graph", select_examples(args.graph_data, "graph"))]
    if args.terminal_data:
        sources.append(("terminal", list(read_jsonl(args.terminal_data / "validation.jsonl"))[:4]))
    server = VllmServer(VllmConfig(model=str(args.base), host="127.0.0.1", port=int(os.environ.get("LORA_SERVING_PORT", "8001")), tensor_parallel_size=1,
                         max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.80,
                         tool_call_parser="gemma4", reasoning_parser="gemma4", enable_auto_tool_choice=True,
                         enable_lora=True, startup_timeout=1200,
                         extra_args=['--no-enable-prefix-caching']),
                        lora_modules={"first_lora": str(args.adapter), "zero_lora": str(zero_adapter)})
    rows = []
    try:
        server.start()
        client = OpenAI(base_url=server.base_url, api_key="EMPTY", timeout=120)
        models = [m.id for m in client.models.list().data]
        if not {'first_lora', 'zero_lora'}.issubset(models):
            raise RuntimeError("A trained/control adapter is absent from served model IDs")
        # A registered adapter may still have no effect. Compare first-token
        # probabilities on identical token IDs. A zero-update adapter distinguishes
        # weight effects from switching to the adapter-enabled numerical path.
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        effects = []
        for kind, examples in sources:
            example = examples[0]
            prompt = completion_token_ids(tokenizer, example)
            distributions = []
            for name in (str(args.base), str(args.base), 'zero_lora', 'zero_lora', 'first_lora'):
                response = client.completions.create(model=name, prompt=prompt, max_tokens=1,
                    temperature=0, seed=42, logprobs=5)
                if len(response.choices) != 1 or response.usage.prompt_tokens != len(prompt):
                    raise RuntimeError('Server did not evaluate exactly the supplied token-ID prompt')
                probabilities = response.choices[0].logprobs.top_logprobs[0]
                if not probabilities:
                    raise RuntimeError('No token log probabilities returned for adapter-effect check')
                distributions.append(probabilities)
            effects.append({'kind': kind, 'group': example['metadata']['group'],
                'prompt_tokens': len(prompt),
                'prompt_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                'base_repeat': probability_difference(distributions[0], distributions[1]),
                'zero_vs_base': probability_difference(distributions[0], distributions[2]),
                **adapter_effect(distributions[2], distributions[3], distributions[4]),
                'distribution_order': ['base', 'base_repeat', 'zero_lora', 'zero_repeat', 'trained_lora'],
                'distributions': distributions})
        for kind, examples in sources:
            for index, example in enumerate(examples):
                expected = (example["messages"][-1]["tool_calls"][0]["function"] if kind != "terminal" else None)
                history = copy.deepcopy(example["messages"][:-1])
                for message in history:
                    for call in message.get("tool_calls", []):
                        if isinstance(call["function"]["arguments"], dict):
                            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
                for name in (str(args.base), "first_lora"):
                    response = client.chat.completions.create(model=name, messages=history,
                         tools=example["tools"], tool_choice="auto", temperature=0, max_tokens=512, seed=42,
                         extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    message = response.choices[0].message
                    calls = message.tool_calls or []
                    actual = calls[0].function if calls else None
                    schema_valid = False
                    arguments = None
                    if actual:
                        try:
                            arguments = json.loads(actual.arguments)
                            definition = next(t["function"] for t in example["tools"] if t["function"]["name"] == actual.name)
                            schema_valid = validate_arguments(arguments, definition["parameters"])
                        except (ValueError, StopIteration, TypeError):
                            pass
                    rows.append({"kind": kind, "index": index, "model": name, "group": example["metadata"]["group"],
                                 "schema_valid": schema_valid, "expected_tool_match": bool(expected and actual and actual.name == expected["name"]),
                                 "exact_arguments_match": bool(expected and arguments == expected["arguments"]),
                                 "clean_final_answer": bool(not calls and message.content and response.choices[0].finish_reason == "stop"),
                                 "output_limit_reached": response.choices[0].finish_reason == "length",
                                 "response": response.model_dump(mode="json")})
        report = {"served_model_ids": models, "adapter_loaded": True,
                  "zero_control_B_matrices": zeroed_matrices,
                  "effect_protocol": "Prefix caching disabled so repeated probes perform identical full prefills; "
                                     "trained adapter compared to zero-update adapter with a repeated-zero control.",
                  "adapter_effect_checked": any(e['measurable_effect'] for e in effects),
                  "adapter_effect_probes": effects, "requests": rows,
                  "summary": {kind: {name: {"schema_valid": sum(r["schema_valid"] for r in rows if r["kind"] == kind and r["model"] == name),
                                  "expected_tool_match": sum(r["expected_tool_match"] for r in rows if r["kind"] == kind and r["model"] == name),
                                  "clean_final_answer": sum(r["clean_final_answer"] for r in rows if r["kind"] == kind and r["model"] == name),
                                  "output_limit_reached": sum(r["output_limit_reached"] for r in rows if r["kind"] == kind and r["model"] == name),
                                  "actions": sum(1 for r in rows if r["kind"] == kind and r["model"] == name)}
                            for name in (str(args.base), "first_lora")} for kind, _ in sources},
                  "scope": "Fixed held-out tool/terminal contexts, thinking disabled for both models. Not an issue-resolution benchmark."}
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "requests"}, indent=2))
        if not report['adapter_effect_checked']:
            raise RuntimeError('Adapter loading succeeded but no measurable effect was found')
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"error_type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
        raise
    finally:
        try:
            if Path(server.log_path).is_file():
                shutil.copyfile(server.log_path, args.output / "vllm.log")
        finally:
            server.stop()


if __name__ == "__main__":
    main()
