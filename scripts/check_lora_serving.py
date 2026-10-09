"""Load a trained adapter with official quantized vLLM and compare fixed held-out actions."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from src.sft_data import read_jsonl


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
    sources = [("swe", select_examples(args.data, "swe")), ("graph", select_examples(args.graph_data, "graph"))]
    if args.terminal_data:
        sources.append(("terminal", list(read_jsonl(args.terminal_data / "validation.jsonl"))[:4]))
    server = VllmServer(VllmConfig(model=str(args.base), host="127.0.0.1", port=int(os.environ.get("LORA_SERVING_PORT", "8001")), tensor_parallel_size=1,
                         max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.80,
                         tool_call_parser="gemma4", reasoning_parser="gemma4", enable_auto_tool_choice=True,
                         enable_lora=True, startup_timeout=1200), lora_modules={"first_lora": str(args.adapter)})
    rows = []
    try:
        server.start()
        client = OpenAI(base_url=server.base_url, api_key="EMPTY", timeout=120)
        models = [m.id for m in client.models.list().data]
        if "first_lora" not in models:
            raise RuntimeError("Trained adapter is absent from served model IDs")
        # A registered adapter may still have no effect. Compare first-token
        # probabilities on identical prompts, including a repeated-base control.
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        effects = []
        for kind, examples in sources:
            example = examples[0]
            prompt = tokenizer.apply_chat_template(example['messages'][:-1], tools=example['tools'],
                tokenize=True, add_generation_prompt=True, enable_thinking=True)
            distributions = []
            for name in (str(args.base), str(args.base), 'first_lora'):
                response = client.completions.create(model=name, prompt=prompt, max_tokens=1,
                    temperature=0, seed=42, logprobs=5)
                probabilities = response.choices[0].logprobs.top_logprobs[0]
                if not probabilities:
                    raise RuntimeError('No token log probabilities returned for adapter-effect check')
                distributions.append(probabilities)
            shared = set.intersection(*(set(d) for d in distributions))
            base_jitter = max((abs(distributions[0][t] - distributions[1][t]) for t in shared), default=0)
            difference = max((abs(distributions[0][t] - distributions[2][t]) for t in shared), default=0)
            changed_tokens = set(distributions[0]) != set(distributions[2])
            effects.append({'kind': kind, 'group': example['metadata']['group'],
                'base_repeat_max_logprob_difference': base_jitter,
                'adapter_max_logprob_difference': difference, 'top_tokens_changed': changed_tokens,
                'measurable_effect': changed_tokens or difference > max(1e-5, base_jitter * 10),
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
