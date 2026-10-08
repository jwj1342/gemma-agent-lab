"""Build graph component demonstrations using executed official tools, without gold patches.

Run in the competition evaluation environment on an allocated node.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from src.sft_data import SYSTEM, action_examples, split_for_group


def call(name, arguments, number):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"graph-{number}", "type": "function", "function": {"name": name, "arguments": arguments}}]}


def observation(message, response):
    c = message["tool_calls"][0]
    return {"role": "tool", "tool_call_id": c["id"], "name": c["function"]["name"], "content": response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-graphs", type=int, default=16)
    parser.add_argument("--eval-benchmark", type=Path, default=Path("configs/evaluation/public-v1.json"))
    args = parser.parse_args()
    from swegemma.tools.graph import get_code_neighbors, get_code_subgraph, search_similar_code
    from swegemma.graph.graph_utils import clear_caches
    if args.output.exists():
        raise ValueError("Output already exists")
    args.output.mkdir(parents=True)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    # Reserve the historically tested snapshot entirely; select other snapshots without using issue patches.
    reserved = {t["base_commit"] for t in tasks if t["instance_id"] == "rich_3454"}
    benchmark_hash = None
    if not args.eval_benchmark.exists():
        raise ValueError("Frozen evaluation benchmark is missing; provide --eval-benchmark explicitly")
    if args.eval_benchmark.exists():
        benchmark = json.loads(args.eval_benchmark.read_text())
        protected = set(benchmark["cohorts"]["dev"] + benchmark["cohorts"]["holdout"])
        reserved.update(t["base_commit"] for t in tasks if t["instance_id"] in protected)
        benchmark_hash = hashlib.sha256(args.eval_benchmark.read_bytes()).hexdigest()
    unique = {(t["repo"], t["base_commit"]): t for t in tasks if t["base_commit"] not in reserved}
    selected = sorted(unique.values(), key=lambda t: hashlib.sha256(t["base_commit"].encode()).hexdigest())[:args.max_graphs]
    records = {"train": [], "validation": []}
    executed = []
    for task in selected:
        repo, commit = task["repo"], task["base_commit"]
        stem = repo.split("/")[-1] + "_" + commit
        path = args.graphs / (stem + ".json")
        graph = json.loads(path.read_text())
        degree = Counter()
        for edge in graph.get("edges", graph.get("links", [])):
            degree[edge["source"]] += 1
            degree[edge["target"]] += 1
        nodes = sorted((n["id"] for n in graph["nodes"] if 1 < degree[n["id"]] < 15 and not n["id"].startswith(("tests.", "test_", "benchmarks."))),
                       key=lambda n: hashlib.sha256(n.encode()).hexdigest())[:4]
        group = repo + "." + commit[:8]
        split = split_for_group(group)
        ctx = SimpleNamespace(repo=repo, task={"repo": repo, "base_commit": commit}, graph_dir=args.graphs,
                              embeddings_dir=args.embeddings, harness=SimpleNamespace(max_stdout_chars=5000),
                              check_budget=lambda **kwargs: None, budget=None)
        for node in nodes:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                         f"Repository: {repo}. Investigate how symbol {node} connects to nearby code. "
                         "Query its calls neighbors, then inspect the induced graph on it and up to two returned symbols."}]
            first = call("get_code_neighbors", {"node": node, "edge_type": "calls", "max_neighbors": 5}, 1)
            result = get_code_neighbors(ctx, **first["tool_calls"][0]["function"]["arguments"])
            parsed = json.loads(result)
            if parsed.get("status") != "ok" or not parsed.get("neighbors"):
                continue
            messages.extend([first, observation(first, result)])
            second = call("get_code_subgraph", {"nodes": [node] + parsed["neighbors"][:2]}, 2)
            subresult = get_code_subgraph(ctx, **second["tool_calls"][0]["function"]["arguments"])
            if json.loads(subresult).get("status") != "ok":
                raise ValueError("Official subgraph tool failed")
            messages.extend([second, observation(second, subresult)])
            similar = call("search_similar_code", {"query": node, "k": 3}, 5)
            similar_result = search_similar_code(ctx, **similar["tool_calls"][0]["function"]["arguments"])
            if json.loads(similar_result).get("status") != "ok":
                raise ValueError("Official cached-vector retrieval failed")
            messages.extend([similar, observation(similar, similar_result)])
            metadata = {"source": "competition public graph", "group": group, "split": split,
                        "graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "symbol": node,
                        "verification": "official swegemma 0.2.7 graph tools executed; component exercise, no repair claim"}
            records[split].extend(action_examples(messages, metadata))
            # The same tool's actual empty response supplies a grounded fallback demonstration.
            missing = node + ".__missing_symbol__"
            empty_call = call("get_code_neighbors", {"node": missing, "edge_type": "calls", "max_neighbors": 5}, 3)
            empty_result = get_code_neighbors(ctx, **empty_call["tool_calls"][0]["function"]["arguments"])
            if json.loads(empty_result).get("neighbors") != []:
                raise ValueError("Expected a genuinely missing graph symbol")
            fallback = call("run_command", {"command": "grep -R -n --include='*.py' " + repr(node.split(".")[-1]) + " . | head -20"}, 4)
            fallback_messages = [messages[0], {"role": "user", "content": f"Find source evidence for {node}; the graph lookup used {missing}."},
                                 empty_call, observation(empty_call, empty_result), fallback]
            records[split].append({"messages": fallback_messages, "tools": next(action_examples(messages, metadata))["tools"],
                                   "metadata": {**metadata, "component": "empty-result fallback"}})
            executed.append({**metadata, "neighbor_result": parsed, "subgraph_result": json.loads(subresult),
                             "similar_result": json.loads(similar_result),
                             "empty_result": json.loads(empty_result)})
        clear_caches()
    for split, rows in records.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {"harness_version": "0.2.7", "excluded_snapshot_commits": sorted(reserved), "graphs": len(selected),
                "eval_benchmark_sha256": benchmark_hash,
                "actions": {k: len(v) for k, v in records.items()}, "executed_components": executed,
                "scope": "Graph query arguments, induced subgraph selection, and empty-result fallback. No end-to-end repairs generated."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "executed_components"}, indent=2))


if __name__ == "__main__":
    main()
