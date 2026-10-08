"""Evaluate Localization v1 against complete source trees at task base commits.

The benchmark never exposes reference patches to the ranker.  Gold paths are used
only after ranking, and fusion weights are selected on the deterministic dev split
before the test split is reported.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LOCALIZER_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "localization-v1"
    / "skills"
    / "localization-v1"
    / "scripts"
    / "localize.py"
)
MODEL_PATH = LOCALIZER_PATH.parents[1] / "resources" / "model.json"
PATCH_PATH_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
REPOSITORIES = {
    "fastapi/fastapi": "fastapi.git",
    "Textualize/rich": "rich.git",
    "psf/requests": "requests.git",
    "encode/httpx": "httpx.git",
}


def load_localizer():
    spec = importlib.util.spec_from_file_location("localization_v1_runtime", LOCALIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LOCALIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


LOCALIZER = load_localizer()


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    commit: str
    issue: str
    gold_paths: tuple[str, ...]


def gold_source_paths(patch: str) -> tuple[str, ...]:
    result = []
    for path in set(PATCH_PATH_RE.findall(patch or "")):
        normalized = "/" + path.lower()
        filename = PurePosixPath(path).name.lower()
        if not path.endswith((".py", ".pyi")):
            continue
        if "/tests/" in normalized or "/testing/" in normalized:
            continue
        if filename.startswith("test_") or filename.endswith("_test.py") or filename == "conftest.py":
            continue
        result.append(path)
    return tuple(sorted(result))


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold = gold_source_paths(row.get("patch", ""))
        if gold:
            tasks.append(
                Task(
                    row["instance_id"],
                    row["repo"],
                    row["base_commit"],
                    (row.get("problem_statement", "") + "\n" + row.get("hints_text", "")).strip(),
                    gold,
                )
            )
    return tasks


def stable_hash(text: str) -> int:
    value = 2_166_136_261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16_777_619) & 0xFFFFFFFF
    return value


def split_tasks(tasks: Iterable[Task], seed: int = 52) -> dict[str, list[Task]]:
    by_repo: dict[str, list[Task]] = collections.defaultdict(list)
    for task in tasks:
        by_repo[task.repo].append(task)
    splits = {"train": [], "dev": [], "test": []}
    for repo in sorted(by_repo):
        rows = sorted(by_repo[repo], key=lambda task: task.instance_id)
        random.Random(seed + stable_hash(repo)).shuffle(rows)
        if len(rows) < 5:
            splits["train"].extend(rows)
            continue
        dev_count = max(1, round(len(rows) * 0.15))
        test_count = max(1, round(len(rows) * 0.15))
        train_count = len(rows) - dev_count - test_count
        splits["train"].extend(rows[:train_count])
        splits["dev"].extend(rows[train_count : train_count + dev_count])
        splits["test"].extend(rows[train_count + dev_count :])
    return splits


def git_output(git_dir: Path, *args: str, input_data: bytes | None = None) -> bytes:
    process = subprocess.run(
        ["git", f"--git-dir={git_dir}", *args],
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    return process.stdout


def tree_at(git_dir: Path, commit: str) -> dict[str, str]:
    output = git_output(git_dir, "ls-tree", "-r", "-z", commit)
    tree: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, kind, oid = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8", errors="replace")
        if kind != "blob" or not path.endswith((".py", ".pyi")) or LOCALIZER.TEST_RE.search(path):
            continue
        tree[path] = oid
    return tree


def read_blobs(git_dir: Path, object_ids: Iterable[str]) -> dict[str, str]:
    requested = sorted(set(object_ids))
    raw = git_output(git_dir, "cat-file", "--batch", input_data=("\n".join(requested) + "\n").encode("ascii"))
    cursor = 0
    result: dict[str, str] = {}
    for expected in requested:
        line_end = raw.index(b"\n", cursor)
        header = raw[cursor:line_end].decode("ascii", errors="replace")
        cursor = line_end + 1
        oid, kind, size_text = header.split()
        if oid != expected or kind != "blob":
            raise RuntimeError(f"Unexpected cat-file response: {header}")
        size = int(size_text)
        result[oid] = raw[cursor : cursor + size].decode("utf-8", errors="replace")
        cursor += size + 1
    return result


def load_snapshots(tasks: list[Task], repositories: Path):
    snapshots: dict[tuple[str, str], list] = {}
    inventories: dict[tuple[str, str], set[str]] = {}
    stats = {}
    for repo, bare_name in REPOSITORIES.items():
        git_dir = repositories / bare_name
        commits = sorted({task.commit for task in tasks if task.repo == repo})
        if not commits:
            continue
        trees = {commit: tree_at(git_dir, commit) for commit in commits}
        blobs = read_blobs(git_dir, (oid for tree in trees.values() for oid in tree.values()))
        document_cache = {}
        for commit, tree in trees.items():
            documents = []
            for path, oid in sorted(tree.items()):
                key = (path, oid)
                if key not in document_cache:
                    document_cache[key] = LOCALIZER.make_document(path, blobs[oid])
                documents.append(document_cache[key])
            snapshots[(repo, commit)] = documents
            inventories[(repo, commit)] = set(tree)
        stats[repo] = {
            "commits": len(commits),
            "unique_blobs": len(blobs),
            "min_candidates": min(map(len, trees.values())),
            "max_candidates": max(map(len, trees.values())),
        }
    return snapshots, inventories, stats


def rank_for(
    task: Task,
    documents: list,
    model,
    mode: str,
    weights: dict[str, float] | None,
    score_cache: dict[str, dict[str, list[float]]],
) -> list[str]:
    components = score_cache.get(task.instance_id)
    if components is None:
        query = LOCALIZER.tokenize(task.issue)
        components = {
            "bm25": LOCALIZER.bm25_scores(documents, query),
            "exact": [LOCALIZER.exact_score(document, task.issue) for document in documents],
            "legacy": [model.score(task.issue, document.path) for document in documents],
        }
        score_cache[task.instance_id] = components
    if mode in {"bm25", "legacy"}:
        final = components[mode]
    elif mode == "hybrid":
        final = LOCALIZER.reciprocal_rank_fusion(
            components, weights or LOCALIZER.DEFAULT_FUSION_WEIGHTS
        )
    else:
        raise ValueError(f"Unknown ranking mode: {mode}")
    order = sorted(
        range(len(documents)), key=lambda index: (-final[index], documents[index].path)
    )
    return [documents[index].path for index in order]


def metrics(rows: list[dict]) -> dict[str, float | int]:
    ranks = [row["rank"] for row in rows if row["rank"] is not None]
    count = len(ranks)
    return {
        "tasks": len(rows),
        "scorable_tasks": count,
        "new_file_only_tasks": len(rows) - count,
        "hit_at_1": sum(rank <= 1 for rank in ranks) / count if count else 0.0,
        "hit_at_3": sum(rank <= 3 for rank in ranks) / count if count else 0.0,
        "hit_at_5": sum(rank <= 5 for rank in ranks) / count if count else 0.0,
        "hit_at_10": sum(rank <= 10 for rank in ranks) / count if count else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks) / count if count else 0.0,
        "mean_rank": sum(ranks) / count if count else 0.0,
    }


def evaluate(
    tasks: list[Task],
    snapshots,
    inventories,
    model,
    mode: str,
    score_cache: dict[str, dict[str, list[float]]],
    weights=None,
):
    rows = []
    for index, task in enumerate(tasks, 1):
        inventory = inventories[(task.repo, task.commit)]
        existing_gold = sorted(set(task.gold_paths) & inventory)
        rank = None
        top = []
        if existing_gold:
            ranked = rank_for(
                task,
                snapshots[(task.repo, task.commit)],
                model,
                mode,
                weights,
                score_cache,
            )
            gold = set(existing_gold)
            rank = next(position for position, path in enumerate(ranked, 1) if path in gold)
            top = ranked[:10]
        rows.append(
            {
                "instance_id": task.instance_id,
                "repo": task.repo,
                "gold_paths": list(task.gold_paths),
                "existing_gold_paths": existing_gold,
                "rank": rank,
                "top_10": top,
            }
        )
        if index % 10 == 0:
            print(f"{mode}: {index}/{len(tasks)}", flush=True)
    return metrics(rows), rows


def objective(value: dict[str, float | int]) -> tuple[float, float, float]:
    return (
        0.5 * float(value["hit_at_1"]) + 0.5 * float(value["hit_at_5"]),
        float(value["mrr"]),
        float(value["hit_at_10"]),
    )


def compare(candidate: list[dict], control: list[dict]) -> dict[str, int]:
    control_by_id = {row["instance_id"]: row for row in control}
    wins = ties = losses = 0
    for row in candidate:
        candidate_rank = row["rank"]
        control_rank = control_by_id[row["instance_id"]]["rank"]
        if candidate_rank is None or control_rank is None:
            continue
        if candidate_rank < control_rank:
            wins += 1
        elif candidate_rank > control_rank:
            losses += 1
        else:
            ties += 1
    return {"rank_wins": wins, "rank_ties": ties, "rank_losses": losses}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "public" / "tasks.jsonl")
    parser.add_argument("--repositories", type=Path, default=ROOT / "data" / "raw" / "source-repos")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "localization-v1" / "full-tree-benchmark.json")
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    started = time.perf_counter()
    tasks = load_tasks(args.tasks)
    splits = split_tasks(tasks, args.seed)
    snapshots, inventories, repository_stats = load_snapshots(tasks, args.repositories)
    model = LOCALIZER.LegacyModel.load(MODEL_PATH)
    if model is None:
        raise RuntimeError(f"Missing model: {MODEL_PATH}")

    score_cache: dict[str, dict[str, list[float]]] = {}
    dev_results = {}
    dev_rows = {}
    for mode in ("bm25", "legacy"):
        dev_results[mode], dev_rows[mode] = evaluate(
            splits["dev"], snapshots, inventories, model, mode, score_cache
        )

    candidates = []
    for exact_weight in (0.4, 0.8, 1.2):
        for legacy_weight in (0.05, 0.1, 0.15, 0.25, 0.5):
            weights = {"bm25": 1.0, "exact": exact_weight, "legacy": legacy_weight}
            value, _rows = evaluate(
                splits["dev"], snapshots, inventories, model, "hybrid", score_cache, weights
            )
            candidates.append((objective(value), weights, value))
    _, selected_weights, selected_dev = max(candidates, key=lambda item: (item[0], -item[1]["legacy"], -item[1]["exact"]))
    dev_results["hybrid"] = selected_dev
    _, dev_rows["hybrid"] = evaluate(
        splits["dev"], snapshots, inventories, model, "hybrid", score_cache, selected_weights
    )
    portable_safe_weights = {"bm25": 1.0, "exact": 0.4, "legacy": 0.0}
    dev_results["portable_safe"], dev_rows["portable_safe"] = evaluate(
        splits["dev"],
        snapshots,
        inventories,
        model,
        "hybrid",
        score_cache,
        portable_safe_weights,
    )

    test_results = {}
    test_rows = {}
    for mode in ("bm25", "legacy"):
        test_results[mode], test_rows[mode] = evaluate(
            splits["test"], snapshots, inventories, model, mode, score_cache
        )
    test_results["hybrid"], test_rows["hybrid"] = evaluate(
        splits["test"], snapshots, inventories, model, "hybrid", score_cache, selected_weights
    )
    test_results["portable_safe"], test_rows["portable_safe"] = evaluate(
        splits["test"],
        snapshots,
        inventories,
        model,
        "hybrid",
        score_cache,
        portable_safe_weights,
    )

    payload = {
        "benchmark": "full source tree at each public task base commit",
        "label_policy": "reference patch is used only after ranking; new-file-only tasks are reported but excluded from retrieval denominators",
        "split_policy": "original Localization v1 deterministic repository-stratified 70/15/15 split",
        "seed": args.seed,
        "task_counts": {name: len(rows) for name, rows in splits.items()},
        "repository_stats": repository_stats,
        "model": {
            "path": MODEL_PATH.relative_to(ROOT).as_posix(),
            "portable_features_only": True,
            "repository_specific_priors_enabled": False,
        },
        "selection": {
            "split": "dev",
            "objective": "0.5 * Hit@1 + 0.5 * Hit@5; MRR then Hit@10 tie-breakers",
            "grid": {
                "bm25": [1.0],
                "exact": [0.4, 0.8, 1.2],
                "legacy": [0.05, 0.1, 0.15, 0.25, 0.5],
            },
            "selected_weights": selected_weights,
            "portable_safe_weights": portable_safe_weights,
        },
        "dev": dev_results,
        "test": test_results,
        "paired_rank_comparison": {
            "dev_hybrid_vs_bm25": compare(dev_rows["hybrid"], dev_rows["bm25"]),
            "test_hybrid_vs_bm25": compare(test_rows["hybrid"], test_rows["bm25"]),
            "dev_portable_safe_vs_bm25": compare(
                dev_rows["portable_safe"], dev_rows["bm25"]
            ),
            "test_portable_safe_vs_bm25": compare(
                test_rows["portable_safe"], test_rows["bm25"]
            ),
        },
        "task_results": {"dev": dev_rows, "test": test_rows},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selection": payload["selection"], "dev": dev_results, "test": test_results}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
