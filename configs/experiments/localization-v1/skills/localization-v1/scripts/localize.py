#!/usr/bin/env python3
"""Offline issue-to-file localizer derived from the archived Localization v1 model.

The runtime is deliberately standard-library-only.  It enumerates the repository
that is actually mounted in the agent sandbox and reports bounded file and symbol
hints.  Full-tree BM25 is the validated default; exact and archived learned scores
remain available for diagnostic ablations.
"""

from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import json
import math
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
SKIP_DIRS = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".venv", "__pycache__", "build", "dist", "env", "node_modules",
    "site-packages", "venv",
}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "has", "have",
    "if", "in", "into", "is", "it", "its", "may", "not", "of", "on",
    "or", "should", "so", "than", "that", "the", "their", "then", "there",
    "these", "this", "to", "use", "using", "was", "we", "when", "which",
    "will", "with", "would", "you", "issue", "bug", "fix", "error", "python",
}
DEFAULT_FUSION_WEIGHTS = {"bm25": 1.0, "exact": 0.8, "legacy": 0.15}
GATE_FEATURE_NAMES = (
    "log_candidates",
    "log_issue_tokens",
    "bm25_margin",
    "bm25_positive_fraction",
    "legacy_margin",
    "legacy_positive_fraction",
    "exact_top",
    "exact_positive_fraction",
    "top1_agree",
    "top3_jaccard",
    "candidate_top_in_bm25",
    "bm25_top_in_candidate",
)


@dataclasses.dataclass(frozen=True)
class Symbol:
    name: str
    line: int


@dataclasses.dataclass(frozen=True)
class Document:
    path: str
    source: str
    symbols: tuple[Symbol, ...]
    tokens: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Result:
    path: str
    line: int
    symbol: str
    score: float
    bm25: float
    exact: float
    legacy: float


@dataclasses.dataclass(frozen=True)
class AdaptiveGate:
    enabled: bool
    candidate_weights: dict[str, float]
    rule: dict
    decision: str

    @classmethod
    def load(cls, path: Path) -> "AdaptiveGate":
        if not path.is_file():
            raise FileNotFoundError(f"Localization gate not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported gate schema in {path}")
        if payload.get("fallback") != "bm25" or payload.get("candidate") != "v1_hybrid":
            raise ValueError(f"Unsupported gate policies in {path}")
        weights = payload.get("candidate_weights")
        if not isinstance(weights, dict) or set(weights) != {"bm25", "exact", "legacy"}:
            raise ValueError(f"Invalid candidate weights in {path}")
        candidate_weights = {name: float(value) for name, value in weights.items()}
        if any(not math.isfinite(value) or value < 0 for value in candidate_weights.values()):
            raise ValueError(f"Invalid candidate weight value in {path}")
        rule = payload.get("rule")
        if not isinstance(rule, dict):
            raise ValueError(f"Invalid gate rule in {path}")
        validate_gate_rule(rule)
        return cls(
            enabled=bool(payload.get("enabled", False)),
            candidate_weights=candidate_weights,
            rule=rule,
            decision=str(payload.get("decision", "")),
        )

    def use_v1(self, features: dict[str, float]) -> bool:
        return self.enabled and gate_rule_matches(features, self.rule)


def tokenize(text: str, limit: int | None = None) -> tuple[str, ...]:
    """Tokenize prose, snake_case, and camelCase deterministically."""
    tokens: list[str] = []
    for match in WORD_RE.finditer(text):
        raw = match.group(0)
        pieces = [raw, *raw.split("_"), *CAMEL_RE.sub(" ", raw).split()]
        for piece in pieces:
            token = piece.lower().strip("_")
            if len(token) >= 2 and token not in STOP_WORDS and not token.isdigit():
                tokens.append(token)
                if limit is not None and len(tokens) >= limit:
                    return tuple(tokens)
    return tuple(tokens)


def unique_issue_tokens(text: str, limit: int = 64) -> tuple[str, ...]:
    tokens = tokenize(text)
    counts = collections.Counter(tokens)
    first = {token: index for index, token in enumerate(dict.fromkeys(tokens))}
    return tuple(sorted(counts, key=lambda token: (-min(counts[token], 4), first[token], token))[:limit])


def path_tokens(path: str) -> tuple[str, ...]:
    raw = re.split(r"[/_.-]+", path.removesuffix(".py"))
    tokens: list[str] = []
    for part in raw:
        tokens.extend(CAMEL_RE.sub(" ", part).lower().split())
    return tuple(dict.fromkeys(token for token in tokens if token and token not in {"src", "py"}))


def stable_hash(text: str, dimension: int = 262_144) -> int:
    value = 2_166_136_261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16_777_619) & 0xFFFFFFFF
    return value % dimension


def symbols_from_source(source: str) -> tuple[Symbol, ...]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    result: list[Symbol] = []

    def visit(node: ast.AST, prefix: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = ".".join((*prefix, child.name))
                result.append(Symbol(name, child.lineno))
                if isinstance(child, ast.ClassDef):
                    visit(child, (*prefix, child.name))
            elif not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(child, prefix)

    visit(tree)
    return tuple(result)


def make_document(path: str, source: str, max_source_tokens: int = 6_000) -> Document:
    normalized = path.replace("\\", "/")
    symbols = symbols_from_source(source)
    path_terms = tokenize(normalized.replace("/", " "))
    symbol_terms = tokenize(" ".join(symbol.name for symbol in symbols))
    source_terms = tokenize(source, max_source_tokens)
    # Path and symbol names are more precise than arbitrary implementation words.
    terms = (*path_terms, *path_terms, *path_terms, *symbol_terms, *symbol_terms, *source_terms)
    return Document(normalized, source, symbols, tuple(terms))


def read_repository(
    root: Path,
    include_tests: bool = False,
    max_files: int = 5_000,
    max_file_bytes: int = 1_000_000,
) -> list[Document]:
    documents: list[Document] = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS and not name.startswith("."))
        for filename in sorted(files):
            if not filename.endswith((".py", ".pyi")):
                continue
            absolute = Path(directory, filename)
            relative = absolute.relative_to(root).as_posix()
            if not include_tests and TEST_RE.search(relative):
                continue
            try:
                if absolute.is_symlink():
                    continue
                if absolute.stat().st_size > max_file_bytes:
                    continue
                source = absolute.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            documents.append(make_document(relative, source))
            if len(documents) >= max_files:
                return documents
    return documents


def bm25_scores(documents: list[Document], query: Iterable[str], b: float = 0.75) -> list[float]:
    if not documents:
        return []
    query_counts = collections.Counter(query)
    lengths = [len(document.tokens) for document in documents]
    average = max(sum(lengths) / len(lengths), 1.0)
    document_frequency = collections.Counter(
        token for document in documents for token in set(document.tokens) if token in query_counts
    )
    scores: list[float] = []
    for document, length in zip(documents, lengths):
        frequencies = collections.Counter(token for token in document.tokens if token in query_counts)
        score = 0.0
        for token, frequency in frequencies.items():
            inverse_frequency = math.log(
                1.0 + (len(documents) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.2 * (1.0 - b + b * length / average)
            score += query_counts[token] * inverse_frequency * frequency * 2.2 / denominator
        scores.append(score)
    return scores


PORTABLE_FEATURE_NAMES = (
    "bm25", "unique_query", "short_length_normalization", "path", "symbols",
    "exact_symbols", "implementation",
)


def portable_features(documents: list[Document], issue: str) -> list[list[float]]:
    """Build portable evidence features without learned repository or token identities."""
    query = tokenize(issue)
    unique_query = tuple(dict.fromkeys(query))
    paths = [dataclasses.replace(doc, tokens=tokenize(doc.path)) for doc in documents]
    symbols = [dataclasses.replace(
        doc, tokens=tokenize(" ".join(symbol.name for symbol in doc.symbols))
    ) for doc in documents]
    fields = [
        bm25_scores(documents, query),
        bm25_scores(documents, unique_query),
        bm25_scores(documents, unique_query, b=0.15),
        bm25_scores(paths, unique_query, b=0.25),
        bm25_scores(symbols, unique_query, b=0.25),
    ]
    # A symbol shared by many files is weaker evidence than a unique definition.
    names_by_file = [set(s.name.rpartition(".")[2].lower() for s in doc.symbols)
                     for doc in documents]
    frequency = collections.Counter(name for names in names_by_file for name in names)
    issue_words = set(match.group().lower() for match in WORD_RE.finditer(issue))
    fields.append([sum(math.log1p(len(documents) / frequency[name]) for name in names
                       if len(name) >= 3 and name in issue_words) for names in names_by_file])
    # Normalize within each issue; rank evidence is comparable across repository sizes.
    normalized = []
    for values in fields:
        maximum = max(values, default=0.0)
        normalized.append([value / maximum if maximum > 0 else 0.0 for value in values])
    normalized.append([
        float(not any(part in {"docs", "examples", "tutorial", "tutorials", "benchmarks"}
                      for part in PurePosixPath(doc.path).parts)) for doc in documents
    ])
    return [list(row) for row in zip(*normalized)]


def portable_scores(features: list[list[float]], weights: list[float]) -> list[float]:
    if len(weights) != len(PORTABLE_FEATURE_NAMES) or any(
        not math.isfinite(value) or value < 0 for value in weights
    ):
        raise ValueError("Invalid portable ranking weights")
    return [sum(value * weight for value, weight in zip(row, weights)) for row in features]


@dataclasses.dataclass(frozen=True)
class PortablePolicy:
    weights: list[float]
    gate: dict
    enabled: bool = False

    @classmethod
    def load(cls, path: Path) -> "PortablePolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("schema_version") != 1
                or payload.get("model_type") != "portable_localization_v1"
                or payload.get("features") != list(PORTABLE_FEATURE_NAMES)):
            raise ValueError("Unsupported portable localization policy")
        if not isinstance(payload.get("enabled"), bool):
            raise ValueError("Portable policy enabled must be a boolean")
        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, list):
            raise ValueError("Portable weights must be a list")
        weights = [float(value) for value in raw_weights]
        portable_scores([], weights)
        if not any(weights):
            raise ValueError("Portable weights cannot all be zero")
        gate = payload.get("gate")
        if not isinstance(gate, dict) or gate.get("kind") not in {"off", "confidence"}:
            raise ValueError("Unsupported portable confidence gate")
        if gate["kind"] == "confidence":
            for name in ("max_bm25_margin", "min_candidate_margin"):
                value = float(gate.get(name, float("nan")))
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Portable gate margins must be finite and in [0, 1]")
                gate[name] = value
        return cls(weights, gate, payload["enabled"])


def portable_gate_matches(bm25_margin: float, candidate_margin: float, gate: dict) -> bool:
    return (gate["kind"] != "off" and bm25_margin <= gate["max_bm25_margin"]
            and candidate_margin >= gate["min_candidate_margin"])


def rank_documents_portable(
    documents: list[Document], issue: str, policy: PortablePolicy, k: int = 10,
    experimental: bool = False,
) -> tuple[list[Result], bool]:
    if not policy.enabled and not experimental:
        return rank_documents(documents, issue, mode="bm25", k=k), False
    documents = sorted(documents, key=lambda doc: doc.path)
    if not documents:
        return [], False
    features = portable_features(documents, issue)
    baseline = [row[0] for row in features]
    candidate = portable_scores(features, policy.weights)
    use_v1 = portable_gate_matches(
        normalized_margin(baseline, descending_order(baseline)),
        normalized_margin(candidate, descending_order(candidate)), policy.gate,
    )
    components = {"bm25": baseline, "exact": [0.0] * len(documents),
                  "legacy": [0.0] * len(documents)}
    return results_from_scores(documents, issue, components,
                               candidate if use_v1 else baseline, k), use_v1


def exact_score(document: Document, issue: str) -> float:
    issue_lower = issue.lower()
    path_lower = document.path.lower()
    stem = PurePosixPath(path_lower).stem
    score = 0.0
    if path_lower in issue_lower:
        score += 12.0
    if stem and re.search(rf"(?<![a-z0-9_]){re.escape(stem)}(?![a-z0-9_])", issue_lower):
        score += 4.0
    for token in path_tokens(document.path):
        if len(token) >= 3 and re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", issue_lower):
            score += 0.5
    for symbol in document.symbols:
        leaf = symbol.name.rpartition(".")[2].lower()
        if len(leaf) >= 3 and re.search(rf"(?<![a-z0-9_]){re.escape(leaf)}(?![a-z0-9_])", issue_lower):
            score += 3.0
    return score


def validate_gate_rule(rule: dict) -> None:
    kind = rule.get("kind")
    if kind == "off":
        return
    conditions = rule.get("conditions", [rule])
    if kind not in {"stump", "all"} or not isinstance(conditions, list) or not conditions:
        raise ValueError("Invalid localization gate rule")
    if len(conditions) > 2:
        raise ValueError("Localization gate supports at most two conditions")
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("Invalid localization gate condition")
        if condition.get("feature") not in GATE_FEATURE_NAMES:
            raise ValueError(f"Unsupported localization gate feature: {condition.get('feature')}")
        if condition.get("operator") not in {"<=", ">"}:
            raise ValueError(f"Unsupported localization gate operator: {condition.get('operator')}")
        threshold = float(condition.get("threshold"))
        if not math.isfinite(threshold):
            raise ValueError("Localization gate threshold must be finite")


def gate_rule_matches(features: dict[str, float], rule: dict) -> bool:
    if rule.get("kind") == "off":
        return False
    conditions = rule.get("conditions", [rule])
    for condition in conditions:
        value = features[condition["feature"]]
        threshold = float(condition["threshold"])
        matched = value <= threshold if condition["operator"] == "<=" else value > threshold
        if not matched:
            return False
    return True


def descending_order(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def normalized_margin(values: list[float], order: list[int]) -> float:
    if len(order) < 2:
        return 1.0
    first, second = values[order[0]], values[order[1]]
    return (first - second) / max(abs(first), abs(second), 1e-9)


def reciprocal_position(order: list[int], target: int, cap: int = 50) -> float:
    try:
        position = order.index(target) + 1
    except ValueError:
        return 0.0
    return 1.0 / min(position, cap)


def gate_features(
    issue: str,
    components: dict[str, list[float]],
    candidate_weights: dict[str, float],
) -> dict[str, float]:
    """Build label-free features shared by training and adaptive inference."""
    bm25 = components["bm25"]
    legacy = components["legacy"]
    exact = components["exact"]
    if not bm25:
        raise ValueError("Cannot build gate features for an empty repository")
    candidate = reciprocal_rank_fusion(components, candidate_weights)
    bm25_order = descending_order(bm25)
    candidate_order = descending_order(candidate)
    legacy_order = descending_order(legacy)
    exact_order = descending_order(exact)
    count = len(bm25)
    bm25_top = bm25_order[0]
    candidate_top = candidate_order[0]
    top3_bm25 = set(bm25_order[:3])
    top3_candidate = set(candidate_order[:3])
    union = top3_bm25 | top3_candidate
    return {
        "log_candidates": math.log1p(count),
        "log_issue_tokens": math.log1p(len(tokenize(issue))),
        "bm25_margin": normalized_margin(bm25, bm25_order),
        "bm25_positive_fraction": sum(value > 0 for value in bm25) / count,
        "legacy_margin": normalized_margin(legacy, legacy_order),
        "legacy_positive_fraction": sum(value > 0 for value in legacy) / count,
        "exact_top": exact[exact_order[0]],
        "exact_positive_fraction": sum(value > 0 for value in exact) / count,
        "top1_agree": float(bm25_top == candidate_top),
        "top3_jaccard": len(top3_bm25 & top3_candidate) / max(len(union), 1),
        "candidate_top_in_bm25": reciprocal_position(bm25_order, candidate_top),
        "bm25_top_in_candidate": reciprocal_position(candidate_order, bm25_top),
    }


class LegacyModel:
    """Portable inference view of the archived v1 hashed pairwise checkpoint."""

    def __init__(self, dimension: int, weights: dict[int, float]):
        self.dimension = dimension
        self.weights = weights

    @classmethod
    def load(cls, path: Path | None) -> "LegacyModel | None":
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"Localization model not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") != "hashed_pairwise_file_localizer_v1":
            raise ValueError(f"Unsupported model type in {path}")
        if payload.get("feature_set") != "no_issue_path_v3":
            raise ValueError(f"Unsupported feature set in {path}")
        dimension = int(payload["dimension"])
        if dimension <= 0:
            raise ValueError(f"Invalid model dimension in {path}: {dimension}")
        weights: dict[int, float] = {}
        for item in payload.get("weights", []):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"Invalid sparse weight entry in {path}: {item!r}")
            index, weight = int(item[0]), float(item[1])
            if not 0 <= index < dimension or not math.isfinite(weight):
                raise ValueError(f"Invalid sparse weight in {path}: {item!r}")
            weights[index] = weight
        if not weights:
            raise ValueError(f"Localization model has no weights: {path}")
        return cls(dimension, weights)

    def score(self, issue: str, path: str) -> float:
        issue_terms = unique_issue_tokens(issue)
        candidate_terms = path_tokens(path)
        candidate_set = set(candidate_terms)
        stem = PurePosixPath(path).stem.lower()
        issue_lower = issue.lower()
        features: dict[int, float] = {}

        def add(name: str, value: float = 1.0) -> None:
            index = stable_hash(name, self.dimension)
            features[index] = features.get(index, 0.0) + value

        # Repository/path priors from v1 are intentionally omitted.  They memorize the
        # four public repositories and do not transfer to unseen evaluation repositories.
        for issue_term in issue_terms:
            if issue_term in candidate_set:
                add("lexical_overlap")
                add(f"exact_token={issue_term}", 0.5)
            for candidate_term in candidate_terms:
                if issue_term != candidate_term:
                    add(f"token_pair={issue_term}:{candidate_term}", 0.15)
        if stem and re.search(rf"(?<![a-z0-9_]){re.escape(stem)}(?![a-z0-9_])", issue_lower):
            add("stem_mentioned", 2.0)
        if path.lower() in issue_lower:
            add("path_mentioned", 4.0)
        add(f"depth={min(path.count('/'), 5)}", 0.15)
        return sum(self.weights.get(index, 0.0) * value for index, value in features.items())


def reciprocal_rank_fusion(scores: dict[str, list[float]], weights: dict[str, float]) -> list[float]:
    if not scores:
        return []
    size = len(next(iter(scores.values())))
    fused = [0.0] * size
    for name, values in scores.items():
        weight = weights.get(name, 0.0)
        if weight <= 0:
            continue
        # Sparse lexical components use zero for "no evidence", so do not let
        # thousands of zero-score files receive arbitrary tie-breaking credit.
        # Signed learned scores are relative, however: a less-negative score is
        # still useful evidence and must remain in the component ranking.
        if any(value < 0 for value in values):
            active = range(len(values))
        else:
            active = (index for index, value in enumerate(values) if value > 0)
        ranked = sorted(active, key=lambda index: (-values[index], index))
        for rank, index in enumerate(ranked, 1):
            fused[index] += weight / (60.0 + rank)
    return fused


def best_symbol(document: Document, issue: str) -> tuple[int, str]:
    if not document.symbols:
        return 1, "<module>"
    issue_lower = issue.lower()
    query = set(tokenize(issue))
    scored = []
    for symbol in document.symbols:
        leaf = symbol.name.rpartition(".")[2]
        overlap = len(query & set(tokenize(symbol.name)))
        mention = 5 if re.search(rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])", issue, re.I) else 0
        scored.append((mention + overlap, -symbol.line, symbol))
    _, _, chosen = max(scored)
    return chosen.line, chosen.name


def score_documents(
    documents: list[Document],
    issue: str,
    model: LegacyModel | None = None,
) -> dict[str, list[float]]:
    query = tokenize(issue)
    return {
        "bm25": bm25_scores(documents, query),
        "exact": [exact_score(document, issue) for document in documents],
        "legacy": [model.score(issue, document.path) if model else 0.0 for document in documents],
    }


def results_from_scores(
    documents: list[Document],
    issue: str,
    component_scores: dict[str, list[float]],
    final: list[float],
    k: int,
) -> list[Result]:
    order = sorted(range(len(documents)), key=lambda index: (-final[index], documents[index].path))[: max(1, k)]
    results = []
    for index in order:
        line, symbol = best_symbol(documents[index], issue)
        results.append(
            Result(
                path=documents[index].path,
                line=line,
                symbol=symbol,
                score=final[index],
                bm25=component_scores["bm25"][index],
                exact=component_scores["exact"][index],
                legacy=component_scores["legacy"][index],
            )
        )
    return results


def rank_documents(
    documents: list[Document],
    issue: str,
    model: LegacyModel | None = None,
    mode: str = "hybrid",
    weights: dict[str, float] | None = None,
    k: int = 10,
) -> list[Result]:
    component_scores = score_documents(documents, issue, model)
    if mode == "bm25":
        final = component_scores["bm25"]
    elif mode == "legacy":
        final = component_scores["legacy"]
    elif mode == "hybrid":
        final = reciprocal_rank_fusion(component_scores, weights or DEFAULT_FUSION_WEIGHTS)
    else:
        raise ValueError(f"Unknown ranking mode: {mode}")
    return results_from_scores(documents, issue, component_scores, final, k)


def rank_documents_adaptive(
    documents: list[Document],
    issue: str,
    gate: AdaptiveGate,
    model: LegacyModel | None,
    k: int = 10,
) -> tuple[list[Result], bool, dict[str, float]]:
    if not gate.enabled:
        results = rank_documents(documents, issue, model=None, mode="bm25", k=k)
        return results, False, {}
    if model is None:
        raise ValueError("An enabled localization gate requires the v1 model")
    components = score_documents(documents, issue, model)
    features = gate_features(issue, components, gate.candidate_weights)
    use_v1 = gate.use_v1(features)
    final = (
        reciprocal_rank_fusion(components, gate.candidate_weights)
        if use_v1
        else components["bm25"]
    )
    return results_from_scores(documents, issue, components, final, k), use_v1, features


def find_root(given: str | None) -> Path:
    candidates = [given, "/workspace", os.environ.get("PWD"), os.getcwd()]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            path = Path(candidate).resolve()
            if given or (path / ".git").exists() or str(path) == "/workspace":
                return path
    return Path(given or ".").resolve()


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "model.json"


def default_gate_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "gate.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root")
    parser.add_argument("--issue")
    parser.add_argument("--issue-file")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mode", choices=("adaptive", "bm25", "legacy", "hybrid", "portable"), default="adaptive")
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--gate", type=Path, default=default_gate_path())
    parser.add_argument("--policy", type=Path,
                        default=default_gate_path().with_name("portable-policy.json"))
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("text", nargs="*")
    args = parser.parse_args(argv)
    issue = args.issue or " ".join(args.text)
    if not issue and args.issue_file:
        issue = Path(args.issue_file).read_text(encoding="utf-8", errors="replace")
    if not issue:
        issue = sys.stdin.read()
    if not issue.strip():
        parser.error("an issue is required")
    root = find_root(args.root)
    documents = read_repository(root, include_tests=args.include_tests)
    if not documents:
        parser.error(f"no Python source files found under {root}")
    decision = args.mode
    portable = PortablePolicy.load(args.policy) if args.mode in {"adaptive", "portable"} and args.policy.is_file() else None
    if args.mode == "portable" or (args.mode == "adaptive" and portable and portable.enabled):
        if portable is None:
            parser.error(f"portable policy not found: {args.policy}")
        results, use_v1 = rank_documents_portable(
            documents, issue, portable, min(max(args.k, 1), 50),
            experimental=args.mode == "portable",
        )
        decision = f"portable(v1={'on' if use_v1 else 'off'},experimental={not portable.enabled})"
    elif args.mode == "adaptive":
        gate = AdaptiveGate.load(args.gate)
        model = LegacyModel.load(args.model) if gate.enabled else None
        results, use_v1, _features = rank_documents_adaptive(
            documents, issue, gate=gate, model=model, k=min(max(args.k, 1), 50)
        )
        decision = f"adaptive(v1={'on' if use_v1 else 'off'})"
    else:
        model = LegacyModel.load(args.model) if args.mode != "bm25" else None
        results = rank_documents(
            documents, issue, model=model, mode=args.mode, k=min(max(args.k, 1), 50)
        )
    print(f"# localization-v1: {decision} top {len(results)} of {len(documents)} files in {root}")
    print("# rank score path:line symbol [bm25 exact legacy]")
    for rank, result in enumerate(results, 1):
        print(
            f"{rank:>2} {result.score:.6f} {result.path}:{result.line} {result.symbol} "
            f"[{result.bm25:.3f} {result.exact:.3f} {result.legacy:.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
