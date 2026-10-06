"""Graph inspection and semantic code search tools for SWE-gemma."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, cast

from swegemma.tools.base import (
    budget_gated,
    error_response,
    ok_response,
    parse_tool_response,
)

logger = logging.getLogger(__name__)


def _coerce_int(val: Any, default: int, param_name: str) -> int:
    """Coerce an integer tool parameter that may arrive as a string or None."""
    if val is None:
        return default
    if isinstance(val, bool):
        raise TypeError(f'{param_name} must be an integer, got boolean {val}')
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ('none', 'null', 'nil'):
            return default
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
            raise ValueError(
                f'Invalid integer for {param_name}: {val!r}'
            ) from None
    raise ValueError(f'Invalid integer for {param_name}: {val!r}')


def _get_max_stdout_chars(ctx: Any) -> int | None:
    """Resolve max_stdout_chars from context harness or fallback attribute."""
    harness = getattr(ctx, 'harness', None)
    if harness is not None and hasattr(harness, 'max_stdout_chars'):
        return harness.max_stdout_chars
    return getattr(ctx, 'max_stdout_chars', 5000)


def _allocate_code_budgets(lengths: list[int], max_chars: int) -> list[int]:
    """Divide max_chars across results with unused characters rolling over to larger nodes."""
    if not lengths:
        return []
    if max_chars <= 0:
        return [0] * len(lengths)
    if sum(lengths) <= max_chars:
        return list(lengths)

    remaining = max_chars
    unallocated = list(range(len(lengths)))
    budgets = [0] * len(lengths)

    while unallocated:
        share = remaining // len(unallocated)
        fitting = [i for i in unallocated if lengths[i] <= share]
        if fitting:
            for i in fitting:
                budgets[i] = lengths[i]
                remaining -= lengths[i]
            fitting_set = set(fitting)
            unallocated = [i for i in unallocated if i not in fitting_set]
        else:
            remainder = remaining % len(unallocated)
            for rank_idx, i in enumerate(unallocated):
                budgets[i] = share + (1 if rank_idx < remainder else 0)
            break

    return budgets


@budget_gated
def get_code_neighbors(
    ctx: Any,
    node: str,
    edge_type: str | None = None,
    max_neighbors: int = 50,
) -> str:
    """Discover structural code graph neighbors (callers, callees, definitions) for a given code node.

    Args:
        ctx: Context instance.
        node: Name of the function, method, or class node (e.g. 'parse_request' or 'app.routes.get_user').
        edge_type: Optional edge relation filter (e.g. 'CALLS', 'DEFINED_IN', 'IMPORTS').
        max_neighbors: Maximum number of neighbor nodes to return (default: 50).

    Returns:
        JSON string with list of neighbor node names and count.
    """
    task = getattr(ctx, 'task', None)
    task_repo = (
        task.get('repo')
        if isinstance(task, dict)
        else getattr(task, 'repo', None)
        if task
        else None
    )
    repo_name = (
        getattr(ctx, 'repo', None)
        or task_repo
        or getattr(ctx, 'graph_repo_key', None)
    )
    if not repo_name:
        return error_response(
            error_type='MissingRepoContext',
            error_message='No repository configured for graph queries.',
        )

    try:
        from swegemma import graph as sg

        coerced_max_neighbors = _coerce_int(max_neighbors, 50, 'max_neighbors')
        coerced_edge_type = edge_type
        if isinstance(coerced_edge_type, str) and coerced_edge_type.strip().lower() in (
            '',
            'none',
            'null',
            'nil',
        ):
            coerced_edge_type = None

        base_commit = getattr(task, 'base_commit', None) if task else None
        if not base_commit and isinstance(task, dict):
            base_commit = task.get('base_commit')
        if base_commit and str(repo_name).endswith(f'_{base_commit}'):
            base_commit = None

        node_arg = node if ';' in str(node) else f'{repo_name};{node}'
        neighbors = sg.get_neighbor(
            node=node_arg,
            max_neighbors=coerced_max_neighbors,
            edge_type=coerced_edge_type,
            sstable_path=str(ctx.graph_dir)
            if ctx.graph_dir
            else sg.DEFAULT_REPO_GRAPHS_SSTABLE,
            embeddings_sstable_path=str(ctx.embeddings_dir)
            if ctx.embeddings_dir
            else sg.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
            use_local_storage=True,
            repo_name=repo_name,
            graph_dir=str(ctx.graph_dir) if ctx.graph_dir else None,
            embeddings_dir=str(ctx.embeddings_dir)
            if ctx.embeddings_dir
            else None,
            base_commit=base_commit,
        )
        raw_neighbors = (
            neighbors if isinstance(neighbors, list) else neighbors.get(node_arg, [])
        )
        neighbor_list = [str(n) for n in raw_neighbors]
        max_chars = _get_max_stdout_chars(ctx)
        is_truncated = False
        if max_chars is not None:
            kept_neighbors: list[str] = []
            used_chars = 0
            for idx, n_str in enumerate(neighbor_list):
                if idx > 0 and used_chars + len(n_str) > max_chars:
                    is_truncated = True
                    break
                if idx == 0 and len(n_str) > max_chars:
                    kept_neighbors.append(n_str[:max_chars])
                    is_truncated = True
                    break
                kept_neighbors.append(n_str)
                used_chars += len(n_str)
            neighbor_list = kept_neighbors

        return ok_response(
            node=node,
            neighbors=neighbor_list,
            count=len(neighbor_list),
            is_truncated=is_truncated,
        )
    except Exception as e:
        return error_response(
            error_type='GraphLookupError',
            error_message=str(e),
        )


@budget_gated
def search_similar_code(
    ctx: Any,
    query: str,
    k: int = 10,
) -> str:
    """Find semantically similar code functions and classes using graph vector embeddings.

    Args:
        ctx: Context instance.
        query: Function name or code snippet to search for similar implementations.
        k: Number of most similar code nodes to return (default: 10).

    Returns:
        JSON string containing matching node names, code snippets, and similarity scores.
    """
    task = getattr(ctx, 'task', None)
    task_repo = (
        task.get('repo')
        if isinstance(task, dict)
        else getattr(task, 'repo', None)
        if task
        else None
    )
    repo_name = (
        getattr(ctx, 'repo', None)
        or task_repo
        or getattr(ctx, 'graph_repo_key', None)
    )
    if not repo_name:
        return error_response(
            error_type='MissingRepoContext',
            error_message='No repository configured for graph queries.',
        )

    try:
        from swegemma import graph as sg

        coerced_k = _coerce_int(k, 10, 'k')

        base_commit = getattr(task, 'base_commit', None) if task else None
        if not base_commit and isinstance(task, dict):
            base_commit = task.get('base_commit')
        if base_commit and str(repo_name).endswith(f'_{base_commit}'):
            base_commit = None

        similar_nodes = sg.get_similar_nodes(
            node=query,
            repo_name=repo_name,
            k=coerced_k,
            graphs_sstable_path=str(ctx.graph_dir)
            if ctx.graph_dir
            else sg.DEFAULT_REPO_GRAPHS_SSTABLE,
            embeddings_sstable_path=str(ctx.embeddings_dir)
            if ctx.embeddings_dir
            else sg.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
            use_local_storage=True,
            graph_dir=str(ctx.graph_dir) if ctx.graph_dir else None,
            embeddings_dir=str(ctx.embeddings_dir)
            if ctx.embeddings_dir
            else None,
            base_commit=base_commit,
        )
        max_chars = _get_max_stdout_chars(ctx)
        raw_codes = [str(item.get('code') or '') for item in similar_nodes]
        lengths = [len(c) for c in raw_codes]
        is_truncated = False
        if max_chars is not None and sum(lengths) > max_chars:
            budgets = _allocate_code_budgets(lengths, max_chars)
            is_truncated = True
        else:
            budgets = lengths

        formatted = [
            {
                'node_name': item['node_name'],
                'code': raw_codes[i][: budgets[i]],
                'similarity': round(float(item.get('similarity', 0.0)), 4),
            }
            for i, item in enumerate(similar_nodes)
        ]
        return ok_response(
            query=query,
            results=formatted,
            count=len(formatted),
            is_truncated=is_truncated,
        )
    except Exception as e:
        return error_response(
            error_type='SimilaritySearchError',
            error_message=str(e),
        )


@budget_gated
def get_code_subgraph(
    ctx: Any,
    nodes: list[str],
) -> str:
    """Extract the relationship graph (nodes and connecting edges) for a focal set of code elements.

    Args:
        ctx: Context instance.
        nodes: List of function/class node names to include in the induced subgraph.

    Returns:
        JSON string detailing the subgraph nodes and connecting directed edges.
    """
    task = getattr(ctx, 'task', None)
    task_repo = (
        task.get('repo')
        if isinstance(task, dict)
        else getattr(task, 'repo', None)
        if task
        else None
    )
    repo_name = (
        getattr(ctx, 'repo', None)
        or task_repo
        or getattr(ctx, 'graph_repo_key', None)
    )
    if not repo_name:
        return error_response(
            error_type='MissingRepoContext',
            error_message='No repository configured for graph queries.',
        )

    try:
        from swegemma import graph as sg

        coerced_nodes: Any = nodes
        if isinstance(coerced_nodes, str):
            stripped = coerced_nodes.strip()
            if stripped.startswith('['):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        coerced_nodes = [str(x) for x in parsed]
                except Exception:
                    pass
            if isinstance(coerced_nodes, str):
                coerced_nodes = [
                    part.strip().strip('"\'')
                    for part in stripped.strip('[]').split(',')
                    if part.strip().strip('"\'')
                ]

        base_commit = getattr(task, 'base_commit', None) if task else None
        if not base_commit and isinstance(task, dict):
            base_commit = task.get('base_commit')
        if base_commit and str(repo_name).endswith(f'_{base_commit}'):
            base_commit = None

        full_graph = sg.get_graph(
            repo_name=repo_name,
            sstable_path=str(ctx.graph_dir)
            if ctx.graph_dir
            else sg.DEFAULT_REPO_GRAPHS_SSTABLE,
            embeddings_sstable_path=str(ctx.embeddings_dir)
            if ctx.embeddings_dir
            else sg.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
            use_local_storage=True,
            graph_dir=str(ctx.graph_dir) if ctx.graph_dir else None,
            embeddings_dir=str(ctx.embeddings_dir)
            if ctx.embeddings_dir
            else None,
            base_commit=base_commit,
        )
        subgraph = sg.get_induced_subgraph(full_graph, cast(Any, coerced_nodes))

        edges = []
        for u, v, data in subgraph.edges(data=True):
            edges.append(
                {
                    'from': str(u),
                    'to': str(v),
                    'type': data.get('type') or data.get('relation', 'connected'),
                }
            )

        nodes_list = [str(n) for n in subgraph.nodes()]
        total_node_count = len(nodes_list)
        total_edge_count = len(edges)
        max_chars = _get_max_stdout_chars(ctx)
        is_truncated = False
        if max_chars is not None:
            node_budget = max_chars // 2 if edges else max_chars
            kept_nodes: list[str] = []
            used_node_chars = 0
            for idx, n_str in enumerate(nodes_list):
                if idx > 0 and used_node_chars + len(n_str) > node_budget:
                    is_truncated = True
                    break
                if idx == 0 and len(n_str) > node_budget:
                    kept_nodes.append(n_str[:node_budget])
                    is_truncated = True
                    break
                kept_nodes.append(n_str)
                used_node_chars += len(n_str)
            nodes_list = kept_nodes

            edge_budget = max(0, max_chars - used_node_chars)
            kept_edges: list[dict[str, Any]] = []
            used_edge_chars = 0
            for e in edges:
                e_len = len(e['from']) + len(e['to']) + len(str(e['type']))
                if used_edge_chars + e_len > edge_budget:
                    is_truncated = True
                    break
                kept_edges.append(e)
                used_edge_chars += e_len
            edges = kept_edges

        return ok_response(
            nodes=nodes_list,
            edges=edges,
            node_count=total_node_count,
            edge_count=total_edge_count,
            is_truncated=is_truncated,
        )
    except Exception as e:
        return error_response(
            error_type='SubgraphExtractionError',
            error_message=str(e),
        )


def make_get_code_neighbors(ctx: Any) -> Callable:
    def get_code_neighbors_tool(
        node: str,
        edge_type: str | None = None,
        max_neighbors: int = 50,
    ) -> dict[str, Any]:
        """Discover structural code graph neighbors (callers, callees, definitions) for a given code node.

        Args:
            node: Name of the function, method, or class node (e.g. 'parse_request' or 'app.routes.get_user').
            edge_type: Optional edge relation filter (e.g. 'CALLS', 'DEFINED_IN', 'IMPORTS').
            max_neighbors: Maximum number of neighbor nodes to return (default: 50).

        Returns:
            Dictionary with list of neighbor node names and count.
        """
        return parse_tool_response(
            get_code_neighbors(ctx, node, edge_type, max_neighbors)
        )

    get_code_neighbors_tool.__name__ = 'get_code_neighbors'
    return get_code_neighbors_tool


def make_search_similar_code(ctx: Any) -> Callable:
    def search_similar_code_tool(
        query: str,
        k: int = 10,
    ) -> dict[str, Any]:
        """Find semantically similar code functions and classes using graph vector embeddings.

        Args:
            query: Function name or code snippet to search for similar implementations.
            k: Number of most similar code nodes to return (default: 10).

        Returns:
            Dictionary containing matching node names, code snippets, and similarity scores.
        """
        return parse_tool_response(search_similar_code(ctx, query, k))

    search_similar_code_tool.__name__ = 'search_similar_code'
    return search_similar_code_tool


def make_get_code_subgraph(ctx: Any) -> Callable:
    def get_code_subgraph_tool(nodes: list[str]) -> dict[str, Any]:
        """Extract the relationship graph (nodes and connecting edges) for a focal set of code elements.

        Args:
            nodes: List of function/class node names to include in the induced subgraph.

        Returns:
            Dictionary detailing the subgraph nodes and connecting directed edges.
        """
        return parse_tool_response(get_code_subgraph(ctx, nodes))

    get_code_subgraph_tool.__name__ = 'get_code_subgraph'
    return get_code_subgraph_tool
