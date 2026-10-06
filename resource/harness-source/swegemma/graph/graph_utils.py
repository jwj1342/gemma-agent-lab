"""Graph utilities for constructing and inspecting repository graphs."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import cachetools
import networkx as nx
import numpy as np

from swegemma.graph import embedding_utils

logger = logging.getLogger(__name__)

DEFAULT_REPO_GRAPHS_SSTABLE = embedding_utils.DEFAULT_REPO_GRAPHS_SSTABLE
FALLBACK_REPO_GRAPHS_SSTABLE = embedding_utils.FALLBACK_REPO_GRAPHS_SSTABLE

_LOCK = threading.Lock()
_SSTABLE_CACHE: dict[str, Any] = {}
_GRAPH_CACHE: cachetools.LRUCache = cachetools.LRUCache(maxsize=100)


def clear_caches() -> None:
    """Clears the in-memory SSTable and graph caches."""
    with _LOCK:
        _SSTABLE_CACHE.clear()
        _GRAPH_CACHE.clear()
        logger.info('Graph utils caches cleared.')


def clear_cache_for_repo(repo_name: str | bytes) -> None:
    """Clears the in-memory graph cache for a specific repository."""
    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else str(repo_name)
    )
    with _LOCK:
        keys_to_remove = [
            k
            for k in _GRAPH_CACHE
            if k[0] == repo_str or str(k[0]).startswith(f'{repo_str}_')
        ]
        for k in keys_to_remove:
            del _GRAPH_CACHE[k]
        logger.info('Graph cache cleared for repo: %s', repo_str)


class CustomMultiEdgeView(nx.classes.reportviews.OutMultiEdgeView):
    """Custom OutMultiEdgeView that supports 2-tuple indexing (u, v) for MultiDiGraph."""

    def __getitem__(self, e: Any) -> Any:
        if isinstance(e, tuple) and len(e) == 2:
            u, v = e
            return self._adjdict[u][v]
        return super().__getitem__(e)


class CustomMultiDiGraph(nx.MultiDiGraph):
    """Custom MultiDiGraph whose edges property returns CustomMultiEdgeView."""

    @property
    def edges(self) -> CustomMultiEdgeView:
        return CustomMultiEdgeView(self)


def _clean_for_json(obj: Any, include_embeddings: bool = False) -> Any:
    """Recursively cleans an object for JSON serialization, handling bytes and numpy types."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
            if k_str == 'embedding' and not include_embeddings:
                continue
            if k_str == 'embedding' and isinstance(v, np.ndarray):
                cleaned[k_str] = v.tolist()
            else:
                cleaned[k_str] = _clean_for_json(v, include_embeddings)
        return cleaned
    if isinstance(obj, (set, frozenset)):
        return sorted(
            [_clean_for_json(item, include_embeddings) for item in obj],
            key=str,
        )
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(item, include_embeddings) for item in obj]
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_graph_to_json(
    graph: nx.DiGraph | nx.MultiDiGraph,
    output_dir: str = 'data/graphs',
    include_embeddings: bool = False,
) -> str:
    """Saves a networkx graph topology to a local JSON file using node_link_data.

    By default, node 'embedding' vectors are excluded from the JSON to maintain
    clean separation of structure (.json) and embedding features (.npz).

    Args:
        graph: A networkx DiGraph or MultiDiGraph to serialize.
        output_dir: Output base directory or target .json file path.
        include_embeddings: Whether to include node embedding vectors in JSON.

    Returns:
        The path of the written JSON file.
    """
    repo_name = getattr(graph, 'graph', {}).get('repo_name')
    if isinstance(repo_name, bytes):
        repo_name = repo_name.decode('utf-8', errors='replace')

    if output_dir.endswith('.json'):
        file_path = output_dir
    elif repo_name:
        file_path = embedding_utils.repo_to_file_path(
            output_dir, str(repo_name), '.json'
        )
    else:
        raise ValueError(
            "Graph missing required 'repo_name' attribute in graph.graph for JSON"
            ' output.'
        )

    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    raw_data = nx.node_link_data(graph, edges='edges')
    clean_data = _clean_for_json(raw_data, include_embeddings=include_embeddings)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2)

    logger.info(
        'Saved graph for repo %s to JSON file %s',
        repo_name or 'unknown',
        file_path,
    )
    return file_path


def load_graph_from_json(file_path: str) -> CustomMultiDiGraph:
    """Loads a CustomMultiDiGraph from a JSON file serialized via node_link_data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Graph JSON file not found at: {file_path}')

    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    edge_key = (
        'edges'
        if 'edges' in data
        else ('links' if 'links' in data else 'edges')
    )
    base_g = nx.node_link_graph(data, edges=edge_key)
    g = CustomMultiDiGraph(base_g)
    if hasattr(base_g, 'graph') and isinstance(base_g.graph, dict):
        g.graph.update(base_g.graph)

    for _n, attrs in g.nodes(data=True):
        if 'embedding' in attrs and attrs['embedding'] is not None:
            attrs['embedding'] = np.array(attrs['embedding'], dtype=np.float32)
    return g


def _discover_embedding_dim(
    node_names: list[str],
    repo_name: str,
    sstable_path: str,
    embeddings_sstable_path: str,
    use_local_storage: bool,
    base_commit: str | None = None,
) -> int:
    """Discovers the embedding dimension by trying to embed nodes."""
    for name in node_names:
        try:
            emb = embedding_utils.embed(
                name,
                repo_name,
                graphs_sstable_path=sstable_path,
                embeddings_sstable_path=embeddings_sstable_path,
                use_local_storage=use_local_storage,
                base_commit=base_commit,
            )
            if emb is not None and len(emb) > 0:
                return len(emb)
        except Exception:
            pass
    return embedding_utils.DEFAULT_EMBEDDING_DIM


def get_graph(
    repo_name: str | bytes,
    sstable_path: str = DEFAULT_REPO_GRAPHS_SSTABLE,
    embeddings_sstable_path: str = embedding_utils.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
    use_local_storage: bool = False,
    *,
    graph_dir: str | None = None,
    embeddings_dir: str | None = None,
    base_commit: str | None = None,
) -> CustomMultiDiGraph:
    """Returns a networkx MultiDiGraph for the given repository name."""
    if isinstance(repo_name, bytes):
        repo_name = repo_name.decode('utf-8', errors='replace')

    if graph_dir is not None:
        sstable_path = graph_dir
        use_local_storage = True
    if embeddings_dir is not None:
        embeddings_sstable_path = embeddings_dir
        use_local_storage = True

    effective_commit = (
        None
        if (base_commit and repo_name.endswith(f'_{base_commit}'))
        else base_commit
    )
    effective_repo = (
        f'{repo_name}_{effective_commit}' if effective_commit else repo_name
    )

    cache_key = (
        effective_repo,
        sstable_path,
        embeddings_sstable_path,
        use_local_storage,
    )
    with _LOCK:
        if cache_key in _GRAPH_CACHE:
            return _GRAPH_CACHE[cache_key]

    if sstable_path.endswith('.json'):
        local_path = sstable_path
    else:
        base_dir = (
            sstable_path
            if (
                (
                    use_local_storage
                    and sstable_path
                    not in (
                        DEFAULT_REPO_GRAPHS_SSTABLE,
                        FALLBACK_REPO_GRAPHS_SSTABLE,
                    )
                )
                or embedding_utils.is_local_path(sstable_path)
            )
            else 'data/graphs'
        )
        local_path = embedding_utils.resolve_local_file_path(
            base_dir, repo_name, '.json', base_commit=effective_commit
        )

    if not os.path.exists(local_path):
        raise KeyError(
            f'Repository {repo_name} not found in local files at {local_path}'
        )

    try:
        g = load_graph_from_json(local_path)
    except Exception as e:
        raise KeyError(
            f'Failed to load graph JSON for {repo_name} at {local_path}: {e}'
        ) from e

    g.graph['repo_name'] = repo_name

    # Discover target dimension for fallback
    target_dim = next(
        (
            len(attrs['embedding'])
            for _, attrs in g.nodes(data=True)
            if 'embedding' in attrs
            and attrs['embedding'] is not None
            and len(attrs['embedding']) > 0
        ),
        None,
    )
    if target_dim is None:
        node_names = [str(n) for n in g.nodes()]
        target_dim = _discover_embedding_dim(
            node_names,
            repo_name,
            sstable_path,
            embeddings_sstable_path,
            use_local_storage=True,
            base_commit=effective_commit,
        )

    for n, attrs in g.nodes(data=True):
        if 'name' not in attrs:
            attrs['name'] = str(n)
        if 'text' not in attrs:
            attrs['text'] = ''
        if (
            'embedding' not in attrs
            or attrs['embedding'] is None
            or (
                isinstance(attrs['embedding'], np.ndarray)
                and len(attrs['embedding']) == 0
            )
        ):
            try:
                emb = embedding_utils.embed(
                    str(n),
                    repo_name,
                    graphs_sstable_path=sstable_path,
                    embeddings_sstable_path=embeddings_sstable_path,
                    use_local_storage=True,
                    base_commit=effective_commit,
                )
            except Exception as e:
                logger.warning(
                    'embedding_utils.embed failed for local node %s in %s: %s',
                    n,
                    repo_name,
                    e,
                )
                emb = None
            if emb is None:
                emb = np.zeros(target_dim, dtype=np.float32)
            attrs['embedding'] = emb

    with _LOCK:
        _GRAPH_CACHE[cache_key] = g
    return g


def get_neighbor(
    node: str | bytes | list[str | bytes],
    graph: nx.DiGraph | nx.MultiDiGraph | None = None,
    max_neighbors: int = 50,
    edge_type: str | None = None,
    sstable_path: str = DEFAULT_REPO_GRAPHS_SSTABLE,
    embeddings_sstable_path: str = embedding_utils.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
    use_local_storage: bool = False,
    *,
    repo_name: str | bytes | None = None,
    graph_dir: str | None = None,
    embeddings_dir: str | None = None,
    base_commit: str | None = None,
) -> Any:
    """Returns a list of neighboring node names or a dict of lists if node is a list."""
    if max_neighbors <= 0:
        raise ValueError('max_neighbors must be greater than 0.')

    if isinstance(node, list):
        return {
            n: get_neighbor(
                n,
                graph=graph,
                max_neighbors=max_neighbors,
                edge_type=edge_type,
                sstable_path=sstable_path,
                embeddings_sstable_path=embeddings_sstable_path,
                use_local_storage=use_local_storage,
                repo_name=repo_name,
                graph_dir=graph_dir,
                embeddings_dir=embeddings_dir,
                base_commit=base_commit,
            )
            for n in node
        }

    node_str = (
        node.decode('utf-8', errors='replace')
        if isinstance(node, bytes)
        else str(node)
    )
    if repo_name is not None and isinstance(repo_name, bytes):
        repo_name = repo_name.decode('utf-8', errors='replace')

    graph_repo_name = getattr(graph, 'graph', {}).get('repo_name')
    if isinstance(graph_repo_name, bytes):
        graph_repo_name = graph_repo_name.decode('utf-8', errors='replace')

    eff_repo = repo_name or graph_repo_name
    target_node = node_str

    if ';' in node_str:
        parts = node_str.split(';', 1)
        if graph is None:
            eff_repo = eff_repo or parts[0]
            graph = get_graph(
                eff_repo,
                sstable_path=sstable_path,
                embeddings_sstable_path=embeddings_sstable_path,
                use_local_storage=use_local_storage,
                graph_dir=graph_dir,
                embeddings_dir=embeddings_dir,
                base_commit=base_commit,
            )
            target_node = parts[1]
        elif not graph.has_node(node_str) and graph.has_node(parts[1]):
            eff_repo = eff_repo or parts[0]
            target_node = parts[1]
        elif graph.has_node(node_str):
            target_node = node_str
            eff_repo = eff_repo or parts[0]
    elif graph is None:
        if eff_repo:
            graph = get_graph(
                eff_repo,
                sstable_path=sstable_path,
                embeddings_sstable_path=embeddings_sstable_path,
                use_local_storage=use_local_storage,
                graph_dir=graph_dir,
                embeddings_dir=embeddings_dir,
                base_commit=base_commit,
            )
        else:
            raise ValueError(
                "If graph is None, node must be in 'repo_name;node_name' format."
            )

    if graph is None:
        raise ValueError('graph cannot be None at this point.')

    if not graph.has_node(target_node):
        return []

    neighbors: list[str] = []
    for _u, v, data in graph.out_edges(target_node, data=True):
        if edge_type is None or data.get('type') == edge_type:
            neighbor_name = (
                f'{eff_repo};{v}'
                if eff_repo and not str(v).startswith(f'{eff_repo};')
                else str(v)
            )
            if neighbor_name not in neighbors:
                neighbors.append(neighbor_name)

    for u, _v, data in graph.in_edges(target_node, data=True):
        if edge_type is None or data.get('type') == edge_type:
            neighbor_name = (
                f'{eff_repo};{u}'
                if eff_repo and not str(u).startswith(f'{eff_repo};')
                else str(u)
            )
            if neighbor_name not in neighbors:
                neighbors.append(neighbor_name)

    neighbors.sort()
    return neighbors[:max_neighbors]


def get_induced_subgraph(
    graph: nx.DiGraph | nx.MultiDiGraph,
    subset_nodes: list[str | bytes] | None = None,
) -> nx.DiGraph | nx.MultiDiGraph:
    """Returns the induced subgraph for the given subset of nodes."""
    if not subset_nodes:
        raise ValueError('subset_nodes cannot be empty.')
    valid_nodes = []
    for raw_n in subset_nodes:
        n = (
            raw_n.decode('utf-8', errors='replace')
            if isinstance(raw_n, bytes)
            else str(raw_n)
        )
        if graph.has_node(n):
            valid_nodes.append(n)
        elif ';' in n:
            bare = n.split(';', 1)[1]
            if graph.has_node(bare):
                valid_nodes.append(bare)
    return graph.subgraph(valid_nodes).copy()
