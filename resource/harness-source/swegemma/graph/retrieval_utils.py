"""Retrieval utilities for finding similar code nodes in repository graphs using embeddings."""

from __future__ import annotations

import logging
import threading
from typing import Any

import cachetools
import networkx as nx
import numpy as np

from swegemma.graph import embedding_utils, graph_utils

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# In-memory caches
# Maps (repo_name, str(node_name), k, str(node_content), graphs_sstable_path,
# embeddings_sstable_path, use_local_storage) -> list of similar node dicts
_SIMILARITY_CACHE: cachetools.LRUCache = cachetools.LRUCache(maxsize=10000)

# Maps (repo_name, graphs_sstable_path, embeddings_sstable_path, use_local_storage) -> dict
# mapping node_name -> list of top-K similar node dicts
_PRECOMPUTED_GRAPHS: dict[
    tuple[Any, ...], dict[str, list[dict[str, Any]]]
] = {}

# Maps (repo_name, graphs_sstable_path, embeddings_sstable_path, use_local_storage) -> dict
# mapping node_name -> node text
_PRECOMPUTED_TEXTS: dict[tuple[Any, ...], dict[str, str]] = {}


def clear_caches(repo_name: str | bytes | None = None) -> None:
    """Clears the in-memory similarity and precomputed caches.

    Args:
        repo_name: If provided, clears only the caches associated with this repo.
            If None, clears all caches.
    """
    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else repo_name
    )
    with _LOCK:
        if repo_str is None:
            _SIMILARITY_CACHE.clear()
            _PRECOMPUTED_GRAPHS.clear()
            _PRECOMPUTED_TEXTS.clear()
            graph_utils.clear_caches()
            embedding_utils.clear_caches()
            logger.info('All retrieval caches cleared.')
        else:
            g_keys = [
                k
                for k in _PRECOMPUTED_GRAPHS
                if (isinstance(k, tuple) and (k[0] == repo_str or str(k[0]).startswith(f'{repo_str}_')))
                or k == repo_str
                or str(k).startswith(f'{repo_str}_')
            ]
            for k in g_keys:
                del _PRECOMPUTED_GRAPHS[k]
            t_keys = [
                k
                for k in _PRECOMPUTED_TEXTS
                if (isinstance(k, tuple) and (k[0] == repo_str or str(k[0]).startswith(f'{repo_str}_')))
                or k == repo_str
                or str(k).startswith(f'{repo_str}_')
            ]
            for k in t_keys:
                del _PRECOMPUTED_TEXTS[k]
            keys_to_remove = [
                k
                for k in _SIMILARITY_CACHE
                if isinstance(k, tuple)
                and len(k) > 0
                and (k[0] == repo_str or str(k[0]).startswith(f'{repo_str}_'))
            ]
            for k in keys_to_remove:
                del _SIMILARITY_CACHE[k]
            graph_utils.clear_cache_for_repo(repo_str)
            embedding_utils.clear_caches(repo_str)
            logger.info('Retrieval caches cleared for repo: %s', repo_str)


def _finalize_embedding_matrix(
    node_names: list[str],
    raw_embeddings: list[np.ndarray | None],
) -> np.ndarray:
    """Converts a list of embedding arrays (or None) into a 2D float32 matrix."""
    if not node_names:
        return np.empty(
            (0, embedding_utils.DEFAULT_EMBEDDING_DIM), dtype=np.float32
        )

    detected_dim = next(
        (
            len(e)
            for e in raw_embeddings
            if e is not None and len(e) > 0
        ),
        embedding_utils.DEFAULT_EMBEDDING_DIM,
    )
    embeddings_list: list[np.ndarray] = []
    for emb in raw_embeddings:
        if emb is None or len(emb) == 0:
            embeddings_list.append(np.zeros(detected_dim, dtype=np.float32))
        else:
            arr = np.array(emb, dtype=np.float32).flatten()
            if arr.shape[0] != detected_dim:
                adjusted = np.zeros(detected_dim, dtype=np.float32)
                dim = min(arr.shape[0], detected_dim)
                adjusted[:dim] = arr[:dim]
                embeddings_list.append(adjusted)
            else:
                embeddings_list.append(arr)

    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    if embeddings_array.ndim == 1:
        embeddings_array = embeddings_array.reshape(len(node_names), -1)
    return embeddings_array


def _extract_nodes_and_embeddings(
    graph: Any,
    repo_name: str | bytes,
    graphs_sstable_path: str = embedding_utils.DEFAULT_REPO_GRAPHS_SSTABLE,
    embeddings_sstable_path: str = embedding_utils.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
    use_local_storage: bool = False,
    *,
    embeddings_dir: str | None = None,
    graph_dir: str | None = None,
    base_commit: str | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """Extracts node names, code texts, and single-precision embedding vectors from various graph representations.

    Args:
        graph: A GraphStruct, networkx.DiGraph, networkx.Graph, or dict representation of a graph.
        repo_name: The name of the repository.
        graphs_sstable_path: Path to the repo_graphs SSTable or local graphs path.
        embeddings_sstable_path: Path to the function embeddings SSTable or local NPZ.
        use_local_storage: Whether to read embeddings from local files.
        embeddings_dir: Optional local embeddings directory override.
        graph_dir: Optional local graphs directory override.
        base_commit: Optional base commit hash for snapshot-specific lookup.

    Returns:
        A tuple of (node_names, node_texts, embeddings_array).
    """
    if embeddings_dir is not None:
        embeddings_sstable_path = embeddings_dir
        use_local_storage = True
    if graph_dir is not None:
        graphs_sstable_path = graph_dir
        use_local_storage = True

    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else str(repo_name)
    )

    node_names: list[str] = []
    node_texts: list[str] = []
    raw_embeddings: list[np.ndarray | None] = []

    if graph is None:
        return (
            [],
            [],
            np.empty(
                (0, embedding_utils.DEFAULT_EMBEDDING_DIM), dtype=np.float32
            ),
        )

    # 1. networkx.DiGraph or networkx.Graph
    if isinstance(graph, (nx.DiGraph, nx.Graph)):
        for n, data in graph.nodes(data=True):
            name = data.get('name', str(n))
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            text = data.get('text', data.get('code', ''))
            if text is None:
                text = ''
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            emb = data.get('embedding')

            if emb is None or (isinstance(emb, np.ndarray) and emb.size == 0):
                emb = embedding_utils.embed(
                    text or name,
                    repo_str,
                    graphs_sstable_path=graphs_sstable_path,
                    embeddings_sstable_path=embeddings_sstable_path,
                    use_local_storage=use_local_storage,
                    base_commit=base_commit,
                )
                if emb is None and text:
                    emb = embedding_utils.embed(
                        name,
                        repo_str,
                        graphs_sstable_path=graphs_sstable_path,
                        embeddings_sstable_path=embeddings_sstable_path,
                        use_local_storage=use_local_storage,
                        base_commit=base_commit,
                    )

            node_names.append(str(name))
            node_texts.append(str(text))
            raw_embeddings.append(
                np.array(emb, dtype=np.float32) if emb is not None else None
            )

    # 2. GraphStruct
    elif (
        hasattr(graph, 'nodes')
        and isinstance(graph.nodes, dict)
        and 'nodes' in graph.nodes
    ):
        nodes_feat = graph.nodes['nodes']
        raw_names = nodes_feat.get('name', [])
        raw_texts = nodes_feat.get('text', [])
        raw_embs = nodes_feat.get('embedding')

        node_names = [
            s.decode('utf-8', errors='replace') if isinstance(s, bytes) else str(s)
            for s in raw_names
        ]
        node_texts = [
            s.decode('utf-8', errors='replace') if isinstance(s, bytes) else str(s)
            for s in raw_texts
        ]
        while len(node_texts) < len(node_names):
            node_texts.append('')

        if raw_embs is not None and len(raw_embs) == len(node_names):
            for emb in raw_embs:
                raw_embeddings.append(np.array(emb, dtype=np.float32))
        else:
            for i, name in enumerate(node_names):
                text = node_texts[i]
                emb = embedding_utils.embed(
                    text or name,
                    repo_str,
                    graphs_sstable_path=graphs_sstable_path,
                    embeddings_sstable_path=embeddings_sstable_path,
                    use_local_storage=use_local_storage,
                    base_commit=base_commit,
                )
                if emb is None and text:
                    emb = embedding_utils.embed(
                        name,
                        repo_str,
                        graphs_sstable_path=graphs_sstable_path,
                        embeddings_sstable_path=embeddings_sstable_path,
                        use_local_storage=use_local_storage,
                        base_commit=base_commit,
                    )
                raw_embeddings.append(
                    np.array(emb, dtype=np.float32) if emb is not None else None
                )

    # 3. dict representation
    elif isinstance(graph, dict):
        nodes_dict = graph.get('nodes', graph)
        if isinstance(nodes_dict, dict):
            if 'name' in nodes_dict:
                raw_names = nodes_dict.get('name', [])
                raw_texts = nodes_dict.get('text', nodes_dict.get('code', []))
                if raw_texts is None:
                    raw_texts = []
                raw_embs = nodes_dict.get('embedding')

                node_names = [
                    s.decode('utf-8', errors='replace')
                    if isinstance(s, bytes)
                    else str(s)
                    for s in raw_names
                ]
                node_texts = [
                    s.decode('utf-8', errors='replace')
                    if isinstance(s, bytes)
                    else str(s)
                    for s in raw_texts
                ]
                while len(node_texts) < len(node_names):
                    node_texts.append('')

                if raw_embs is not None and len(raw_embs) == len(node_names):
                    for emb in raw_embs:
                        raw_embeddings.append(np.array(emb, dtype=np.float32))
                else:
                    for i, name in enumerate(node_names):
                        text = node_texts[i]
                        emb = embedding_utils.embed(
                            text or name,
                            repo_str,
                            graphs_sstable_path=graphs_sstable_path,
                            embeddings_sstable_path=embeddings_sstable_path,
                            use_local_storage=use_local_storage,
                            base_commit=base_commit,
                        )
                        if emb is None and text:
                            emb = embedding_utils.embed(
                                name,
                                repo_str,
                                graphs_sstable_path=graphs_sstable_path,
                                embeddings_sstable_path=embeddings_sstable_path,
                                use_local_storage=use_local_storage,
                                base_commit=base_commit,
                            )
                        raw_embeddings.append(
                            np.array(emb, dtype=np.float32)
                            if emb is not None
                            else None
                        )
            else:
                for n, data in nodes_dict.items():
                    if isinstance(data, dict):
                        name_val = data.get('name', str(n))
                        name = (
                            name_val.decode('utf-8', errors='replace')
                            if isinstance(name_val, bytes)
                            else str(name_val)
                        )
                        text_val = data.get('text', data.get('code', ''))
                        if text_val is None:
                            text_val = ''
                        text = (
                            text_val.decode('utf-8', errors='replace')
                            if isinstance(text_val, bytes)
                            else str(text_val)
                        )
                        emb = data.get('embedding')

                        if emb is None:
                            emb = embedding_utils.embed(
                                text or name,
                                repo_str,
                                graphs_sstable_path=graphs_sstable_path,
                                embeddings_sstable_path=embeddings_sstable_path,
                                use_local_storage=use_local_storage,
                                base_commit=base_commit,
                            )
                            if emb is None and text:
                                emb = embedding_utils.embed(
                                    name,
                                    repo_str,
                                    graphs_sstable_path=graphs_sstable_path,
                                    embeddings_sstable_path=embeddings_sstable_path,
                                    use_local_storage=use_local_storage,
                                    base_commit=base_commit,
                                )

                        node_names.append(name)
                        node_texts.append(text)
                        raw_embeddings.append(
                            np.array(emb, dtype=np.float32)
                            if emb is not None
                            else None
                        )

    embeddings_array = _finalize_embedding_matrix(node_names, raw_embeddings)
    return node_names, node_texts, embeddings_array


def precompute_top_k_similar_nodes(
    repo_name: str | bytes,
    k: int = 10,
    graphs_sstable_path: str = embedding_utils.DEFAULT_REPO_GRAPHS_SSTABLE,
    embeddings_sstable_path: str = embedding_utils.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
    use_local_storage: bool = False,
    *,
    chunk_size: int = 1000,
    graph_dir: str | None = None,
    embeddings_dir: str | None = None,
    base_commit: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Pre-computes and caches the top-K similar nodes for each node in a graph.

    Args:
        repo_name: The name of the repository graph to precompute.
        k: The number of nearest neighbors to compute for each node.
        graphs_sstable_path: Path to the repo_graphs SSTable or local graphs path.
        embeddings_sstable_path: Path to the function embeddings SSTable or local NPZ.
        use_local_storage: Whether to read embeddings from local files.
        chunk_size: Batch size for matrix multiplication.
        graph_dir: Optional local graphs directory override.
        embeddings_dir: Optional local embeddings directory override.
        base_commit: Optional base commit hash for snapshot-specific precomputation.

    Returns:
        A dictionary mapping each node_name to its top-K similar nodes in sorted order.
    """
    if k <= 0:
        raise ValueError('k must be greater than 0.')

    if graph_dir is not None:
        graphs_sstable_path = graph_dir
        use_local_storage = True
    if embeddings_dir is not None:
        embeddings_sstable_path = embeddings_dir
        use_local_storage = True

    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else str(repo_name)
    )
    effective_commit = (
        None
        if (base_commit and repo_str.endswith(f'_{base_commit}'))
        else base_commit
    )
    effective_repo = (
        f'{repo_str}_{effective_commit}' if effective_commit else repo_str
    )

    logger.info('Precomputing top-%d similar nodes for repo: %s', k, repo_str)
    graph = graph_utils.get_graph(
        repo_str,
        sstable_path=graphs_sstable_path,
        embeddings_sstable_path=embeddings_sstable_path,
        use_local_storage=use_local_storage,
        base_commit=effective_commit,
    )
    names, texts, embeddings = _extract_nodes_and_embeddings(
        graph,
        repo_str,
        graphs_sstable_path=graphs_sstable_path,
        embeddings_sstable_path=embeddings_sstable_path,
        use_local_storage=use_local_storage,
        base_commit=effective_commit,
    )

    cache_key = (
        effective_repo,
        graphs_sstable_path,
        embeddings_sstable_path,
        use_local_storage,
    )
    if not names:
        logger.warning('No nodes found in graph for repo %s.', repo_str)
        with _LOCK:
            _PRECOMPUTED_GRAPHS[cache_key] = {}
            _PRECOMPUTED_TEXTS[cache_key] = {}
        return {}

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed_embeddings = embeddings / norms

    n_nodes = len(names)
    effective_k = min(k, n_nodes - 1) if n_nodes > 1 else 0

    repo_results: dict[str, list[dict[str, Any]]] = {}
    if effective_k > 0:
        batch_size = chunk_size
        for batch_start in range(0, n_nodes, batch_size):
            batch_end = min(batch_start + batch_size, n_nodes)
            batch_embeddings = normed_embeddings[batch_start:batch_end]
            sim_batch = np.dot(batch_embeddings, normed_embeddings.T)

            for local_i in range(batch_end - batch_start):
                global_i = batch_start + local_i
                node_name = names[global_i]
                sim_batch[local_i, global_i] = -np.inf

                valid_indices = [j for j in range(n_nodes) if j != global_i]
                sorted_indices = sorted(
                    valid_indices,
                    key=lambda j, s=sim_batch, loc_i=local_i: (
                        -float(s[loc_i, j]),
                        names[j],
                    ),
                )
                top_indices = sorted_indices[:effective_k]

                similar_nodes = [
                    {
                        'node_name': names[j],
                        'code': texts[j],
                        'similarity': float(sim_batch[local_i, j]),
                        'embedding': embeddings[j],
                    }
                    for j in top_indices
                ]
                repo_results[node_name] = similar_nodes
    else:
        for node_name in names:
            repo_results[node_name] = []

    with _LOCK:
        _PRECOMPUTED_TEXTS[cache_key] = {
            names[i]: texts[i] for i in range(len(names))
        }
        _PRECOMPUTED_GRAPHS[cache_key] = repo_results
    logger.info(
        'Successfully precomputed top-%d similar nodes for %d nodes in repo %s.',
        k,
        n_nodes,
        repo_str,
    )
    return repo_results


def get_similar_nodes(
    node: Any,
    repo_name: str | bytes | None = None,
    k: int = 10,
    use_memoization: bool = True,
    graph: Any | None = None,
    graphs_sstable_path: str = embedding_utils.DEFAULT_REPO_GRAPHS_SSTABLE,
    embeddings_sstable_path: str = embedding_utils.DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
    use_local_storage: bool = False,
    *,
    graph_dir: str | None = None,
    embeddings_dir: str | None = None,
    base_commit: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieves the top-K similar code nodes within the same repo based on graph embeddings.

    Args:
        node: The target node to find similarities for. Can be a dictionary
            containing 'node_name' (and optionally 'repo_name', 'code', 'text'), a
            string or bytes representing 'node_name' or 'repo_name;node_name', or an
            object with those attributes.
        repo_name: The name of the repository. Required if not provided within `node`.
        k: The number of nearest neighbors to retrieve (default K=10).
        use_memoization: Whether to use in-memory memoization/caching (default True).
        graph: Optional graph object to scope the retrieval to (e.g. an induced subgraph).
        graphs_sstable_path: Path to the repo_graphs SSTable or local graphs dir.
        embeddings_sstable_path: Path to the function embeddings SSTable or local embeddings dir.
        use_local_storage: Whether to read embeddings from local files.
        graph_dir: Optional local graphs directory override.
        embeddings_dir: Optional local embeddings directory override.
        base_commit: Optional base commit hash for snapshot-specific lookup.

    Returns:
        A list of dictionaries representing the most similar nodes, sorted by similarity.
        Each dictionary contains 'node_name', 'code', 'similarity', and 'embedding'.
    """
    if k <= 0:
        raise ValueError('k must be greater than 0.')

    if graph_dir is not None:
        graphs_sstable_path = graph_dir
        use_local_storage = True
    if embeddings_dir is not None:
        embeddings_sstable_path = embeddings_dir
        use_local_storage = True

    if isinstance(repo_name, bytes):
        repo_name = repo_name.decode('utf-8', errors='replace')

    node_name = None
    node_content = None

    if isinstance(node, (str, bytes)):
        node_str = (
            node.decode('utf-8', errors='replace')
            if isinstance(node, bytes)
            else node
        )
        if ';' in node_str:
            parts = node_str.split(';', 1)
            repo_name = repo_name or parts[0]
            node_name = parts[1]
        else:
            node_name = node_str
    elif isinstance(node, dict):
        node_name = node.get('node_name') or node.get('name')
        if isinstance(node_name, bytes):
            node_name = node_name.decode('utf-8', errors='replace')
        repo_name = repo_name or node.get('repo_name')
        if isinstance(repo_name, bytes):
            repo_name = repo_name.decode('utf-8', errors='replace')
        node_content = node.get('code') or node.get('text')
        if isinstance(node_content, bytes):
            node_content = node_content.decode('utf-8', errors='replace')
    elif (
        hasattr(node, 'node_name')
        or hasattr(node, 'code')
        or hasattr(node, 'text')
    ):
        node_name = getattr(node, 'node_name', None)
        if isinstance(node_name, bytes):
            node_name = node_name.decode('utf-8', errors='replace')
        repo_name = repo_name or getattr(node, 'repo_name', None)
        if isinstance(repo_name, bytes):
            repo_name = repo_name.decode('utf-8', errors='replace')
        node_content = getattr(node, 'code', None) or getattr(
            node, 'text', None
        )
        if isinstance(node_content, bytes):
            node_content = node_content.decode('utf-8', errors='replace')

    if node_name and ';' in node_name:
        parts = node_name.split(';', 1)
        repo_name = repo_name or parts[0]
        node_name = parts[1]

    if not repo_name:
        raise ValueError(
            'repo_name must be provided either as an argument or within the node'
            ' object.'
        )

    if not node_name and not node_content:
        raise ValueError(
            'Either node_name or code content must be provided to find similar'
            ' nodes.'
        )

    effective_commit = (
        None
        if (base_commit and str(repo_name).endswith(f'_{base_commit}'))
        else base_commit
    )
    effective_repo = (
        f'{repo_name}_{effective_commit}' if effective_commit else str(repo_name)
    )

    is_custom_graph = graph is not None
    names = None
    texts = None
    embeddings = None

    # 1. Check precomputed graph cache
    if not is_custom_graph and use_memoization:
        precompute_key = (
            effective_repo,
            graphs_sstable_path,
            embeddings_sstable_path,
            use_local_storage,
        )
        with _LOCK:
            has_precomputed = (
                precompute_key in _PRECOMPUTED_GRAPHS
                and node_name is not None
                and node_name in _PRECOMPUTED_GRAPHS[precompute_key]
            )
            cached_list = (
                _PRECOMPUTED_GRAPHS[precompute_key].get(node_name, [])
                if has_precomputed and node_name is not None
                else []
            )
            total_precomputed = (
                len(_PRECOMPUTED_GRAPHS[precompute_key])
                if precompute_key in _PRECOMPUTED_GRAPHS
                else 0
            )
            has_cached_text = (
                precompute_key in _PRECOMPUTED_TEXTS
                and node_name is not None
                and node_name in _PRECOMPUTED_TEXTS[precompute_key]
            )
            cached_text = (
                _PRECOMPUTED_TEXTS[precompute_key].get(node_name)
                if has_cached_text and node_name is not None
                else None
            )

        if has_precomputed:
            is_modified = False
            if node_content:
                if has_cached_text:
                    if node_content != cached_text:
                        is_modified = True
                else:
                    temp_graph = graph_utils.get_graph(
                        repo_name,
                        sstable_path=graphs_sstable_path,
                        embeddings_sstable_path=embeddings_sstable_path,
                        use_local_storage=use_local_storage,
                        base_commit=effective_commit,
                    )
                    names, texts, embeddings = _extract_nodes_and_embeddings(
                        temp_graph,
                        repo_name,
                        graphs_sstable_path=graphs_sstable_path,
                        embeddings_sstable_path=embeddings_sstable_path,
                        use_local_storage=use_local_storage,
                        base_commit=effective_commit,
                    )
                    if node_name and node_name in names:
                        target_index = names.index(node_name)
                        if node_content != texts[target_index]:
                            is_modified = True

            if not is_modified:
                logger.info(
                    'Retrieving similar nodes for %s from precomputed cache.',
                    node_name,
                )
                if (
                    len(cached_list) >= k
                    or len(cached_list) == total_precomputed - 1
                ):
                    return cached_list[:k]
            else:
                logger.info(
                    'Node %s exists in precomputed cache but content was modified.'
                    ' Bypassing precomputed cache.',
                    node_name,
                )

    # 2. Check memoization cache
    cache_key = (
        effective_repo,
        str(node_name),
        k,
        str(node_content),
        graphs_sstable_path,
        embeddings_sstable_path,
        use_local_storage,
    )
    if not is_custom_graph and use_memoization:
        with _LOCK:
            if cache_key in _SIMILARITY_CACHE:
                logger.info(
                    'Retrieving similar nodes for %s from memoization cache.',
                    node_name,
                )
                return _SIMILARITY_CACHE[cache_key]

    # 3. Retrieve graph from graph_utils and extract nodes/embeddings if not already retrieved
    if graph is None:
        graph = graph_utils.get_graph(
            repo_name,
            sstable_path=graphs_sstable_path,
            embeddings_sstable_path=embeddings_sstable_path,
            use_local_storage=use_local_storage,
            base_commit=effective_commit,
        )

    if names is None or texts is None or embeddings is None:
        names, texts, embeddings = _extract_nodes_and_embeddings(
            graph,
            repo_name,
            graphs_sstable_path=graphs_sstable_path,
            embeddings_sstable_path=embeddings_sstable_path,
            use_local_storage=use_local_storage,
            base_commit=effective_commit,
        )

    if not names:
        logger.warning(
            'Graph for repo %s is empty or has no nodes. Returning empty similarity'
            ' list.',
            repo_name,
        )
        return []

    # 4. Obtain target embedding
    target_embedding = None
    target_index = -1

    if node_name and node_name in names:
        target_index = names.index(node_name)
        existing_text = texts[target_index]
        if node_content and node_content != existing_text:
            logger.info(
                'Node %s exists but content was edited. Calling'
                ' embedding_utils.embed.',
                node_name,
            )
            target_embedding = embedding_utils.embed(
                node_content,
                repo_name,
                graphs_sstable_path=graphs_sstable_path,
                embeddings_sstable_path=embeddings_sstable_path,
                use_local_storage=use_local_storage,
                base_commit=effective_commit,
            )
            if target_embedding is not None:
                target_embedding = np.array(target_embedding, dtype=np.float32)
        else:
            target_embedding = embeddings[target_index]
    elif node_content or node_name:
        query_text = node_content or node_name
        logger.info(
            'Node %s not in graph or new content provided. Calling'
            ' embedding_utils.embed.',
            node_name,
        )
        target_embedding = embedding_utils.embed(
            query_text,
            repo_name,
            graphs_sstable_path=graphs_sstable_path,
            embeddings_sstable_path=embeddings_sstable_path,
            use_local_storage=use_local_storage,
            base_commit=effective_commit,
        )
        if target_embedding is not None:
            target_embedding = np.array(target_embedding, dtype=np.float32)

    if target_embedding is None:
        logger.warning(
            'Could not obtain embedding for node %s in repo %s.',
            node_name,
            repo_name,
        )
        return []

    # 5. Compute cosine similarity
    target_norm = np.linalg.norm(target_embedding)
    if target_norm == 0:
        target_norm = 1e-12
    normed_target = target_embedding / target_norm

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed_embeddings = embeddings / norms

    sims = np.dot(normed_embeddings, normed_target.T).flatten()

    if target_index != -1:
        sims[target_index] = -np.inf

    n_nodes = len(names)
    effective_k = min(k, n_nodes - 1 if target_index != -1 else n_nodes)
    if effective_k <= 0:
        return []

    valid_indices = [j for j in range(n_nodes) if j != target_index]
    sorted_indices = sorted(
        valid_indices, key=lambda j: (-float(sims[j]), names[j])
    )
    top_indices = sorted_indices[:effective_k]

    results = [
        {
            'node_name': names[j],
            'code': texts[j],
            'similarity': float(sims[j]),
            'embedding': embeddings[j],
        }
        for j in top_indices
    ]

    if not is_custom_graph and use_memoization:
        with _LOCK:
            _SIMILARITY_CACHE[cache_key] = results

    return results
