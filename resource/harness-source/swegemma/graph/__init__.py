"""swegemma.graph module for local repository code graph loading, traversal, and retrieval."""

from swegemma.graph.embedding_utils import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
    DEFAULT_REPO_GRAPHS_SSTABLE,
    FALLBACK_FUNC_EMBEDDINGS_SSTABLE,
    FALLBACK_REPO_GRAPHS_SSTABLE,
    embed,
    is_local_path,
    load_embeddings_from_npz,
    repo_to_file_path,
    save_embeddings_to_npz,
)
from swegemma.graph.graph_utils import (
    CustomMultiDiGraph,
    CustomMultiEdgeView,
    clear_cache_for_repo,
    get_graph,
    get_induced_subgraph,
    get_neighbor,
    load_graph_from_json,
    save_graph_to_json,
)
from swegemma.graph.retrieval_utils import (
    clear_caches,
    get_similar_nodes,
    precompute_top_k_similar_nodes,
)

__all__ = [
    'DEFAULT_EMBEDDING_DIM',
    'DEFAULT_FUNC_EMBEDDINGS_SSTABLE',
    'DEFAULT_REPO_GRAPHS_SSTABLE',
    'FALLBACK_FUNC_EMBEDDINGS_SSTABLE',
    'FALLBACK_REPO_GRAPHS_SSTABLE',
    'CustomMultiDiGraph',
    'CustomMultiEdgeView',
    'clear_cache_for_repo',
    'clear_caches',
    'embed',
    'get_graph',
    'get_induced_subgraph',
    'get_neighbor',
    'get_similar_nodes',
    'is_local_path',
    'load_embeddings_from_npz',
    'load_graph_from_json',
    'precompute_top_k_similar_nodes',
    'repo_to_file_path',
    'save_embeddings_to_npz',
    'save_graph_to_json',
]
