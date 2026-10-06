"""Embedding utilities for graph embedding lookup from local storage."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from typing import Any

import cachetools
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 256
DEFAULT_REPO_GRAPHS_SSTABLE = '/cns/iz-d/home/research-graph/public/ttl=4y/python_repos/gh_kaggle_v12/repo_graphs@351'
FALLBACK_REPO_GRAPHS_SSTABLE = '/cns/iz-d/home/research-graph-nonconf/public/ttl=4y/g4swe/gh_kaggle_v12/repo_graphs@351'
DEFAULT_FUNC_EMBEDDINGS_SSTABLE = '/cns/iz-d/home/research-graph/public/ttl=4y/python_repos/gh_kaggle_v11/final/github_kaggle_func_node_embeddings.sst@1392'
FALLBACK_FUNC_EMBEDDINGS_SSTABLE = '/cns/iz-d/home/research-graph-nonconf/public/ttl=4y/g4swe/gh_kaggle_v12/final/github_kaggle_func_node_embeddings.sst@1392'

_LOCK = threading.Lock()
_ACTIVE_STATE: tuple[Any, ...] | None = None
_ACTIVE_REPO: str | None = None
_CACHE: cachetools.LRUCache = cachetools.LRUCache(maxsize=50000)
_REPO_CACHES: dict[str, dict[str, np.ndarray]] = {}


def is_local_path(path: str | None) -> bool:
    """Returns True if the path targets local storage rather than CNS or SSTable."""
    if not path:
        return False
    if any(path.endswith(ext) for ext in ('.json', '.npz')):
        return True
    if any(d in path for d in ('data/graphs', 'data/embeddings')):
        return True
    return os.path.exists(path)


def repo_to_file_path(base_dir: str, repo_name: str | bytes, ext: str) -> str:
    """Converts a repository name (e.g. 'owner/repo') into a local filesystem path."""
    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else str(repo_name)
    )
    if not ext.startswith('.'):
        ext = '.' + ext
    return os.path.join(base_dir, f'{repo_str}{ext}')


def resolve_local_file_path(
    base_dir: str,
    repo_name: str | bytes,
    ext: str,
    base_commit: str | None = None,
) -> str:
    """Resolves a local file path across nested and flat/snapshot filename conventions."""
    if base_dir.endswith(ext):
        return base_dir
    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else str(repo_name)
    )
    if not ext.startswith('.'):
        ext = '.' + ext

    effective_commit = (
        None
        if (base_commit and repo_str.endswith(f'_{base_commit}'))
        else base_commit
    )
    repo_short = repo_str.rsplit('/', maxsplit=1)[-1]
    repo_slug = repo_str.replace('/', '_')

    candidates: list[str] = []
    if effective_commit:
        candidates.extend([
            os.path.join(base_dir, f'{repo_short}_{effective_commit}{ext}'),
            os.path.join(base_dir, f'{repo_slug}_{effective_commit}{ext}'),
            repo_to_file_path(base_dir, f'{repo_str}_{effective_commit}', ext),
        ])
    candidates.extend([
        repo_to_file_path(base_dir, repo_str, ext),
        os.path.join(base_dir, f'{repo_slug}{ext}'),
        os.path.join(base_dir, f'{repo_short}{ext}'),
    ])
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


def clear_caches(repo_name: str | bytes | None = None) -> None:
    """Clears the in-memory embedding cache for a specific repo or all repos."""
    global _ACTIVE_STATE, _ACTIVE_REPO  # noqa: PLW0603
    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else repo_name
    )
    with _LOCK:
        if repo_str is not None:
            if repo_str == _ACTIVE_REPO or (
                _ACTIVE_REPO and _ACTIVE_REPO.startswith(f'{repo_str}_')
            ):
                _CACHE.clear()
                _ACTIVE_STATE = None
                _ACTIVE_REPO = None
            for k in list(_REPO_CACHES.keys()):
                if k == repo_str or k.startswith(f'{repo_str}_'):
                    _REPO_CACHES.pop(k, None)
        else:
            _CACHE.clear()
            _ACTIVE_STATE = None
            _ACTIVE_REPO = None
            _REPO_CACHES.clear()


def save_embeddings_to_npz(
    repo_name: str | bytes,
    node_embeddings: dict[str | bytes, Any],
    output_dir: str = 'data/embeddings',
) -> str:
    """Saves a dictionary of node embeddings to a compressed NumPy (.npz) file.

    Args:
        repo_name: The name of the repository (e.g. '0913ktg/vits_korean_speaker').
        node_embeddings: Dictionary mapping node names/texts to float32 NumPy arrays.
        output_dir: Base directory where embeddings are locally saved.

    Returns:
        The full path to the written .npz file.
    """
    if not repo_name or node_embeddings is None:
        raise ValueError(
            'repo_name and node_embeddings cannot be empty/None when saving'
            ' embeddings.'
        )
    repo_str = (
        repo_name.decode('utf-8', errors='replace')
        if isinstance(repo_name, bytes)
        else str(repo_name)
    )
    file_path = repo_to_file_path(output_dir, repo_str, '.npz')
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    target_dim = next(
        (
            len(v)
            for v in node_embeddings.values()
            if v is not None and len(v) > 0
        ),
        DEFAULT_EMBEDDING_DIM,
    )

    cleaned_embeddings: dict[str, np.ndarray] = {}
    for k, v in node_embeddings.items():
        key_str = (
            k.decode('utf-8', errors='replace') if isinstance(k, bytes) else str(k)
        )
        if v is not None:
            cleaned_embeddings[key_str] = np.array(v, dtype=np.float32)
        else:
            cleaned_embeddings[key_str] = np.zeros(target_dim, dtype=np.float32)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **cleaned_embeddings)
    buffer.seek(0)

    with open(file_path, 'wb') as f:
        f.write(buffer.read())

    clear_caches(repo_str)
    logger.info(
        'Saved %d embeddings for repo %s to %s using NumPy v%s.',
        len(cleaned_embeddings),
        repo_str,
        file_path,
        np.__version__,
    )
    return file_path


def load_embeddings_from_npz(file_path: str) -> dict[str, np.ndarray]:
    """Loads node embeddings from a compressed NumPy (.npz) file.

    Args:
        file_path: Path to the .npz archive.

    Returns:
        Dictionary mapping node names/texts to float32 NumPy arrays.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Embeddings file not found at: {file_path}')
    with open(file_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    try:
        with np.load(buffer, allow_pickle=False) as data:
            result = {
                str(k): np.array(data[k], dtype=np.float32) for k in data.files
            }
    except ValueError as exc:
        raise ValueError(
            f'Failed to load pure float32 numerical arrays from {file_path}'
            ' (allow_pickle=False). Archive may contain unsupported object types or'
            f' corrupted headers: {exc}'
        ) from exc
    return result


def embed(
    node_content: str | bytes | None,
    repo_name: str | bytes | None,
    graphs_sstable_path: str = DEFAULT_REPO_GRAPHS_SSTABLE,
    embeddings_sstable_path: str = DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
    use_local_storage: bool = False,
    *,
    embeddings_dir: str | None = None,
    graph_dir: str | None = None,
    base_commit: str | None = None,
) -> np.ndarray | None:
    """Returns the numerical single-precision embedding vector for the given code content/node.

    Utilizes an in-memory bounded LRU cache per graph. Ensures only the active
    graph's data is loaded into memory at any given time.

    Args:
        node_content: The node name (e.g. 'commons.shift_1d') or code content string.
        repo_name: The repository name (e.g. '0913ktg/vits_korean_speaker').
        graphs_sstable_path: Path to the repo_graphs SSTable or local graphs path.
        embeddings_sstable_path: Path to the function embeddings SSTable or local NPZ.
        use_local_storage: Whether to read embeddings from local files.
        embeddings_dir: Optional local embeddings directory override.
        graph_dir: Optional local graphs directory override.
        base_commit: Optional base commit hash for snapshot-specific lookup.

    Returns:
        A numpy array of float32, or None if not found.
    """
    if not node_content or not repo_name:
        return None

    if isinstance(node_content, bytes):
        node_content = node_content.decode('utf-8', errors='replace')
    if isinstance(repo_name, bytes):
        repo_name = repo_name.decode('utf-8', errors='replace')

    if embeddings_dir is not None:
        embeddings_sstable_path = embeddings_dir
        use_local_storage = True
    if graph_dir is not None:
        graphs_sstable_path = graph_dir
        use_local_storage = True

    if node_content.startswith(f'{repo_name};'):
        node_content = node_content[len(f'{repo_name};') :]

    effective_commit = (
        None
        if (base_commit and repo_name.endswith(f'_{base_commit}'))
        else base_commit
    )
    if effective_commit and node_content.startswith(
        f'{repo_name}_{effective_commit};'
    ):
        node_content = node_content[len(f'{repo_name}_{effective_commit};') :]

    effective_repo = (
        f'{repo_name}_{effective_commit}' if effective_commit else repo_name
    )

    global _ACTIVE_STATE, _ACTIVE_REPO  # noqa: PLW0603

    with _LOCK:
        current_state = (
            effective_repo,
            graphs_sstable_path,
            embeddings_sstable_path,
            'local',
        )

        if current_state != _ACTIVE_STATE:
            _CACHE.clear()
            _ACTIVE_STATE = None
            _ACTIVE_REPO = None

            if embeddings_sstable_path.endswith('.npz'):
                local_path = embeddings_sstable_path
            else:
                base_emb_dir = (
                    embeddings_sstable_path
                    if (
                        (
                            use_local_storage
                            and embeddings_sstable_path
                            not in (
                                DEFAULT_FUNC_EMBEDDINGS_SSTABLE,
                                FALLBACK_FUNC_EMBEDDINGS_SSTABLE,
                            )
                        )
                        or is_local_path(embeddings_sstable_path)
                    )
                    else 'data/embeddings'
                )
                local_path = resolve_local_file_path(
                    base_emb_dir, repo_name, '.npz', base_commit=effective_commit
                )

            if not os.path.exists(local_path):
                logger.warning('Local embeddings NPZ file not found at %s', local_path)
                _ACTIVE_STATE = current_state
                _ACTIVE_REPO = effective_repo
                _REPO_CACHES[effective_repo] = {}
            else:
                try:
                    loaded_embs = load_embeddings_from_npz(local_path)
                    repo_cache: dict[str, np.ndarray] = {}
                    for k, v in loaded_embs.items():
                        _CACHE[k] = v
                        repo_cache[k] = v
                        if ';' in k:
                            bare_k = k.split(';', 1)[1]
                            if bare_k not in _CACHE:
                                _CACHE[bare_k] = v
                                repo_cache[bare_k] = v

                    # Also index node texts from local graph JSON if available (matching CNS mode)
                    base_graph_dir = (
                        graphs_sstable_path
                        if (
                            (
                                use_local_storage
                                and graphs_sstable_path
                                not in (
                                    DEFAULT_REPO_GRAPHS_SSTABLE,
                                    FALLBACK_REPO_GRAPHS_SSTABLE,
                                )
                            )
                            or is_local_path(graphs_sstable_path)
                        )
                        else os.path.join(
                            os.path.dirname(os.path.dirname(local_path)), 'graphs'
                        )
                    )
                    local_graph_path = resolve_local_file_path(
                        base_graph_dir,
                        repo_name,
                        '.json',
                        base_commit=effective_commit,
                    )
                    if os.path.exists(local_graph_path):
                        try:
                            with open(local_graph_path, encoding='utf-8') as gf:
                                gdata = json.load(gf)
                            for n_obj in gdata.get('nodes', []):
                                if isinstance(n_obj, dict):
                                    nid = str(
                                        n_obj.get('name') or n_obj.get('id') or ''
                                    )
                                    ntext = n_obj.get('text') or n_obj.get('code')
                                    if (
                                        nid
                                        and ntext
                                        and isinstance(ntext, str)
                                        and nid in repo_cache
                                    ):
                                        _CACHE[ntext] = repo_cache[nid]
                                        repo_cache[ntext] = repo_cache[nid]
                        except Exception:
                            pass

                    _ACTIVE_STATE = current_state
                    _ACTIVE_REPO = effective_repo
                    _REPO_CACHES[effective_repo] = repo_cache
                except Exception as e:
                    logger.warning(
                        'Failed to load local embeddings from %s: %s',
                        local_path,
                        e,
                        exc_info=True,
                    )
                    _ACTIVE_STATE = current_state
                    _ACTIVE_REPO = effective_repo
                    _REPO_CACHES[effective_repo] = {}

        if node_content not in _CACHE:
            _CACHE[node_content] = None

        return _CACHE.get(node_content)
