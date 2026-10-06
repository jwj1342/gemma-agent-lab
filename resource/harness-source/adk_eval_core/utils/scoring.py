"""Scoring and evaluation metric adapter utilities backed by scikit-learn."""

from __future__ import annotations

import logging
from typing import Any

from adk_eval_core.errors import InvalidScorerError, MetricResolutionError

logger = logging.getLogger(__name__)


def make_scorer(*args: Any, **kwargs: Any) -> Any:
    """Delegate to sklearn.metrics.make_scorer."""
    try:
        from sklearn.metrics import make_scorer as _make_scorer

        return _make_scorer(*args, **kwargs)
    except ImportError as e:
        raise MetricResolutionError("scikit-learn is required for make_scorer") from e


def resolve_scorer(metric: str | Any) -> Any:
    """Resolve a metric to a scikit-learn scorer.

    Args:
        metric: Either a scorer name string (e.g. ``"roc_auc"``,
            ``"neg_mean_squared_error"``) or a ``_Scorer`` object
            returned by :func:`sklearn.metrics.make_scorer`.

    Returns:
        A scikit-learn ``_Scorer`` with ``._score_func``, ``._sign``,
        and ``._kwargs`` attributes.

    Raises:
        ValueError: If *metric* is a string that is not registered in
            scikit-learn's scorer registry.
    """
    if isinstance(metric, str):
        try:
            from sklearn.metrics import get_scorer as _get_scorer

            return _get_scorer(metric)
        except ImportError as e:
            raise MetricResolutionError(
                "scikit-learn is required for resolve_scorer with metric names",
                metric_name=metric,
            ) from e
    return metric


def scorer_direction(scorer: Any) -> bool:
    """Return ``True`` if higher scores are better for *scorer*."""
    return getattr(scorer, "_sign", 1) == 1


def scorer_name(scorer: Any, default: str = "") -> str:
    """Return a human-readable name for the scorer.

    If *default* is provided it is returned as-is. Otherwise the
    name is derived from the underlying score function.
    """
    if default:
        return default
    fn = getattr(scorer, "_score_func", None)
    if fn is not None:
        return getattr(fn, "__name__", "unknown")
    return "unknown"


def score_arrays(
    scorer: Any,
    y_true: Any,
    y_pred: Any,
) -> float:
    """Score predictions using a scorer's underlying function.

    Handles single-target auto-squeeze: if *y_true* and *y_pred* are
    2-D arrays with a single column they are flattened to 1-D before
    scoring, since most scikit-learn metrics expect 1-D inputs for
    single-target problems.
    """
    import numpy as np

    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    # If scorer is a string name, resolve it
    if isinstance(scorer, str):
        scorer = resolve_scorer(scorer)

    if hasattr(scorer, "_score_func"):
        kwargs = getattr(scorer, "_kwargs", {}) or {}
        sign = getattr(scorer, "_sign", 1)
        val: Any = scorer._score_func(y_true, y_pred, **kwargs)
        return float(sign * val)
    elif callable(scorer):
        val: Any = scorer(y_true, y_pred)
        return float(val)
    else:
        raise InvalidScorerError(f"Invalid scorer type: {type(scorer)}")


__all__ = [
    "make_scorer",
    "resolve_scorer",
    "score_arrays",
    "scorer_direction",
    "scorer_name",
]
