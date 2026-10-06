"""Warning configuration and suppression utilities for SWE-gemma."""

from __future__ import annotations

import warnings

__all__ = [
    '_configure_warning_filters',
    '_suppress_warnings',
]


def _suppress_warnings() -> None:
    """Suppress noisy third-party authentication and framework warnings during evaluation.

    Encapsulates warning suppression into runtime evaluation execution to isolate
    module-level import side-effects.
    """
    warnings.filterwarnings('ignore', module=r'authlib\.')
    warnings.filterwarnings(
        'ignore', message=r'.*PLUGGABLE_AUTH.*', category=UserWarning
    )
    warnings.filterwarnings(
        'ignore', message=r'.*Pydantic serializer warnings.*', category=UserWarning
    )
    warnings.filterwarnings(
        'ignore',
        message=r'.*PydanticSerializationUnexpectedValue.*',
        category=UserWarning,
    )
    warnings.filterwarnings(
        'ignore', module=r'pydantic(\..*)?', category=UserWarning
    )


# Alias for backward compatibility / explicit naming
_configure_warning_filters = _suppress_warnings
