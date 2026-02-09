import os
from typing import Optional, Dict

_cache: Dict[str, Optional[str]] = {}


def get(name: str) -> Optional[str]:
    """Return the cached value for name. None if not yet ensured or env var was absent."""
    return _cache.get(name)


def ensure(name: str) -> str:
    """Fetch env var, cache it, raise EnvironmentError if unset/empty."""
    cached = _cache.get(name)
    if cached is not None:
        return cached
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Environment variable '{name}' is not set."
        )
    _cache[name] = value
    return value
