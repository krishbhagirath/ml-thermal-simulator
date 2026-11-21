"""Whitelist-only unpickler for the untrusted UCR dataset pickles.

`pickle.load` can execute arbitrary code. These files only need numpy arrays inside a dict,
so any other global reference is refused.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("builtins", "dict"),
        ("builtins", "list"),
        ("collections", "OrderedDict"),
    }
)


class UnsafePickleError(pickle.UnpicklingError):
    """Raised when a pickle references a global outside the whitelist."""


class _WhitelistUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise UnsafePickleError(f"refusing to load global {module}.{name}")


def load_pickle(path: str | Path) -> Any:
    """Load a pickle allowing only numpy-array reconstruction and plain containers."""
    with open(path, "rb") as f:
        return _WhitelistUnpickler(f).load()


def loads_pickle(data: bytes) -> Any:
    import io

    return _WhitelistUnpickler(io.BytesIO(data)).load()
