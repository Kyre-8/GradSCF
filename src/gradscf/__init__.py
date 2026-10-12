"""Core interfaces for GradSCF.

The root exposes domain namespaces only. Numerical operations, configuration
objects and advanced helpers live in their owning modules. Namespaces are lazy
so importing gradscf does not load the electronic-structure stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_NAMESPACE_EXPORTS = {
    "bse",
    "gw",
    "cc",
    "mp",
    "solvers",
    "ci",
    "fci",
    "integrals",
    "gto",
    "scf",
    "dft",
    "ofdft",
    "tdscf",
    "tddft",
    "model",
    "tools",
    "data",
}

__all__ = sorted(_NAMESPACE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _NAMESPACE_EXPORTS:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}; use its owning domain module")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
