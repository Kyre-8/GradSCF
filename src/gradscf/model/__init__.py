"""Neural-network model subpackage for GradSCF.

Hosts the machine-learned model code:

- ``gradscf.model.neural_xc``: neural XC functional models
- ``gradscf.model.neural_d``: neural dispersion corrections
- ``gradscf.model.nnao``: MACE-conditioned neural-network basis sets (NNAO)

Note: the vendored ``mace_jax`` upstream project ships inside
``gradscf/model/nnao/vendor/mace-jax`` but remains a separate top-level package at
runtime (installed via the ``nnao`` extra); its sources are unchanged.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = ("neural_xc", "neural_d", "nnao")


def __getattr__(name):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_SUBMODULES)
