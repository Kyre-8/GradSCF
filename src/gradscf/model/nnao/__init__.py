"""Neural contracted bases with model-owned integration and training.

Imports are lazy so workflows can configure JAX before loading numerical code.
MACE remains optional for basis assembly and cache/control tests.
"""
from importlib import import_module

_EXPORTS = {
    'NeuralBasis': ('.basis', 'NeuralBasis'),
    'DirectBasis': ('.basis', 'DirectBasis'),
    'prepare_basis': ('.basis', 'prepare_basis'),
    'prepare_direct_basis': ('.basis', 'prepare_direct_basis'),
    'supported_elements': ('.basis', 'supported_elements'),
    'MACEBasisModel': ('.encoder', 'MACEBasisModel'),
    'build_graph': ('.graph', 'build_graph'),
    'prepare_grimme_basis': ('.grimme', 'prepare_grimme_basis'),
}
_SUBMODULES = ('integrals', 'training', 'workflows')
__all__ = [*_EXPORTS, *_SUBMODULES]


def __getattr__(name):
    if name in _SUBMODULES:
        value = import_module('.'+name, __name__)
    elif name in _EXPORTS:
        module, member = _EXPORTS[name]
        value = getattr(import_module(module, __name__), member)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
