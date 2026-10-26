"""Model-owned integral execution using gradscf.integrals native kernels."""
from importlib import import_module

_EXPORTS = {'prepare_native_context': '.context',
            'physical_inputs': '.fixed_exponents',
            'native_projection': '.variable_exponents'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
