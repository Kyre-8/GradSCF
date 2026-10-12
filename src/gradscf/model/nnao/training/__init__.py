"""NNAO-specific training, independent of the Neural XC trainer."""
from importlib import import_module

_EXPORTS = {'BasisRHF': ('.objective', 'BasisRHF'),
            'NativeJointEngine': ('.native', 'NativeJointEngine'),
            'JointAdam': ('.native', 'JointAdam'),
            'run_training': ('.trainer', 'run_training')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, member = _EXPORTS[name]
    value = getattr(import_module(module, __name__), member)
    globals()[name] = value
    return value
