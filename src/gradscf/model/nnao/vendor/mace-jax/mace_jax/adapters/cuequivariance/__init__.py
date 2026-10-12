"""Cue-equivariant adapters for Flax."""

from . import _compat as _compat  # noqa: F401
from .fully_connected_tensor_product import FullyConnectedTensorProduct
from .linear import Linear
from .symmetric_contraction import SymmetricContraction
from .tensor_product import TensorProduct

__all__ = [
    'TensorProduct',
    'FullyConnectedTensorProduct',
    'Linear',
    'SymmetricContraction',
]
