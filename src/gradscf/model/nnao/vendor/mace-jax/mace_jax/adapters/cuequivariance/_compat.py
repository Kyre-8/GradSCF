"""Compatibility patches for cuequivariance runtime behaviour."""

from __future__ import annotations

from functools import wraps


def _patch_guarded_comparison(cls, method_name: str) -> None:
    """Make rich-comparison methods return ``NotImplemented`` for other types."""
    method = getattr(cls, method_name, None)
    if method is None or getattr(method, '_mace_jax_compat_patch', False):
        return

    @wraps(method)
    def wrapped(self, value):
        if not isinstance(value, cls):
            return NotImplemented
        return method(self, value)

    wrapped._mace_jax_compat_patch = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)


def _patch_type_checked_comparisons() -> None:
    """Patch cuequivariance comparison helpers that assert on foreign types."""
    try:
        from cuequivariance.group_theory.equivariant_polynomial import (
            EquivariantPolynomial,
        )
        from cuequivariance.segmented_polynomials.operation import Operation
        from cuequivariance.segmented_polynomials.segmented_polynomial import (
            SegmentedPolynomial,
        )
        from cuequivariance.segmented_polynomials.segmented_tensor_product import (
            SegmentedTensorProduct,
        )
    except Exception:
        return

    for cls in (
        Operation,
        SegmentedPolynomial,
        SegmentedTensorProduct,
        EquivariantPolynomial,
    ):
        _patch_guarded_comparison(cls, '__eq__')
        _patch_guarded_comparison(cls, '__lt__')


_patch_type_checked_comparisons()
