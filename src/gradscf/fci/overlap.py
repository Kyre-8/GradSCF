"""Real FCI wavefunction overlap between two orthonormal orbital frames."""
from numbers import Integral
import jax.numpy as jnp
from .cistring import FCISpace
from .hamiltonian import DEFAULT_WORKSPACE, coefficient_array


def _string_overlaps(matrix, strings, occupied):
    if occupied == 0:
        return jnp.ones((1,1), dtype=matrix.dtype)
    indices = jnp.asarray([[p for p in range(matrix.shape[0]) if string & (1 << p)]
                           for string in strings], dtype=jnp.int32)
    minors = matrix[indices[:,None,:,None], indices[None,:,None,:]]
    return jnp.linalg.det(minors)


def overlap(bra, ket, space, *, orbital_overlap=None,
            max_workspace_elements=DEFAULT_WORKSPACE):
    """Return ``<bra|ket>`` including the supplied left/right orbital overlap.

    ``space`` is one shared :class:`FCISpace`; coefficients use its alpha-by-
    beta string layout (or the equivalent flat layout). ``orbital_overlap[p,q]``
    is ``<phi_left[p]|phi_right[q]>``. Each frame is assumed orthonormal at its
    own geometry. The cross matrix need not be symmetric or invertible.
    Both CI tensors and the cross matrix remain differentiable. No state
    normalization, phase choice or state assignment is performed here.

    ``None`` represents identical orbitals and uses a direct coefficient inner
    product, for any supported occupancy. A supplied matrix, including a
    numerical identity matrix, is a differentiable orbital map: this first
    implementation requires at most three occupied orbitals per spin, where
    polynomial determinants have validated first derivatives at singular minors.
    Larger cross-frame overlaps are explicitly unsupported. Inputs are real.

    The capacity guard bounds the largest determinant-minor tensor, string
    transformation or CI intermediate before array conversion. It is not a
    bound on total process memory. No many-body overlap matrix is constructed.
    """
    if not isinstance(space, FCISpace):
        raise TypeError('Expected an FCISpace')
    if (not isinstance(max_workspace_elements, Integral)
            or isinstance(max_workspace_elements, bool) or max_workspace_elements < 1):
        raise ValueError('max_workspace_elements must be a positive integer')
    if orbital_overlap is not None and max(space.nelec) > 3:
        raise NotImplementedError('Cross-frame overlap supports at most three occupied orbitals per spin')
    required = space.size
    if orbital_overlap is not None:
        required = max(required, *(count**2*max(1,occupied**2)
                                   for count,occupied in zip(space.shape,space.nelec)))
    if required > max_workspace_elements:
        raise ValueError('FCI overlap tensor exceeds max_workspace_elements')
    left, right = coefficient_array(bra,space), coefficient_array(ket,space)
    if orbital_overlap is None:
        return jnp.sum(left*right)
    matrix = jnp.asarray(orbital_overlap)
    if matrix.shape != (space.norb,space.norb):
        raise ValueError('orbital_overlap must have shape (norb,norb)')
    if jnp.iscomplexobj(matrix):
        raise NotImplementedError('FCI overlap currently requires real orbital overlaps')
    matrix = matrix.astype(jnp.result_type(matrix,left,right))
    alpha = _string_overlaps(matrix,space.alpha_strings,space.nelec[0])
    beta = (alpha if space.nelec[0] == space.nelec[1] else
            _string_overlaps(matrix,space.beta_strings,space.nelec[1]))
    return jnp.sum(left*(alpha@right@beta.T))
