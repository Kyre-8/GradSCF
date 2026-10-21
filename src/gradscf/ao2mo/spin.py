"""Frozen-orbital selection and real spin-orbital integral expansion."""
from numbers import Integral

import jax.numpy as jnp

from gradscf.ao2mo.reference import validate_unrestricted_integrals


def unrestricted_frozen_indices(nmo, nocc, frozen=None):
    """A core count/shared list, or (alpha_list, beta_list) for separate spaces."""
    if len(nocc) != 2 or any(not isinstance(n, Integral) or isinstance(n, bool)
                             or not 0 <= n <= nmo for n in nocc):
        raise ValueError("nocc must be (nalpha, nbeta) with integer occupations")
    separate = (isinstance(frozen, (tuple, list)) and len(frozen) == 2
                and all(isinstance(f, (tuple, list)) for f in frozen))
    choices = frozen if separate else (frozen, frozen)
    return tuple(frozen_indices(nmo, n, f) for n, f in zip(nocc, choices))


def spin_orbital_integrals(h1, eri):
    """Dense real chemists' spin integrals; all alpha orbitals precede beta.

    Reference implementation: O((2*nmo)**4) storage. No spin-flip matrix
    elements are inserted. The beta-alpha block is the pair transpose of gab.
    """
    (ha, hb), (gaa, gab, gbb) = validate_unrestricted_integrals(h1, eri)
    n = ha.shape[0]
    dtype = jnp.result_type(ha, hb, gaa, gab, gbb)
    h = jnp.zeros((2*n, 2*n), dtype=dtype).at[:n, :n].set(ha).at[n:, n:].set(hb)
    g = jnp.zeros((2*n,)*4, dtype=dtype)
    g = g.at[:n, :n, :n, :n].set(gaa).at[:n, :n, n:, n:].set(gab)
    g = g.at[n:, n:, :n, :n].set(gab.transpose(2, 3, 0, 1))
    return h, g.at[n:, n:, n:, n:].set(gbb)


def frozen_indices(nmo, nocc, frozen):
    if frozen is None:
        return ()
    if isinstance(frozen, Integral) and not isinstance(frozen, bool):
        if frozen < 0 or frozen > nocc:
            raise ValueError("frozen core count must lie between 0 and nocc")
        return tuple(range(frozen))
    try:
        indices = tuple(frozen)
    except TypeError as error:
        raise ValueError("frozen must be a core count or orbital index list") from error
    if any(
        not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= nmo
        for i in indices
    ):
        raise ValueError("Frozen orbital indices must be integers in [0, nmo)")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate frozen orbital indices")
    return tuple(sorted(int(i) for i in indices))
