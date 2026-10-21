"""Restricted CC blocks from a real orthonormal MO Hamiltonian."""

from typing import NamedTuple
import jax.numpy as jnp
from gradscf.ao2mo import validate_integrals, frozen_indices


class CCIntegrals(NamedTuple):
    fock: object
    mo_energy: object
    oooo: object
    ovoo: object
    ovov: object
    oovv: object
    ovvo: object
    ovvv: object
    vvvv: object
    reference_energy: object

    def get_ovvv(self):
        return self.ovvv

    @property
    def nocc(self):
        return self.oooo.shape[0]

    @property
    def nvir(self):
        return self.fock.shape[0] - self.nocc


def prepare_integrals(h1, eri, *, nocc, nuclear_repulsion=0.0, frozen=None):
    h, g = validate_integrals(h1, eri)
    if not isinstance(nocc, int) or not 0 <= nocc <= h.shape[0]:
        raise ValueError("Invalid restricted nocc")
    frozen = frozen_indices(h.shape[0], nocc, frozen)
    # All occupied electrons contribute to F and E_HF, including frozen cores.
    f = (
        h
        + 2 * jnp.einsum("pqii->pq", g[:, :, :nocc, :nocc])
        - jnp.einsum("piiq->pq", g[:, :nocc, :nocc, :])
    )
    e = jnp.trace((h + f)[:nocc, :nocc]) + nuclear_repulsion
    occ = [i for i in range(nocc) if i not in frozen]
    vir = [i for i in range(nocc, h.shape[0]) if i not in frozen]
    active = jnp.asarray(occ + vir, dtype=jnp.int32)
    f = f[jnp.ix_(active, active)]
    o = jnp.asarray(occ, dtype=jnp.int32)
    v = jnp.asarray(vir, dtype=jnp.int32)
    # Extract only the CC blocks; a complete active-space ERI copy is unnecessary.
    return CCIntegrals(
        f,
        jnp.diag(f),
        g[jnp.ix_(o, o, o, o)],
        g[jnp.ix_(o, v, o, o)],
        g[jnp.ix_(o, v, o, v)],
        g[jnp.ix_(o, o, v, v)],
        g[jnp.ix_(o, v, v, o)],
        g[jnp.ix_(o, v, v, v)],
        g[jnp.ix_(v, v, v, v)],
        e,
    )


def denominators(ints):
    n = ints.nocc
    d = ints.mo_energy[:n, None] - ints.mo_energy[None, n:]
    return d, d[:, None, :, None] + d[None, :, None, :]
