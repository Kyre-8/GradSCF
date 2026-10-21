"""Restricted CC blocks from a real orthonormal MO Hamiltonian."""

from typing import NamedTuple
from numbers import Integral
import jax
import jax.numpy as jnp
from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, array_storage_bytes,
                             resolve_memory_mb, validate_reserved_bytes, workspace_bytes)
from gradscf.ao2mo import validate_integrals, frozen_indices, transform_eri_blocks
from gradscf.ao2mo.blocks import _array_source


_BLOCK_SPACES = ((0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 0, 1),
                 (0, 0, 1, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 1, 1, 1))


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


def from_orbitals(coefficients, fock, nocc, reference_energy, *, frozen=None,
                  eri=None, eri_pair_matrix=None, df_factors=None, direct_basis=None,
                  max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Prepare seven restricted CC blocks from a supplied real orbital frame.

    ``fock`` is the complete MO-frame Fock before freezing; ``reference_energy``
    includes all HF electrons and nuclear repulsion. Frozen orbitals are removed
    only from the correlation space. No SCF, J/K, symmetrization or orbital
    rotation is performed. Native integrals promote Fock/energy to float64;
    array inputs preserve their supplied dtypes and raw differentiation chart.

    One AO2MO batch shares the original decimal-MB budget with the known inputs,
    selected Fock/energy and caller ``reserved_bytes``. Bare vvvv remains resident.
    Amplitudes, solver workspaces and complete AD storage remain caller-owned.
    """
    max_memory = resolve_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    c, f, energy = map(jnp.asarray, (coefficients, fock, reference_energy))
    if c.ndim != 2 or f.shape != (c.shape[1], c.shape[1]) or energy.ndim != 0:
        raise ValueError('Supply AO/MO coefficients, a full MO Fock and scalar reference_energy')
    if any(jnp.iscomplexobj(x) for x in (c, f, energy)):
        raise NotImplementedError('Restricted CC requires real orbitals, Fock and reference energy')
    if any(not jnp.issubdtype(x.dtype, jnp.floating) for x in (c, f, energy)):
        raise ValueError('Restricted CC inputs must have floating-point dtype')
    if isinstance(nocc, bool) or not isinstance(nocc, Integral) or not 0 <= nocc <= c.shape[1]:
        raise ValueError('Invalid restricted nocc')
    options = dict(eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
                   direct_basis=direct_basis)
    if sum(x is not None for x in options.values()) != 1:
        raise ValueError('Supply exactly one AO integral representation')
    source = None
    if direct_basis is None:
        keyword = next(key for key, value in options.items() if value is not None)
        _, source = _array_source(c.shape[0], eri, eri_pair_matrix, df_factors)
        if jnp.iscomplexobj(source):
            raise NotImplementedError('Restricted CC requires real AO integrals')
        if not jnp.issubdtype(source.dtype, jnp.floating):
            raise ValueError('Restricted CC AO integrals must have floating-point dtype')
        options = {keyword: source}
    frozen = frozen_indices(c.shape[1], nocc, frozen)
    occupied = tuple(i for i in range(nocc) if i not in frozen)
    virtual = tuple(i for i in range(nocc, c.shape[1]) if i not in frozen)
    nactive = len(occupied)+len(virtual)
    dtype = jnp.dtype(jnp.float64) if direct_basis is not None else f.dtype
    # Check known gather/output storage before creating active-space copies.
    planned = (jax.ShapeDtypeStruct((c.shape[0], nactive), c.dtype),
               jax.ShapeDtypeStruct((nactive, nactive), dtype),
               jax.ShapeDtypeStruct((nactive,), dtype))
    promotion = nactive*nactive*f.dtype.itemsize if f.dtype != dtype else 0
    if direct_basis is not None and energy.dtype != dtype:
        promotion += dtype.itemsize
    device = getattr(c, 'device', None)
    device = device() if callable(device) else device
    if device is None:
        device = getattr(jax.config, 'jax_default_device', None)
    workspace_bytes(max_memory, device=device, resident_bytes=reserved_bytes+
        array_storage_bytes(c, f, energy, source, *planned)+8*nactive+promotion)
    o, v = (jnp.asarray(indices, dtype=jnp.int32) for indices in (occupied, virtual))
    active = jnp.concatenate((o, v))
    selected_fock = f[jnp.ix_(active, active)].astype(dtype)
    energies = jnp.diag(selected_fock)
    selected_energy = energy.astype(jnp.float64) if direct_basis is not None else energy
    resident = array_storage_bytes(c, f, energy, selected_fock, energies, selected_energy, o, v, active)
    blocks = transform_eri_blocks((c[:, o], c[:, v]), _BLOCK_SPACES, **options,
        max_memory=max_memory, reserved_bytes=reserved_bytes+resident)
    return CCIntegrals(selected_fock, energies, *blocks, selected_energy)


def denominators(ints):
    n = ints.nocc
    d = ints.mo_energy[:n, None] - ints.mo_energy[None, n:]
    return d, d[:, None, :, None] + d[None, :, None, :]
