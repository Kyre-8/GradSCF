"""Complex, full-matrix mode-A qsGW on a complete periodic source k mesh.

All Hartree, exchange and screening terms use the same finite Coulomb-factor
Hamiltonian. The reference MO frame stays fixed; iteration rotates both ends
of each momentum-linked product vertex. This is an eager forward method.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ...solvers.nonlinear import NonlinearConfig, iterate_eager
from ..types import GWResult
from .source import (build_correlation, correlation_at, hartree_exchange,
                     require_eager, validate_model)


def effective_fock(model, energy, rotation, *, nw=32, eta=1e-3, linear_config=None):
    """Return the Hermitian mode-A effective Fock in the CURRENT eigenframe."""
    cache = build_correlation(model, energy, rotation=rotation, nw=nw, eta=eta,
                              linear_config=linear_config)
    u = jnp.asarray(rotation)
    occupied = u[:, :, :model.nocc]
    density = occupied@occupied.conj().swapaxes(-1, -2)
    hartree, exchange = hartree_exchange(model, density)
    static = u.conj().swapaxes(-1, -2)@(model.hcore+hartree+exchange)@u
    matrices = []
    for k in range(len(energy)):
        # Each endpoint reuses the module-level correlation executable. An
        # outer lax.map closure would capture a new full cache each iteration.
        rows = []
        for p in range(energy.shape[-1]):
            sigma = correlation_at(cache, k, energy[k, p])
            rows.append(.5*(sigma[p]+sigma[:, p].conj()))
        rows = jnp.stack(rows)
        mode_a = (rows+rows.conj().T)/2
        value = static[k]+mode_a
        matrices.append((value+value.conj().T)/2)
    return jnp.stack(matrices)


def qsgw_cd_kpoints(model, *, nw=32, eta=1e-3, max_iter=40, tol=1e-6,
                     tol_density=1e-6, damping=.2, linear_config=None,
                     diis_space=6, diis_start=2, callback=None, differentiation=None):
    """Iterate the full Hermitian quasiparticle effective Hamiltonian.

    ``tol`` bounds the unmixed energy update AND current-frame F-diag(E)
    residual in Ha; ``tol_density`` bounds one-spin density changes. Fock
    damping changes the trajectory, never the residual certificate. Shared
    eager DIIS mixes full Hamiltonians in the fixed reference MO frame,
    including complex matrix elements; diis_space=0 selects a plain map.
    callback(iteration,reference_frame_fock,residual_norm) is observational.
    Returned energies/orbitals are the state actually evaluated and checked.
    Degenerate levels retain their full matrix self-energy; no inverse-gap
    orbital derivatives are supplied. There is no periodic outer AD/JIT.
    """
    validate_model(model)
    if differentiation is not None:
        raise NotImplementedError('Periodic qsGW is eager forward only; outer AD is unsupported')
    if (not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1
            or not np.isfinite(tol) or tol <= 0
            or not np.isfinite(tol_density) or tol_density <= 0
            or not np.isfinite(damping) or not 0 <= damping < 1):
        raise ValueError('Use positive iteration/tolerances and damping in [0,1)')
    require_eager()
    if not isinstance(diis_space, int) or isinstance(diis_space, bool) or diis_space < 0:
        raise ValueError('diis_space must be a nonnegative integer')
    nk, nmo = model.energy.shape
    shape = (nk, nmo, nmo)
    size = nk*nmo*nmo
    pack = lambda matrix: jnp.concatenate((matrix.real.reshape(-1), matrix.imag.reshape(-1)))
    def unpack(value):
        matrix = value[:size].reshape(shape)+1j*value[size:].reshape(shape)
        return .5*(matrix+matrix.conj().swapaxes(-1, -2))
    def valid(value):
        matrix = np.asarray(value[:size]).reshape(shape)+1j*np.asarray(value[size:]).reshape(shape)
        energy = np.linalg.eigvalsh(.5*(matrix+matrix.conj().swapaxes(-1, -2)))
        return np.min(energy[:, model.nocc:])-np.max(energy[:, :model.nocc]) > 0
    last = {}
    def physical(value):
        energy, rotation = jnp.linalg.eigh(unpack(value))
        fock = effective_fock(model, energy, rotation, nw=nw, eta=eta,
                              linear_config=linear_config)
        diagonal = jax.vmap(jnp.diag)(energy)
        d_fock = float(jnp.max(jnp.abs(fock-diagonal)))
        new_energy, local_rotation = jnp.linalg.eigh(fock)
        proposed_rotation = rotation@local_rotation
        density = rotation[:, :, :model.nocc]@rotation[:, :, :model.nocc].conj().swapaxes(-1, -2)
        new_density = proposed_rotation[:, :, :model.nocc]@proposed_rotation[:, :, :model.nocc].conj().swapaxes(-1, -2)
        d_energy = float(jnp.max(jnp.abs(new_energy-energy)))
        d_density = float(jnp.max(jnp.abs(new_density-density)))
        last.update(energy=energy, rotation=rotation, fock=fock, diagonal=diagonal,
                    errors=(d_fock, d_energy, d_density))
        return pack(rotation@fock@rotation.conj().swapaxes(-1, -2))
    def check(state, mapped, norm):
        d_fock, d_energy, d_density = last['errors']
        return d_fock < tol and d_energy < tol and d_density < tol_density
    observer = (None if callback is None else
                lambda iteration, state, norm: callback(iteration, unpack(state), norm))
    solved = iterate_eager(physical, pack(jnp.asarray(model.reference_fock)),
        valid_state=valid, check=check, callback=observer,
        config=NonlinearConfig(maxiter=max_iter, residual_tol=tol, energy_tol=tol,
            damping=damping, diis_space=max(1, diis_space), diis_start=diis_start))
    if not solved.converged:
        d_fock, d_energy, d_density = last['errors']
        raise ArithmeticError(f'Periodic qsGW did not converge in {solved.iterations} iterations '
            f'(unmixed Fock {d_fock:.3e} Ha, energy {d_energy:.3e} Ha, density {d_density:.3e}); '
            'no fallback result is returned')
    energy, rotation = last['energy'], last['rotation']
    computed = jnp.ones_like(energy, dtype=bool)
    return GWResult(mo_energy=energy, mo_coeff=model.coeff@rotation,
        converged=True, converged_mask=computed, qp_computed_mask=computed,
        nw=int(nw), qp_residual=jnp.max(jnp.abs(last['fock']-last['diagonal']), axis=-1),
        screening_energy=energy, g_orbitals=tuple(range(nmo)),
        screening_occupied=tuple(range(model.nocc)),
        screening_virtual=tuple(range(model.nocc, nmo)),
        method='periodic_qsgw_full_cd_matrix_mode_a')


__all__ = ['effective_fock', 'qsgw_cd_kpoints']
