"""Quasiparticle self-consistent GW (qsGW) -- Stage 4 (molecular, restricted).

qsGW replaces the dynamical self-energy by a static, Hermitian effective
potential and iterates orbitals + energies to self-consistency:

    v^qsGW_mn = (Re Sigma^c_mn(e_m) + Re Sigma^c_mn(e_n))/2  [mode A]
    F^qsGW     = h + J[rho] + Sigma^x + v^qsGW     [effective Fock]
    (C, e)     = eig(F^qsGW)  ->  rho -> iterate

Unlike G0W0/evGW there is no Dyson root search: the quasiparticle
energies are the eigenvalues of the effective Fock, and the whole
mean-field potential v^mf (v_xc or -K) is replaced by Sigma^x + v^qsGW.

The off-diagonal self-energy is evaluated with the contour-deformation
kernels (:func:`gradscf.gw.self_energy.sigma_imag_matrix` and
:func:`gradscf.gw.self_energy.sigma_residue_matrix`). The endpoint average
is the mode-A mapping of Kotani et al., PRB 76, 165106 (2007), Eq. (10).
This driver supports real molecular orbitals; Re is the Hermitian part
of the self-energy (elementwise real part for these symmetric matrices).

References
----------
- S. V. Faleev, M. van Schilfgaarde and T. Kotani, "All-electron
  self-consistent GW approximation: Application to Si, MnO, and NiO",
  Phys. Rev. Lett. 93, 126406 (2004). DOI:10.1103/PhysRevLett.93.126406
- M. van Schilfgaarde, T. Kotani and S. V. Faleev, "Quasiparticle
  self-consistent GW theory", Phys. Rev. B 74, 245125 (2006).
  DOI:10.1103/PhysRevB.74.245125
- T. Kotani, M. van Schilfgaarde and S. V. Faleev, "Quasiparticle
  self-consistent GW method: a basis for the independent-particle
  approximation", Phys. Rev. B 76, 165106 (2007).
  DOI:10.1103/PhysRevB.76.165106
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..scf.autodiff import SCFDifferentiationConfig
from ..solvers.nonlinear import attach_root
from .outer_response import linear_config, require_occupation_gap, require_valid
from gradscf.integrals.molecular.jk import build_j_from_df
from .freq import scaled_legendre_grid
from .g0w0 import _exchange_mo, _mo_factors
from .polarizability import rho_response_iw
from .screened import screened_w_imag_axis_matrix
from .self_energy import sigma_imag_matrix, sigma_residue_matrix
from .types import GWResult


def _static_self_energy(
    *,
    b_mn: Array,
    b_ov: Array,
    mo_energy: Array,
    nocc: int,
    ef: Array,
    freqs: Array,
    wts: Array,
    eta: float,
) -> Array:
    """Mode-A endpoint average of the correlation self-energy (real MOs)."""
    nmo = mo_energy.shape[0]

    def response_fn(omega):
        return rho_response_iw(omega, mo_energy, b_ov, spin_factor=4.0)

    wmn_full = screened_w_imag_axis_matrix(b_mn, response_fn, freqs)  # (nw, nq, m, n)
    w_static = screened_w_imag_axis_matrix(b_mn, response_fn, jnp.zeros(1))[0]
    static = dict(wmn_static=w_static,
                  occupation_sign=jnp.where(jnp.arange(nmo) < nocc, -1., 1.))
    channels = ((mo_energy, b_ov, 2.0),)

    def at_endpoint(args):
        m, omega = args
        sigma = sigma_imag_matrix(omega, wmn_full, mo_energy, ef, freqs, wts, eta, **static)
        sigma = sigma + sigma_residue_matrix(omega, mo_energy, b_mn, channels, ef, eta, **static)
        # Re Sigma is the Hermitian part at this frequency. Retain only
        # the required row instead of stacking nmo full self-energy matrices.
        return 0.5 * (sigma[m] + sigma[:, m].conj()).real

    # One evaluation per orbital energy, rather than nmo**2 midpoints.
    rows = jax.lax.map(at_endpoint, (jnp.arange(nmo), mo_energy))
    return 0.5 * (rows + rows.T)


def qsgw_cd_restricted(
    *,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    df_factors: Array,
    hcore_matrix: Array,
    nw: int = 100,
    eta: float = 1e-3,
    max_iter: int = 20,
    tol: float = 1e-6,
    tol_density: float = 1e-5,
    damping: float = 0.0,
    differentiation: SCFDifferentiationConfig | None = None,
) -> GWResult:
    """Spin-restricted qsGW with contour deformation.

    Parameters
    ----------
    mo_energy, mo_coeff:
        Real mean-field starting orbitals/energies (HF or DFT).
    nocc:
        Number of doubly occupied orbitals.
    df_factors:
        Low-rank ERI factors ``(naux, nao, nao)``.
    hcore_matrix:
        One-electron core Hamiltonian (AO).
    nw, eta:
        Imaginary-grid size and broadening.
    max_iter, tol, tol_density, damping:
        Outer self-consistency controls. ``damping`` mixes eigenvalues;
        ``tol`` bounds the undamped energy update and the largest entry of
        ``F_MO - diag(energy)`` (Ha). ``tol_density`` bounds the density change.

    Returns
    -------
    :class:`gradscf.gw.types.GWResult` with updated ``mo_coeff``.

    Notes
    -----
    The Hartree term is rebuilt from the qsGW density each iteration; the
    starting spectrum is only an iteration guess. By default the driver is
    eager. ``differentiation=SCFDifferentiationConfig(mode="implicit")``
    enables JIT/JVP/VJP through the converged symmetric effective Fock root,
    including orbital, density, and screening response. The input coefficients
    define an S-orthonormal frame and its spanned subspace; its matrix elements
    retain their input dependence. All orbital eigenvalues must be isolated
    by more than ``1e-8 * (1 + max(abs(energy)))`` Ha in this response path.
    Degenerate spectra are rejected; no eigenvector-gap regularization is used.
    Failed response solves return NaNs under the shared linear-solver policy.
    Both evaluation paths require occupied and virtual orbitals and a finite
    positive occupation gap. The returned ``screening_energy`` records the
    final independent-particle spectrum in the returned orbital frame.
    Convergence on a chosen discrete CD grid is separate from basis/grid accuracy.
    """
    inputs = (mo_energy, mo_coeff, df_factors, hcore_matrix)
    if differentiation is None and any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(inputs)):
        raise NotImplementedError("qsGW default evaluation is eager; supply differentiation for implicit AD/JIT.")
    if any(jnp.iscomplexobj(x) for x in inputs):
        raise NotImplementedError("qsGW currently supports real molecular inputs only.")
    if max_iter < 1 or tol <= 0.0 or tol_density <= 0.0:
        raise ValueError("max_iter, tol and tol_density must be positive.")
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    hcore = jnp.asarray(hcore_matrix, dtype=jnp.float64)
    df_factors = jnp.asarray(df_factors)
    nocc = int(nocc)
    if not 0 < nocc < mo_energy.size:
        raise ValueError('Restricted qsGW requires occupied and virtual orbitals.')
    require_occupation_gap(mo_energy, nocc, "qsGW")
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1).")

    if differentiation is not None:
        return _implicit_qsgw(mo_energy=mo_energy, coeff=coeff, nocc=nocc,
            df_factors=df_factors, hcore=hcore, nw=int(nw), eta=float(eta),
            max_iter=int(max_iter), tol=float(tol), tol_density=float(tol_density),
            damping=float(damping), config=differentiation)

    freqs, wts = scaled_legendre_grid(nw)
    energy = mo_energy
    last_delta = None
    for _ in range(int(max_iter)):
        require_occupation_gap(energy, nocc, "qsGW")
        b_mn = _mo_factors(df_factors, coeff)
        b_ov = b_mn[:, :nocc, nocc:]
        occ_coeff = coeff[:, :nocc]
        density = 2.0 * occ_coeff @ occ_coeff.T
        j_mat = build_j_from_df(df_factors, density)
        vx_mo = _exchange_mo(b_mn, nocc)

        ef = 0.5 * (energy[nocc - 1] + energy[nocc])
        v_qsgw = _static_self_energy(
            b_mn=b_mn, b_ov=b_ov, mo_energy=energy, nocc=nocc, ef=ef,
            freqs=freqs, wts=wts, eta=float(eta),
        )
        fock_mo = coeff.T @ (hcore + j_mat) @ coeff + vx_mo + v_qsgw
        fock_mo = 0.5 * (fock_mo + fock_mo.T)
        d_fock = float(jnp.max(jnp.abs(fock_mo - jnp.diag(energy))))
        new_energy, rotation = jnp.linalg.eigh(fock_mo)
        new_coeff = coeff @ rotation
        # Damping can make the accepted step arbitrarily small even when
        # the actual fixed-point energy residual is still large.
        d_energy = float(jnp.max(jnp.abs(new_energy - energy)))
        if damping > 0.0:
            # Mix the Fock eigenvalues; orbitals follow the new rotation.
            new_energy = (1.0 - damping) * new_energy + damping * energy
        new_density = 2.0 * new_coeff[:, :nocc] @ new_coeff[:, :nocc].T
        d_density = float(jnp.max(jnp.abs(new_density - density)))
        last_delta = (d_energy, d_density, d_fock)
        if d_energy < tol and d_density < tol_density and d_fock < tol:
            # Return the state whose effective Fock was actually evaluated
            # and checked, not an unchecked next iterate.
            return GWResult(
                mo_energy=energy,
                mo_coeff=coeff,
                converged=True,
                sigma_qp=None,
                nw=int(nw),
                screening_energy=energy,
                method="qsgw",
            )
        energy, coeff = new_energy, new_coeff
    raise ArithmeticError(
        f"qsGW did not converge in {max_iter} iterations "
        f"(max|undamped de|={last_delta[0]:.3e}, max|drho|={last_delta[1]:.3e}, "
        f"max|F_MO-diag(e)|={last_delta[2]:.3e} Ha). "
        "Increase max_iter or damping; no fallback result is returned."
    )


def _implicit_qsgw(*, mo_energy, coeff, nocc, df_factors, hcore, nw, eta,
                   max_iter, tol, tol_density, damping, config):
    """Gauge-free symmetric Fock root in the starting orthonormal MO frame.

    The input C0 defines the fixed metric/subspace (C0.T S C0 = I). All
    matrix elements and the output AO density retain C0's parameter response.
    Eigenvectors are differentiated only at isolated orbital eigenvalues;
    degenerate orbital manifolds require a separate spectral-block response.
    """
    response_config = linear_config(config)
    nmo = mo_energy.size
    rows, cols = jnp.triu_indices(nmo)
    b_frame = _mo_factors(df_factors, coeff)
    h_frame = coeff.T @ hcore @ coeff
    freqs, wts = scaled_legendre_grid(nw)

    def unpack(state):
        matrix = jnp.zeros((nmo, nmo), dtype=state.dtype).at[rows, cols].set(state)
        return matrix.at[cols, rows].set(state)

    def mapping(state):
        fock = unpack(state)
        energy, rotation = jnp.linalg.eigh(fock)
        require_occupation_gap(energy, nocc, "qsGW")
        b_mn = _mo_factors(b_frame, rotation)
        density = 2 * rotation[:, :nocc] @ rotation[:, :nocc].T
        static = _static_self_energy(
            b_mn=b_mn, b_ov=b_mn[:, :nocc, nocc:], mo_energy=energy, nocc=nocc,
            ef=.5 * (energy[nocc - 1] + energy[nocc]), freqs=freqs, wts=wts, eta=eta)
        new_fock = h_frame + build_j_from_df(b_frame, density)
        new_fock = new_fock + rotation @ (_exchange_mo(b_mn, nocc) + static) @ rotation.T
        return .5 * (new_fock + new_fock.T), energy, rotation, density

    def residual(state):
        return state - mapping(state)[0][rows, cols]

    def body(loop):
        state, iteration, _, _ = loop
        new_fock, energy, rotation, density = mapping(state)
        new_energy, new_rotation = jnp.linalg.eigh(new_fock)
        new_density = 2 * new_rotation[:, :nocc] @ new_rotation[:, :nocc].T
        fock_error = jnp.max(jnp.abs(rotation.T @ new_fock @ rotation - jnp.diag(energy)))
        energy_error = jnp.max(jnp.abs(new_energy - energy))
        density_error = jnp.max(jnp.abs(coeff @ (new_density - density) @ coeff.T))
        done = (fock_error < tol) & (energy_error < tol) & (density_error < tol_density)
        mixed = new_rotation @ jnp.diag((1 - damping) * new_energy + damping * energy) @ new_rotation.T
        return jnp.where(done, state, mixed[rows, cols]), iteration + 1, fock_error, done

    initial = jnp.diag(mo_energy)[rows, cols]
    def numerical_forward(seed):
        return jax.lax.while_loop(
            lambda loop: (loop[1] < max_iter) & ~loop[3], body,
            (seed, jnp.array(0), jnp.array(jnp.inf), jnp.array(False)))
    # Keep the forward eigensolver trajectory out of the implicit response,
    # following the shared nonlinear solver's captured-parameter isolation.
    forward, parameters = jax.closure_convert(numerical_forward, initial)
    state = forward(jax.lax.stop_gradient(initial),
                    *jax.tree.map(jax.lax.stop_gradient, parameters))
    seed, _, error, done = jax.tree.map(jax.lax.stop_gradient, state)
    energies = jnp.linalg.eigvalsh(unpack(seed))
    gap_ok = jnp.all(jnp.diff(energies) > 1e-8 * (1 + jnp.max(jnp.abs(energies))))
    require_valid(done & gap_ok, error, 'qsGW (isolated orbital spectrum required)')
    state = attach_root(residual, seed, config=response_config, converged=done & gap_ok)
    energy, rotation = jnp.linalg.eigh(unpack(state))
    return GWResult(mo_energy=energy, mo_coeff=coeff @ rotation, converged=done,
                    sigma_qp=None, nw=nw, screening_energy=energy, method="qsgw")


__all__ = ["qsgw_cd_restricted"]
