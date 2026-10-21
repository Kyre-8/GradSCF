"""Restricted open-shell SCF with common spatial orbitals and Roothaan Fock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..dft.xc import hybrid_coeff, parse_recipe, xc_type
from .core import _build_density_from_occ, _contains_jax_tracer, _host_float_unless_traced
from .core import _diagonalize_fock, _orthogonalizer, _validate_density_matrix
from .facade import _BaseKS
from gradscf.scf.inputs.assembly import build_uks_integral_inputs
from .convergence import convergence_reached
from .rks import RKSResult, _PYSCF_LIKE_DIIS_SPACE, _diis_extrapolate
from .rks import _apply_level_shift, _orthonormal_diis_error
from .uks import UKSConfig, _raw_fock_and_energy_for_state


@dataclass(frozen=True)
class ROKSConfig(UKSConfig):
    """LDA/GGA/global-hybrid ROKS controls; full real AO integrals only."""

    conv_tol_grad: float = 1e-7


@dataclass(frozen=True)
class ROKSResult(RKSResult):
    """Common orbitals, 0/1/2 occupations, and spin-resolved ROKS matrices."""

    density_matrix_alpha: Array
    density_matrix_beta: Array
    fock_matrix_alpha: Array
    fock_matrix_beta: Array
    mo_energy_alpha: Array
    mo_energy_beta: Array
    orbital_gradient_norm: float


def roothaan_fock(fock_alpha: Array, fock_beta: Array,
                  density_major: Array, density_minor: Array, overlap: Array) -> Array:
    """Effective Fock for closed/open/virtual subspaces (majority spin first).

    The diagonal blocks use (Fa+Fb)/2; closed-open, open-virtual, and
    closed-virtual couplings use Fb, Fa, and (Fa+Fb)/2, respectively.
    """

    closed = density_minor @ overlap
    opened = (density_major - density_minor) @ overlap
    virtual = jnp.eye(overlap.shape[0], dtype=overlap.dtype) - density_major @ overlap
    average = (fock_alpha + fock_beta) * 0.5
    diagonal = sum(p.T @ average @ p for p in (closed, opened, virtual))
    cross = (opened.T @ fock_beta @ closed + opened.T @ fock_alpha @ virtual
             + virtual.T @ average @ closed)
    return diagonal + cross + cross.T


def _ro_occupations(energy: Array, majority_energy: Array, ncore: int, nopen: int) -> Array:
    # PySCF ordering: closed orbitals by effective energy, open orbitals by
    # majority-spin energy among the remaining orbitals.
    closed = jnp.argsort(energy)[:ncore]
    occ = jnp.zeros_like(energy).at[closed].set(2.0)
    opened = jnp.argsort(jnp.where(occ == 2, jnp.inf, majority_energy))[:nopen]
    return occ.at[opened].set(1.0)


def _ro_spin_occ(occ: Array, alpha_major: bool) -> Array:
    major = (occ > 0).astype(occ.dtype)
    minor = (occ == 2).astype(occ.dtype)
    return jnp.stack((major, minor) if alpha_major else (minor, major))


def _ro_gradient(coeff: Array, occ_spin: Array, fock_spin: Array) -> Array:
    fmo = jax.vmap(lambda f: coeff.T @ f @ coeff)(fock_spin)
    rotations = (1 - occ_spin[:, :, None]) * occ_spin[:, None, :]
    return jnp.linalg.norm(jnp.sum(rotations * fmo, axis=0))


class _ROIteration(NamedTuple):
    coeff: Array
    occ: Array
    density: Array
    energy: Array
    xc_energy: Array
    fock_spin: Array
    fock: Array
    fock_last: Array
    fock_hist: Array
    err_hist: Array
    head: Array
    count: Array
    cycles: Array
    converged: Array
    gradient: Array


def run_roks_from_integrals(
    *, overlap: Array, hcore: Array, eri: Array, nalpha: int, nbeta: int,
    nuclear_repulsion: float | Array, ao: Array, ao_deriv1: Array, grid_weights: Array,
    init_density_alpha: Array | None = None, init_density_beta: Array | None = None,
    init_mo_coeff: Array | None = None, config: ROKSConfig | None = None,
    grid_ao: Any | None = None,
) -> ROKSResult:
    """Solve high-spin, common-orbital ROKS using real AO integrals.

    This is the restricted open-shell KS convention used by PySCF, not a
    multiplet-sum excited-singlet ROKS method. Electron counts are static.
    """

    cfg = ROKSConfig() if config is None else config
    recipe = parse_recipe(cfg.xc_spec)
    recipe.require_complete_support()
    if recipe.exact_exchange.omega is not None:
        raise NotImplementedError("ROKS range-separated exchange operators are not implemented.")
    s, h, eri = map(jnp.asarray, (overlap, hcore, eri))
    nao = h.shape[0]
    if any(not isinstance(n, int) or n < 0 or n > nao for n in (nalpha, nbeta)) or nalpha + nbeta == 0:
        raise ValueError("Invalid occupation counts for restricted open-shell SCF.")
    if cfg.max_cycle < 1 or cfg.conv_tol < 0 or min(cfg.conv_tol_density, cfg.conv_tol_grad) <= 0:
        raise ValueError("ROKS requires positive max_cycle, density/gradient tolerances, and nonnegative conv_tol.")
    if cfg.convergence_metric not in ("energy_and_residual", "energy"):
        raise ValueError("ROKS requires energy_and_residual or energy convergence.")
    if cfg.jk_backend != "full":
        raise NotImplementedError("ROKS currently requires jk_backend='full'.")
    if any(jnp.iscomplexobj(a) for a in (s, h, eri, init_density_alpha, init_density_beta, init_mo_coeff)):
        raise ValueError("ROKS currently supports real orbitals and integrals only.")
    kind = xc_type(cfg.xc_spec)
    if kind not in {"HF", "LDA", "GGA"}:
        raise NotImplementedError("ROKS supports HF, LDA, and GGA/global hybrids only.")
    alpha = jnp.asarray(hybrid_coeff(cfg.xc_spec), dtype=h.dtype)
    enuc = jnp.asarray(nuclear_repulsion, dtype=h.dtype)
    ao, deriv, weights = map(jnp.asarray, (ao, ao_deriv1, grid_weights))
    x = _orthogonalizer(s, cfg.orthogonalization_eps)
    alpha_major = nalpha >= nbeta
    ncore, nopen = min(nalpha, nbeta), abs(nalpha - nbeta)

    def evaluate(density):
        energy, exc, fa, fb = _raw_fock_and_energy_for_state(
            density_a=density[0], density_b=density[1], ao=ao, ao_deriv1=deriv,
            weights=weights, h=h, eri=eri, df_factors=None, enuc=enuc,
            alpha=alpha, cfg=cfg, xc_kind=kind,
            grid_ao=grid_ao,
        )
        fs = jnp.stack([fa, fb])
        major, minor = (0, 1) if alpha_major else (1, 0)
        effective = roothaan_fock(fs[major], fs[minor], density[major], density[minor], s)
        return energy, exc, fs, effective

    def fill(energy, coeff, fs):
        major_fock = fs[0 if alpha_major else 1]
        majority_energy = jnp.diag(coeff.T @ major_fock @ coeff)
        occ = _ro_occupations(energy, majority_energy, ncore, nopen)
        dm = jax.vmap(lambda o: _build_density_from_occ(coeff, o))(_ro_spin_occ(occ, alpha_major))
        return occ, dm

    if (init_density_alpha is None) != (init_density_beta is None):
        raise ValueError("Provide both alpha and beta initial densities, or neither.")
    initial_fock = h
    initial_spin_fock = jnp.stack([h, h])
    if init_density_alpha is not None:
        initial = jnp.stack([
            _validate_density_matrix(dm, nao=nao, dtype=h.dtype, label=label, method="ROKS")
            for dm, label in ((init_density_alpha, "init_density_alpha"),
                              (init_density_beta, "init_density_beta"))
        ])
        _, _, initial_spin_fock, initial_fock = evaluate(initial)
    energy0, coeff0 = _diagonalize_fock(initial_fock, x)
    if init_mo_coeff is not None:
        coeff0 = jnp.asarray(init_mo_coeff)
        if coeff0.shape != (nao, nao):
            raise ValueError("init_mo_coeff must have shape (nao, nao).")
        energy0 = jnp.diag(coeff0.T @ initial_fock @ coeff0)
    occ0, dm0 = fill(energy0, coeff0, initial_spin_fock)
    e0, exc0, fs0, f0 = evaluate(dm0)
    state = _ROIteration(
        coeff0, occ0, dm0, e0, exc0, fs0, f0, f0,
        jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao, nao), dtype=h.dtype),
        jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, nao * nao), dtype=h.dtype),
        jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32), jnp.asarray(False),
        _ro_gradient(coeff0, _ro_spin_occ(occ0, alpha_major), fs0),
    )

    def advance(old):
        f = jnp.where(old.cycles < 2, (1 - cfg.damping) * old.fock + cfg.damping * old.fock_last,
                      old.fock)
        f, fh, eh, head, count = _diis_extrapolate(
            f, _orthonormal_diis_error(f, old.density.sum(axis=0), s, x),
            old.fock_hist, old.err_hist, old.head, old.count,
        )
        shifted = _apply_level_shift(f, s, old.density.sum(axis=0), jnp.asarray(cfg.level_shift))
        eps, coeff = _diagonalize_fock(shifted, x)
        occ, density = fill(eps, coeff, old.fock_spin)
        energy, exc, fs, effective = evaluate(density)
        gradient = _ro_gradient(coeff, _ro_spin_occ(occ, alpha_major), fs)
        delta_dm = jnp.sqrt(jnp.mean((density - old.density) ** 2))
        converged = convergence_reached(
            energy - old.energy, delta_dm, gradient, conv_tol=cfg.conv_tol,
            conv_tol_density=cfg.conv_tol_density, conv_tol_grad=cfg.conv_tol_grad,
            energy_only=cfg.convergence_metric == "energy", has_prior_cycle=old.cycles > 0,
        )
        return _ROIteration(coeff, occ, density, energy, exc, fs, effective, f,
                            fh, eh, head, count, old.cycles + 1, converged, gradient)

    def step(old, _):
        return jax.lax.cond(old.converged, lambda x: x, advance, old), None

    final, _ = jax.lax.scan(step, state, xs=None, length=cfg.max_cycle)
    traced = _contains_jax_tracer(final)
    # Unshifted orbital energies are evaluated in the returned common orbitals.
    eps = jnp.diag(final.coeff.T @ final.fock @ final.coeff)
    eps_spin = jax.vmap(lambda f: jnp.diag(final.coeff.T @ f @ final.coeff))(final.fock_spin)
    return ROKSResult(
        converged=final.converged if traced else bool(final.converged),
        cycles=final.cycles if traced else int(final.cycles),
        total_energy=_host_float_unless_traced(final.energy),
        electronic_energy=_host_float_unless_traced(final.energy - enuc),
        nuclear_repulsion=_host_float_unless_traced(enuc),
        xc_energy=_host_float_unless_traced(final.xc_energy),
        exact_exchange_fraction=float(hybrid_coeff(cfg.xc_spec)),
        mo_energy=eps, mo_coeff=final.coeff, mo_occ=final.occ,
        density_matrix=final.density.sum(axis=0), fock_matrix=final.fock,
        overlap_matrix=s, hcore_matrix=h,
        density_matrix_alpha=final.density[0], density_matrix_beta=final.density[1],
        fock_matrix_alpha=final.fock_spin[0], fock_matrix_beta=final.fock_spin[1],
        mo_energy_alpha=eps_spin[0], mo_energy_beta=eps_spin[1],
        orbital_gradient_norm=_host_float_unless_traced(final.gradient),
    )


@dataclass
class ROKS(_BaseKS):
    """PySCF-style restricted open-shell ground-state facade."""

    conv_tol_grad: float = 1e-7
    scf_result: ROKSResult | None = None

    def _config(self) -> ROKSConfig:
        return ROKSConfig(xc_spec=self.xc, max_cycle=self.max_cycle, conv_tol=self.conv_tol,
                          conv_tol_density=self.conv_tol_density, conv_tol_grad=self.conv_tol_grad,
                          damping=self.damp, level_shift=self.level_shift,
                          grid_block_size=self.grid_block_size, grid_max_memory=self.grid_max_memory)

    def kernel(self) -> Any:
        self._configure_jax_cache()
        self.converged = False
        cfg = self._config()
        inputs = build_uks_integral_inputs(
            atom=self._spec(), basis=self.mol.basis, config=cfg, xc_spec=self.xc,
            unit=self.mol.unit, charge=self.mol.charge, spin=self.mol.spin, cart=self.mol.cart,
            grids_level=self.grids_level, max_l=self.max_l,
            integral_backend=self.integral_backend, grid_ao_backend=self.grid_ao_backend,
            libcint_geometry_grad_policy=self.geometry_grad_policy,
            init_guess=self.init_guess, chkfile=self.chkfile, init_guess_sap_basis=self.sap_basis,
            init_guess_chkfile_project=self.init_guess_chkfile_project, verbose=self.mol.verbose,
        )
        kwargs = dict(overlap=inputs.overlap, hcore=inputs.hcore, eri=inputs.eri,
                      ao=inputs.ao, ao_deriv1=inputs.ao_deriv1, grid_weights=inputs.grid_weights,
                      grid_ao=inputs.grid_ao,
                      nuclear_repulsion=inputs.nuclear_repulsion,
                      init_density_alpha=inputs.init_density_alpha,
                      init_density_beta=inputs.init_density_beta)
        if self.execution_device != "auto":
            from ..tools.device import resolve_execution_device
            device = resolve_execution_device(self.execution_device)
            kwargs = jax.tree_util.tree_map(
                lambda a: jax.device_put(a, device), kwargs,
            )
        result = run_roks_from_integrals(**kwargs, nalpha=inputs.nalpha, nbeta=inputs.nbeta, config=cfg)
        self.scf_result = result
        self._sync_from_scf_result(result)
        self._scf_inputs = inputs
        self._cached_scf_key = self._scf_signature()
        return self.e_tot

    def make_rdm1(self) -> Array:
        if self.mo_coeff is None or self.mo_occ is None:
            raise RuntimeError("Run restricted open-shell SCF before make_rdm1().")
        occ = _ro_spin_occ(jnp.asarray(self.mo_occ), self.mol.spin >= 0)
        return jax.vmap(lambda o: _build_density_from_occ(self.mo_coeff, o))(occ)

    def TDA(self, **kwargs: Any) -> Any:
        raise NotImplementedError("TDA for restricted open-shell references is not implemented.")

    def _ensure_reference(self) -> Any:
        raise NotImplementedError("Response for restricted open-shell references is not implemented.")

    def TDDFT(self, **kwargs: Any) -> Any:
        raise NotImplementedError("TDDFT for restricted open-shell references is not implemented.")

    def density_fit(self) -> "ROKS":
        raise NotImplementedError("Restricted open-shell density fitting is not implemented.")

    def direct_scf(self) -> "ROKS":
        raise NotImplementedError("Restricted open-shell direct SCF is not implemented.")
