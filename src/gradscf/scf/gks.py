"""Complex generalized Kohn-Sham SCF in an alpha-then-beta AO basis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..dft.xc import grid_value_and_grad, hybrid_coeff, parse_recipe, xc_type
from .core import _contains_jax_tracer, _host_float_unless_traced
from .facade import _BaseKS
from gradscf.scf.inputs.assembly import build_uks_integral_inputs
from .convergence import convergence_reached
from .rks import RKSResult, _PYSCF_LIKE_DIIS_SPACE, _diis_extrapolate
from .uks import UKSConfig
from .uks import _unrestricted_vxc_matrices, _unrestricted_xc_energy_and_potential_on_grid


@dataclass(frozen=True)
class GKSConfig(UKSConfig):
    """Generalized SCF controls: collinear LDA/GGA or noncollinear LDA XC."""

    collinear: str = "col"
    conv_tol_grad: float = 1e-7


@dataclass(frozen=True)
class GKSResult(RKSResult):
    """Spinor orbitals/density/Fock of size 2*nao and single occupations."""

    orbital_gradient_norm: float


def _hermitian(a: Array) -> Array:
    return (a + a.conj().T) * 0.5


def _spin_diagonal(a: Array) -> Array:
    zero = jnp.zeros_like(a)
    return jnp.block([[a, zero], [zero, a]])


def _validate_hermitian(a: Array, shape: tuple[int, int], name: str) -> None:
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not _contains_jax_tracer(a):
        tolerance = 100 * jnp.finfo(a.real.dtype).eps
        if not bool(jnp.all(jnp.isfinite(a))) or not bool(jnp.allclose(a, a.conj().T, atol=tolerance, rtol=0)):
            raise ValueError(f"{name} must be finite and Hermitian.")


def _validate_mode(config: GKSConfig) -> str:
    recipe = parse_recipe(config.xc_spec)
    recipe.require_complete_support()
    if recipe.exact_exchange.omega is not None:
        raise NotImplementedError("GKS range-separated exchange operators are not implemented.")
    if config.collinear not in {"col", "ncol"}:
        raise NotImplementedError("GKS supports col and ncol; mcol is not implemented.")
    kind = xc_type(config.xc_spec)
    if kind not in {"HF", "LDA", "GGA"}:
        raise NotImplementedError("GKS supports LDA/GGA/global hybrids; meta-GGA is not implemented.")
    if config.collinear == "ncol" and kind not in {"HF", "LDA"}:
        raise NotImplementedError("Noncollinear GKS currently supports LDA (and HF) only.")
    if config.jk_backend != "full":
        raise NotImplementedError("Generalized SCF currently requires full real AO integrals.")
    return kind


def generalized_jk(eri: Array, density: Array) -> tuple[Array, Array]:
    """Build spin-diagonal Coulomb and all four complex exchange blocks.

    Spatial AO ERIs use chemists' notation. The spatial AOs and ERIs are real;
    the spinor density may be complex Hermitian, including nonzero ab blocks.
    """

    eri, density = jnp.asarray(eri), jnp.asarray(density)
    n = density.shape[0]//2
    total = density[:n, :n] + density[n:, n:]
    if eri.ndim in (1,2):
        from gradscf.integrals.molecular.jk import build_jk_from_packed
        coulomb,_=build_jk_from_packed(eri,total.T)
        _,exchange=build_jk_from_packed(eri,density.reshape(2,n,2,n).transpose(0,2,1,3))
        exchange=exchange.transpose(0,2,1,3).reshape(2*n,2*n)
        return _hermitian(_spin_diagonal(coulomb)),_hermitian(exchange)
    coulomb = jnp.einsum("pqrs,sr->pq", eri, total, precision=Precision.HIGHEST)
    blocks = density.reshape(2, n, 2, n)
    from gradscf.integrals.molecular.jk import exchange_matrix
    exchange = exchange_matrix(eri, blocks.transpose(0,2,1,3)).transpose(0,2,1,3)
    return _hermitian(_spin_diagonal(coulomb)), _hermitian(exchange.reshape(2*n, 2*n))


def _generalized_xc(density, ao, deriv, weights, cfg, kind):
    n = ao.shape[1]
    if kind == "HF":
        return jnp.asarray(0., dtype=density.real.dtype), jnp.zeros_like(density)
    if cfg.collinear == "col":
        exc, va, vb, ga, gb = _unrestricted_xc_energy_and_potential_on_grid(
            ao=ao, ao_deriv1=deriv, weights=weights,
            density_a=density[:n, :n].real, density_b=density[n:, n:].real,
            xc_spec=cfg.xc_spec, density_floor=cfg.density_floor,
            potential_clip=cfg.potential_clip, xc_kind=kind,
        )
        a, b = _unrestricted_vxc_matrices(
            ao=ao, ao_deriv1=deriv, ao_laplacian=jnp.zeros_like(ao), weights=weights,
            vxc_rho_a=va, vxc_rho_b=vb, vxc_grad_a=ga, vxc_grad_b=gb, xc_kind=kind,
        )
        zero = jnp.zeros_like(a)
        return exc, jnp.block([[a, zero], [zero, b]])

    # Local spin-density eigenvalues (rho +/- |m|)/2 define noncollinear LDA.
    local = jnp.einsum("gp,apbq,gq->gab", ao, density.reshape(2, n, 2, n), ao,
                       precision=Precision.HIGHEST)
    total = (local[:, 0, 0] + local[:, 1, 1]).real
    magnetization = jnp.stack([2 * local[:, 0, 1].real, -2 * local[:, 0, 1].imag,
                              (local[:, 0, 0] - local[:, 1, 1]).real], axis=1)
    norm2 = jnp.sum(magnetization**2, axis=1)
    # Safe zero-magnetization limit, matching PySCF's 1e-20 direction cutoff.
    cutoff = max(1e-20, float(jnp.finfo(density.real.dtype).tiny)**.5)
    safe_norm = jnp.sqrt(jnp.maximum(norm2, cutoff**2))
    magnitude = jnp.where(norm2 >= cutoff**2, safe_norm, 0.)
    rho = jnp.stack([(total + magnitude) * .5, (total - magnitude) * .5], axis=1)
    values, potentials = grid_value_and_grad(cfg.xc_spec, "LDA")(rho)
    potentials = jnp.where((total > cfg.density_floor)[:, None], potentials, 0.)
    if cfg.potential_clip is not None:
        potentials = jnp.clip(potentials, -cfg.potential_clip, cfg.potential_clip)
    scalar = (potentials[:, 0] + potentials[:, 1]) * .5
    spin = (potentials[:, 0] - potentials[:, 1]) * .5
    direction = jnp.where((norm2 >= cutoff**2)[:, None], magnetization / safe_norm[:, None], 0.)
    vx, vy, vz = (spin[:, None] * direction).T

    def integrate(value):
        return jnp.einsum("g,gp,gq->pq", weights * value, ao, ao, precision=Precision.HIGHEST)

    ab = integrate(vx - 1j * vy)
    vxc = jnp.block([[integrate(scalar + vz), ab], [ab.conj().T, integrate(scalar - vz)]])
    return jnp.dot(weights, values), vxc


def generalized_energy_and_fock(
    *, density: Array, hcore: Array, eri: Array, nuclear_repulsion: float | Array,
    ao: Array, ao_deriv1: Array, grid_weights: Array, config: GKSConfig,
) -> tuple[Array, Array, Array]:
    """Return total energy, semilocal XC energy, and Hermitian spinor Fock."""

    kind = _validate_mode(config)
    density, h = jnp.asarray(density), jnp.asarray(hcore)
    if h.shape[0] == density.shape[0] // 2:
        h = _spin_diagonal(h)
    j, k = generalized_jk(eri, density)
    exc, vxc = _generalized_xc(density, jnp.asarray(ao), jnp.asarray(ao_deriv1),
                             jnp.asarray(grid_weights), config, kind)
    potential = j - hybrid_coeff(config.xc_spec) * k
    energy = (jnp.einsum("pq,qp->", density, h + .5 * potential,
                        precision=Precision.HIGHEST).real + exc + nuclear_repulsion)
    return energy, exc, _hermitian(h + potential + vxc)


class _GeneralizedIteration(NamedTuple):
    coeff: Array
    density: Array
    energy: Array
    xc_energy: Array
    fock: Array
    last_fock: Array
    fock_hist: Array
    err_hist: Array
    head: Array
    count: Array
    cycles: Array
    converged: Array
    gradient: Array


def run_gks_from_integrals(
    *, overlap: Array, hcore: Array, eri: Array, nelectron: int,
    nuclear_repulsion: float | Array, ao: Array, ao_deriv1: Array, grid_weights: Array,
    init_density: Array | None = None, config: GKSConfig | None = None,
) -> GKSResult:
    """Generalized SCF: spatial overlap/ERIs, spatial or 2*nao spinor hcore.

    Only total electron number is constrained. Spin mixing is preserved in
    complex densities, exchange blocks, eigensolutions, and DIIS residuals.
    """

    cfg = GKSConfig() if config is None else config
    _validate_mode(cfg)
    s, h, eri = map(jnp.asarray, (overlap, hcore, eri))
    n = s.shape[0]
    from gradscf.integrals.molecular.eri import packed_eri_shape
    if s.shape != (n,n) or eri.shape not in {(n,n,n,n),packed_eri_shape(n,1),packed_eri_shape(n,2)}:
        raise ValueError("Generalized SCF requires spatial overlap and full/s4/s8 ERIs.")
    if jnp.iscomplexobj(s) or jnp.iscomplexobj(eri) or jnp.iscomplexobj(ao) or jnp.iscomplexobj(ao_deriv1):
        raise ValueError("Generalized SCF currently requires real spatial AOs and integrals.")
    if not isinstance(nelectron, Integral) or not 0 < nelectron <= 2*n:
        raise ValueError("Invalid total electron count for generalized SCF.")
    if cfg.max_cycle < 1 or cfg.conv_tol < 0 or min(cfg.conv_tol_density, cfg.conv_tol_grad) <= 0:
        raise ValueError("Generalized SCF requires positive iterations, density/gradient tolerances, and nonnegative conv_tol.")
    if cfg.convergence_metric not in ("energy_and_residual", "energy"):
        raise ValueError("Generalized SCF requires energy_and_residual or energy convergence.")
    ao, deriv, weights = map(jnp.asarray, (ao, ao_deriv1, grid_weights))
    real_dtype = jnp.result_type(s, h.real, eri, ao, deriv, weights, nuclear_repulsion, jnp.float32)
    if init_density is not None:
        real_dtype = jnp.result_type(real_dtype, jnp.asarray(init_density).real)
    complex_dtype = jnp.result_type(real_dtype, jnp.complex64)
    s, h = s.astype(real_dtype), h.astype(complex_dtype)
    eri, ao, deriv, weights = (a.astype(real_dtype) for a in (eri, ao, deriv, weights))
    _validate_hermitian(s, (n, n), "overlap")
    if h.shape == (n, n):
        h = _spin_diagonal(h)
    _validate_hermitian(h, (2*n, 2*n), "hcore")
    if ao.shape != (weights.size, n) or deriv.shape != (4, weights.size, n):
        raise ValueError("AO values/first derivatives must match the grid and spatial AO dimensions.")
    eig, vec = jnp.linalg.eigh(s)
    if not _contains_jax_tracer(eig) and bool(jnp.any(eig <= 0)):
        raise ValueError("overlap must be positive definite.")
    x = _spin_diagonal((vec * jnp.maximum(eig, cfg.orthogonalization_eps)**-.5) @ vec.T)
    s2 = _spin_diagonal(s)
    occ = jnp.zeros((2*n,), dtype=real_dtype).at[:nelectron].set(1.)

    def diagonalize(fock):
        _, c = jnp.linalg.eigh(_hermitian(x.conj().T @ fock @ x))
        return x @ c

    def density_from(coeff):
        return _hermitian((coeff * occ) @ coeff.conj().T)

    def evaluate(dm):
        return generalized_energy_and_fock(
            density=dm, hcore=h, eri=eri, nuclear_repulsion=nuclear_repulsion,
            ao=ao, ao_deriv1=deriv, grid_weights=weights, config=cfg,
        )

    def gradient(c, f):
        fmo = c.conj().T @ f @ c
        return jnp.linalg.norm((1 - occ[:, None]) * occ[None, :] * fmo)

    c0 = diagonalize(h)
    dm0 = density_from(c0) if init_density is None else jnp.asarray(init_density, dtype=complex_dtype)
    _validate_hermitian(dm0, (2*n, 2*n), "init_density")
    e0, xc0, f0 = evaluate(dm0)
    initial = _GeneralizedIteration(
        c0, dm0, e0, xc0, f0, f0,
        jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, 2*n, 2*n), dtype=complex_dtype),
        jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, 2*(2*n)**2), dtype=real_dtype),
        jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
        jnp.asarray(False), gradient(c0, f0),
    )

    def advance(old):
        f = jnp.where(old.cycles < 2, (1-cfg.damping)*old.fock + cfg.damping*old.last_fock, old.fock)
        residual = x.conj().T @ (f @ old.density @ s2 - s2 @ old.density @ f) @ x
        # Real DIIS metric Re(<e_i|e_j>) retains both parts of complex errors.
        error = jnp.concatenate([residual.real.ravel(), residual.imag.ravel()])
        f, fh, eh, head, count = _diis_extrapolate(
            f, error, old.fock_hist, old.err_hist, old.head, old.count,
        )
        shifted = _hermitian(f + cfg.level_shift * (s2 - s2 @ old.density @ s2))
        coeff = diagonalize(shifted)
        dm = density_from(coeff)
        energy, exc, raw = evaluate(dm)
        grad = gradient(coeff, raw)
        delta = jnp.sqrt(jnp.mean(jnp.abs(dm - old.density)**2))
        converged = convergence_reached(
            energy - old.energy, delta, grad, conv_tol=cfg.conv_tol,
            conv_tol_density=cfg.conv_tol_density, conv_tol_grad=cfg.conv_tol_grad,
            energy_only=cfg.convergence_metric == "energy", has_prior_cycle=old.cycles > 0,
        )
        return _GeneralizedIteration(coeff, dm, energy, exc, raw, f, fh, eh, head, count,
                                     old.cycles+1, converged, grad)

    def step(old, _):
        return jax.lax.cond(old.converged, lambda x: x, advance, old), None

    final, _ = jax.lax.scan(step, initial, xs=None, length=cfg.max_cycle)
    traced = _contains_jax_tracer(final)
    return GKSResult(
        converged=final.converged if traced else bool(final.converged),
        cycles=final.cycles if traced else int(final.cycles),
        total_energy=_host_float_unless_traced(final.energy),
        electronic_energy=_host_float_unless_traced(final.energy - nuclear_repulsion),
        nuclear_repulsion=_host_float_unless_traced(nuclear_repulsion),
        xc_energy=_host_float_unless_traced(final.xc_energy),
        exact_exchange_fraction=float(hybrid_coeff(cfg.xc_spec)),
        mo_energy=jnp.diag(final.coeff.conj().T @ final.fock @ final.coeff).real,
        mo_coeff=final.coeff, mo_occ=occ, density_matrix=final.density,
        fock_matrix=final.fock, overlap_matrix=s2, hcore_matrix=h,
        orbital_gradient_norm=_host_float_unless_traced(final.gradient),
    )


@dataclass
class GKS(_BaseKS):
    """Generalized ground-state facade; mol.spin controls only the initial guess."""

    collinear: str = "col"
    conv_tol_grad: float = 1e-7
    scf_result: GKSResult | None = None

    def _config(self) -> GKSConfig:
        return GKSConfig(xc_spec=self.xc, collinear=self.collinear, max_cycle=self.max_cycle,
                         conv_tol=self.conv_tol, conv_tol_density=self.conv_tol_density,
                         conv_tol_grad=self.conv_tol_grad, damping=self.damp, level_shift=self.level_shift,
                         max_memory=self.max_memory,
                         grid_block_size=self.grid_block_size, grid_max_memory=self.grid_max_memory)

    def kernel(self, dm0: Array | None = None) -> Any:
        cfg = self._config()
        _validate_mode(cfg)
        if cfg.grid_block_size > 0:
            raise NotImplementedError('Generalized SCF does not support streamed grid AO data; use grid_block_size=-1.')
        self._configure_jax_cache()
        supplied = dm0 if dm0 is not None else (None if isinstance(self.init_guess, str) else self.init_guess)
        inputs = build_uks_integral_inputs(
            atom=self._spec(), basis=self.mol.basis, config=replace(cfg, grid_block_size=-1),
            xc_spec=self.xc, unit=self.mol.unit, charge=self.mol.charge, spin=self.mol.spin,
            cart=self.mol.cart, grids_level=self.grids_level, max_l=self.max_l,
            integral_backend=self.integral_backend, grid_ao_backend=self.grid_ao_backend,
            libcint_geometry_grad_policy=self.geometry_grad_policy,
            init_guess=self.init_guess if supplied is None else "hcore", chkfile=self.chkfile,
            init_guess_sap_basis=self.sap_basis,
            init_guess_chkfile_project=self.init_guess_chkfile_project, verbose=self.mol.verbose,
        )
        if supplied is None and inputs.init_density_alpha is not None:
            a, b = inputs.init_density_alpha, inputs.init_density_beta
            zero = jnp.zeros_like(a)
            supplied = jnp.block([[a, zero], [zero, b]])
        kwargs = dict(overlap=inputs.overlap, hcore=inputs.hcore, eri=inputs.eri,
                      nuclear_repulsion=inputs.nuclear_repulsion, ao=inputs.ao,
                      ao_deriv1=inputs.ao_deriv1, grid_weights=inputs.grid_weights, init_density=supplied)
        if self.execution_device != "auto":
            from ..tools.device import resolve_execution_device
            device = resolve_execution_device(self.execution_device)
            kwargs = jax.tree_util.tree_map(lambda a: jax.device_put(a, device), kwargs)
        result = run_gks_from_integrals(**kwargs, nelectron=self.mol.nelectron, config=cfg)
        self.scf_result = result
        self._sync_from_scf_result(result)
        return self.e_tot

    def run(self, dm0: Array | None = None) -> "GKS":
        self.kernel(dm0=dm0)
        return self

    def make_rdm1(self) -> Array:
        if self.mo_coeff is None or self.mo_occ is None:
            raise RuntimeError("Run generalized SCF before make_rdm1().")
        return (self.mo_coeff * self.mo_occ) @ self.mo_coeff.conj().T

    def _ensure_reference(self) -> Any:
        raise NotImplementedError("Generalized spinor response is not implemented.")

    def TDA(self, **kwargs: Any) -> Any:
        return self._ensure_reference()

    def TDDFT(self, **kwargs: Any) -> Any:
        return self._ensure_reference()

    def density_fit(self) -> "GKS":
        raise NotImplementedError("Generalized density fitting is not implemented.")

    def direct_scf(self) -> "GKS":
        raise NotImplementedError("Generalized direct SCF is not implemented.")
