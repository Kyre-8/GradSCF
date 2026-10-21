from __future__ import annotations

from ..solvers import LinearOperator, EigenSolverConfig, EigenResponseConfig, LinearSolverConfig, solve_hermitian

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any
from typing import Literal
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from .cisd import unrestricted_cisd_second_order_correction
from ._unrestricted_semilocal_response import (
    UnrestrictedSemilocalResponseFunctional,
    build_unrestricted_semilocal_response_action,
    _build_blocked_spin_response_action,
)
from ..solvers.eigen.rpa import implicit_differential_davidson_lowest_tdhf
from .defaults import (
    FULL_TDDFT_DAVIDSON_MAX_CYCLE,
    PYSCF_TD_DAVIDSON_MAX_CYCLE,
    PYSCF_TD_DAVIDSON_TOL,
    PYSCF_TD_POSITIVE_EIG_THRESHOLD,
)
from ..solvers.eigen.davidson import (
    _davidson_search_nroots, _solver_dtype,
)
from ..solvers.eigen.response import (
    EigenGradientMode,
)
from ._utils import (
    _resolve_xc_functional,
)
from .response import (
    _needs_exchange_terms, _raise_if_strict_local_hf_response,
    _map_response_vectors, _restricted_fock_action,
    _jk_from_df_factors, _j_from_df_factors,
)
from .response_options import (
    ResponseKernelOptions, normalize_response_kernel_options, plan_response_memory,
)


def _pytree_dataclass(cls):
    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls_, aux_data, children):
        del aux_data
        return cls_(*children)

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    return jax.tree_util.register_pytree_node_class(cls)


@_pytree_dataclass
@dataclass(frozen=True)
class UnrestrictedTDAResult:
    """Excitation energies and alpha/beta amplitudes from unrestricted TDA."""

    excitation_energies: Array
    amplitudes_alpha: Array
    amplitudes_beta: Array
    posthoc_correction: Array | None = None
    converged: Array | bool = True


@_pytree_dataclass
@dataclass(frozen=True)
class UnrestrictedTDDFTResult:
    """Excitation energies and (X, Y) amplitudes from unrestricted Casida TDDFT."""

    excitation_energies: Array
    x_amplitudes_alpha: Array
    x_amplitudes_beta: Array
    y_amplitudes_alpha: Array
    y_amplitudes_beta: Array
    posthoc_correction: Array | None = None
    converged: Array | bool = True


@dataclass(frozen=True)
class _UnrestrictedResponseOperatorData:
    orbital_energy_differences_alpha: Array
    orbital_energy_differences_beta: Array
    orbo_alpha: Array
    orbv_alpha: Array
    orbo_beta: Array
    orbv_beta: Array
    ao_response_action_fn: Callable[[Array, Array], tuple[Array, Array]]
    ao_mo_response_action_fn: Callable[..., tuple[Array, Array]] | None = None
    xc_response_action_fn: Callable[[Array, Array], tuple[Array, Array]] | None = None
    fock_occupied: tuple[Array, Array] | None = None
    fock_virtual: tuple[Array, Array] | None = None
    memory_plan: Any | None = None


def _unrestricted_orbital_data(
    molecule: Any,
    occupation_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    mo_coeff = jnp.asarray(molecule.mo_coeff)
    mo_occ = jnp.asarray(molecule.mo_occ)
    mo_energy = jnp.asarray(molecule.mo_energy)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise NotImplementedError(
            "Expected unrestricted orbitals with shape (2, nao, nmo)."
        )

    nmo = int(mo_coeff.shape[-1])

    def _channel_partition(
        spin: int,
        *,
        nocc_hint: Any | None,
    ) -> tuple[Array, Array, Array]:
        if nocc_hint is not None:
            nocc = int(nocc_hint)
            nocc = max(0, min(nocc, nmo))
            occ_idx = jnp.arange(nocc)
            vir_idx = jnp.arange(nocc, nmo)
        else:
            occ_values = mo_occ[spin]
            if isinstance(occ_values, jax.core.Tracer):
                raise ValueError(
                    "JIT-traced unrestricted TDDFT/TDA requires static "
                    "nocc_alpha/nocc_beta on the molecule."
                )
            host_occ = np.asarray(jax.device_get(occ_values))
            occ_idx = jnp.asarray(np.where(host_occ > occupation_tolerance)[0], dtype=jnp.int32)
            vir_idx = jnp.asarray(np.where(host_occ <= occupation_tolerance)[0], dtype=jnp.int32)
        orbo = mo_coeff[spin][:, occ_idx]
        orbv = mo_coeff[spin][:, vir_idx]
        de = mo_energy[spin][vir_idx][None, :] - mo_energy[spin][occ_idx][:, None]
        return orbo, orbv, de

    orbo_a, orbv_a, de_a = _channel_partition(
        0,
        nocc_hint=getattr(molecule, "nocc_alpha", None),
    )
    orbo_b, orbv_b, de_b = _channel_partition(
        1,
        nocc_hint=getattr(molecule, "nocc_beta", None),
    )
    has_alpha_channel = de_a.shape[0] > 0 and de_a.shape[1] > 0
    has_beta_channel = de_b.shape[0] > 0 and de_b.shape[1] > 0
    if not (has_alpha_channel or has_beta_channel):
        raise ValueError(
            "Need at least one occupied-virtual excitation channel across alpha/beta spins."
        )
    return orbo_a, orbv_a, orbo_b, orbv_b, de_a, de_b


def _spin_densities_on_grid(molecule: Any) -> tuple[Array, Array]:
    rdm1 = jnp.asarray(molecule.rdm1)
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError("Unrestricted response requires spin-resolved rdm1 with shape (2, nao, nao).")
    provider = getattr(molecule, "grid_ao", None)
    if provider is not None:
        density = provider.map(lambda ao, dao, lap, coords, weights:
            jnp.einsum("spq,gp,gq->gs", rdm1, ao, ao, precision=Precision.HIGHEST), deriv=0)
        return density[:, 0], density[:, 1]
    ao = jnp.asarray(molecule.ao)
    rho_a = jnp.einsum("pq,rp,rq->r", rdm1[0], ao, ao, precision=Precision.HIGHEST)
    rho_b = jnp.einsum("pq,rp,rq->r", rdm1[1], ao, ao, precision=Precision.HIGHEST)
    return rho_a, rho_b


def _normalize_spin_kernel_values(raw_kernel: Any, dtype: Any) -> tuple[Array, Array, Array]:
    if isinstance(raw_kernel, (tuple, list)):
        if len(raw_kernel) != 3:
            raise ValueError("spin kernel tuple/list must contain (f_aa, f_ab, f_bb).")
        f_aa, f_ab, f_bb = raw_kernel
    else:
        kernel = jnp.asarray(raw_kernel, dtype=dtype)
        if kernel.ndim == 1:
            return kernel, kernel, kernel
        if kernel.ndim == 2 and kernel.shape[-1] == 3:
            return kernel[:, 0], kernel[:, 1], kernel[:, 2]
        if kernel.ndim == 3 and kernel.shape[-2:] == (2, 2):
            f_aa = kernel[:, 0, 0]
            f_ab = 0.5 * (kernel[:, 0, 1] + kernel[:, 1, 0])
            f_bb = kernel[:, 1, 1]
            return f_aa, f_ab, f_bb
        raise ValueError(
            "Unsupported spin-kernel format. Expected 1D scalar kernel, "
            "(ngrids, 3), (ngrids, 2, 2), or tuple/list (f_aa, f_ab, f_bb)."
        )
    return (
        jnp.asarray(f_aa, dtype=dtype),
        jnp.asarray(f_ab, dtype=dtype),
        jnp.asarray(f_bb, dtype=dtype),
    )


def _spin_resolved_kernel_on_grid(
    molecule: Any,
    resolved_xc: Any,
    density_alpha: Array,
    density_beta: Array,
    dtype: Any,
) -> tuple[Array, Array, Array]:
    feature_kind = str(
        getattr(resolved_xc, "response_feature_kind", "LDA")
    ).upper()
    if feature_kind != "LDA":
        raise ValueError(
            "Unrestricted GGA response requires spin_grid_response_hvp; "
            "a density-only spin kernel is not a complete GGA response."
        )
    spin_grid_kernel = getattr(resolved_xc, "spin_grid_kernel", None)
    if callable(spin_grid_kernel):
        raw_kernel = spin_grid_kernel(molecule)
    else:
        spin_local_kernel = getattr(resolved_xc, "spin_local_kernel", None)
        if callable(spin_local_kernel):
            raw_kernel = spin_local_kernel(density_alpha, density_beta)
        else:
            raise ValueError(
                "Unrestricted strict TDDFT response requires spin-resolved XC kernels "
                "(spin_grid_kernel or spin_local_kernel). Scalar local_kernel/grid_kernel "
                "fallback is an approximation and is disabled."
            )
    return _normalize_spin_kernel_values(raw_kernel, dtype=dtype)


def _unrestricted_transition_density(orbo: Array, orbv: Array, values: Array, *, bottom: bool) -> Array:
    values = jnp.asarray(values)
    if bottom:
        return jnp.einsum(
            "nia,pi,qa->npq",
            values,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    return jnp.einsum(
        "nia,pa,qi->npq",
        values,
        orbv,
        orbo,
        precision=Precision.HIGHEST,
    )


def _unrestricted_project_response(
    response_ao: Array,
    orbo: Array,
    orbv: Array,
    *,
    bottom: bool,
) -> Array:
    if bottom:
        return jnp.einsum(
            "npq,pi,qa->nia",
            response_ao,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    return jnp.einsum(
        "npq,qi,pa->nia",
        response_ao,
        orbo,
        orbv,
        precision=Precision.HIGHEST,
    )


def _jk_from_full_eri(eri: Array, density: Array) -> tuple[Array, Array]:
    if eri.ndim in (1,2):
        from gradscf.integrals.molecular.jk import build_jk_from_packed
        return build_jk_from_packed(eri,density)
    j_mat = jnp.einsum("pqrs,nrs->npq", eri, density, precision=Precision.HIGHEST)
    k_mat = jnp.einsum("prqs,nrs->npq", eri, density, precision=Precision.HIGHEST)
    return j_mat, k_mat


def _j_from_full_eri(eri: Array, density: Array) -> Array:
    if eri.ndim in (1,2):return _jk_from_full_eri(eri,density)[0]
    return jnp.einsum("pqrs,nrs->npq", eri, density, precision=Precision.HIGHEST)


def _unrestricted_ao_response_action(
    molecule: Any,
    hybrid_fraction: Array,
    *,
    include_exchange: bool,
    dtype: Any,
    response_options: ResponseKernelOptions | None = None,
    eri_cache_bytes: int = 0,
) -> Callable[[Array, Array], tuple[Array, Array]]:
    response_options = normalize_response_kernel_options(response_options)
    alpha = jnp.asarray(hybrid_fraction, dtype=dtype)
    from ..scf.rsh import response_rsh_coefficients
    rsh = response_rsh_coefficients(molecule)
    eri_lr = None
    mode = response_options.two_electron_mode
    direct_basis = getattr(molecule, "direct_basis", None)
    use_direct = mode in {"auto", "direct"} and direct_basis is not None
    if rsh is not None:
        from .response import _validate_restricted_rsh_mode
        _validate_restricted_rsh_mode(molecule, mode)
        _, long_range, short_range = rsh
        alpha = jnp.asarray(short_range, dtype=dtype)
        if not use_direct:
            eri_lr = jnp.asarray(molecule.eri_lr, dtype=dtype)
        include_exchange = True

    factors = getattr(molecule, "df_factors", None)
    has_df = factors is not None and int(jnp.asarray(factors).size) > 0
    if mode == "ris":
        raise NotImplementedError("Unrestricted response does not implement RIS fitting.")
    if mode == "df" and not has_df:
        raise ValueError('Unrestricted response mode="df" requires df_factors.')
    if use_direct:
        if rsh is not None:
            from ..scf.rsh import _direct_rsh_jk
            jk_fn = lambda density: _direct_rsh_jk(
                direct_basis, density, rsh, screening_threshold=response_options.direct_tol,
                workers=response_options.direct_workers)
            alpha = jnp.asarray(1., dtype=dtype)
        else:
            from ..integrals.backends.native.jk import NativeDirectBasis, _selected_jk
            jk_fn = (direct_basis.prepare_jk(
                cache_bytes=eri_cache_bytes, screening_threshold=response_options.direct_tol,
                workers=response_options.direct_workers, with_k=include_exchange)
                if type(direct_basis) is NativeDirectBasis else
                lambda density: _selected_jk(
                    direct_basis, density, with_k=include_exchange,
                    screening_threshold=response_options.direct_tol, workers=response_options.direct_workers))
        j_fn = lambda density: jk_fn(density)[0]
    elif mode in {"auto", "df"} and has_df:
        source = jnp.asarray(factors, dtype=dtype)
        jk_fn = lambda density: _jk_from_df_factors(source, density)
        j_fn = lambda density: _j_from_df_factors(source, density)
    else:
        source = getattr(molecule, "eri_pair_matrix", None)
        if source is None or not int(jnp.asarray(source).size):
            source = getattr(molecule, "rep_tensor", None)
        if source is None or not int(jnp.asarray(source).size):
            raise ValueError("Unrestricted response requires direct_basis or AO ERIs.")
        source = jnp.asarray(source, dtype=dtype)
        jk_fn = lambda density: _jk_from_full_eri(source, density)
        j_fn = lambda density: _j_from_full_eri(source, density)

    def action(density_alpha: Array, density_beta: Array) -> tuple[Array, Array]:
        density_alpha = jnp.asarray(density_alpha, dtype=dtype)
        density_beta = jnp.asarray(density_beta, dtype=dtype)
        if not include_exchange:
            j_total = j_fn(density_alpha + density_beta)
            return j_total, j_total
        j_spin, k_spin = jk_fn(jnp.concatenate((density_alpha, density_beta), axis=0))
        j_alpha, j_beta = jnp.split(j_spin, 2)
        k_alpha, k_beta = jnp.split(k_spin, 2)
        j_total = j_alpha + j_beta
        exchange_a, exchange_b = alpha*k_alpha, alpha*k_beta
        if eri_lr is not None:
            exchange_a = exchange_a+(long_range-short_range)*_jk_from_full_eri(eri_lr, density_alpha)[1]
            exchange_b = exchange_b+(long_range-short_range)*_jk_from_full_eri(eri_lr, density_beta)[1]
        return j_total-exchange_a, j_total-exchange_b

    return action


def _unrestricted_df_mo_response_action(
    df_factors: Array,
    orbo_alpha: Array,
    orbv_alpha: Array,
    orbo_beta: Array,
    orbv_beta: Array,
    hybrid_fraction: Any,
    *,
    include_exchange: bool,
    dtype: Any,
) -> Callable[..., tuple[Array, Array]]:
    alpha = jnp.asarray(hybrid_fraction, dtype=dtype)
    factors = jnp.asarray(df_factors, dtype=dtype)

    def spin_factors(orbo: Array, orbv: Array):
        orbo = jnp.asarray(orbo, dtype=dtype)
        orbv = jnp.asarray(orbv, dtype=dtype)
        b_ov = jnp.einsum("Qpq,pi,qa->Qia", factors, orbo, orbv, precision=Precision.HIGHEST)
        b_vo = jnp.einsum("Qpq,pa,qi->Qai", factors, orbv, orbo, precision=Precision.HIGHEST)
        if include_exchange:
            b_oo = jnp.einsum("Qpq,pi,qj->Qij", factors, orbo, orbo, precision=Precision.HIGHEST)
            b_vv = jnp.einsum("Qpq,pa,qb->Qab", factors, orbv, orbv, precision=Precision.HIGHEST)
        else:
            b_oo = None
            b_vv = None
        return b_ov, b_vo, b_oo, b_vv, int(orbo.shape[1]), int(orbv.shape[1])

    fac_a = spin_factors(orbo_alpha, orbv_alpha)
    fac_b = spin_factors(orbo_beta, orbv_beta)

    def aux_projection(fac, values: Array, *, bottom_density: bool) -> Array:
        b_ov, b_vo, _, _, nocc, nvir = fac
        x = jnp.asarray(values, dtype=dtype).reshape(-1, nocc, nvir)
        if bottom_density:
            return jnp.einsum("Qia,nia->nQ", b_ov, x, precision=Precision.HIGHEST)
        return jnp.einsum("Qai,nia->nQ", b_vo, x, precision=Precision.HIGHEST)

    def project_j(fac, rho_aux: Array, original_shape: tuple[int, ...], *, bottom_projection: bool) -> Array:
        b_ov, b_vo, _, _, nocc, nvir = fac
        if bottom_projection:
            out = jnp.einsum("Qia,nQ->nia", b_ov, rho_aux, precision=Precision.HIGHEST)
        else:
            out = jnp.einsum("Qai,nQ->nia", b_vo, rho_aux, precision=Precision.HIGHEST)
        return out.reshape(original_shape[:-2] + (nocc, nvir))

    def exchange_projection(fac, values: Array, *, bottom_density: bool, bottom_projection: bool) -> Array:
        b_ov, b_vo, b_oo, b_vv, nocc, nvir = fac
        x = jnp.asarray(values, dtype=dtype).reshape(-1, nocc, nvir)
        if bottom_density == bottom_projection:
            assert b_oo is not None and b_vv is not None
            return jnp.einsum("Qij,Qab,njb->nia", b_oo, b_vv, x, precision=Precision.HIGHEST)
        return jnp.einsum("Qaj,Qib,njb->nia", b_vo, b_ov, x, precision=Precision.HIGHEST)

    def action(
        alpha_values: Array,
        beta_values: Array,
        *,
        bottom_density: bool,
        bottom_projection: bool,
    ) -> tuple[Array, Array]:
        alpha_shape = jnp.asarray(alpha_values).shape
        beta_shape = jnp.asarray(beta_values).shape
        rho_aux = aux_projection(fac_a, alpha_values, bottom_density=bottom_density)
        rho_aux = rho_aux + aux_projection(fac_b, beta_values, bottom_density=bottom_density)
        out_alpha = project_j(fac_a, rho_aux, alpha_shape, bottom_projection=bottom_projection)
        out_beta = project_j(fac_b, rho_aux, beta_shape, bottom_projection=bottom_projection)
        if include_exchange:
            out_alpha = out_alpha - alpha * exchange_projection(
                fac_a,
                alpha_values,
                bottom_density=bottom_density,
                bottom_projection=bottom_projection,
            ).reshape(alpha_shape)
            out_beta = out_beta - alpha * exchange_projection(
                fac_b,
                beta_values,
                bottom_density=bottom_density,
                bottom_projection=bottom_projection,
            ).reshape(beta_shape)
        return out_alpha, out_beta

    return action


def _unrestricted_grid_xc_response_action(
    molecule: Any,
    resolved_xc: Any | None,
    orbo_a: Array,
    orbv_a: Array,
    orbo_b: Array,
    orbv_b: Array,
    *,
    dtype: Any,
    block_size: int = 256,
) -> Callable[[Array, Array], tuple[Array, Array]] | None:
    if resolved_xc is None:
        return None
    rho_a, rho_b = _spin_densities_on_grid(molecule)
    f_aa, f_ab, f_bb = _spin_resolved_kernel_on_grid(
        molecule,
        resolved_xc,
        rho_a,
        rho_b,
        dtype,
    )
    f_aa, f_ab, f_bb = tuple(jnp.broadcast_to(value, molecule.grid.weights.shape)
                              for value in (f_aa, f_ab, f_bb))

    def block_hvp(start, size, tangent_a, tangent_b):
        aa, ab, bb = (jax.lax.dynamic_slice_in_dim(value, start, size)
                      for value in (f_aa, f_ab, f_bb))
        return (aa[None, None, :]*tangent_a + ab[None, None, :]*tangent_b,
                ab[None, None, :]*tangent_a + bb[None, None, :]*tangent_b)

    return _build_blocked_spin_response_action(
        molecule, orbo_a, orbv_a, orbo_b, orbv_b, block_hvp,
        feature_kind="LDA", dtype=dtype, block_size=block_size,
    )


def _unrestricted_memory_plan(molecule, resolved_xc, de_a, de_b, options, *,
                              nroots=1, max_subspace=None, rpa=True):
    if isinstance(resolved_xc, str):
        resolved_xc = UnrestrictedSemilocalResponseFunctional(resolved_xc)
    standard_hf = (type(resolved_xc) is UnrestrictedSemilocalResponseFunctional
                   and resolved_xc.xc_spec == "hf")
    return plan_response_memory(
        molecule.mo_coeff.shape[-2], de_a.size + de_b.size, nroots,
        jnp.result_type(de_a, de_b), options.max_memory,
        max_subspace=max_subspace, rpa=rpa, spin_channels=2,
        direct_basis=(getattr(molecule, "direct_basis", None)
                      if options.two_electron_mode in {"auto", "direct"} else None),
        integral_cache=("none" if getattr(resolved_xc, "range_separated_coefficients", None) is not None
                        else options.integral_cache), direct_workers=options.direct_workers,
        direct_tol=options.direct_tol,
        grid_block_size=options.grid_block_size if resolved_xc is not None and not standard_hf else 0,
    )


def _build_unrestricted_response_operator_data(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    need_b_terms: bool = True,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
) -> _UnrestrictedResponseOperatorData:
    response_options = normalize_response_kernel_options(response_kernel_options)
    from ..scf.rsh import response_rsh_coefficients
    rsh = response_rsh_coefficients(molecule)
    resolved_xc = _resolve_xc_functional(molecule, xc_functional, xc_params)
    if isinstance(resolved_xc, str):
        resolved_xc = UnrestrictedSemilocalResponseFunctional(resolved_xc)
    functional_rsh = getattr(resolved_xc, "range_separated_coefficients", None)
    if rsh is not None or functional_rsh is not None:
        from ..scf.molecules import RestrictedMolecule
        if isinstance(molecule, RestrictedMolecule):
            raise NotImplementedError("Range-separated unrestricted TD requires an unrestricted reference.")
        if rsh is None or functional_rsh != rsh:
            raise ValueError("A full RSH reference requires a matching complete unrestricted response functional.")
        response_rsh_coefficients(molecule, xc_spec=getattr(resolved_xc, "xc_spec", None))
    _raise_if_strict_local_hf_response(resolved_xc)
    orbo_a, orbv_a, orbo_b, orbv_b, de_a, de_b = _unrestricted_orbital_data(
        molecule,
        occupation_tolerance,
    )
    fock_occupied = fock_virtual = None
    fock = getattr(molecule, "fock_matrix", None)
    if fock is not None:
        fock = jnp.asarray(fock)
        nao = orbo_a.shape[0]
        if fock.shape != (2, nao, nao) or jnp.iscomplexobj(fock):
            raise ValueError("Noncanonical unrestricted response requires real fock_matrix (2,nao,nao).")
        fock_occupied = tuple(o.T @ f @ o for o, f in zip((orbo_a, orbo_b), fock))
        fock_virtual = tuple(v.T @ f @ v for v, f in zip((orbv_a, orbv_b), fock))
        de_a, de_b = tuple(jnp.diag(v)[None, :] - jnp.diag(o)[:, None]
                          for o, v in zip(fock_occupied, fock_virtual))
    memory_plan = _unrestricted_memory_plan(
        molecule, resolved_xc, de_a, de_b, response_options, rpa=need_b_terms)
    hybrid_fraction_raw = getattr(molecule, "exact_exchange_fraction", 0.0)
    hybrid_fraction: Any = (
        float(hybrid_fraction_raw)
        if isinstance(hybrid_fraction_raw, (int, float, np.number))
        else jnp.asarray(hybrid_fraction_raw, dtype=jnp.result_type(de_a, de_b))
    )
    if resolved_xc is not None:
        hybrid_fraction_raw = getattr(resolved_xc, "exact_exchange_fraction", hybrid_fraction)
        hybrid_fraction = (
            float(hybrid_fraction_raw)
            if isinstance(hybrid_fraction_raw, (int, float, np.number))
            else jnp.asarray(hybrid_fraction_raw, dtype=jnp.result_type(de_a, de_b))
        )
    ao_response_action_fn = _unrestricted_ao_response_action(
        molecule,
        hybrid_fraction,
        include_exchange=_needs_exchange_terms(hybrid_fraction),
        dtype=jnp.result_type(de_a, de_b),
        response_options=response_options,
        eri_cache_bytes=memory_plan.eri_cache_bytes,
    )
    df_factors = getattr(molecule, "df_factors", None)
    ao_mo_response_action_fn = None
    use_direct = (response_options.two_electron_mode in {"auto", "direct"}
                  and getattr(molecule, "direct_basis", None) is not None)
    if (not use_direct and response_options.two_electron_mode in {"auto", "df"}
            and df_factors is not None and int(jnp.asarray(df_factors).size) > 0):
        ao_mo_response_action_fn = _unrestricted_df_mo_response_action(
            df_factors,
            orbo_a,
            orbv_a,
            orbo_b,
            orbv_b,
            hybrid_fraction,
            include_exchange=_needs_exchange_terms(hybrid_fraction),
            dtype=jnp.result_type(de_a, de_b),
        )
    spin_grid_response_hvp = (
        None
        if resolved_xc is None
        else getattr(resolved_xc, "spin_grid_response_hvp", None)
    )
    explicit_spin_hvp = (
        None
        if resolved_xc is None
        else getattr(resolved_xc, "spin_grid_response_hvp_fn", spin_grid_response_hvp)
    )
    if explicit_spin_hvp is None:
        spin_grid_response_hvp = None
    elif callable(explicit_spin_hvp):
        spin_grid_response_hvp = explicit_spin_hvp
    standard_hf = (type(resolved_xc) is UnrestrictedSemilocalResponseFunctional
                   and resolved_xc.xc_spec == "hf")
    ao_kernel_action = getattr(resolved_xc, "ao_kernel_action", None)
    if standard_hf:
        xc_response_action_fn = None
    elif callable(ao_kernel_action):
        def xc_response_action_fn(alpha, beta):
            da = _unrestricted_transition_density(orbo_a, orbv_a, alpha, bottom=False)
            db = _unrestricted_transition_density(orbo_b, orbv_b, beta, bottom=False)
            response = jax.vmap(lambda d: ao_kernel_action(molecule, d))(jnp.stack([da, db], axis=1))
            return (_unrestricted_project_response(response[:, 0], orbo_a, orbv_a, bottom=False),
                    _unrestricted_project_response(response[:, 1], orbo_b, orbv_b, bottom=False))
    elif callable(spin_grid_response_hvp):
        xc_response_action_fn = build_unrestricted_semilocal_response_action(
            molecule,
            orbo_a,
            orbv_a,
            orbo_b,
            orbv_b,
            spin_grid_response_hvp,
            feature_kind=str(getattr(resolved_xc, "response_feature_kind", "LDA")),
            dtype=jnp.result_type(de_a, de_b),
            block_size=memory_plan.grid_block_size,
        )
    else:
        xc_response_action_fn = _unrestricted_grid_xc_response_action(
            molecule,
            resolved_xc,
            orbo_a,
            orbv_a,
            orbo_b,
            orbv_b,
            dtype=jnp.result_type(de_a, de_b),
            block_size=memory_plan.grid_block_size,
        )
    return _UnrestrictedResponseOperatorData(
        orbital_energy_differences_alpha=de_a,
        orbital_energy_differences_beta=de_b,
        orbo_alpha=orbo_a,
        orbv_alpha=orbv_a,
        orbo_beta=orbo_b,
        orbv_beta=orbv_b,
        ao_response_action_fn=ao_response_action_fn,
        ao_mo_response_action_fn=ao_mo_response_action_fn,
        xc_response_action_fn=xc_response_action_fn,
        fock_occupied=fock_occupied,
        fock_virtual=fock_virtual,
        memory_plan=memory_plan,
    )


def _unrestricted_dimensions(
    data: _UnrestrictedResponseOperatorData,
) -> tuple[int, int, int, int, int]:
    nocca, nvira = data.orbital_energy_differences_alpha.shape
    noccb, nvirb = data.orbital_energy_differences_beta.shape
    naa = int(nocca * nvira)
    nbb = int(noccb * nvirb)
    return int(nocca), int(nvira), int(noccb), int(nvirb), int(naa + nbb)


def _split_unrestricted_rows(
    data: _UnrestrictedResponseOperatorData,
    rows: Array,
) -> tuple[Array, Array]:
    nocca, nvira, noccb, nvirb, dim = _unrestricted_dimensions(data)
    rows = jnp.asarray(rows).reshape(-1, dim)
    batch = int(rows.shape[0])
    naa = int(nocca * nvira)
    alpha = rows[:, :naa].reshape(batch, nocca, nvira)
    beta = rows[:, naa:].reshape(batch, noccb, nvirb)
    return alpha, beta


def _join_unrestricted_rows(alpha: Array, beta: Array) -> Array:
    batch = int(alpha.shape[0])
    return jnp.concatenate(
        [
            alpha.reshape(batch, -1),
            beta.reshape(batch, -1),
        ],
        axis=-1,
    )


def _unrestricted_xc_action(
    data: _UnrestrictedResponseOperatorData,
    alpha: Array,
    beta: Array,
) -> tuple[Array, Array]:
    if data.xc_response_action_fn is None:
        return jnp.zeros_like(alpha), jnp.zeros_like(beta)
    return data.xc_response_action_fn(alpha, beta)


def _unrestricted_ao_mo_action(
    data: _UnrestrictedResponseOperatorData,
    alpha: Array,
    beta: Array,
    *,
    bottom_density: bool,
    bottom_projection: bool,
) -> tuple[Array, Array]:
    if data.ao_mo_response_action_fn is not None:
        return data.ao_mo_response_action_fn(
            alpha,
            beta,
            bottom_density=bottom_density,
            bottom_projection=bottom_projection,
        )
    density_alpha = _unrestricted_transition_density(
        data.orbo_alpha,
        data.orbv_alpha,
        alpha,
        bottom=bottom_density,
    )
    density_beta = _unrestricted_transition_density(
        data.orbo_beta,
        data.orbv_beta,
        beta,
        bottom=bottom_density,
    )
    response_alpha, response_beta = data.ao_response_action_fn(density_alpha, density_beta)
    out_alpha = _unrestricted_project_response(
        response_alpha,
        data.orbo_alpha,
        data.orbv_alpha,
        bottom=bottom_projection,
    )
    out_beta = _unrestricted_project_response(
        response_beta,
        data.orbo_beta,
        data.orbv_beta,
        bottom=bottom_projection,
    )
    return out_alpha, out_beta


def _unrestricted_a_action(
    data: _UnrestrictedResponseOperatorData,
    rows: Array,
) -> Array:
    alpha, beta = _split_unrestricted_rows(data, rows)
    out_alpha, out_beta = _unrestricted_ao_mo_action(
        data,
        alpha,
        beta,
        bottom_density=False,
        bottom_projection=False,
    )
    xc_alpha, xc_beta = _unrestricted_xc_action(data, alpha, beta)
    fock_alpha, fock_beta = _unrestricted_fock_actions(data, alpha, beta)
    out_alpha = out_alpha + xc_alpha + fock_alpha
    out_beta = out_beta + xc_beta + fock_beta
    return _join_unrestricted_rows(out_alpha, out_beta)


def _unrestricted_b_action(
    data: _UnrestrictedResponseOperatorData,
    rows: Array,
) -> Array:
    alpha, beta = _split_unrestricted_rows(data, rows)
    out_alpha, out_beta = _unrestricted_ao_mo_action(
        data,
        alpha,
        beta,
        bottom_density=True,
        bottom_projection=False,
    )
    xc_alpha, xc_beta = _unrestricted_xc_action(data, alpha, beta)
    out_alpha = out_alpha + xc_alpha
    out_beta = out_beta + xc_beta
    return _join_unrestricted_rows(out_alpha, out_beta)


def _unrestricted_tda_diagonal(data: _UnrestrictedResponseOperatorData) -> Array:
    return jnp.concatenate(
        [
            data.orbital_energy_differences_alpha.reshape(-1),
            data.orbital_energy_differences_beta.reshape(-1),
        ],
        axis=0,
    )


def _unrestricted_fock_actions(data, alpha, beta):
    gaps = (data.orbital_energy_differences_alpha, data.orbital_energy_differences_beta)
    return tuple(_restricted_fock_action(SimpleNamespace(
        delta_eps=gap,
        fock_occupied=None if data.fock_occupied is None else data.fock_occupied[spin],
        fock_virtual=None if data.fock_virtual is None else data.fock_virtual[spin],
    ), values) for spin, (gap, values) in enumerate(zip(gaps, (alpha, beta))))


def _unrestricted_tdhf_action(
    data: _UnrestrictedResponseOperatorData,
    x_rows: Array,
    y_rows: Array,
) -> tuple[Array, Array]:
    x_alpha, x_beta = _split_unrestricted_rows(data, x_rows)
    y_alpha, y_beta = _split_unrestricted_rows(data, y_rows)
    fock_x_alpha, fock_x_beta = _unrestricted_fock_actions(data, x_alpha, x_beta)
    fock_y_alpha, fock_y_beta = _unrestricted_fock_actions(data, y_alpha, y_beta)
    if data.ao_mo_response_action_fn is not None:
        upper_x_alpha, upper_x_beta = data.ao_mo_response_action_fn(
            x_alpha,
            x_beta,
            bottom_density=False,
            bottom_projection=False,
        )
        upper_y_alpha, upper_y_beta = data.ao_mo_response_action_fn(
            y_alpha,
            y_beta,
            bottom_density=True,
            bottom_projection=False,
        )
        lower_x_alpha, lower_x_beta = data.ao_mo_response_action_fn(
            x_alpha,
            x_beta,
            bottom_density=False,
            bottom_projection=True,
        )
        lower_y_alpha, lower_y_beta = data.ao_mo_response_action_fn(
            y_alpha,
            y_beta,
            bottom_density=True,
            bottom_projection=True,
        )
        xc_alpha, xc_beta = _unrestricted_xc_action(
            data,
            x_alpha + y_alpha,
            x_beta + y_beta,
        )
        upper_alpha = (
            upper_x_alpha
            + upper_y_alpha
            + xc_alpha
            + fock_x_alpha
        )
        upper_beta = (
            upper_x_beta
            + upper_y_beta
            + xc_beta
            + fock_x_beta
        )
        lower_alpha = (
            lower_x_alpha
            + lower_y_alpha
            + xc_alpha
            + fock_y_alpha
        )
        lower_beta = (
            lower_x_beta
            + lower_y_beta
            + xc_beta
            + fock_y_beta
        )
        return _join_unrestricted_rows(upper_alpha, upper_beta), _join_unrestricted_rows(
            lower_alpha,
            lower_beta,
        )

    density_alpha = _unrestricted_transition_density(
        data.orbo_alpha,
        data.orbv_alpha,
        x_alpha,
        bottom=False,
    ) + _unrestricted_transition_density(
        data.orbo_alpha,
        data.orbv_alpha,
        y_alpha,
        bottom=True,
    )
    density_beta = _unrestricted_transition_density(
        data.orbo_beta,
        data.orbv_beta,
        x_beta,
        bottom=False,
    ) + _unrestricted_transition_density(
        data.orbo_beta,
        data.orbv_beta,
        y_beta,
        bottom=True,
    )
    response_alpha, response_beta = data.ao_response_action_fn(density_alpha, density_beta)
    upper_alpha = _unrestricted_project_response(
        response_alpha,
        data.orbo_alpha,
        data.orbv_alpha,
        bottom=False,
    )
    upper_beta = _unrestricted_project_response(
        response_beta,
        data.orbo_beta,
        data.orbv_beta,
        bottom=False,
    )
    lower_alpha = _unrestricted_project_response(
        response_alpha,
        data.orbo_alpha,
        data.orbv_alpha,
        bottom=True,
    )
    lower_beta = _unrestricted_project_response(
        response_beta,
        data.orbo_beta,
        data.orbv_beta,
        bottom=True,
    )
    xc_alpha, xc_beta = _unrestricted_xc_action(
        data,
        x_alpha + y_alpha,
        x_beta + y_beta,
    )
    upper_alpha = upper_alpha + xc_alpha + fock_x_alpha
    upper_beta = upper_beta + xc_beta + fock_x_beta
    lower_alpha = lower_alpha + xc_alpha + fock_y_alpha
    lower_beta = lower_beta + xc_beta + fock_y_beta
    return _join_unrestricted_rows(upper_alpha, upper_beta), _join_unrestricted_rows(
        lower_alpha,
        lower_beta,
    )


def build_unrestricted_tda_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
) -> tuple[Callable[[Array], Array], Array, Array, Array]:
    data = _build_unrestricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        need_b_terms=False,
        response_kernel_options=response_kernel_options,
    )
    diagonal = _unrestricted_tda_diagonal(data)

    def vind(rows: Array) -> Array:
        _, _, _, _, dim = _unrestricted_dimensions(data)
        values = jnp.asarray(rows).reshape(-1, dim)
        return _map_response_vectors(lambda block: _unrestricted_a_action(data, block),
                                     values, data.memory_plan.batch_size)

    vind.memory_plan = data.memory_plan

    return (
        vind,
        diagonal,
        data.orbital_energy_differences_alpha,
        data.orbital_energy_differences_beta,
    )


def build_unrestricted_tdhf_operator(
    molecule: Any,
    xc_functional: Any | None = None,
    *,
    xc_params: Any | None = None,
    occupation_tolerance: float = 1e-8,
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None,
) -> tuple[Callable[[Array], Array], Array, Array]:
    data = _build_unrestricted_response_operator_data(
        molecule,
        xc_functional,
        xc_params=xc_params,
        occupation_tolerance=occupation_tolerance,
        response_kernel_options=response_kernel_options,
    )
    _, _, _, _, dim = _unrestricted_dimensions(data)

    def vind(rows: Array) -> Array:
        rows = jnp.asarray(rows).reshape(-1, 2 * dim)

        def block_action(block):
            upper, lower = _unrestricted_tdhf_action(data, block[:, :dim], block[:, dim:])
            return jnp.concatenate([upper, -lower], axis=-1)

        return _map_response_vectors(block_action, rows, data.memory_plan.batch_size)

    vind.memory_plan = data.memory_plan

    return (
        vind,
        data.orbital_energy_differences_alpha,
        data.orbital_energy_differences_beta,
    )


def _finalize_unrestricted_tda_result(
    eigvals: Array,
    eigvecs: Array,
    *,
    nroots: int,
    excitation_threshold: float,
    de_a: Array,
    de_b: Array,
    converged=True,
) -> UnrestrictedTDAResult:
    nocca, nvira = de_a.shape
    noccb, nvirb = de_b.shape
    naa = int(nocca * nvira)
    valid = eigvals > excitation_threshold
    order = jnp.argsort(jnp.where(valid, eigvals, jnp.inf))
    keep = order[:nroots]
    mask = valid[keep]
    energies = jnp.where(mask, eigvals[keep], 0.0)
    amps = eigvecs[:, keep].T * mask[:, None]
    amplitudes_alpha = amps[:, :naa].reshape(nroots, nocca, nvira)
    amplitudes_beta = amps[:, naa:].reshape(nroots, noccb, nvirb)
    return UnrestrictedTDAResult(
        excitation_energies=energies,
        amplitudes_alpha=amplitudes_alpha,
        amplitudes_beta=amplitudes_beta,
        converged=converged,
    )


def solve_unrestricted_tda_from_operator(
    de_a: Array,
    de_b: Array,
    vind_rows: Callable[[Array], Array],
    diagonal: Array,
    *,
    nstates: int | None = None,
    excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD,
    davidson_tol: float = PYSCF_TD_DAVIDSON_TOL,
    davidson_max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
    davidson_max_subspace: int | None = None,
    tda_gradient_mode: EigenGradientMode = "eigenvalue_only",
    eigenvector_adjoint_tol: float = 1e-6,
    eigenvector_adjoint_max_iter: int = 64,
) -> UnrestrictedTDAResult:
    dim = int(jnp.asarray(diagonal).size)
    nroots = dim if nstates is None else min(int(nstates), dim)
    if tda_gradient_mode not in {"eigenvalue_only", "implicit_eigenvector"}:
        raise ValueError(f"Unsupported TDA gradient mode {tda_gradient_mode!r}.")

    def matrix_action(vectors):
        return vind_rows(jnp.asarray(vectors).T).T

    diag = jnp.asarray(diagonal).reshape(dim)
    operator = LinearOperator(
        (dim, dim),
        diag.dtype,
        lambda v: matrix_action(v[:, None])[:, 0],
        diagonal=diag,
        matmat=matrix_action,
    )
    solved = solve_hermitian(
        operator,
        config=EigenSolverConfig(
            nroots=nroots,
            atol=davidson_tol,
            maxiter=davidson_max_iter,
            max_subspace=davidson_max_subspace,
            value_min=excitation_threshold,
        ),
        response=EigenResponseConfig(
            target=(
                "eigenpairs"
                if tda_gradient_mode == "implicit_eigenvector"
                else "eigenvalues"
            ),
            linear_config=LinearSolverConfig(
                rtol=eigenvector_adjoint_tol, maxiter=eigenvector_adjoint_max_iter
            ),
        ),
    )
    eigvals, eigvecs = solved.values, solved.vectors
    converged = jnp.all(solved.converged) & jnp.all(eigvals > excitation_threshold)
    return _finalize_unrestricted_tda_result(
        eigvals,
        eigvecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        de_a=de_a,
        de_b=de_b,
        converged=converged,
    )


def _finalize_unrestricted_casida_result(
    w: Array,
    x_vecs: Array,
    y_vecs: Array,
    *,
    nroots: int,
    excitation_threshold: float,
    matrix_eps: float,
    de_a: Array,
    de_b: Array,
    converged=True,
) -> UnrestrictedTDDFTResult:
    nocca, nvira = de_a.shape
    noccb, nvirb = de_b.shape
    naa = int(nocca * nvira)
    valid = jnp.isfinite(w) & (w > excitation_threshold)
    order = jnp.argsort(jnp.where(valid, w, jnp.inf))
    keep = order[:nroots]
    keep_mask = valid[keep]

    energies = jnp.where(keep_mask, w[keep], 0.0)
    x = x_vecs[:, keep]
    y = y_vecs[:, keep]
    x = x * keep_mask[jnp.newaxis, :]
    y = y * keep_mask[jnp.newaxis, :]
    norm = jnp.sum(jnp.abs(x) ** 2, axis=0) - jnp.sum(jnp.abs(y) ** 2, axis=0)
    scale = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(norm), matrix_eps))
    x = x * scale[jnp.newaxis, :]
    y = y * scale[jnp.newaxis, :]

    x_alpha = x[:naa, :].T.reshape(nroots, nocca, nvira)
    x_beta = x[naa:, :].T.reshape(nroots, noccb, nvirb)
    y_alpha = y[:naa, :].T.reshape(nroots, nocca, nvira)
    y_beta = y[naa:, :].T.reshape(nroots, noccb, nvirb)
    return UnrestrictedTDDFTResult(
        excitation_energies=energies,
        x_amplitudes_alpha=x_alpha,
        x_amplitudes_beta=x_beta,
        y_amplitudes_alpha=y_alpha,
        y_amplitudes_beta=y_beta,
        converged=jnp.asarray(converged) & jnp.all(keep_mask),
    )


def solve_unrestricted_casida_from_tdhf_operator(
    de_a: Array,
    de_b: Array,
    tdhf_vind_rows: Callable[[Array], Array],
    *,
    nstates: int | None = None,
    excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD,
    matrix_eps: float = 1e-10,
    davidson_tol: float = PYSCF_TD_DAVIDSON_TOL,
    davidson_max_iter: int = FULL_TDDFT_DAVIDSON_MAX_CYCLE,
    davidson_max_subspace: int | None = None,
) -> UnrestrictedTDDFTResult:
    dim = int(jnp.asarray(de_a).size + jnp.asarray(de_b).size)
    nroots = dim if nstates is None else min(int(nstates), dim)
    search_nroots = _davidson_search_nroots(nroots, dim)
    dtype = _solver_dtype(jnp.result_type(de_a, de_b))

    def tdhf_vind(values: Array) -> Array:
        values = jnp.asarray(values, dtype=dtype).reshape(-1, 2 * dim)
        return tdhf_vind_rows(values)

    diagonal = jnp.concatenate([jnp.ravel(de_a), jnp.ravel(de_b)]).astype(dtype)
    w, x_vecs, y_vecs, converged = implicit_differential_davidson_lowest_tdhf(
        tdhf_vind,
        nroots=search_nroots,
        size=dim,
        diag=diagonal,
        tol=davidson_tol,
        max_iter=davidson_max_iter,
        max_subspace=davidson_max_subspace,
        matrix_eps=matrix_eps,
    )
    return _finalize_unrestricted_casida_result(
        w,
        x_vecs,
        y_vecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        matrix_eps=matrix_eps,
        de_a=de_a,
        de_b=de_b,
        converged=converged,
    )


def _unrestricted_solver_memory_plan(driver, de_a, de_b, nstates, *, rpa=False):
    dim = int(de_a.size + de_b.size)
    nroots = dim if nstates is None else min(int(nstates), dim)
    return _unrestricted_memory_plan(
        driver.molecule,
        _resolve_xc_functional(driver.molecule, driver.xc_functional, driver.xc_params),
        de_a, de_b, normalize_response_kernel_options(driver.response_kernel_options),
        nroots=nroots, max_subspace=driver.davidson_max_subspace, rpa=rpa,
    )


@dataclass(frozen=True)
class UnrestrictedTDA:
    """PySCF-like unrestricted TDA driver."""

    molecule: Any
    xc_functional: Any | None = None
    xc_params: Any | None = None
    occupation_tolerance: float = 1e-8
    excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD
    eigensolver: Literal["auto", "davidson"] = "auto"
    davidson_tol: float = PYSCF_TD_DAVIDSON_TOL
    davidson_max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE
    davidson_max_subspace: int | None = None
    tda_gradient_mode: EigenGradientMode = "eigenvalue_only"
    eigenvector_adjoint_tol: float = 1e-6
    eigenvector_adjoint_max_iter: int = 64
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None

    def gen_tda_vind(self):
        vind, _, _, _ = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            response_kernel_options=self.response_kernel_options,
        )
        return vind

    def _posthoc_correction(self, result: UnrestrictedTDAResult) -> Array | None:
        resolved_xc = _resolve_xc_functional(
            self.molecule,
            self.xc_functional,
            self.xc_params,
        )
        if resolved_xc is None:
            return None
        correction_fn = getattr(resolved_xc, "post_tda_correction", None)
        if not callable(correction_fn):
            return None
        try:
            correction = correction_fn(
                self.molecule,
                result,
                occupation_tolerance=self.occupation_tolerance,
            )
        except AttributeError as exc:
            if "does not expose" not in str(exc):
                raise
            return None
        correction = jnp.asarray(correction, dtype=result.excitation_energies.dtype)
        if correction.ndim == 0:
            correction = jnp.full_like(result.excitation_energies, correction)
        elif correction.shape != result.excitation_energies.shape:
            raise ValueError(
                "post_tda_correction must return a scalar or shape "
                f"{result.excitation_energies.shape}, got {correction.shape}."
            )
        return correction

    def _apply_posthoc_correction(
        self,
        result: UnrestrictedTDAResult,
    ) -> UnrestrictedTDAResult:
        correction = self._posthoc_correction(result)
        if correction is None:
            return result
        return replace(
            result,
            excitation_energies=result.excitation_energies + correction,
            posthoc_correction=correction,
        )

    def kernel(self, nstates: int | None = None) -> UnrestrictedTDAResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )
        vind, diagonal, de_a, de_b = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            response_kernel_options=self.response_kernel_options,
        )
        memory = _unrestricted_solver_memory_plan(self, de_a, de_b, nstates)
        result = solve_unrestricted_tda_from_operator(
            de_a,
            de_b,
            vind,
            diagonal,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=memory.max_subspace,
            tda_gradient_mode=self.tda_gradient_mode,
            eigenvector_adjoint_tol=self.eigenvector_adjoint_tol,
            eigenvector_adjoint_max_iter=self.eigenvector_adjoint_max_iter,
        )
        return self._apply_posthoc_correction(result)


@dataclass(frozen=True)
class UnrestrictedCasidaTDDFT:
    """PySCF-like unrestricted Casida TDDFT driver."""

    molecule: Any
    xc_functional: Any | None = None
    xc_params: Any | None = None
    occupation_tolerance: float = 1e-8
    excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD
    matrix_eps: float = 1e-10
    eigensolver: Literal["auto", "davidson"] = "auto"
    davidson_tol: float = PYSCF_TD_DAVIDSON_TOL
    davidson_max_iter: int = FULL_TDDFT_DAVIDSON_MAX_CYCLE
    davidson_max_subspace: int | None = None
    tda_gradient_mode: EigenGradientMode = "eigenvalue_only"
    eigenvector_adjoint_tol: float = 1e-6
    eigenvector_adjoint_max_iter: int = 64
    response_kernel_options: ResponseKernelOptions | dict[str, Any] | None = None

    def tda(self, nstates: int | None = None) -> UnrestrictedTDAResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )
        vind, diagonal, de_a, de_b = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            response_kernel_options=self.response_kernel_options,
        )
        memory = _unrestricted_solver_memory_plan(self, de_a, de_b, nstates)
        result = solve_unrestricted_tda_from_operator(
            de_a,
            de_b,
            vind,
            diagonal,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=memory.max_subspace,
            tda_gradient_mode=self.tda_gradient_mode,
            eigenvector_adjoint_tol=self.eigenvector_adjoint_tol,
            eigenvector_adjoint_max_iter=self.eigenvector_adjoint_max_iter,
        )
        return self._apply_posthoc_correction(result, use_tda=True)

    def gen_tda_vind(self):
        vind, _, _, _ = build_unrestricted_tda_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            response_kernel_options=self.response_kernel_options,
        )
        return vind

    def gen_tdhf_vind(self):
        vind, _, _ = build_unrestricted_tdhf_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            response_kernel_options=self.response_kernel_options,
        )
        return vind

    def _posthoc_correction(
        self,
        result: UnrestrictedTDAResult | UnrestrictedTDDFTResult,
        *,
        use_tda: bool,
    ) -> Array | None:
        resolved_xc = _resolve_xc_functional(
            self.molecule,
            self.xc_functional,
            self.xc_params,
        )
        if resolved_xc is None:
            return None
        method_name = "post_tda_correction" if use_tda else "post_tddft_correction"
        correction_fn = getattr(resolved_xc, method_name, None)
        if not callable(correction_fn):
            return None
        try:
            correction = correction_fn(
                self.molecule,
                result,
                occupation_tolerance=self.occupation_tolerance,
            )
        except AttributeError as exc:
            if "does not expose" not in str(exc):
                raise
            return None
        correction = jnp.asarray(correction, dtype=result.excitation_energies.dtype)
        if correction.ndim == 0:
            correction = jnp.full_like(result.excitation_energies, correction)
        elif correction.shape != result.excitation_energies.shape:
            raise ValueError(
                f"{method_name} must return a scalar or shape "
                f"{result.excitation_energies.shape}, got {correction.shape}."
            )
        return correction

    def _apply_posthoc_correction(
        self,
        result: UnrestrictedTDAResult | UnrestrictedTDDFTResult,
        *,
        use_tda: bool,
    ) -> UnrestrictedTDAResult | UnrestrictedTDDFTResult:
        correction = self._posthoc_correction(result, use_tda=use_tda)
        if correction is None:
            return result
        return replace(
            result,
            excitation_energies=result.excitation_energies + correction,
            posthoc_correction=correction,
        )

    def kernel(self, nstates: int | None = None) -> UnrestrictedTDDFTResult:
        mode = str(self.eigensolver).lower()
        if mode not in {"auto", "davidson"}:
            raise ValueError(
                f"Unsupported eigensolver={self.eigensolver!r}. Choose one of {{'auto', 'davidson'}}."
            )
        vind_tdhf, de_a, de_b = build_unrestricted_tdhf_operator(
            self.molecule,
            self.xc_functional,
            xc_params=self.xc_params,
            occupation_tolerance=self.occupation_tolerance,
            response_kernel_options=self.response_kernel_options,
        )
        memory = _unrestricted_solver_memory_plan(self, de_a, de_b, nstates, rpa=True)
        result = solve_unrestricted_casida_from_tdhf_operator(
            de_a,
            de_b,
            vind_tdhf,
            nstates=nstates,
            excitation_threshold=self.excitation_threshold,
            matrix_eps=self.matrix_eps,
            davidson_tol=self.davidson_tol,
            davidson_max_iter=self.davidson_max_iter,
            davidson_max_subspace=memory.max_subspace,
        )
        return self._apply_posthoc_correction(result, use_tda=False)
