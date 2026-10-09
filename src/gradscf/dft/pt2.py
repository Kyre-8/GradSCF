"""Shared second-order correlation features and orbital response contractions."""
from __future__ import annotations
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import Precision
from gradscf.integrals import eri_pair_matrix_to_mo_eri_slices
from gradscf.integrals.layouts import _metadata_arrays, _mo_pair_products
from ..df import df_factors_to_mo_eri_slices

def _local_pt2_feature_from_restricted_orbitals(
    ao: Any,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    *,
    rep_tensor: Any | None = None,
    eri_ovov: Any | None = None,
    eri_pair_matrix: Any | None = None,
    df_factors: Any | None = None,
    nocc: int | None = None,
    occupation_tolerance: float = 1e-8,
    density_floor: float = 1e-12,
) -> jnp.ndarray:
    ao_arr = jnp.asarray(ao)
    mo_coeff_arr = jnp.asarray(mo_coeff)
    mo_occ_arr = jnp.asarray(mo_occ)
    mo_energy_arr = jnp.asarray(mo_energy)

    if mo_coeff_arr.ndim == 3:
        mo_coeff_arr = mo_coeff_arr[0]
    if mo_occ_arr.ndim == 2:
        mo_occ_arr = mo_occ_arr[0]
    if mo_energy_arr.ndim == 2:
        mo_energy_arr = mo_energy_arr[0]

    nocc_int = (
        int(nocc)
        if nocc is not None
        else int(jnp.count_nonzero(mo_occ_arr > occupation_tolerance))
    )
    nmo = int(mo_coeff_arr.shape[1])
    if nocc_int <= 0 or nocc_int >= nmo:
        raise ValueError("PT2 local feature requires at least one occupied and one virtual orbital.")

    orbo = mo_coeff_arr[:, :nocc_int]
    orbv = mo_coeff_arr[:, nocc_int:]
    eps_occ = mo_energy_arr[:nocc_int]
    eps_vir = mo_energy_arr[nocc_int:]

    eri_ovov_arr = None if eri_ovov is None else jnp.asarray(eri_ovov)
    if eri_ovov_arr is None:
        if df_factors is not None:
            factors = jnp.asarray(df_factors)
            if factors.size != 0:
                eri_ovov_arr, _, _ = df_factors_to_mo_eri_slices(
                    factors,
                    mo_coeff_arr,
                    nocc_int,
                    include_oovv=False,
                )
        if eri_ovov_arr is None and eri_pair_matrix is not None:
            pair = jnp.asarray(eri_pair_matrix)
            if pair.size != 0:
                eri_ovov_arr, _, _ = eri_pair_matrix_to_mo_eri_slices(
                    pair,
                    mo_coeff_arr,
                    nocc=nocc_int,
                    include_oovv=False,
                )
        if eri_ovov_arr is None:
            if rep_tensor is None:
                raise ValueError(
                    "PT2 local feature requires rep_tensor, eri_ovov, eri_pair_matrix, "
                    "or df_factors."
                )
            rep = jnp.asarray(rep_tensor)
            if rep.size == 0:
                raise ValueError(
                    "PT2 local feature cannot be constructed from an empty rep_tensor without df_factors."
                )
            eri_ovov_arr = jnp.einsum(
                "pqrs,pi,qa,rj,sb->iajb",
                rep,
                orbo,
                orbv,
                orbo,
                orbv,
                precision=Precision.HIGHEST,
            )

    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    denom = jnp.where(jnp.abs(denom) > density_floor, denom, -density_floor)
    direct = eri_ovov_arr
    exchange = jnp.transpose(eri_ovov_arr, (0, 3, 2, 1))
    pair_weights = (2.0 * direct - exchange) / denom

    rho_o = jnp.einsum("rp,pi->ri", ao_arr, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao_arr, orbv, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)

    if df_factors is not None and jnp.asarray(df_factors).size != 0:
        factors = jnp.asarray(df_factors)
        grid_aux = jnp.einsum(
            "Qpq,gp,gq->gQ",
            factors,
            ao_arr,
            ao_arr,
            precision=Precision.HIGHEST,
        )
        qjb = jnp.einsum(
            "Qrs,rj,sb->Qjb",
            factors,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
        pair_potential = jnp.einsum(
            "gQ,Qjb->gjb",
            grid_aux,
            qjb,
            precision=Precision.HIGHEST,
        )
    elif eri_pair_matrix is not None and jnp.asarray(eri_pair_matrix).size != 0:
        pair = jnp.asarray(eri_pair_matrix)
        rows, cols, _, multiplicity = _metadata_arrays(
            int(mo_coeff_arr.shape[0]),
            ao_arr.dtype,
        )
        grid_pair = ao_arr[:, rows] * ao_arr[:, cols] * multiplicity[None, :]
        ov = _mo_pair_products(orbo, orbv, rows, cols)
        pair_potential = jnp.einsum(
            "gP,PQ,jbQ->gjb",
            grid_pair,
            pair,
            ov,
            precision=Precision.HIGHEST,
        )
    else:
        rep = jnp.asarray(rep_tensor)
        pair_potential = jnp.einsum(
            "gp,gq,pqrs,rj,sb->gjb",
            ao_arr,
            ao_arr,
            rep,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
    local_energy = jnp.einsum(
        "ria,rjb,iajb->r",
        rho_ov,
        pair_potential,
        pair_weights,
        precision=Precision.HIGHEST,
    )
    return jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)


def _local_pt2_feature_and_fock_response_from_restricted_orbitals(
    ao: Any,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    *,
    rep_tensor: Any | None = None,
    eri_ovov: Any | None = None,
    eri_pair_matrix: Any | None = None,
    df_factors: Any | None = None,
    nocc: int | None = None,
    occupation_tolerance: float = 1e-8,
    density_floor: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return fixed-reference restricted PT2 local energy and AO Fock response.

    The response is the derivative of the fixed-source linearization
    ``p_g[D] = Tr[D R_g]`` with respect to the spin-summed AO density ``D``.
    At the reference closed-shell density it reconstructs the cached local PT2
    feature while avoiding AD through MP2 amplitudes during SCF/training.
    """

    ao_arr = jnp.asarray(ao)
    mo_coeff_arr = jnp.asarray(mo_coeff)
    mo_occ_arr = jnp.asarray(mo_occ)
    mo_energy_arr = jnp.asarray(mo_energy)

    if mo_coeff_arr.ndim == 3:
        mo_coeff_arr = mo_coeff_arr[0]
    if mo_occ_arr.ndim == 2:
        mo_occ_arr = mo_occ_arr[0]
    if mo_energy_arr.ndim == 2:
        mo_energy_arr = mo_energy_arr[0]

    nocc_int = int(nocc) if nocc is not None else int(jnp.count_nonzero(mo_occ_arr > occupation_tolerance))
    nmo = int(mo_coeff_arr.shape[1])
    if nocc_int <= 0 or nocc_int >= nmo:
        raise ValueError("PT2 local feature requires at least one occupied and one virtual orbital.")

    orbo = mo_coeff_arr[:, :nocc_int]
    orbv = mo_coeff_arr[:, nocc_int:]
    eps_occ = mo_energy_arr[:nocc_int]
    eps_vir = mo_energy_arr[nocc_int:]

    eri_ovov_arr = None if eri_ovov is None else jnp.asarray(eri_ovov)
    if eri_ovov_arr is None:
        if df_factors is not None:
            factors = jnp.asarray(df_factors)
            if factors.size != 0:
                eri_ovov_arr, _, _ = df_factors_to_mo_eri_slices(
                    factors,
                    mo_coeff_arr,
                    nocc_int,
                    include_oovv=False,
                )
        if eri_ovov_arr is None and eri_pair_matrix is not None:
            pair = jnp.asarray(eri_pair_matrix)
            if pair.size != 0:
                eri_ovov_arr, _, _ = eri_pair_matrix_to_mo_eri_slices(
                    pair,
                    mo_coeff_arr,
                    nocc=nocc_int,
                    include_oovv=False,
                )
        if eri_ovov_arr is None:
            if rep_tensor is None:
                raise ValueError(
                    "PT2 local feature requires rep_tensor, eri_ovov, eri_pair_matrix, "
                    "or df_factors."
                )
            rep = jnp.asarray(rep_tensor)
            if rep.size == 0:
                raise ValueError(
                    "PT2 local feature cannot be constructed from an empty rep_tensor without df_factors."
                )
            eri_ovov_arr = jnp.einsum(
                "pqrs,pi,qa,rj,sb->iajb",
                rep,
                orbo,
                orbv,
                orbo,
                orbv,
                precision=Precision.HIGHEST,
            )

    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    denom = jnp.where(jnp.abs(denom) > density_floor, denom, -density_floor)
    direct = eri_ovov_arr
    exchange = jnp.transpose(eri_ovov_arr, (0, 3, 2, 1))
    pair_weights = (2.0 * direct - exchange) / denom

    rho_o = jnp.einsum("rp,pi->ri", ao_arr, orbo, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao_arr, orbv, precision=Precision.HIGHEST)

    if df_factors is not None and jnp.asarray(df_factors).size != 0:
        factors = jnp.asarray(df_factors)
        grid_aux = jnp.einsum(
            "Qpq,gp,gq->gQ",
            factors,
            ao_arr,
            ao_arr,
            precision=Precision.HIGHEST,
        )
        qjb = jnp.einsum(
            "Qrs,rj,sb->Qjb",
            factors,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )
        pair_potential = jnp.einsum(
            "gQ,Qjb->gjb",
            grid_aux,
            qjb,
            precision=Precision.HIGHEST,
        )
    elif eri_pair_matrix is not None and jnp.asarray(eri_pair_matrix).size != 0:
        pair = jnp.asarray(eri_pair_matrix)
        rows, cols, _, multiplicity = _metadata_arrays(int(mo_coeff_arr.shape[0]), ao_arr.dtype)
        grid_pair = ao_arr[:, rows] * ao_arr[:, cols] * multiplicity[None, :]
        ov = _mo_pair_products(orbo, orbv, rows, cols)
        pair_potential = jnp.einsum(
            "gP,PQ,jbQ->gjb",
            grid_pair,
            pair,
            ov,
            precision=Precision.HIGHEST,
        )
    else:
        rep = jnp.asarray(rep_tensor)
        pair_potential = jnp.einsum(
            "gp,gq,pqrs,rj,sb->gjb",
            ao_arr,
            ao_arr,
            rep,
            orbo,
            orbv,
            precision=Precision.HIGHEST,
        )

    source_occ = jnp.einsum(
        "ga,gjb,iajb->gi",
        rho_v,
        pair_potential,
        pair_weights,
        precision=Precision.HIGHEST,
    )
    local_energy = jnp.einsum("gi,gi->g", rho_o, source_occ, precision=Precision.HIGHEST)
    local_energy = jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)

    occ_gram_inv = jnp.linalg.pinv(
        jnp.einsum("pi,pj->ij", orbo, orbo, precision=Precision.HIGHEST)
    )
    source_ao = jnp.einsum(
        "pi,ij,gj->gp",
        orbo,
        occ_gram_inv,
        source_occ,
        precision=Precision.HIGHEST,
    )
    response = 0.25 * (
        ao_arr[:, :, None] * source_ao[:, None, :]
        + source_ao[:, :, None] * ao_arr[:, None, :]
    )
    response = jnp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
    return local_energy, response


def _local_pt2_feature_from_unrestricted_orbitals(
    ao: Any,
    mo_coeff: Any,
    mo_occ: Any,
    mo_energy: Any,
    *,
    rep_tensor: Any | None = None,
    eri_pair_matrix: Any | None = None,
    df_factors: Any | None = None,
    occupation_tolerance: float = 1e-8,
    density_floor: float = 1e-12,
    return_total_energy: bool = False,
    return_fock_response: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    ao_arr = jnp.asarray(ao)
    mo_coeff_arr = jnp.asarray(mo_coeff)
    mo_occ_arr = jnp.asarray(mo_occ)
    mo_energy_arr = jnp.asarray(mo_energy)

    if mo_coeff_arr.ndim != 3 or mo_coeff_arr.shape[0] != 2:
        raise ValueError(
            "Unrestricted PT2 local feature expects mo_coeff with shape (2, nao, nmo)."
        )
    if mo_occ_arr.ndim != 2 or mo_occ_arr.shape[0] != 2:
        raise ValueError(
            "Unrestricted PT2 local feature expects mo_occ with shape (2, nmo)."
        )
    if mo_energy_arr.ndim != 2 or mo_energy_arr.shape[0] != 2:
        raise ValueError(
            "Unrestricted PT2 local feature expects mo_energy with shape (2, nmo)."
        )

    occ_a = jnp.where(mo_occ_arr[0] > occupation_tolerance)[0]
    vir_a = jnp.where(mo_occ_arr[0] <= occupation_tolerance)[0]
    occ_b = jnp.where(mo_occ_arr[1] > occupation_tolerance)[0]
    vir_b = jnp.where(mo_occ_arr[1] <= occupation_tolerance)[0]

    occ_coeff_a = mo_coeff_arr[0][:, occ_a]
    occ_coeff_b = mo_coeff_arr[1][:, occ_b]
    vir_coeff_a = mo_coeff_arr[0][:, vir_a]
    vir_coeff_b = mo_coeff_arr[1][:, vir_b]
    occ_coeff = jnp.concatenate([occ_coeff_a, occ_coeff_b], axis=1)
    vir_coeff = jnp.concatenate([vir_coeff_a, vir_coeff_b], axis=1)
    occ_spin = jnp.concatenate(
        [
            jnp.zeros((occ_a.shape[0],), dtype=jnp.int32),
            jnp.ones((occ_b.shape[0],), dtype=jnp.int32),
        ]
    )
    vir_spin = jnp.concatenate(
        [
            jnp.zeros((vir_a.shape[0],), dtype=jnp.int32),
            jnp.ones((vir_b.shape[0],), dtype=jnp.int32),
        ]
    )
    eps_occ = jnp.concatenate([mo_energy_arr[0][occ_a], mo_energy_arr[1][occ_b]])
    eps_vir = jnp.concatenate([mo_energy_arr[0][vir_a], mo_energy_arr[1][vir_b]])

    nocc_total = int(occ_coeff.shape[1])
    nvir_total = int(vir_coeff.shape[1])
    zero_local = jnp.zeros((int(ao_arr.shape[0]),), dtype=ao_arr.dtype)
    zero_total = jnp.asarray(0.0, dtype=ao_arr.dtype)
    zero_response = jnp.zeros(
        (2, int(ao_arr.shape[0]), int(ao_arr.shape[1]), int(ao_arr.shape[1])),
        dtype=ao_arr.dtype,
    )
    if nocc_total < 2 or nvir_total < 2:
        outputs = [zero_local]
        if return_fock_response:
            outputs.append(zero_response)
        if return_total_energy:
            outputs.append(zero_total)
        return tuple(outputs) if len(outputs) > 1 else zero_local

    pair = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
    factors = None if df_factors is None else jnp.asarray(df_factors)
    rep = None if rep_tensor is None else jnp.asarray(rep_tensor)

    if factors is not None and factors.size != 0:
        b_ov = jnp.einsum(
            "Qpq,pi,qa->Qia",
            factors,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )
        direct_spatial = jnp.einsum("Qia,Qjb->iajb", b_ov, b_ov, precision=Precision.HIGHEST)
    elif pair is not None and pair.size != 0:
        rows, cols, _, _ = _metadata_arrays(int(mo_coeff_arr.shape[1]), ao_arr.dtype)
        ov = _mo_pair_products(occ_coeff, vir_coeff, rows, cols)
        direct_spatial = jnp.einsum(
            "iaP,PQ,jbQ->iajb",
            ov,
            pair,
            ov,
            precision=Precision.HIGHEST,
        )
    else:
        if rep is None or rep.size == 0:
            raise ValueError(
                "Unrestricted PT2 local feature requires rep_tensor, eri_pair_matrix, or df_factors."
            )
        direct_spatial = jnp.einsum(
            "pqrs,pi,qa,rj,sb->iajb",
            rep,
            occ_coeff,
            vir_coeff,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )

    exchange_spatial = jnp.transpose(direct_spatial, (0, 3, 2, 1))
    mask_direct = (
        (occ_spin[:, None, None, None] == vir_spin[None, :, None, None])
        & (occ_spin[None, None, :, None] == vir_spin[None, None, None, :])
    )
    mask_exchange = (
        (occ_spin[:, None, None, None] == vir_spin[None, None, None, :])
        & (occ_spin[None, None, :, None] == vir_spin[None, :, None, None])
    )
    direct = direct_spatial * mask_direct.astype(direct_spatial.dtype)
    exchange = exchange_spatial * mask_exchange.astype(exchange_spatial.dtype)

    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, None, :, None]
        - eps_vir[None, :, None, None]
        - eps_vir[None, None, None, :]
    )
    denom = jnp.where(jnp.abs(denom) > density_floor, denom, -density_floor)
    amplitudes = (direct - exchange) / denom
    pair_weights = 0.5 * amplitudes
    total_energy = jnp.sum(direct * pair_weights)

    rho_o = jnp.einsum("rp,pi->ri", ao_arr, occ_coeff, precision=Precision.HIGHEST)
    rho_v = jnp.einsum("rp,pa->ra", ao_arr, vir_coeff, precision=Precision.HIGHEST)
    rho_ov = jnp.einsum("ri,ra->ria", rho_o, rho_v, precision=Precision.HIGHEST)

    if factors is not None and factors.size != 0:
        grid_aux = jnp.einsum(
            "Qpq,gp,gq->gQ",
            factors,
            ao_arr,
            ao_arr,
            precision=Precision.HIGHEST,
        )
        qjb = jnp.einsum(
            "Qrs,rj,sb->Qjb",
            factors,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )
        pair_potential = jnp.einsum(
            "gQ,Qjb->gjb",
            grid_aux,
            qjb,
            precision=Precision.HIGHEST,
        )
    elif pair is not None and pair.size != 0:
        rows, cols, _, multiplicity = _metadata_arrays(int(mo_coeff_arr.shape[1]), ao_arr.dtype)
        grid_pair = ao_arr[:, rows] * ao_arr[:, cols] * multiplicity[None, :]
        ov = _mo_pair_products(occ_coeff, vir_coeff, rows, cols)
        pair_potential = jnp.einsum(
            "gP,PQ,jbQ->gjb",
            grid_pair,
            pair,
            ov,
            precision=Precision.HIGHEST,
        )
    else:
        pair_potential = jnp.einsum(
            "gp,gq,pqrs,rj,sb->gjb",
            ao_arr,
            ao_arr,
            rep,
            occ_coeff,
            vir_coeff,
            precision=Precision.HIGHEST,
        )

    local_energy = jnp.einsum(
        "ria,rjb,iajb->r",
        rho_ov,
        pair_potential,
        pair_weights,
        precision=Precision.HIGHEST,
    )
    local_energy = jnp.nan_to_num(local_energy, nan=0.0, posinf=0.0, neginf=0.0)
    total_energy = jnp.nan_to_num(total_energy, nan=0.0, posinf=0.0, neginf=0.0)
    outputs = [local_energy]
    if return_fock_response:
        source_occ = jnp.einsum(
            "ga,gjb,iajb->gi",
            rho_v,
            pair_potential,
            pair_weights,
            precision=Precision.HIGHEST,
        )
        nocc_a = int(occ_coeff_a.shape[1])
        source_occ_a = source_occ[:, :nocc_a]
        source_occ_b = source_occ[:, nocc_a:]

        def spin_response(
            occ_coeff_spin: jnp.ndarray,
            source_occ_spin: jnp.ndarray,
        ) -> jnp.ndarray:
            if int(occ_coeff_spin.shape[1]) <= 0:
                return zero_response[0]
            occ_gram_inv = jnp.linalg.pinv(
                jnp.einsum(
                    "pi,pj->ij",
                    occ_coeff_spin,
                    occ_coeff_spin,
                    precision=Precision.HIGHEST,
                )
            )
            source_ao = jnp.einsum(
                "pi,ij,gj->gp",
                occ_coeff_spin,
                occ_gram_inv,
                source_occ_spin,
                precision=Precision.HIGHEST,
            )
            response = 0.5 * (
                ao_arr[:, :, None] * source_ao[:, None, :]
                + source_ao[:, :, None] * ao_arr[:, None, :]
            )
            return jnp.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)

        outputs.append(
            jnp.stack(
                [
                    spin_response(occ_coeff_a, source_occ_a),
                    spin_response(occ_coeff_b, source_occ_b),
                ]
            )
        )
    if return_total_energy:
        outputs.append(total_energy)
    return tuple(outputs) if len(outputs) > 1 else local_energy

__all__ = [
    '_local_pt2_feature_and_fock_response_from_restricted_orbitals',
    '_local_pt2_feature_from_restricted_orbitals',
    '_local_pt2_feature_from_unrestricted_orbitals',
]
