"""Integral and grid payloads consumed by restricted and unrestricted SCF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from jaxtyping import Array
from gradscf.integrals.basis import CartesianBasis

GeometryGradPolicy = Literal["analytic", "error", "zero"]

@dataclass(frozen=True)
class RKSIntegralInputs:
    """AO integrals, grid data, and JK data required by the RKS SCF kernel."""

    basis: CartesianBasis
    overlap: Array
    hcore: Array
    eri: Array | None
    eri_pair_matrix: Array | None
    df_factors: Array | None
    direct_basis: CartesianBasis | None
    nelectron: int
    nuclear_repulsion: float | Array
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None
    dipole_integrals: Array | None
    init_density: Array | None = None
    init_mo_coeff: Array | None = None
    init_mo_occ: Array | None = None
    init_mo_energy: Array | None = None
    molecule_charge: int = 0
    geometry_is_traced: bool = False
    integral_backend: str = "cpu"
    grid_ao_backend: str = "jax"
    eri_lr: Array | None = None
    dispersion_energy: Array | None = None
    grid_ao: Any | None = None

    def as_rks_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by run_rks_from_integrals."""

        kwargs = {
            "overlap": self.overlap,
            "hcore": self.hcore,
            "eri": self.eri,
            "eri_pair_matrix": self.eri_pair_matrix,
            "nelectron": self.nelectron,
            "nuclear_repulsion": self.nuclear_repulsion,
            "ao": self.ao,
            "ao_deriv1": self.ao_deriv1,
            "grid_weights": self.grid_weights,
            "df_factors": self.df_factors,
            "direct_basis": self.direct_basis,
            "init_density": self.init_density,
            "init_mo_coeff": self.init_mo_coeff,
            "init_mo_occ": self.init_mo_occ,
            "init_mo_energy": self.init_mo_energy,
        }
        if self.grid_ao is not None:
            kwargs["grid_ao"] = self.grid_ao
        if self.eri_lr is not None or self.dispersion_energy is not None:
            kwargs.update(eri_lr=self.eri_lr, dispersion_energy=self.dispersion_energy)
        return kwargs

    def response_eri_pair_matrix(self) -> Array | None:
        """Return packed AO-pair ERI data for response assembly when available."""

        if self.eri_pair_matrix is not None:
            return self.eri_pair_matrix
        if self.direct_basis is not None:
            if hasattr(self.direct_basis,'response_eri_pair_matrix'):
                return self.direct_basis.response_eri_pair_matrix()
            # Resolve the shared public/legacy patch point only when needed.
            from gradscf.scf.inputs.assembly import eri_pair_matrix_packed

            return eri_pair_matrix_packed(self.direct_basis)
        return None


@dataclass(frozen=True)
class UKSIntegralInputs:
    """AO/grid data for UKS, with full/packed ERIs or an AO J/K provider."""

    basis: CartesianBasis
    overlap: Array
    hcore: Array
    eri: Array
    nalpha: int
    nbeta: int
    nuclear_repulsion: float | Array
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None
    dipole_integrals: Array
    df_factors: Array | None = None
    direct_basis: Any | None = None
    init_density_alpha: Array | None = None
    init_density_beta: Array | None = None
    init_mo_coeff_alpha: Array | None = None
    init_mo_coeff_beta: Array | None = None
    init_mo_occ_alpha: Array | None = None
    init_mo_occ_beta: Array | None = None
    init_mo_energy_alpha: Array | None = None
    init_mo_energy_beta: Array | None = None
    total_electrons: int = 0
    molecule_charge: int = 0
    geometry_is_traced: bool = False
    integral_backend: str = "cpu"
    grid_ao_backend: str = "jax"
    eri_lr: Array | None = None
    dispersion_energy: Array | None = None
    grid_ao: Any | None = None

    def as_uks_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by run_uks_from_integrals."""

        kwargs = {
            "overlap": self.overlap,
            "hcore": self.hcore,
            "eri": self.eri,
            "df_factors": self.df_factors,
            "nalpha": self.nalpha,
            "nbeta": self.nbeta,
            "nuclear_repulsion": self.nuclear_repulsion,
            "ao": self.ao,
            "ao_deriv1": self.ao_deriv1,
            "grid_weights": self.grid_weights,
            "init_density_alpha": self.init_density_alpha,
            "init_density_beta": self.init_density_beta,
            "init_mo_coeff_alpha": self.init_mo_coeff_alpha,
            "init_mo_coeff_beta": self.init_mo_coeff_beta,
            "init_mo_occ_alpha": self.init_mo_occ_alpha,
            "init_mo_occ_beta": self.init_mo_occ_beta,
            "init_mo_energy_alpha": self.init_mo_energy_alpha,
            "init_mo_energy_beta": self.init_mo_energy_beta,
        }
        if self.grid_ao is not None:
            kwargs["grid_ao"] = self.grid_ao
        if self.eri_lr is not None or self.dispersion_energy is not None:
            kwargs.update(eri_lr=self.eri_lr, dispersion_energy=self.dispersion_energy)
        if self.direct_basis is not None:
            kwargs["direct_basis"] = self.direct_basis
        return kwargs
