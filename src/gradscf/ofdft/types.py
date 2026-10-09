"""Explicit OFDFT arrays: all continuous physical inputs are PyTree leaves."""
from dataclasses import dataclass, field
from typing import NamedTuple
from math import isfinite

from ..scf._pytree import pytree_dataclass
from ..scf.autodiff import SCFDifferentiationConfig


@pytree_dataclass(static_fields=('representation', 'mesh'))
@dataclass(frozen=True)
class OFDFTInputs:
    representation: str
    weights: object
    coordinates: object
    nelectron: object
    external_potential: object
    nuclear_repulsion: object = 0.
    overlap: object = None
    kinetic_matrix: object = None
    ao: object = None
    ao_gradient: object = None
    eri: object = None
    df_factors: object = None
    lattice: object = None
    mesh: tuple = ()


@pytree_dataclass(static_fields=('mesh',))
@dataclass(frozen=True)
class KineticFeatures:
    rho: object
    grad_rho: object
    weights: object
    coordinates: object
    vw_energy: object
    phi: object = None
    gvectors: object = None
    volume: object = None
    mesh: tuple = ()


@dataclass(frozen=True)
class OFDFTConfig:
    maxiter: int = 400
    tolerance: float = 1e-8
    xc: str | None = 'svwn'
    differentiation: SCFDifferentiationConfig = field(default_factory=SCFDifferentiationConfig)

    def __post_init__(self):
        if self.maxiter < 1 or not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('Positive iteration limit and finite positive tolerance required.')


class EnergyComponents(NamedTuple):
    kinetic: object
    external: object
    hartree: object
    xc: object
    nuclear: object

    @property
    def total(self):
        return sum(self)


class OFDFTResult(NamedTuple):
    total_energy: object
    density: object
    amplitude: object
    coefficients: object
    chemical_potential: object
    electron_number: object
    grid_electron_number: object
    residual_norm: object
    converged: object
    iterations: object
    components: EnergyComponents
