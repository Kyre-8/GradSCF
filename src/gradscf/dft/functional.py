"""External JAX energy functionals with SCF-current, differentiable inputs."""
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp

from .derivatives import xc_energy_and_potential_from_density, xc_kernel_action
from ..tools.features import (
    _ao_and_derivatives, _ao_laplacian, _restricted_rho_kernel,
    _spin_density_and_gradient, _spin_tau, _spin_laplacian,
)


@dataclass(frozen=True)
class DensityInputs:
    """Lazy physical quantities evaluated from the CURRENT trial density.

    Atomic units; rho/grad_rho/tau/laplacian_rho are spin sums. Their *_spin
    counterparts have leading alpha/beta axis. tau = 1/2 sum |grad psi|^2.
    Fixed-geometry AO/grid arrays are reused; density-derived values are never
    cached across SCF steps. MO energies/orbitals and reference HFX/PT2 caches
    are deliberately absent: they are not functions of this trial D alone.
    """
    density_matrix: Any
    _molecule: Any

    @property
    def total_density_matrix(self):
        return self.density_matrix.sum(axis=0)

    @property
    def ao(self):
        return self._molecule.ao

    @property
    def ao_deriv1(self):
        return _ao_and_derivatives(self._molecule)[1]

    @property
    def weights(self):
        return self._molecule.grid.weights

    @property
    def coordinates(self):
        return self._molecule.grid.coords

    @property
    def atom_coords(self):
        return self._molecule.atom_coords

    @property
    def atom_charges(self):
        return self._molecule.atom_charges

    @property
    def overlap_matrix(self):
        return self._molecule.overlap_matrix

    @property
    def rho_spin(self):
        return jax.vmap(lambda d: _restricted_rho_kernel(self.ao, d[None]))(self.density_matrix)

    @property
    def rho(self):
        return _restricted_rho_kernel(self.ao, self.density_matrix)

    @property
    def grad_rho_spin(self):
        return jax.vmap(lambda d: _spin_density_and_gradient(self.ao, self.ao_deriv1, d)[1])(self.density_matrix)

    @property
    def grad_rho(self):
        return self.grad_rho_spin.sum(axis=0)

    @property
    def tau_spin(self):
        return jax.vmap(lambda d: _spin_tau(self.ao_deriv1, d))(self.density_matrix)

    @property
    def tau(self):
        return self.tau_spin.sum(axis=0)

    @property
    def laplacian_rho_spin(self):
        lapl = _ao_laplacian(self._molecule)
        return jax.vmap(lambda d: _spin_laplacian(self.ao, self.ao_deriv1, lapl, d))(self.density_matrix)

    @property
    def laplacian_rho(self):
        return self.laplacian_rho_spin.sum(axis=0)


@dataclass(frozen=True)
class Functional:
    """External features + external scalar energy; no network architecture imposed.

    input_fn(DensityInputs) -> arbitrary array PyTree
    energy_fn(params, inputs) -> scalar total XC energy in Hartree
    init_fn(key, inputs) -> params (optional if parameters are supplied)

    All learned parameters must be explicit in params. Pure JAX/Flax/Haiku
    callbacks are supported. The energy may couple grid points; the response
    is the full density-matrix Hessian action, not a local projection. Real
    density functionals only. Include all desired baseline XC in energy_fn;
    no separate hybrid fraction or implicit baseline is added. TD response
    uses the adiabatic Hessian of the real symmetric density functional.
    Current-dependent/complex-orbital and hybrid response are not inferred.
    """
    input_fn: Callable
    energy_fn: Callable
    init_fn: Callable | None = None
    name: str = 'external_xc'
    exact_exchange_fraction: float = 0.0
    include_hfx_channel: ClassVar[bool] = False
    require_converged_scf: ClassVar[bool] = True

    def __post_init__(self):
        if self.exact_exchange_fraction != 0.:
            raise ValueError('A separate exact-exchange fraction requires an explicit hybrid adapter.')

    def inputs_for_density(self, molecule, density):
        dm = jnp.asarray(density)
        if jnp.iscomplexobj(dm):
            raise ValueError('Functional currently requires real density matrices.')
        if dm.ndim == 2:
            dm = jnp.stack([.5*dm, .5*dm])
        if dm.ndim != 3 or dm.shape[0] != 2 or dm.shape[1:] != (molecule.ao.shape[1],)*2:
            raise ValueError('Density must have shape (nao,nao) or (2,nao,nao).')
        dm = .5*(dm+jnp.swapaxes(dm,-1,-2))
        return self.input_fn(DensityInputs(dm, molecule))

    def apply(self, params, inputs):
        value = jnp.asarray(self.energy_fn(params, inputs))
        if value.shape != () or jnp.iscomplexobj(value):
            raise ValueError('energy_fn must return a real scalar XC energy in Hartree.')
        return value

    def init_from_molecule(self, rng, molecule):
        if self.init_fn is None:
            raise ValueError('Supply params or init_fn for the external functional.')
        return self.init_fn(rng, self.inputs_for_density(molecule, molecule.rdm1))

    def energy_for_density(self, params, molecule, density):
        return self.apply(params, self.inputs_for_density(molecule, density))

    def energy_from_molecule(self, params, molecule):
        return self.energy_for_density(params, molecule, molecule.rdm1)

    def scf_xc_energy_and_alpha_for_density(self, params, molecule, density):
        return self.energy_for_density(params, molecule, density), jnp.asarray(0., dtype=density.dtype)

    scf_spin_xc_energy_for_density = energy_for_density

    def potential(self, params, molecule, density):
        return xc_energy_and_potential_from_density(params, molecule=molecule, density=density,
            xc_energy_fn=self.energy_for_density).vxc_matrix

    def kernel_action(self, params, molecule, density, tangent):
        return xc_kernel_action(params, molecule=molecule, density=density,
            tangent=tangent, xc_energy_fn=self.energy_for_density)

    def bind_to_molecule_for_response(self, params, molecule):
        return _BoundFunctional(self, params)



@dataclass(frozen=True)
class _BoundFunctional:
    functional: Functional
    params: Any
    exact_exchange_fraction: float = 0.0

    def ao_kernel_action(self, molecule, tangent):
        density = molecule.rdm1
        if tangent.ndim == 2 and density.ndim == 3:
            density = density.sum(axis=0)
        return self.functional.kernel_action(self.params, molecule, density, tangent)
