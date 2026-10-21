"""External JAX energy functionals with SCF-current, differentiable inputs."""
from dataclasses import dataclass
from functools import cached_property
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
    A grid provider forms small physical features blockwise. Explicitly asking
    for ao/ao_deriv1 materializes those full-grid arrays for legacy callbacks.
    Values are cached only within this trial-density input object, never across
    SCF steps. MO energies/orbitals and reference HFX/PT2 caches
    are deliberately absent: they are not functions of this trial D alone.
    """
    density_matrix: Any
    _molecule: Any

    @property
    def total_density_matrix(self):
        return self.density_matrix.sum(axis=0)

    @cached_property
    def ao(self):
        provider = getattr(self._molecule, "grid_ao", None)
        if provider is not None:
            return provider.map(lambda ao, dao, lap, coords, weights: ao, deriv=0)
        return self._molecule.ao

    @cached_property
    def ao_deriv1(self):
        provider = getattr(self._molecule, "grid_ao", None)
        if provider is not None:
            values = provider.map(
                lambda ao, dao, lap, coords, weights: jnp.moveaxis(dao, 0, 1), deriv=1)
            return jnp.moveaxis(values, 1, 0)
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

    @cached_property
    def rho_spin(self):
        provider = getattr(self._molecule, "grid_ao", None)
        if provider is not None:
            def density(ao, dao, lap, coords, weights):
                values = jax.vmap(lambda d: _restricted_rho_kernel(ao, d[None]))(self.density_matrix)
                return values.T
            return provider.map(density, deriv=0).T
        return jax.vmap(lambda d: _restricted_rho_kernel(self.ao, d[None]))(self.density_matrix)

    @property
    def rho(self):
        return self.rho_spin.sum(axis=0)

    @cached_property
    def _gradient_tau_spin(self):
        provider = getattr(self._molecule, "grid_ao", None)
        if provider is None:
            gradient = jax.vmap(lambda d: _spin_density_and_gradient(
                self.ao, self.ao_deriv1, d)[1])(self.density_matrix)
            tau = jax.vmap(lambda d: _spin_tau(self.ao_deriv1, d))(self.density_matrix)
            return gradient, tau

        def quantities(ao, dao, lap, coords, weights):
            gradient = jax.vmap(lambda d: _spin_density_and_gradient(ao, dao, d)[1])(
                self.density_matrix)
            tau = jax.vmap(lambda d: _spin_tau(dao, d))(self.density_matrix)
            return jnp.moveaxis(gradient, 0, 1), tau.T

        gradient, tau = provider.map(quantities, deriv=1)
        return jnp.moveaxis(gradient, 1, 0), tau.T

    @property
    def grad_rho_spin(self):
        return self._gradient_tau_spin[0]

    @property
    def grad_rho(self):
        return self.grad_rho_spin.sum(axis=0)

    @property
    def tau_spin(self):
        return self._gradient_tau_spin[1]

    @property
    def tau(self):
        return self.tau_spin.sum(axis=0)

    @cached_property
    def laplacian_rho_spin(self):
        provider = getattr(self._molecule, "grid_ao", None)
        if provider is not None:
            def laplacian(ao, dao, lap, coords, weights):
                values = jax.vmap(lambda d: _spin_laplacian(ao, dao, lap, d))(self.density_matrix)
                return values.T
            return provider.map(laplacian, deriv=2).T
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
    is the full density-matrix Hessian action, not a local projection. Molecular
    inputs use real symmetric densities; periodic inputs use complex Hermitian
    densities and a real energy per cell. Include all desired baseline XC in
    energy_fn; no separate hybrid fraction or implicit baseline is added.
    Molecular TD response uses the adiabatic Hessian. Periodic external models
    support static q=0 DFPT. Finite q additionally requires q_energy_fn; TD,
    band-query and hybrid operators are not inferred from a scalar callback.

    q_energy_fn(params, periodic_inputs, reference_density, transfer, delta_q)
    optionally supplies a real XC energy per primitive cell for the physical
    density D + Z_q + Z_q.adjoint(), with delta_q[s,k] the row k+q / column k
    blocks of Z_q. It must agree with energy_fn at delta_q=0 and describe the
    same translation-invariant functional on the commensurate supercell.
    Include BOTH q and -q terms even when q is self-inverse; no factor of 1/2
    is applied to Z_q. The callback receives raw periodic arrays, not input_fn
    features: diagonal-k features alone do not define a finite-q extension.
    GradSCF derives the kernel by AD, including feature/parameter derivatives.
    """
    input_fn: Callable
    energy_fn: Callable
    init_fn: Callable | None = None
    name: str = 'external_xc'
    exact_exchange_fraction: float = 0.0
    q_energy_fn: Callable | None = None
    include_hfx_channel: ClassVar[bool] = False
    require_converged_scf: ClassVar[bool] = True

    def __post_init__(self):
        if self.exact_exchange_fraction != 0.:
            raise ValueError('A separate exact-exchange fraction requires an explicit hybrid adapter.')

    def inputs_for_density(self, molecule, density):
        if getattr(molecule,'kpoints',None) is not None:
            from ..pbc.functional import density_inputs
            return self.input_fn(density_inputs(molecule,density))
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
        return self.init_for_density(rng,molecule,molecule.rdm1)

    def init_for_density(self, rng, molecule, density):
        if self.init_fn is None:
            raise ValueError('Supply params or init_fn for the external functional.')
        return self.init_fn(rng, self.inputs_for_density(molecule, density))

    def energy_for_density(self, params, molecule, density):
        return self.apply(params, self.inputs_for_density(molecule, density))

    def energy_from_molecule(self, params, molecule):
        return self.energy_for_density(params, molecule, molecule.rdm1)

    def scf_xc_energy_and_alpha_for_density(self, params, molecule, density):
        return self.energy_for_density(params, molecule, density), jnp.asarray(0., dtype=density.dtype)

    scf_spin_xc_energy_for_density = energy_for_density

    def potential(self, params, molecule, density):
        return xc_energy_and_potential_from_density(params, molecule=molecule, density=density,
            xc_energy_fn=self.energy_for_density,density_weights=_density_weights(molecule,density)).vxc_matrix

    def kernel_action(self, params, molecule, density, tangent):
        return xc_kernel_action(params, molecule=molecule, density=density,
            tangent=tangent, xc_energy_fn=self.energy_for_density,density_weights=_density_weights(molecule,density))

    def q_kernel_action(self, params, inputs, density, transfer, tangent):
        """Complex-linear nonzero-q XC Hessian action from q_energy_fn."""
        from ..pbc.functional import finite_q_kernel
        return finite_q_kernel(self,params,inputs,density,transfer)(tangent)

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


def _density_weights(molecule,density):
    if getattr(molecule,'kpoints',None) is not None:
        from ..pbc.kpoints import weights_for
        weights=weights_for(molecule)
        return weights[0] if jnp.ndim(density)==2 else weights
    return None


def custom_xc(spec: str | None = None, *, input_fn: Callable | None = None,
              energy_fn: Callable | None = None, init_fn: Callable | None = None,
              name: str | None = None):
    """Construct a complete traditional recipe or an external energy functional.

    ``custom_xc(spec)`` retains the recipe's exchange and extra physical terms.
    ``custom_xc(input_fn=..., energy_fn=...)`` returns the existing Functional:
    inputs may be any PyTree and energy_fn(params, inputs) returns total XC
    energy in Hartree (per cell for PBC). Both handwritten formulas and neural
    models use this same contract. Supply parameters explicitly to SCF/Trainer;
    no baseline or exact exchange is inferred from a scalar energy callback.
    """
    if spec is not None:
        if not isinstance(spec, str) or any(value is not None for value in
                                           (input_fn, energy_fn, init_fn)):
            raise TypeError('Use either a traditional spec recipe or input_fn/energy_fn callbacks.')
        from .xc import TraditionalXCFunctional
        return TraditionalXCFunctional(spec, name=name)
    if not callable(input_fn) or not callable(energy_fn):
        raise TypeError('Custom energy models require callable input_fn and energy_fn.')
    if init_fn is not None and not callable(init_fn):
        raise TypeError('init_fn must be callable or None.')
    return Functional(input_fn, energy_fn, init_fn,
                      name='custom_xc' if name is None else name)
