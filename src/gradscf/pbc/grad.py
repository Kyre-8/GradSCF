"""Shared periodic geometry derivatives for KS/HF and OFDFT energies.

Lattice vectors are rows. Forces hold the cell fixed; stress uses symmetric
homogeneous strain at fixed fractional positions and fractional k points.
Mesh and Ewald image topology remain fixed. Positive stress is tensile and
pressure is -trace(stress)/3. Units: Eh, Eh/Bohr, Eh/Bohr**3.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from ..scf.forces import _coordinates_array


class CellDerivatives(NamedTuple):
    energy: object
    forces: object
    stress: object


def _checked_energy(value,density,converged,reference):
    if reference is not None:
        reference=jax.lax.stop_gradient(reference)
        converged &= jnp.linalg.norm(density-reference)<=1e-6*(1+jnp.linalg.norm(reference))
    return value*jnp.where(converged,1.,jnp.nan)


def energy_and_derivatives(energy_fn,params,coordinates,lattice):
    """Differentiate energy_fn(params, Cartesian coordinates, row lattice).

    For symmetric strain e, R(e)=R@(I+e).T and A(e)=A@(I+e).T.
    The callback owns SCF convergence and the chosen thermodynamic potential.
    This array interface preserves parameter and geometry AD, subject to the
    derivative orders supported by the callback; it does not freeze a density.
    """
    coordinates=_coordinates_array(coordinates)
    lattice=jnp.asarray(lattice)
    if lattice.shape!=(3,3) or not jnp.issubdtype(lattice.dtype,jnp.floating):
        raise ValueError('lattice must be a real floating (3,3) array in Bohr')
    if not isinstance(lattice,jax.core.Tracer) and (not np.isfinite(lattice).all() or np.linalg.det(lattice)<=0):
        raise ValueError('lattice must be finite and right handed')
    energy,(gr,ga)=jax.value_and_grad(energy_fn,argnums=(1,2))(params,coordinates,lattice)
    virial=gr.T@coordinates+ga.T@lattice
    stress=.5*(virial+virial.T)/jnp.linalg.det(lattice)
    return CellDerivatives(energy,-gr,stress)


class Gradients:
    """Eager periodic facade; kernel returns dE/dR, result also holds stress.

    Each kernel call rebuilds inputs and reconverges from the default seed using
    the source settings. If a previous result exists, its density must agree
    within 1e-6*(1+norm(reference)); a different root is rejected, not followed.
    Finite-temperature KS/HF differentiates free energy at fixed kBT and spin
    populations. Full k points are used even for a time-reversal-reduced source.
    stress() returns the latest result, running kernel once if necessary.
    """
    def __init__(self,source):
        self.source=source
        self.result=None

    def kernel(self):
        self.result=None
        if not jax.config.x64_enabled:
            raise ValueError('Periodic geometry derivatives require JAX float64')
        from .scf import _SCF
        from ..ofdft.api import OFDFT
        source=self.source
        if isinstance(source,_SCF):
            from .scf import validate_reference
            from ..integrals.periodic.fft import build_inputs
            if source.result is not None:
                validate_reference(source,integer=False,equal_weights=False,external=True)
            cell=source.cell
            params=source.params
            reference=None if source.result is None else source.result.density_spin
            def energy(p,r,a):
                data=build_inputs(cell,parameters=cell.parameters_at(r),lattice=a,
                                  kpts=source.kpts,kweights=source.kweights)
                out=source._solve(data,params=p)
                return _checked_energy(out.free_energy,out.density_spin,out.converged,reference)
        elif isinstance(source,OFDFT) and hasattr(source.system,'a'):
            from ..ofdft import inputs_from_cell,run_ofdft
            cell=source.system if hasattr(source.system,'topology') else source.system.build()
            params=(source.kinetic_params,source.xc_params)
            reference=None if source.result is None else source.result.density
            def energy(p,r,a):
                data=inputs_from_cell(cell,representation=source.representation or 'periodic',
                                     parameters=cell.parameters_at(r),lattice=a)
                out=run_ofdft(data,kinetic=source.kinetic,kinetic_params=p[0],
                    xc_energy_fn=source.xc_energy_fn,xc_params=p[1],config=source.config)
                return _checked_energy(out.total_energy,out.density,out.converged,reference)
        else:
            raise TypeError('Gradients requires a periodic SCF or Cell-based OFDFT calculation')
        candidate=energy_and_derivatives(energy,params,cell.coords,cell.lattice)
        if not all(np.isfinite(x).all() for x in candidate):
            raise RuntimeError('Periodic geometry derivative failed: check SCF convergence, backward residuals and reference state')
        self.result=candidate
        return -self.result.forces

    def stress(self):
        if self.result is None:self.kernel()
        return self.result.stress
