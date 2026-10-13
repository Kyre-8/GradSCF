"""Periodic atomic-coordinate response and Pulay-correct force derivatives.

The ground-state adapters supply the stationary residual. Geometry enters
through the existing integral builders; no SCF loop or linear solver is copied.
"""
import jax
import jax.numpy as jnp
from .core import Problem
from .periodic import periodic_problem
from .ofdft import ofdft_problem


def _ks_forces(d,coordinates,inputs_at,**options):
    """Stationary KS/HF forces, with W=D F S^-1 including Fermi occupations."""
    from ..pbc._kpoint import energy_and_fock
    from ..pbc.kpoints import weights_for
    data=inputs_at(coordinates)
    f=energy_and_fock(d,data,**options)[1]
    w=d@f@jnp.linalg.inv(data.overlap)
    w=.5*(w+w.conj().swapaxes(-1,-2))
    def lagrangian(r):
        varied=inputs_at(r)
        value=energy_and_fock(d,varied,**options)[0]
        return value-jnp.einsum('k,skij,kji->',weights_for(varied),w,varied.overlap).real
    return -jax.grad(lagrangian)(coordinates)


def atomic_problem(source,*,tolerance=1e-7):
    """Return a periodic geometry Problem with Cartesian coordinates in Bohr.

    response(tangent=displacements) returns density and force derivatives.
    The force derivative is minus a force-constant action; finite-temperature
    KS forces differentiate free energy at fixed kBT and spin populations.
    The lattice, FFT mesh, Ewald topology and basis exponents stay fixed.
    """
    from ..pbc.scf import _SCF,validate_reference
    from ..ofdft.api import OFDFT
    if isinstance(source,_SCF):
        from ..integrals.periodic.fft import build_inputs
        validate_reference(source,integer=False,equal_weights=False,external=True)
        cell=source.cell
        options=dict(mesh=cell.mesh,xc=source.xc,exxdiv=source.exxdiv,
                     functional=source.functional,params=source.params)
        base=periodic_problem(source.inputs,source.result,nelec=cell.nelec,
                              temperature=source.temperature,tolerance=tolerance,**options)
        def inputs_at(r):
            data=build_inputs(cell,parameters=cell.parameters_at(r),kpts=source.kpts,kweights=source.kweights)
            return data._replace(hcore=data.hcore.astype(jnp.complex128))
        def forces(state,r):
            d=base.quantities(state,args_at(r))['density']
            return _ks_forces(d,r,inputs_at,**options)
    elif isinstance(source,OFDFT) and hasattr(source.system,'a'):
        from ..ofdft import inputs_from_cell
        if source.result is None or not source.converged:
            raise ValueError('Atomic response requires a converged OFDFT reference')
        cell=source.system
        base=ofdft_problem(source.inputs,source.result,kinetic=source.kinetic,
            kinetic_params=source.kinetic_params,xc_energy_fn=source.xc_energy_fn,
            xc_params=source.xc_params,config=source.config,tolerance=tolerance)
        def inputs_at(r):
            return inputs_from_cell(cell,parameters=cell.parameters_at(r),representation=source.inputs.representation)
        def forces(state,r):
            mu=state[-1]/source.inputs.nelectron
            def lagrangian(coords):
                values=base.quantities(state,args_at(coords))
                return values['energy']-mu*values['electron_number']
            return -jax.grad(lagrangian)(r)
    else:
        raise NotImplementedError('atomic_problem currently requires periodic KS/HF or Cell-based OFDFT')

    def args_at(r):return (inputs_at(r),)+base.inputs[1:]
    def quantities(state,r):
        return dict(base.quantities(state,args_at(r)),forces=forces(state,r))
    return Problem(lambda state,r:base.residual(state,args_at(r)),base.solution,
                   cell.coords,quantities,base.converged,tolerance,series_supported=False)
