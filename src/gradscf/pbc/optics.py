"""Gamma-point velocity-gauge optical strengths and normalized broadening.

The output is a per-cell oscillator-strength spectrum, not a macroscopic
absorption coefficient. GTH nonlocal velocity terms are included explicitly.
"""
import jax
import jax.numpy as jnp
import numpy as np
from ..integrals.periodic.ao import reciprocal_grid,fourier_ao
from ..integrals.periodic.pseudo import nonlocal_matrix

HARTREE_TO_EV=27.211386245988


def velocity_matrix(cell,*,parameters=None,kpt=None):
    """AO matrix of p - i[r,V_NL] in atomic units, shape (3,nao,nao).

    Differentiate the nonlocal projectors with respect to a uniform reciprocal
    shift while holding the AO Fourier coefficients fixed. Differentiating the
    entire k-dependent AO matrix would incorrectly include basis-motion terms.
    """
    parameters=cell.parameters if parameters is None else parameters
    k=jnp.zeros(3) if kpt is None else jnp.asarray(kpt)
    vectors=reciprocal_grid(cell.lattice,cell.mesh)+k
    ft=fourier_ao(cell.topology,parameters,vectors)
    volume=jnp.linalg.det(cell.lattice)
    momentum=jnp.stack([ft.conj().T@(vectors[:,axis,None]*ft)/volume for axis in range(3)])
    shifted=lambda q:nonlocal_matrix(vectors+q,ft,parameters.nuclear_coords,cell.pseudopotentials,volume)
    correction=jax.jacfwd(shifted)(jnp.zeros(3))
    return momentum+jnp.moveaxis(correction,-1,0)


def transition_velocity(td):
    """Physical complex transition velocity; overall eigenvector phases are free."""
    mf=td._scf
    from .scf import validate_reference
    validate_reference(mf)
    if mf.unrestricted or np.asarray(mf.kpts).shape!=(1,3) or np.any(np.asarray(mf.kpts)):
        raise NotImplementedError('Optical strengths currently support restricted Gamma references.')
    signature=(id(mf.result),td.singlet,mf.xc,id(mf.inputs))
    if getattr(td,'_solution_reference',None)!=signature:
        raise ValueError('Run the response kernel for the current reference first.')
    if not np.all(np.asarray(td.converged)) or np.any(np.asarray(td.e)<=0):
        raise ValueError('Optical strengths require converged positive-energy excitations.')
    x,y=td.xy
    norm=jnp.sum(jnp.abs(x)**2-jnp.abs(y)**2,axis=0)
    if np.any(np.asarray(norm)<=0):raise ValueError('Excitation norms must be positive.')
    if not td.singlet:return jnp.zeros((len(td.e),3),dtype=complex)
    coeff=mf.result.mo_coeff_spin[0,0];no=mf.cell.nelec[0]
    velocity=velocity_matrix(mf.cell)
    ov=jnp.einsum('pi,dpq,qa->dia',coeff[:,:no].conj(),velocity,coeff[:,no:]).reshape(3,-1)
    # Our spatial X/Y norm is one; PySCF uses one half for restricted states.
    return jnp.sqrt(2.)*((ov@((x-y)/jnp.sqrt(norm)[None,:])).T)


def oscillator_strength(td,*,gauge='velocity'):
    if gauge!='velocity':raise NotImplementedError('Periodic length-gauge strengths are not implemented.')
    velocity=transition_velocity(td)
    return 2./3.*jnp.sum(jnp.abs(velocity)**2,axis=1)/jnp.asarray(td.e)


def broaden_spectrum(energies_hartree,strengths,grid_ev,*,fwhm_ev=.3):
    """Sum unit-area Gaussians; output is oscillator strength per eV."""
    if not np.isfinite(fwhm_ev) or fwhm_ev<=0:raise ValueError('FWHM must be finite and positive.')
    energy=jnp.asarray(energies_hartree)*HARTREE_TO_EV
    strength=jnp.asarray(strengths);grid=jnp.asarray(grid_ev)
    if energy.ndim!=1 or strength.shape!=energy.shape or grid.ndim!=1:
        raise ValueError('Use equal-length energy/strength vectors and a one-dimensional grid.')
    sigma=fwhm_ev/(2*jnp.sqrt(2*jnp.log(2.)))
    profiles=jnp.exp(-.5*((grid[:,None]-energy[None,:])/sigma)**2)/(sigma*jnp.sqrt(2*jnp.pi))
    return profiles@strength
