"""Non-self-consistent semilocal bands from a fixed periodic SCF density."""
import numpy as np
import jax.numpy as jnp
from ..integrals.periodic.fft import build_band_inputs
from ..integrals.periodic.coulomb import coulomb_kernel
from ..dft.libxc_jax.jax_libxc import hybrid_coeff,xc_type
from ._kpoint import grid_xc_potential,project_xc_potential
from .kpoints import weights_for


def get_bands(mf,kpts,*,chunk_size=4):
    from .scf import validate_reference
    validate_reference(mf,integer=False,equal_weights=False)
    if hybrid_coeff(mf.xc)!=0 or xc_type(mf.xc) not in ('LDA','GGA'):
        raise NotImplementedError('Band queries currently support pure LDA/GGA only.')
    points=np.asarray(kpts,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('kpts must be a nonempty finite (nquery,3) array in inverse Bohr.')
    if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('chunk_size must be positive.')
    source=mf.inputs;density=mf.result.density_spin
    ao=source.ao[:,0];nk=len(source.kpoints)
    rho=jnp.einsum('k,kgp,kpq,kgq->g',weights_for(source),ao,density.sum(axis=0),ao.conj()).real
    mesh=mf.cell.mesh
    hartree=jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(mesh))*coulomb_kernel(source.gvectors).reshape(mesh)).reshape(-1).real
    _,potential=grid_xc_potential(density,source,xc=mf.xc)
    energies=[];coefficients=[]
    for start in range(0,len(points),chunk_size):
        target=build_band_inputs(mf.cell,points[start:start+chunk_size],source)
        values=target.ao[:,0]
        j=jnp.einsum('g,kgp,kgq->kpq',source.weights*hartree,values.conj(),values)
        vxc=project_xc_potential(target.ao,source.weights,potential,xc_type(mf.xc))
        fock=target.hcore+j+vxc
        x=jnp.linalg.inv(jnp.linalg.cholesky(target.overlap)).conj().swapaxes(-1,-2)
        e,c=jnp.linalg.eigh(x.conj().swapaxes(-1,-2)@fock@x)
        energies.append(e);coefficients.append(x@c)
    e=jnp.concatenate(energies,axis=1);c=jnp.concatenate(coefficients,axis=1)
    return (e,c) if mf.unrestricted else (e[0],c[0])
