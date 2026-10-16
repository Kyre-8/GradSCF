"""Non-self-consistent bands from a fixed periodic SCF density."""
import numpy as np
import jax.numpy as jnp
from ..integrals.periodic.fft import build_band_inputs,_kpoint_exchange
from ..integrals.periodic.coulomb import coulomb_kernel
from ..dft.libxc_jax.jax_libxc import hybrid_coeff,xc_type
from ._kpoint import grid_xc_potential,project_xc_potential
from .kpoints import weights_for


def get_bands(mf,kpts,*,chunk_size=4):
    from .scf import validate_reference
    validate_reference(mf,integer=False,equal_weights=False,low_dimensional=True)
    alpha=hybrid_coeff(mf.xc)
    if xc_type(mf.xc) not in ('HF','LDA','GGA','MGGA'):
        raise NotImplementedError('Unsupported band XC functional.')
    points=np.asarray(kpts,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('kpts must be a nonempty finite (nquery,3) array in inverse Bohr.')
    scaled=points@np.asarray(mf.cell.lattice).T/(2*np.pi)
    if np.any(abs(scaled[:,mf.cell.dimension:])>1e-12):
        raise ValueError('Band k points must vanish in nonperiodic directions.')
    if alpha and mf.exxdiv!='vcut_sph':
        on_mesh=np.any(np.all(abs(points[:,None]-np.asarray(mf.kpts)[None])<1e-10,axis=-1),axis=1)
        if not np.all(on_mesh):
            raise NotImplementedError('Off-mesh hybrid bands require a matched exxdiv=vcut_sph SCF reference.')
    if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('chunk_size must be positive.')
    source=mf.inputs;density=mf.result.density_spin
    ao=source.ao[:,0];nk=len(source.kpoints)
    rho=jnp.einsum('k,kgp,kpq,kgq->g',weights_for(source),ao,density.sum(axis=0),ao.conj()).real
    mesh=mf.cell.mesh
    hartree=jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(mesh))*coulomb_kernel(source.gvectors,boundary=source.boundary).reshape(mesh)).reshape(-1).real
    _,potential=grid_xc_potential(density,source,xc=mf.xc)
    energies=[];coefficients=[]
    for start in range(0,len(points),chunk_size):
        target=build_band_inputs(mf.cell,points[start:start+chunk_size],source)
        values=target.ao[:,0]
        j=jnp.einsum('g,kgp,kgq->kpq',source.weights*hartree,values.conj(),values)
        vxc=project_xc_potential(target.ao,source.weights,potential,xc_type(mf.xc))
        fock=target.hcore+j+vxc
        if alpha:
            fock-=alpha*_kpoint_exchange(source,density,mesh=mesh,exxdiv=mf.exxdiv,query=target)
        x=jnp.linalg.inv(jnp.linalg.cholesky(target.overlap)).conj().swapaxes(-1,-2)
        e,c=jnp.linalg.eigh(x.conj().swapaxes(-1,-2)@fock@x)
        energies.append(e);coefficients.append(x@c)
    e=jnp.concatenate(energies,axis=1);c=jnp.concatenate(coefficients,axis=1)
    return (e,c) if mf.unrestricted else (e[0],c[0])
