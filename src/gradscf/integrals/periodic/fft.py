"""Gamma-point FFT density fitting without a four-index ERI tensor."""
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
from .ao import bloch_grid
from .coulomb import ewald_energy,coulomb_kernel
from .pseudo import local_potential,nonlocal_matrix


class PeriodicInputs(NamedTuple):
    overlap: object       # (nk,nao,nao), initially nk=1
    hcore: object
    kinetic: object
    ao: object            # (nk,4,ngrid,nao)
    coords: object
    weights: object
    pair_potential: object
    nuclear_repulsion: object
    madelung: object
    kpoints: object
    gvectors: object
    kweights: object = None
    lattice: object = None
    atom_coords: object = None
    atom_charges: object = None
    ionic_charges: object = None
    ao_laplacian: object = None


def _context(cell,parameters,lattice,ft,gv,kpts):
    volume=jnp.linalg.det(lattice);ngrid=int(np.prod(cell.mesh))
    q=gv[None]+kpts[:,None]
    values=-jnp.sum(q*q,axis=-1)[...,None]*ft*ngrid/volume
    laplacian=jnp.fft.ifftn(values.reshape((len(kpts),)+tuple(cell.mesh)+(-1,)),axes=(1,2,3)).reshape(ft.shape)
    return dict(lattice=lattice,atom_coords=parameters.nuclear_coords,
                atom_charges=jnp.asarray(cell.topology.nuclear_charges,dtype=lattice.dtype),
                ionic_charges=jnp.asarray(cell.charges,dtype=lattice.dtype),ao_laplacian=laplacian)


def build_inputs(cell,*,parameters=None,lattice=None,kpts=None,kweights=None):
    """Mesh/topology are static; dynamic parameters and lattice remain explicit."""
    lattice=cell.lattice if lattice is None else jnp.asarray(lattice)
    parameters=cell.parameters if parameters is None else parameters
    mesh=cell.mesh;volume=jnp.linalg.det(lattice);ngrid=int(np.prod(mesh))
    from ...pbc.kpoints import kmesh_shape
    reference_kpts=np.zeros((1,3)) if kpts is None else np.asarray(kpts)
    w=np.full(len(reference_kpts),1/len(reference_kpts)) if kweights is None else np.asarray(kweights,dtype=float)
    if w.shape!=(len(reference_kpts),) or not np.isfinite(w).all() or np.any(w<=0) or not np.isclose(w.sum(),1.,atol=1e-12,rtol=0):
        raise ValueError('k weights must be positive, finite, normalized, and match kpts')
    kmesh=kmesh_shape(cell.lattice,reference_kpts)
    scaled=reference_kpts@np.asarray(cell.lattice).T/(2*np.pi)
    kpts=jnp.asarray(scaled)@(2*jnp.pi*jnp.linalg.inv(lattice).T)
    ao,ft,gv,coords=bloch_grid(cell.topology,parameters,lattice,mesh,kpts)
    if len(reference_kpts)!=1 or np.any(reference_kpts!=0):
        return _assemble_kpoints(cell,parameters,lattice,ao,ft,gv,coords,kpts,kmesh)._replace(kweights=jnp.asarray(w))
    ao=ao.real;coeff=ft[0];values=ao[0,0]
    overlap=(coeff.conj().T@coeff/volume).real
    kinetic=(coeff.conj().T@(.5*jnp.sum(gv*gv,axis=1)[:,None]*coeff)/volume).real
    vg=local_potential(gv,parameters.nuclear_coords,cell.pseudopotentials)
    vr=jnp.fft.ifftn((vg*ngrid/volume).reshape(mesh)).reshape(-1).real
    local=values.T@(vr[:,None]*values)*volume/ngrid
    nonlocal_pp=nonlocal_matrix(gv,coeff,parameters.nuclear_coords,cell.pseudopotentials,volume).real
    pairs=values[:,:,None]*values[:,None,:]
    shape=mesh+pairs.shape[1:]
    coul=coulomb_kernel(gv).reshape(mesh+(1,1))
    potential=jnp.fft.ifftn(jnp.fft.fftn(pairs.reshape(shape),axes=(0,1,2))*coul,axes=(0,1,2)).real.reshape(pairs.shape)
    enuc=ewald_energy(lattice,parameters.nuclear_coords,cell.charges,precision=cell.precision,reference_lattice=cell.lattice)
    mad=-2*ewald_energy(lattice,jnp.zeros((1,3)),jnp.ones(1),precision=cell.precision,reference_lattice=cell.lattice)
    return PeriodicInputs(overlap[None],(kinetic+local+nonlocal_pp)[None],kinetic[None],ao,coords,
        jnp.full(ngrid,volume/ngrid),potential,enuc,mad,kpts,gv,jnp.asarray(w),
        **_context(cell,parameters,lattice,ft,gv,kpts))


def get_jk(inputs,density_spin,*,exxdiv='ewald',with_k=True):
    """Real Gamma spin density (2,nao,nao); Ewald K correction is explicit."""
    if exxdiv not in ('ewald',None):
        raise NotImplementedError('Periodic exchange supports exxdiv=ewald or None.')
    ao=inputs.ao[0,0];weight=inputs.weights[0]
    total=jnp.sum(density_spin,axis=0)
    hartree=jnp.einsum('gp,pq,gq->g',ao,total,ao)
    j=weight*jnp.einsum('g,gpq->pq',hartree,inputs.pair_potential)
    if not with_k:return j,jnp.zeros_like(density_spin)
    k=weight*jnp.einsum('gp,gs,tsr,grq->tpq',ao,ao,density_spin,inputs.pair_potential)
    if exxdiv=='ewald':
        s=inputs.overlap[0];k=k+inputs.madelung*jnp.einsum('pr,trs,sq->tpq',s,density_spin,s)
    return j,k


def _assemble_kpoints(cell,parameters,lattice,ao,ft,gv,coords,kpts,kmesh,*,electrostatic=None):
    volume=jnp.linalg.det(lattice);mesh=cell.mesh;ngrid=int(np.prod(mesh))
    vg=local_potential(gv,parameters.nuclear_coords,cell.pseudopotentials)
    vr=jnp.fft.ifftn((vg*ngrid/volume).reshape(mesh)).reshape(-1).real
    overlaps=[];kinetics=[];cores=[]
    for index,k in enumerate(kpts):
        values=ao[index,0];coeff=ft[index];q=gv+k
        s=coeff.conj().T@coeff/volume
        t=coeff.conj().T@(.5*jnp.sum(q*q,axis=1)[:,None]*coeff)/volume
        v=values.conj().T@(vr[:,None]*values)*volume/ngrid
        v=v+nonlocal_matrix(q,coeff,parameters.nuclear_coords,cell.pseudopotentials,volume)
        overlaps.append(s);kinetics.append(t);cores.append(t+v)
    # Store periodic factors u_k; derivative channels retain (grad + i*k)u_k.
    ao=ao*jnp.exp(-1j*kpts@coords.T)[:,None,:,None]
    if electrostatic is None:
        enuc=ewald_energy(lattice,parameters.nuclear_coords,cell.charges,precision=cell.precision,reference_lattice=cell.lattice)
        supercell=jnp.asarray(kmesh)[:,None]*lattice
        mad=-2*ewald_energy(supercell,jnp.zeros((1,3)),jnp.ones(1),precision=cell.precision,
                            reference_lattice=np.asarray(kmesh)[:,None]*np.asarray(cell.lattice))
    else:
        enuc,mad=electrostatic
    return PeriodicInputs(jnp.stack(overlaps),jnp.stack(cores),jnp.stack(kinetics),ao,coords,
        jnp.full(ngrid,volume/ngrid),None,enuc,mad,kpts,gv,
        **_context(cell,parameters,lattice,ft,gv,kpts))


def get_kpoint_jk(inputs,density_spin,*,mesh,exxdiv='ewald',with_k=True):
    if exxdiv not in ('ewald',None):raise NotImplementedError('Unsupported exchange correction.')
    ao=inputs.ao[:,0];nk=ao.shape[0];weight=inputs.weights[0]
    from ...pbc.kpoints import weights_for
    kw=weights_for(inputs)
    total=density_spin.sum(axis=0)
    rho=jnp.einsum('k,kgp,kpq,kgq->g',kw,ao,total,ao.conj()).real
    vr=jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(mesh))*coulomb_kernel(inputs.gvectors).reshape(mesh)).reshape(-1).real
    j=weight*jnp.einsum('kgp,g,kgq->kpq',ao.conj(),vr,ao)
    if not with_k:return j,jnp.zeros_like(density_spin)
    matrices=[]
    for target in range(nk):
        k=jnp.zeros_like(density_spin[:,target])
        for source in range(nk):
            pairs=ao[source].conj()[:,:,None]*ao[target][:,None,:]
            kernel=coulomb_kernel(inputs.gvectors+inputs.kpoints[target]-inputs.kpoints[source])
            shape=tuple(mesh)+pairs.shape[1:]
            potential=jnp.fft.ifftn(jnp.fft.fftn(pairs.reshape(shape),axes=(0,1,2))*kernel.reshape(tuple(mesh)+(1,1)),axes=(0,1,2)).reshape(pairs.shape)
            k=k+weight*kw[source]*jnp.einsum('gp,gs,tsr,grq->tpq',ao[target].conj(),ao[source],density_spin[:,source],potential)
        if exxdiv=='ewald':
            s=inputs.overlap[target];k=k+inputs.madelung*jnp.einsum('pr,trs,sq->tpq',s,density_spin[:,target],s)
            k=k*jnp.where(jnp.all(jnp.abs(kw-1/nk)<1e-12),1.,jnp.nan)
        matrices.append(k)
    return j,jnp.stack(matrices,axis=1)


def build_band_inputs(cell,kpts,reference):
    """Arbitrary query k points on the reference grid; no SCF mesh interpretation."""
    kpts=jnp.asarray(kpts)
    ao,ft,gv,coords=bloch_grid(cell.topology,cell.parameters,cell.lattice,cell.mesh,kpts)
    return _assemble_kpoints(cell,cell.parameters,cell.lattice,ao,ft,gv,coords,kpts,None,
        electrostatic=(reference.nuclear_repulsion,reference.madelung))
