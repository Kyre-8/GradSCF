"""Gamma-point FFT density fitting without a four-index ERI tensor."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .ao import bloch_grid
from .coulomb import ewald_energy,coulomb_kernel,exchange_kernel,ReducedBoundary
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
    boundary: object = None


def _context(cell,parameters,lattice,ft,gv,kpts):
    volume=jnp.linalg.det(lattice);ngrid=int(np.prod(cell.mesh))
    q=gv[None]+kpts[:,None]
    values=-jnp.sum(q*q,axis=-1)[...,None]*ft*ngrid/volume
    laplacian=jnp.fft.ifftn(values.reshape((len(kpts),)+tuple(cell.mesh)+(-1,)),axes=(1,2,3)).reshape(ft.shape)
    return dict(lattice=lattice,atom_coords=parameters.nuclear_coords,
                atom_charges=jnp.asarray(cell.topology.nuclear_charges,dtype=lattice.dtype),
                ionic_charges=jnp.asarray(cell.charges,dtype=lattice.dtype),ao_laplacian=laplacian,
                boundary=_boundary(cell,lattice))


def _boundary(cell,lattice):
    return None if cell.dimension==3 else ReducedBoundary(lattice,cell.dimension,tuple(cell.mesh))


def _electrostatic(cell,parameters,lattice,kmesh):
    energy=ewald_energy
    if cell.dimension<3:
        from .truncated import slab_energy,wire_energy
        energy=slab_energy if cell.dimension==2 else wire_energy
    enuc=energy(lattice,parameters.nuclear_coords,cell.charges,
                precision=cell.precision,reference_lattice=cell.lattice)
    supercell=jnp.asarray(kmesh)[:,None]*lattice
    mad=-2*energy(supercell,jnp.zeros((1,3)),jnp.ones(1),precision=cell.precision,
        reference_lattice=np.asarray(kmesh)[:,None]*np.asarray(cell.lattice))
    return enuc,mad


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
    if np.any(abs(scaled[:,cell.dimension:])>1e-12):
        raise ValueError('k points must vanish in nonperiodic directions.')
    kpts=jnp.asarray(scaled)@(2*jnp.pi*jnp.linalg.inv(lattice).T)
    ao,ft,gv,coords=bloch_grid(cell.topology,parameters,lattice,mesh,kpts)
    if len(reference_kpts)!=1 or np.any(reference_kpts!=0):
        return _assemble_kpoints(cell,parameters,lattice,ao,ft,gv,coords,kpts,kmesh)._replace(kweights=jnp.asarray(w))
    ao=ao.real;coeff=ft[0];values=ao[0,0]
    overlap=(coeff.conj().T@coeff/volume).real
    kinetic=(coeff.conj().T@(.5*jnp.sum(gv*gv,axis=1)[:,None]*coeff)/volume).real
    boundary=_boundary(cell,lattice)
    vg=local_potential(gv,parameters.nuclear_coords,cell.pseudopotentials,boundary=boundary)
    vr=jnp.fft.ifftn((vg*ngrid/volume).reshape(mesh)).reshape(-1).real
    local=values.T@(vr[:,None]*values)*volume/ngrid
    nonlocal_pp=nonlocal_matrix(gv,coeff,parameters.nuclear_coords,cell.pseudopotentials,volume).real
    pairs=values[:,:,None]*values[:,None,:]
    shape=mesh+pairs.shape[1:]
    coul=coulomb_kernel(gv,boundary=boundary).reshape(mesh+(1,1))
    potential=jnp.fft.ifftn(jnp.fft.fftn(pairs.reshape(shape),axes=(0,1,2))*coul,axes=(0,1,2)).real.reshape(pairs.shape)
    enuc,mad=_electrostatic(cell,parameters,lattice,(1,1,1))
    return PeriodicInputs(overlap[None],(kinetic+local+nonlocal_pp)[None],kinetic[None],ao,coords,
        jnp.full(ngrid,volume/ngrid),potential,enuc,mad,kpts,gv,jnp.asarray(w),
        **_context(cell,parameters,lattice,ft,gv,kpts))


def get_jk(inputs,density_spin,*,exxdiv='ewald',with_k=True,mesh=None):
    """Real Gamma spin density (2,nao,nao); Ewald K correction is explicit."""
    if exxdiv not in ('ewald','vcut_sph',None):
        raise NotImplementedError('Periodic exchange supports exxdiv=ewald, vcut_sph or None.')
    ao=inputs.ao[0,0];weight=inputs.weights[0]
    total=jnp.sum(density_spin,axis=0)
    hartree=jnp.einsum('gp,pq,gq->g',ao,total,ao)
    j=weight*jnp.einsum('g,gpq->pq',hartree,inputs.pair_potential)
    if not with_k:return j,jnp.zeros_like(density_spin)
    potential=inputs.pair_potential
    if exxdiv=='vcut_sph':
        kernel=exchange_kernel(inputs.gvectors,lattice=inputs.lattice,nk=1,
                               exxdiv=exxdiv,boundary=inputs.boundary)
        # The Hartree pair cache uses untruncated electrostatics. Only K changes.
        if mesh is None:
            raise ValueError('Gamma spherical exchange requires the explicit FFT mesh.')
        pairs=ao[:,:,None]*ao[:,None,:]
        potential=_pair_potential(pairs,kernel,mesh).real
    k=weight*jnp.einsum('gp,gs,tsr,grq->tpq',ao,ao,density_spin,potential)
    if exxdiv=='ewald':
        s=inputs.overlap[0];k=k+inputs.madelung*jnp.einsum('pr,trs,sq->tpq',s,density_spin,s)
    return j,k


def _assemble_kpoints(cell,parameters,lattice,ao,ft,gv,coords,kpts,kmesh,*,electrostatic=None):
    volume=jnp.linalg.det(lattice);mesh=cell.mesh;ngrid=int(np.prod(mesh))
    vg=local_potential(gv,parameters.nuclear_coords,cell.pseudopotentials,boundary=_boundary(cell,lattice))
    vr=jnp.fft.ifftn((vg*ngrid/volume).reshape(mesh)).reshape(-1).real
    def matrices(k,values,coeff):
        q=gv+k
        s=coeff.conj().T@coeff/volume
        t=coeff.conj().T@(.5*jnp.sum(q*q,axis=1)[:,None]*coeff)/volume
        v=values.conj().T@(vr[:,None]*values)*volume/ngrid
        v=v+nonlocal_matrix(q,coeff,parameters.nuclear_coords,cell.pseudopotentials,volume)
        return s,t,t+v
    overlaps,kinetics,cores=jax.vmap(matrices)(kpts,ao[:,0],ft)
    # Store periodic factors u_k; derivative channels retain (grad + i*k)u_k.
    ao=ao*jnp.exp(-1j*kpts@coords.T)[:,None,:,None]
    if electrostatic is None:
        enuc,mad=_electrostatic(cell,parameters,lattice,kmesh)
    else:
        enuc,mad=electrostatic
    return PeriodicInputs(overlaps,cores,kinetics,ao,coords,
        jnp.full(ngrid,volume/ngrid),None,enuc,mad,kpts,gv,
        **_context(cell,parameters,lattice,ft,gv,kpts))


def get_kpoint_jk(inputs,density_spin,*,mesh,exxdiv='ewald',with_k=True,targets=None):
    """Full source-k sums with optional representative target contractions."""
    if exxdiv not in ('ewald','vcut_sph',None):raise NotImplementedError('Unsupported exchange correction.')
    ao=inputs.ao[:,0];nk=ao.shape[0];weight=inputs.weights[0]
    from ...pbc.kpoints import weights_for
    kw=weights_for(inputs)
    total=density_spin.sum(axis=0)
    rho=jnp.einsum('k,kgp,kpq,kgq->g',kw,ao,total,ao.conj()).real
    vr=jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(mesh))*coulomb_kernel(inputs.gvectors,boundary=inputs.boundary).reshape(mesh)).reshape(-1).real
    target_ao=ao if targets is None else ao[jnp.asarray(targets)]
    j=weight*jnp.einsum('kgp,g,kgq->kpq',target_ao.conj(),vr,target_ao)
    if not with_k:return j,jnp.zeros((density_spin.shape[0],)+j.shape,dtype=density_spin.dtype)
    return j,_kpoint_exchange(inputs,density_spin,mesh=mesh,exxdiv=exxdiv,targets=targets)


def _pair_potential(pairs,kernel,mesh):
    """FFT Coulomb action on AO-pair columns, shared with geometry response."""
    shape=tuple(mesh)+pairs.shape[1:]
    return jnp.fft.ifftn(jnp.fft.fftn(pairs.reshape(shape),axes=(0,1,2))
        *kernel.reshape(tuple(mesh)+(1,1)),axes=(0,1,2)).reshape(pairs.shape)


def _kpoint_exchange(inputs,density_spin,*,mesh,exxdiv,bra=None,bra_overlap=None,targets=None,query=None):
    """Shared diagonal/finite-q/query exchange on the same source-k density."""
    if query is not None and (bra is not None or bra_overlap is not None or targets is not None):
        raise ValueError('Use either arbitrary query inputs or finite-q/representative targets.')
    from ...pbc.kpoints import weights_for
    ao=inputs.ao[:,0];nk=ao.shape[0];weight=inputs.weights[0];kw=weights_for(inputs)
    bra=ao if bra is None else bra
    bra_overlap=inputs.overlap if bra_overlap is None else bra_overlap
    target_ao=ao if query is None else query.ao[:,0]
    target_bra=bra if query is None else target_ao
    target_kpoints=inputs.kpoints if query is None else query.kpoints
    target_overlap=bra_overlap if query is None else query.overlap
    matrices=[]
    for target in (range(len(target_kpoints)) if targets is None else targets):
        k=jnp.zeros((density_spin.shape[0],target_ao.shape[-1],target_ao.shape[-1]),
                    dtype=jnp.result_type(density_spin,target_bra))
        for source in range(nk):
            pairs=ao[source].conj()[:,:,None]*target_ao[target][:,None,:]
            kernel=exchange_kernel(inputs.gvectors+target_kpoints[target]-inputs.kpoints[source],
                lattice=inputs.lattice,nk=nk,exxdiv=exxdiv,boundary=inputs.boundary)
            potential=_pair_potential(pairs,kernel,mesh)
            k=k+weight*kw[source]*jnp.einsum('gp,gs,tsr,grq->tpq',target_bra[target].conj(),bra[source],density_spin[:,source],potential)
        if exxdiv=='ewald':
            if query is None:
                s=inputs.overlap[target];dm=density_spin[:,target]
            else:
                s=target_overlap[target]
                match=jnp.all(abs(inputs.kpoints-target_kpoints[target])<1e-10,axis=1)
                dm=jnp.einsum('k,skpq->spq',match.astype(density_spin.dtype),density_spin)
            k=k+inputs.madelung*jnp.einsum('pr,trs,sq->tpq',target_overlap[target],dm,s)
        if exxdiv in ('ewald','vcut_sph'):
            k=k*jnp.where(jnp.all(jnp.abs(kw-1/nk)<1e-12),1.,jnp.nan)
        matrices.append(k)
    return jnp.stack(matrices,axis=1)


def build_band_inputs(cell,kpts,reference):
    """Arbitrary query k points on the reference grid; no SCF mesh interpretation."""
    kpts=jnp.asarray(kpts)
    ao,ft,gv,coords=bloch_grid(cell.topology,cell.parameters,cell.lattice,cell.mesh,kpts)
    return _assemble_kpoints(cell,cell.parameters,cell.lattice,ao,ft,gv,coords,kpts,None,
        electrostatic=(reference.nuclear_repulsion,reference.madelung))
