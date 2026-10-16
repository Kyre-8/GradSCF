"""Primitive-cell finite-q atomic sources in a moving Gaussian basis.

The +q ket displacement maps k to k+q; the conjugate -q bra
displacement maps k+q back to k. Both legs and reciprocal folding phases
are retained. No supercell reference or electronic solve is constructed.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .finite_q import QSource
from ..pbc import _kpoint
from ..pbc.kpoints import weights_for,reciprocal_vectors
from ..pbc.momentum import momentum_transfer
from ..pbc.response import _fields,_project,_shifted_ao,_validate_q_domain
from ..integrals.periodic.ao import bloch_grid
from ..integrals.periodic.coulomb import coulomb_kernel
from ..integrals.periodic.pseudo import local_potential,_projector_overlap
from ..integrals.periodic.fft import _pair_potential
from ..dft.libxc_jax.jax_libxc import xc_type,hybrid_coeff


class DisplacementSource(NamedTuple):
    perturbation: QSource
    hcore: object


def _orbitals(cell,inputs,coordinates):
    """Periodic AO/gradient channels and Fourier columns at fixed lattice."""
    values,ft,_,_=bloch_grid(cell.topology,cell.parameters_at(coordinates),cell.lattice,cell.mesh,inputs.kpoints)
    return values*jnp.exp(-1j*inputs.kpoints@inputs.coords.T)[:,None,:,None],ft


def _nonlocal_source(cell,inputs,ft,dft,direction,partner):
    """Two AO legs and two GTH-projector legs, in separable form."""
    volume=jnp.linalg.det(cell.lattice)
    out=jnp.zeros_like(inputs.hcore,dtype=complex)
    for atom,pp in enumerate(cell.pseudopotentials):
        for l,(radius,count,coupling) in enumerate(pp[5:]):
            if not count:continue
            def overlaps(coeff,pos):
                return jax.vmap(lambda k,c:_projector_overlap(
                    inputs.gvectors+k,c,pos,l,radius,count,volume))(inputs.kpoints,coeff)
            pos=cell.coords[atom]
            o,da=jax.jvp(lambda coeff:overlaps(coeff,pos),(ft,),(dft,))
            db=jax.jvp(lambda r:overlaps(ft,r),(pos,),(direction[atom],))[1]
            h=jnp.asarray(coupling)
            product=lambda a,b:jnp.einsum('kimp,ij,kjmq->kpq',a.conj(),h,b)
            out+=product(da,o)+product(o[partner],da[partner])
            out+=product(db[partner],o)+product(o[partner],db)
    return out


def _exchange_source(inputs,density,ao,plus,dao,dplus,overlap,partner,*,mesh,exxdiv):
    """Differentiate the four AO factors of exchange at fixed coefficients."""
    kw=weights_for(inputs);weight=inputs.weights[0];blocks=[]
    a=ao[:,0];b=plus[:,0];da=dao[:,0];db=dplus[:,0]
    for k in range(len(a)):
        block=jnp.zeros_like(density[:,k])
        for s in range(len(a)):
            kernel=coulomb_kernel(inputs.gvectors+inputs.kpoints[k]-inputs.kpoints[s])
            def potential(pairs):return _pair_potential(pairs,kernel,mesh)
            v=potential(a[s].conj()[:,:,None]*a[k][:,None,:])
            dv=potential(b[s].conj()[:,:,None]*db[k][:,None,:]
                         +da[s].conj()[:,:,None]*a[k][:,None,:])
            contract=lambda x,y,d,v:jnp.einsum('gp,gs,tsr,grq->tpq',x.conj(),y,d,v)
            block+=weight*kw[s]*(contract(da[k],a[s],density[:,s],v)
                +contract(b[k],db[s],density[:,s],v)
                +contract(b[k],b[s],density[:,partner[s]],dv))
        if exxdiv=='ewald':
            s=inputs.overlap
            block+=inputs.madelung*(overlap[k]@density[:,k]@s[k]
                                    +s[partner[k]]@density[:,partner[k]]@overlap[k])
        blocks.append(block)
    return jnp.stack(blocks,axis=1)


def displacement_source(source,q,displacements):
    """Return QSource and hcore derivatives for a real atomic amplitude (Bohr).

    q is fractional reciprocal. The reference must be periodic HF/semilocal
    KS on a full equal-weight mesh closed under q. Reference validation is
    eager; displacement amplitudes remain differentiable and may be traced.
    """
    from ..pbc.scf import _SCF,validate_reference
    if not isinstance(source,_SCF):raise TypeError('A periodic HF/KS reference is required')
    validate_reference(source,integer=False,equal_weights=True)
    return _displacement_source(source,q,displacements)


def _displacement_source(source,q,displacements):
    """Array assembly after the public entry point has validated the snapshot."""
    _validate_q_domain(source.inputs, source.exxdiv)
    cell=source.cell;data=source.inputs;d=source.result.density_spin
    q=np.asarray(q,dtype=float)
    if q.shape!=(3,) or not np.isfinite(q).all():raise ValueError('q must be a finite three-vector')
    if np.allclose(q-np.rint(q),0.):raise ValueError('Use atomic_problem for Gamma displacement response')
    direction=jnp.asarray(displacements)
    if direction.shape!=cell.coords.shape or jnp.iscomplexobj(direction):
        raise ValueError('displacements must be real (natom,3) amplitudes in Bohr')
    if not isinstance(direction,jax.core.Tracer) and not np.isfinite(direction).all():
        raise ValueError('displacements must be finite')
    direction=direction.astype(cell.coords.dtype)
    kind=xc_type(source.xc);alpha=hybrid_coeff(source.xc)
    if kind not in ('HF','LDA','GGA','MGGA'):raise NotImplementedError('Unsupported atomic q XC source')
    transfer=momentum_transfer(np.asarray(source.kpts)@np.asarray(cell.lattice).T/(2*np.pi),q)
    partner=transfer.partners;mesh=tuple(cell.mesh);volume=jnp.linalg.det(cell.lattice)
    def orbitals(r):
        return _orbitals(cell,data,r)
    (ao,ft),(dao,dft)=jax.jvp(orbitals,(cell.coords,),(direction,))
    plus=_shifted_ao(data._replace(ao=ao),transfer)
    dplus=_shifted_ao(data._replace(ao=dao),transfer)
    adjoint=lambda a:a.conj().swapaxes(-1,-2)
    left=adjoint(dft)@ft/volume
    right=adjoint(ft)@dft/volume
    overlap=left+right[partner]
    kinetic=.5*jnp.sum((data.gvectors[None]+source.kpts[:,None])**2,axis=-1)
    left=adjoint(dft)@(kinetic[:,:,None]*ft)/volume
    right=adjoint(ft)@(kinetic[:,:,None]*dft)/volume
    hcore=left+right[partner]
    def poisson(rho,wave):
        kernel=coulomb_kernel(data.gvectors+wave).reshape(mesh)
        return jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(mesh))*kernel).ravel()
    def real_potential(vg):return jnp.fft.ifftn((vg*len(data.weights)/volume).reshape(mesh)).ravel()
    def basis_projection(potential,kind):
        return (_project(ao,dao,data.weights,potential,kind)
                +_project(dplus,plus,data.weights,potential,kind))
    wave=jnp.asarray(q)@reciprocal_vectors(cell.lattice)
    v0=real_potential(local_potential(data.gvectors,cell.coords,cell.pseudopotentials)).real
    dv=jax.jvp(lambda r:local_potential(data.gvectors+wave,r,cell.pseudopotentials),
               (cell.coords,),(direction,))[1]
    dv=real_potential(dv)
    hcore+=basis_projection(jnp.stack([v0,v0],axis=1),'LDA')[0]
    hcore+=_project(ao,plus,data.weights,jnp.stack([dv,dv],axis=1),'LDA')[0]
    hcore+=_nonlocal_source(cell,data,ft,dft,direction,partner)
    field_kind='LDA' if kind=='HF' else kind
    fields=_fields(d,ao,dplus,weights_for(data),field_kind)
    fields+=_fields(d[:,partner],dao,plus,weights_for(data),field_kind)
    rho0=_kpoint.grid_xc_variables(d,data,'LDA')
    v0=poisson(rho0.sum(axis=1),jnp.zeros(3)).real
    dv=poisson(fields[:,:2].sum(axis=1),wave)
    fock=hcore[None]+basis_projection(jnp.stack([v0,v0],axis=1),'LDA')
    fock+=_project(ao,plus,data.weights,jnp.stack([dv,dv],axis=1),'LDA')
    if kind!='HF':
        variables=_kpoint.grid_xc_variables(d,data,kind)
        potential,action=jax.linearize(lambda v:_kpoint._point_unrestricted_xc_value_and_grad_kernel(source.xc,kind)(v)[1],variables)
        fock+=basis_projection(potential,kind)
        fock+=_project(ao,plus,data.weights,action(fields.real)+1j*action(fields.imag),kind)
    if alpha:
        fock-=alpha*_exchange_source(data,d,ao,plus,dao,dplus,overlap,partner,mesh=mesh,exxdiv=source.exxdiv)
    return DisplacementSource(QSource(fock,overlap,fields[:,:2].T),hcore)
