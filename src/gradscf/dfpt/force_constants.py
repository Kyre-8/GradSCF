"""Primitive-cell mixed (-q,+q) geometry and relaxed force constants.

The fixed-D,W curvature is assembled from harmonic AO variations. No enlarged
cell, supercell SCF, or four-index electron-repulsion tensor is constructed.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .displacement import _orbitals,displacement_source
from .finite_q import finite_q_problem
from ..pbc import _kpoint
from ..pbc.kpoints import weights_for,reciprocal_vectors
from ..pbc.momentum import momentum_transfer
from ..pbc.response import _fields,_shifted_ao,q_kernel
from ..integrals.periodic.coulomb import coulomb_kernel,ewald_curvature
from ..integrals.periodic.fft import _pair_potential
from ..integrals.periodic.pseudo import local_potential,_projector_overlap
from ..dft.libxc_jax.jax_libxc import xc_type,hybrid_coeff


def _mixed(function,x,u,v):
    return jax.jvp(lambda y:jax.jvp(function,(y,),(u,))[1],(x,),(v,))[1]


def _nonlocal_curvature(cell,data,density,ft,fu,fv,fuv,u,v,p,m):
    """All four AO/projector legs of the separable GTH energy."""
    volume=jnp.linalg.det(cell.lattice);kw=weights_for(data)
    value=jnp.array(0.,dtype=complex)
    for atom,pp in enumerate(cell.pseudopotentials):
        for l,(radius,count,coupling) in enumerate(pp[5:]):
            if not count:continue
            pos=cell.coords[atom]
            def overlap(coeff,r):
                return jnp.stack([_projector_overlap(data.gvectors+k,c,r,l,radius,count,volume)
                                  for k,c in zip(data.kpoints,coeff)])
            o=overlap(ft,pos);au=overlap(fu,pos);av=overlap(fv,pos)
            bu=jax.jvp(lambda r:overlap(ft,r),(pos,),(u[atom],))[1]
            bv=jax.jvp(lambda r:overlap(ft,r),(pos,),(v[atom],))[1]
            aa=overlap(fuv,pos)
            bb=_mixed(lambda r:overlap(ft,r),pos,u[atom],v[atom])
            uv=jax.jvp(lambda r:overlap(fv,r),(pos,),(u[atom],))[1]
            vu=jax.jvp(lambda r:overlap(fu,r),(pos,),(v[atom],))[1]
            ket=aa+bb+uv[p]+vu[m];bra=aa+bb+vu[p]+uv[m]
            h=jnp.asarray(coupling)
            product=lambda a,b:jnp.einsum('kimp,ij,kjmq->kpq',a.conj(),h,b)
            matrix=product(bra,o)+product(o,ket)
            matrix+=product(au[p]+bu,av[p]+bv)+product(av[m]+bv,au[m]+bu)
            value+=jnp.einsum('k,skij,kji->',kw,density,matrix)
    return value


def _pair_variations(ao,d2,up,um,vp,vm):
    """Zero, -q, +q and mixed variations of a transition-density factor."""
    a=ao[:,0];ab=d2[:,0]
    up=up[:,0];um=um[:,0];vp=vp[:,0];vm=vm[:,0]
    pair=lambda x,y:x.conj()[:,:,None]*y[:,None,:]
    def factors(k,l):
        base=pair(a[k],a[l])
        minus=pair(up[k],a[l])+pair(a[k],um[l])
        plus=pair(vm[k],a[l])+pair(a[k],vp[l])
        second=pair(ab[k],a[l])+pair(a[k],ab[l])+pair(up[k],vp[l])+pair(vm[k],um[l])
        return base,minus,plus,second
    return factors


def geometry_curvature(source,q,u,v):
    """Fixed-D,W bilinear geometry curvature, in Eh/Bohr**2.

    Return components (one-body, Hartree, exchange, XC, metric, ionic).
    u is the real amplitude of the -q row and v of the +q column. This is
    an internal numerical assembly, not an SCF/training facade.
    """
    cell=source.cell;data=source.inputs;density=source.result.density_spin
    q=np.asarray(q,dtype=float);kw=weights_for(data);volume=jnp.linalg.det(cell.lattice)
    fractional=np.asarray(source.kpts)@np.asarray(cell.lattice).T/(2*np.pi)
    positive=momentum_transfer(fractional,q);negative=momentum_transfer(fractional,-q)
    p=positive.partners;m=negative.partners
    orbitals=lambda r:_orbitals(cell,data,r)
    (ao,ft),(du,fu)=jax.jvp(orbitals,(cell.coords,),(u,))
    dv,fv=jax.jvp(orbitals,(cell.coords,),(v,))[1]
    d2,fuv=_mixed(orbitals,cell.coords,u,v)
    lift=lambda a,t:_shifted_ao(data._replace(ao=a),t)
    up,um=lift(du,positive),lift(du,negative)
    vp,vm=lift(dv,positive),lift(dv,negative)
    adjoint=lambda a:a.conj().swapaxes(-1,-2)
    def mixed_matrix(weight):
        return (adjoint(fuv)@(weight*ft)+adjoint(ft)@(weight*fuv)
            +adjoint(fu[p])@(weight[p]*fv[p])+adjoint(fv[m])@(weight[m]*fu[m]))/volume
    overlap=mixed_matrix(jnp.ones(ft.shape[:2]+(1,)))
    kinetic=mixed_matrix(.5*jnp.sum((data.gvectors[None]+data.kpoints[:,None])**2,axis=-1)[:,:,None])
    trace=lambda matrix:jnp.einsum('k,skij,kji->',kw,density,matrix)
    kind=xc_type(source.xc);alpha=hybrid_coeff(source.xc)
    field_kind='GGA' if kind=='GGA' else 'LDA'
    fields=lambda left,right:_fields(density,left,right,kw,field_kind)
    second=fields(ao,d2)+fields(d2,ao)+fields(up,vp)+fields(vm,um)
    # Opposite-q density fields carry their own reciprocal folding phases.
    plus_ao=lift(ao,positive);minus_ao=lift(ao,negative)
    first_v=fields(ao,vp)+_fields(density[:,p],dv,plus_ao,kw,field_kind)
    first_u=fields(ao,um)+_fields(density[:,m],du,minus_ao,kw,field_kind)
    variables=_kpoint.grid_xc_variables(density,data,field_kind)
    rho0=variables[:,:2].sum(axis=1);rho2=second[:,:2].sum(axis=1)
    rhou=first_u[:,:2].sum(axis=1);rhov=first_v[:,:2].sum(axis=1)
    mesh=tuple(cell.mesh);wave=jnp.asarray(q)@reciprocal_vectors(cell.lattice)
    ifft=lambda values:jnp.fft.ifftn(values.reshape(mesh)).ravel()
    def poisson(rho,shift):
        return ifft(jnp.fft.fftn(rho.reshape(mesh)).ravel()*coulomb_kernel(data.gvectors+shift))
    potential=lambda coords,g:local_potential(g,coords,cell.pseudopotentials)
    vr=ifft(potential(cell.coords,data.gvectors))*len(data.weights)/volume
    vr2=ifft(_mixed(lambda r:potential(r,data.gvectors),cell.coords,u,v))*len(data.weights)/volume
    vu=ifft(jax.jvp(lambda r:potential(r,data.gvectors-wave),(cell.coords,),(u,))[1])*len(data.weights)/volume
    vv=ifft(jax.jvp(lambda r:potential(r,data.gvectors+wave),(cell.coords,),(v,))[1])*len(data.weights)/volume
    local=jnp.dot(data.weights,rho2*vr+rhou*vv+rhov*vu+rho0*vr2)
    one_body=trace(kinetic)+local+_nonlocal_curvature(cell,data,density,ft,fu,fv,fuv,u,v,p,m)
    hartree=jnp.dot(data.weights,rho2*poisson(rho0,jnp.zeros(3))+rhou*poisson(rhov,wave))
    xc=jnp.array(0.,dtype=complex)
    if kind!='HF':
        potential_xc,action=jax.linearize(lambda z:_kpoint._point_unrestricted_xc_value_and_grad_kernel(source.xc,kind)(z)[1],variables)
        kernel_v=action(first_v.real)+1j*action(first_v.imag)
        xc=jnp.einsum('g,gx,gx->',data.weights,potential_xc,second)+jnp.einsum('g,gx,gx->',data.weights,first_u,kernel_v)
    exchange=jnp.array(0.,dtype=complex)
    if alpha:
        factors=_pair_variations(ao,d2,up,um,vp,vm)
        for k in range(len(data.kpoints)):
            for l in range(len(data.kpoints)):
                a,am,ap,a2=factors(k,l);b,bm,bp,b2=factors(l,k)
                def potential_pair(pair,shift):
                    kernel=coulomb_kernel(data.gvectors+data.kpoints[k]-data.kpoints[l]+shift)
                    return _pair_potential(pair,kernel,mesh)
                contract=lambda left,right:jnp.einsum('spq,gqr,srt,gtp->',density[:,k],left,density[:,l],right)
                term=contract(a2,potential_pair(b,0.))+contract(a,potential_pair(b2,0.))
                term+=contract(am,potential_pair(bp,wave))+contract(ap,potential_pair(bm,-wave))
                exchange-=.5*alpha*kw[k]*kw[l]*data.weights[0]*term
        if source.exxdiv=='ewald':
            su=(adjoint(fu)@ft+adjoint(ft[p])@fu[p])/volume
            sv=(adjoint(fv)@ft+adjoint(ft[p])@fv[p])/volume
            term=density@overlap@density@data.overlap
            term+=density@adjoint(su)@density[:,p]@sv
            exchange-=alpha*data.madelung*jnp.einsum('k,skii->',kw,term)
    w=density@source.result.fock_spin@jnp.linalg.inv(data.overlap)
    w=.5*(w+adjoint(w))
    metric=-jnp.einsum('k,skij,kji->',kw,w,overlap)
    ionic=ewald_curvature(cell.lattice,cell.coords,cell.charges,jnp.asarray(q),u,v,
                          precision=cell.precision,reference_lattice=cell.lattice)
    return jnp.stack([one_body,hartree,exchange,xc,metric,ionic])


def primitive_force_response(source,q,config=None):
    """Return primitive reference forces and a compiled relaxed Phi(q) action."""
    from .atomic import _ks_forces
    from ..integrals.periodic.fft import build_inputs
    from ..pbc.scf import validate_reference
    validate_reference(source,integer=False,equal_weights=True)
    cell=source.cell;data=source.inputs;result=source.result
    fractional=np.asarray(source.kpts)@np.asarray(cell.lattice).T/(2*np.pi)
    transfer=momentum_transfer(fractional,q);p=transfer.partners
    options=dict(mesh=cell.mesh,xc=source.xc,exxdiv=source.exxdiv)
    problem=finite_q_problem(data,result,transfer,temperature=source.temperature,**options)
    interaction=q_kernel(data,result.density_spin,transfer,**options)
    d=result.density_spin;f=result.fock_spin;inverse=jnp.linalg.inv(data.overlap)
    w=d@f@inverse;w=.5*(w+w.conj().swapaxes(-1,-2));kw=weights_for(data)
    def inputs_at(r):
        return build_inputs(cell,parameters=cell.parameters_at(r),kpts=source.kpts,kweights=source.kweights)
    forces=_ks_forces(d,cell.coords,inputs_at,**options)
    def respond(v):
        sv=displacement_source(source,q,v).perturbation
        response=problem.response(tangent=sv,config=config)
        d1=response.density;f1=sv.fock+interaction(d1);s1=sv.overlap
        w1=(d1@f+d[:,p]@f1-w[:,p]@s1)@inverse
        w1+=inverse[p]@(f[:,p]@d1+f1@d-s1@w)
        w1*=.5
        def bilinear(u):
            su=displacement_source(source,q,u).perturbation
            relaxed=jnp.einsum('k,skij,skij->',kw,su.fock.conj(),d1)
            relaxed-=jnp.einsum('k,kij,skij->',kw,su.overlap.conj(),w1)
            value=jnp.sum(geometry_curvature(source,q,u,v))+relaxed
            return jnp.stack([value.real,value.imag])
        parts=jax.jacrev(bilinear)(jnp.zeros_like(cell.coords))
        return (parts[0]+1j*parts[1]).ravel(),response.residual_norm,response.converged
    return forces,jax.jit(respond)
