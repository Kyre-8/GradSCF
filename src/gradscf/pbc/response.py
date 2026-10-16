"""Finite-q FFT Hartree/exchange and semilocal XC kernel actions.

AO response blocks are D_q[s,k]=D[s,k+q,k]. Their cell-periodic density has
the physical phase exp(i*q*r); a real perturbation also includes its adjoint
(-q) component. No four-index response tensor is stored.
"""
import jax
import jax.numpy as jnp
from . import _kpoint
from .kpoints import weights_for,reciprocal_vectors
from ..integrals.periodic.coulomb import coulomb_kernel
from ..integrals.periodic.fft import _kpoint_exchange
from ..dft.libxc_jax.jax_libxc import hybrid_coeff,xc_type


def _validate_q_domain(inputs, exxdiv):
    """Keep array and geometry sources within the same electrostatic model."""
    if getattr(inputs, 'boundary', None) is not None:
        raise NotImplementedError('Finite-q response currently requires 3D electrostatics.')
    if exxdiv not in ('ewald', None):
        raise NotImplementedError('Finite-q response does not support vcut_sph; use exxdiv=ewald or None.')


def _shifted_ao(inputs,transfer):
    g=jnp.asarray(transfer.shifts)@reciprocal_vectors(inputs.lattice)
    phase=jnp.exp(1j*g@inputs.coords.T)
    return inputs.ao[transfer.partners]*phase[:,None,:,None]


def _fields(density,ao,plus,weights,kind):
    # Sum grid fields before moving to the next k point. Full-k contraction
    # intermediates otherwise dominate finite-q geometry response workspaces.
    dtype=jnp.result_type(density,ao,plus,weights)
    def accumulate(total,arrays):
        d,a,b,w=arrays
        d=d.astype(dtype)*w.astype(dtype)
        rho=jnp.einsum('gp,spq,gq->sg',b[0],d,a[0].conj())
        fields=rho.T
        if kind!='LDA':
            gradient=(jnp.einsum('xgp,spq,gq->sgx',b[1:4],d,a[0].conj())
                      +jnp.einsum('gp,spq,xgq->sgx',b[0],d,a[1:4].conj()))
            fields=jnp.concatenate([fields,gradient[0],gradient[1]],axis=1)
        if kind=='MGGA':
            tau=.5*jnp.einsum('xgp,spq,xgq->sg',b[1:4],d,a[1:4].conj())
            fields=jnp.concatenate([fields,tau.T],axis=1)
        return total+fields,None
    initial=jnp.zeros((ao.shape[2],density.shape[0] if kind=='LDA' else (10 if kind=='MGGA' else 8)),
                      dtype=dtype)
    return jax.lax.scan(accumulate,initial,
                        (density.swapaxes(0,1),ao,plus,weights))[0]


def _project(ao,plus,weights,potential,kind):
    def at_k(arrays):
        a,b=arrays;blocks=[]
        for spin in range(2):
            value=jnp.einsum('g,gp,gq->pq',weights*potential[:,spin],b[0].conj(),a[0])
            if kind in ('GGA','MGGA'):
                vg=weights[:,None]*potential[:,2+3*spin:5+3*spin]
                value+=jnp.einsum('gx,xgp,gq->pq',vg,b[1:4].conj(),a[0])
                value+=jnp.einsum('gx,gp,xgq->pq',vg,b[0].conj(),a[1:4])
            if kind=='MGGA':
                value+=.5*jnp.einsum('g,xgp,xgq->pq',weights*potential[:,8+spin],b[1:4].conj(),a[1:4])
            blocks.append(value)
        return jnp.stack(blocks)
    return jax.lax.map(at_k,(ao,plus)).swapaxes(0,1)


def q_density(inputs,transfer,density):
    """Complex spin-density envelopes, shape (2,ngrid), without exp(i*q*r)."""
    return _fields(density,inputs.ao,_shifted_ao(inputs,transfer),weights_for(inputs),'LDA').T


def project_q_potential(inputs,transfer,potential):
    """Project a local periodic envelope (ngrid,) or (2,ngrid) into AO q blocks."""
    v=jnp.asarray(potential)
    if v.shape==inputs.weights.shape:v=jnp.stack([v,v])
    if v.shape!=(2,len(inputs.weights)):raise ValueError('Potential envelope must have shape (ngrid,) or (2,ngrid)')
    return _project(inputs.ao,_shifted_ao(inputs,transfer),inputs.weights,v.T,'LDA')


def q_kernel(inputs,density,transfer,*,mesh,xc='hf',exxdiv='ewald',functional=None,params=None):
    """Return the complex-linear finite-q Fock derivative on a fixed AO basis."""
    _validate_q_domain(inputs, exxdiv)
    if functional is None:
        if params is not None:raise ValueError('params requires an external functional')
        kind=xc_type(xc);alpha=hybrid_coeff(xc)
        external_action=None
    else:
        from .functional import finite_q_kernel
        kind='HF';alpha=0.
        external_action=finite_q_kernel(functional,params,inputs,density,transfer)
    if kind not in ('HF','LDA','GGA','MGGA'):raise NotImplementedError('Unsupported finite-q XC kernel')
    ao=inputs.ao;plus=_shifted_ao(inputs,transfer);kw=weights_for(inputs)
    mesh=tuple(mesh)
    wavevector=jnp.asarray(transfer.q)@reciprocal_vectors(inputs.lattice)
    kernel=coulomb_kernel(inputs.gvectors+wavevector).reshape(mesh)
    xc_action=None
    if kind!='HF':
        variables=_kpoint.grid_xc_variables(density,inputs,kind)
        if kind=='MGGA':
            from ..scf.uks import _point_unrestricted_xc_hessian_kernel
            hessian=_point_unrestricted_xc_hessian_kernel(xc,kind)(variables)
            xc_action=lambda fields:jnp.einsum('gij,gj->gi',hessian,fields)
        else:
            _,xc_action=jax.linearize(lambda v:_kpoint._point_unrestricted_xc_value_and_grad_kernel(xc,kind)(v)[1],variables)
    def apply(d):
        fields=_fields(d,ao,plus,kw,'LDA' if kind=='HF' else kind)
        rho=fields[:,:2].sum(axis=1)
        vh=jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(mesh))*kernel).reshape(-1)
        potential=jnp.stack([vh,vh],axis=1)
        f=_project(ao,plus,inputs.weights,potential,'LDA')
        if xc_action is not None:
            vxc=xc_action(fields.real)+1j*xc_action(fields.imag)
            f+=_project(ao,plus,inputs.weights,vxc,kind)
        if external_action is not None:
            f+=external_action(d)
        if alpha:
            f-=alpha*_kpoint_exchange(inputs,d,mesh=mesh,exxdiv=exxdiv,
                                     bra=plus[:,0],bra_overlap=inputs.overlap[transfer.partners])
        return f
    return apply
