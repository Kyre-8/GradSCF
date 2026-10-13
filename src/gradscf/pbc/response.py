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


def _shifted_ao(inputs,transfer):
    g=jnp.asarray(transfer.shifts)@reciprocal_vectors(inputs.lattice)
    phase=jnp.exp(1j*g@inputs.coords.T)
    return inputs.ao[transfer.partners]*phase[:,None,:,None]


def _fields(density,ao,plus,weights,kind):
    rho=jnp.einsum('k,kgp,skpq,kgq->sg',weights,plus[:,0],density,ao[:,0].conj())
    if kind=='LDA':return rho.T
    gradient=(jnp.einsum('k,kxgp,skpq,kgq->sgx',weights,plus[:,1:4],density,ao[:,0].conj())
              +jnp.einsum('k,kgp,skpq,kxgq->sgx',weights,plus[:,0],density,ao[:,1:4].conj()))
    return jnp.concatenate([rho.T,gradient[0],gradient[1]],axis=1)


def _project(ao,plus,weights,potential,kind):
    blocks=[]
    for spin in range(2):
        value=jnp.einsum('g,kgp,kgq->kpq',weights*potential[:,spin],plus[:,0].conj(),ao[:,0])
        if kind=='GGA':
            vg=weights[:,None]*potential[:,2+3*spin:5+3*spin]
            value+=jnp.einsum('gx,kxgp,kgq->kpq',vg,plus[:,1:4].conj(),ao[:,0])
            value+=jnp.einsum('gx,kgp,kxgq->kpq',vg,plus[:,0].conj(),ao[:,1:4])
        blocks.append(value)
    return jnp.stack(blocks)


def q_density(inputs,transfer,density):
    """Complex spin-density envelopes, shape (2,ngrid), without exp(i*q*r)."""
    return _fields(density,inputs.ao,_shifted_ao(inputs,transfer),weights_for(inputs),'LDA').T


def project_q_potential(inputs,transfer,potential):
    """Project a local periodic envelope (ngrid,) or (2,ngrid) into AO q blocks."""
    v=jnp.asarray(potential)
    if v.shape==inputs.weights.shape:v=jnp.stack([v,v])
    if v.shape!=(2,len(inputs.weights)):raise ValueError('Potential envelope must have shape (ngrid,) or (2,ngrid)')
    return _project(inputs.ao,_shifted_ao(inputs,transfer),inputs.weights,v.T,'LDA')


def q_kernel(inputs,density,transfer,*,mesh,xc='hf',exxdiv='ewald'):
    """Return the complex-linear finite-q Fock derivative on a fixed AO basis."""
    kind=xc_type(xc);alpha=hybrid_coeff(xc)
    if kind not in ('HF','LDA','GGA'):raise NotImplementedError('Finite-q kernels support HF/LDA/GGA/global hybrids')
    if exxdiv not in ('ewald',None):raise NotImplementedError('Unsupported exchange correction')
    ao=inputs.ao;plus=_shifted_ao(inputs,transfer);kw=weights_for(inputs)
    mesh=tuple(mesh)
    wavevector=jnp.asarray(transfer.q)@reciprocal_vectors(inputs.lattice)
    kernel=coulomb_kernel(inputs.gvectors+wavevector).reshape(mesh)
    xc_action=None
    if kind!='HF':
        variables=_kpoint.grid_xc_variables(density,inputs,kind)
        _,xc_action=jax.linearize(lambda v:_kpoint._point_unrestricted_xc_value_and_grad_kernel(xc,kind)(v)[1],variables)
    def apply(d):
        fields=_fields(d,ao,plus,kw,'GGA' if kind=='GGA' else 'LDA')
        rho=fields[:,:2].sum(axis=1)
        vh=jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(mesh))*kernel).reshape(-1)
        potential=jnp.stack([vh,vh],axis=1)
        f=_project(ao,plus,inputs.weights,potential,'LDA')
        if xc_action is not None:
            vxc=xc_action(fields.real)+1j*xc_action(fields.imag)
            f+=_project(ao,plus,inputs.weights,vxc,kind)
        if alpha:
            f-=alpha*_kpoint_exchange(inputs,d,mesh=mesh,exxdiv=exxdiv,
                                     bra=plus[:,0],bra_overlap=inputs.overlap[transfer.partners])
        return f
    return apply
