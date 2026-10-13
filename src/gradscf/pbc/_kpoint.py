"""Complex k-mesh SCF; common DIIS, convergence, XC and AD infrastructure."""
import jax
import jax.numpy as jnp
from ..integrals.periodic.fft import get_kpoint_jk
from ..solvers.nonlinear.diis import diis_extrapolate,DIIS_SPACE
from ..scf.convergence import convergence_reached
from ..scf.uks import _point_unrestricted_xc_value_and_grad_kernel
from ..scf.autodiff import SCFDifferentiationConfig,attach_scf_backward
from ..dft.libxc_jax.jax_libxc import hybrid_coeff,xc_type
from .kpoints import weights_for


def energy_and_fock(density,inputs,*,mesh,xc='hf',exxdiv='ewald',functional=None,params=None):
    alpha=hybrid_coeff(xc) if functional is None else 0.
    j,k=get_kpoint_jk(inputs,density,mesh=mesh,exxdiv=exxdiv,with_k=alpha!=0)
    if functional is None:
        exc,potential=grid_xc_potential(density,inputs,xc=xc)
        vxc=project_xc_potential(inputs.ao,inputs.weights,potential,xc_type(xc))
    else:
        from ..dft.derivatives import xc_energy_and_potential_from_density
        result=xc_energy_and_potential_from_density(params,molecule=inputs,density=density,
            xc_energy_fn=functional.energy_for_density,density_weights=weights_for(inputs))
        exc,vxc=result.xc_energy,result.vxc_matrix
    h=inputs.hcore
    kw=weights_for(inputs)
    energy=jnp.einsum('k,skpq,kqp->',kw,density,h+.5*j).real
    energy-=.5*alpha*jnp.einsum('k,skpq,skqp->',kw,density,k).real
    return energy+exc+inputs.nuclear_repulsion,h+j-alpha*k+vxc


def grid_xc_potential(density,inputs,*,xc):
    """XC fields on the SCF grid, reused for non-self-consistent band queries."""
    kind=xc_type(xc)
    if kind=='HF':return jnp.array(0.),None
    variables=grid_xc_variables(density,inputs,kind)
    energies,potential=_point_unrestricted_xc_value_and_grad_kernel(xc,kind)(variables)
    return jnp.dot(inputs.weights,energies),potential


def grid_xc_variables(density,inputs,kind):
    """Real spin density/gradient variables shared by ground-state and q response."""
    ao=inputs.ao[:,0];deriv=inputs.ao[:,1:4]
    kw=weights_for(inputs)
    rho=jnp.einsum('k,kgp,skpq,kgq->sg',kw,ao,density,ao.conj()).real
    grad=2*jnp.einsum('k,kgp,skpq,kxgq->sgx',kw,ao,density,deriv.conj()).real
    return rho.T if kind=='LDA' else jnp.concatenate([rho.T,grad[0],grad[1]],axis=1)


def project_xc_potential(ao_channels,weights,potential,kind):
    ao=ao_channels[:,0];deriv=ao_channels[:,1:4]
    if potential is None:return jnp.zeros((2,ao.shape[0],ao.shape[-1],ao.shape[-1]),dtype=ao.dtype)
    blocks=[]
    for spin in range(2):
        vr=potential[:,spin]
        block=jnp.einsum('g,kgp,kgq->kpq',weights*vr,ao.conj(),ao)
        if kind=='GGA':
            vg=potential[:,2+3*spin:5+3*spin]
            term=jnp.einsum('gx,kxgp,kgq->kpq',weights[:,None]*vg,deriv.conj(),ao)
            block=block+term+term.conj().swapaxes(-1,-2)
        blocks.append(block)
    return jnp.stack(blocks)


def run_kpoint_scf(inputs,*,mesh,nelec,xc='hf',exxdiv='ewald',max_cycle=100,
                   conv_tol=1e-10,conv_tol_density=1e-8,conv_tol_grad=1e-7,
                   damping=0.,level_shift=0.,init_density=None,differentiation=None,
                   temperature=None,functional=None,params=None,reduction=None):
    from .scf import PeriodicSCFResult
    from .occupations import fermi_density, fermi_occupations, density_entropy
    if functional is None:
        if params is not None:raise ValueError('params requires an external functional')
        if xc_type(xc) not in ('HF','LDA','GGA'):raise NotImplementedError('Unsupported periodic XC.')
    else:
        from ..dft import Functional
        if not isinstance(functional,Functional):raise TypeError('Use dft.Functional for external periodic XC')
    nk,n,_=inputs.overlap.shape
    if reduction is not None and len(reduction.full_to_ir)!=nk:
        raise ValueError('Time-reversal map must cover the full SCF k mesh')
    thermal=temperature is not None
    if max_cycle<1 or not 0<=damping<1:raise ValueError('Invalid cycles or damping.')
    if not thermal and any(v<0 or v>n or v!=int(v) for v in nelec):
        raise ValueError('Zero-temperature SCF requires integer spin populations in [0,nao].')
    if thermal and level_shift!=0:
        raise ValueError('Finite-temperature SCF does not support projector level shifts.')
    config=differentiation or SCFDifferentiationConfig(mode='explicit')
    args=(inputs,jnp.asarray(temperature if thermal else 0.),jnp.asarray(nelec,dtype=float),params)
    s=inputs.overlap
    def transform(s):
        return jnp.linalg.inv(jnp.linalg.cholesky(s)).conj().swapaxes(-1,-2)
    x=transform(s)
    occ=jnp.stack([jnp.broadcast_to((jnp.arange(n)<count).astype(float),(nk,n)) for count in nelec]) if not thermal else None
    def diagonalize(f,metric=x):
        orth=metric.conj().swapaxes(-1,-2)@f@metric
        eps,c=jnp.linalg.eigh(orth)
        return eps,metric@c
    def density_from(c,occupation=occ):return jnp.einsum('skpi,ski,skqi->skpq',c,occupation,c.conj())
    def mapped(f,a,*,reduce=False):
        weights=weights_for(a[0]);overlap=a[0].overlap
        use_reduction=reduce and reduction is not None
        valid=True
        if use_reduction:
            valid=(reduction.consistent(f)&reduction.consistent(overlap)
                   &jnp.all(jnp.abs(weights-weights[reduction.partners])<1e-12))
            f=f[:,reduction.representatives];overlap=overlap[reduction.representatives]
            weights=reduction.reduce_weights(weights)
        if thermal:
            density=fermi_density(f,overlap,a[2],a[1],weights)[0]
        else:
            occupation=occ[:,reduction.representatives] if use_reduction else occ
            density=density_from(diagonalize(f,transform(overlap))[1],occupation)
        return reduction.expand(density)*jnp.where(valid,1.,jnp.nan) if use_reduction else density
    def evaluate(d,a):
        energy,fock=energy_and_fock(d,a[0],mesh=mesh,xc=xc,exxdiv=exxdiv,functional=functional,params=a[3])
        entropy=density_entropy(d,a[0].overlap,weights_for(a[0])) if thermal else jnp.array(0.)
        return energy-a[1]*entropy,fock,energy,entropy
    initial_fock=jnp.broadcast_to(inputs.hcore,(2,nk,n,n)).astype(jnp.complex128)
    eps,c=diagonalize(initial_fock)
    d=mapped(initial_fock,args,reduce=True) if init_density is None else jnp.asarray(init_density,dtype=c.dtype)
    if d.shape!=(2,nk,n,n):raise ValueError('Spin density must have shape (2,nk,nao,nao).')
    e,f,_,_=evaluate(d,args)
    def error(f,d):return x.conj().swapaxes(-1,-2)@(f@d@s-s@d@f)@x
    def pack(a):return jnp.stack([a.real,a.imag])
    history=jnp.zeros((DIIS_SPACE,2,2,nk,n,n))
    errors=jnp.zeros((DIIS_SPACE,2*f.size))
    def step(state,index):
        def advance(state):
            d,e,f,c,eps,fh,eh,head,count,done,cycles=state
            feff,fh,eh,head,count=diis_extrapolate(pack(f),pack(error(f,d)).reshape(-1),fh,eh,head,count)
            feff=feff[0]+1j*feff[1]+level_shift*(s-s@d@s)
            if thermal or reduction is not None:
                target=mapped(feff,args,reduce=True)
            else:
                eps,c=diagonalize(feff)
                target=density_from(c)
            dn=(1-damping)*target+damping*d
            en,fn,_,_=evaluate(dn,args)
            rms=jnp.sqrt(jnp.mean(jnp.abs(dn-d)**2))
            residual=(jnp.linalg.norm(mapped(fn,args,reduce=True)-dn)/jnp.sqrt(dn.size) if thermal
                      else jnp.linalg.norm(error(fn,dn))/jnp.sqrt(2*nk))
            done=convergence_reached(jnp.abs(en-e),rms,residual,conv_tol=conv_tol,
                conv_tol_density=conv_tol_density,conv_tol_grad=conv_tol_grad,has_prior_cycle=index>0)
            return dn,en,fn,c,eps,fh,eh,head,count,done,index+1
        state=jax.lax.cond(state[-2],lambda s:s,advance,state)
        return state,None
    initial=(d,e,f,c,eps,history,errors,jnp.array(0),jnp.array(0),jnp.array(False),jnp.array(0))
    (d,e,f,c,eps,*rest),_=jax.lax.scan(step,initial,jnp.arange(max_cycle))
    converged,cycles=rest[-2:]
    if reduction is not None:
        # Certify the ORIGINAL full fixed point, including symmetry-breaking
        # directions. Implicit backward below also deliberately uses that space.
        converged &= jnp.linalg.norm(mapped(f,args)-d)/jnp.sqrt(d.size)<=max(conv_tol_density,conv_tol_grad)
    if config.mode=='implicit':
        def residual(d,a):
            _,f,_,_=evaluate(d,a)
            return mapped(f,a)-d
        d=attach_scf_backward(args,solution=d,residual=residual,config=config,converged=converged)
    free,f,energy,entropy=evaluate(d,args)
    eps,c=diagonalize(f)
    mu=None
    if thermal:
        population=fermi_occupations(eps,args[2],args[1],weights_for(inputs))
        occ,mu=population.occupations,population.chemical_potential
        converged &= population.converged
    return PeriodicSCFResult(energy,d,c,eps,occ,f,converged,cycles,free,entropy,mu)
