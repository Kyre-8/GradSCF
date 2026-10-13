"""3D Ewald electrostatics with a uniform compensating background."""
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc


def _integer_box(n):
    return jnp.asarray(list(itertools.product(range(-n,n+1),repeat=3)),dtype=float)


def _ewald_setup(reference_lattice,precision):
    """Static image topology shared by ground-state and harmonic electrostatics."""
    ref=np.asarray(reference_lattice)
    length=abs(np.linalg.det(ref))**(1/3)
    eta=2.0/length
    cutoff=np.sqrt(-np.log(precision*1e-2))
    nr=int(np.ceil((cutoff/eta+np.linalg.norm(ref,axis=1).sum())/np.linalg.svd(ref)[1].min()))
    reciprocal=2*np.pi*np.linalg.inv(ref).T
    ng=int(np.ceil(2*eta*cutoff/np.linalg.svd(reciprocal)[1].min()))
    return eta,_integer_box(nr),_integer_box(ng)


def ewald_energy(lattice,coords,charges,*,precision=1e-9,reference_lattice=None):
    # Truncation topology is fixed from the reference lattice during AD.
    eta,real_indices,integers=_ewald_setup(lattice if reference_lattice is None else reference_lattice,precision)
    lattice=jnp.asarray(lattice);coords=jnp.asarray(coords);charges=jnp.asarray(charges)
    volume=jnp.linalg.det(lattice)
    # Wrap arbitrary positions before applying the fixed image list.
    fractional=coords@jnp.linalg.inv(lattice)
    positions=(fractional-jnp.floor(fractional))@lattice
    images=real_indices@lattice
    diff=positions[:,None,None,:]-positions[None,:,None,:]+images[None,None,:,:]
    r2=jnp.sum(diff*diff,axis=-1)
    distance=jnp.sqrt(jnp.where(r2>1e-24,r2,1.))
    real=.5*jnp.sum(charges[:,None,None]*charges[None,:,None]*
        jnp.where(r2>1e-24,erfc(eta*distance)/distance,0.))
    gv=integers@(2*jnp.pi*jnp.linalg.inv(lattice).T)
    g2=jnp.sum(gv*gv,axis=-1);safe=jnp.where(g2>0,g2,1.)
    structure=jnp.exp(-1j*gv@positions.T)@charges
    recip=2*jnp.pi/volume*jnp.sum(jnp.where(g2>0,jnp.exp(-g2/(4*eta*eta))/safe,0.)*jnp.abs(structure)**2)
    return real+recip-eta/jnp.sqrt(jnp.pi)*jnp.sum(charges**2)-jnp.pi*jnp.sum(charges)**2/(2*eta*eta*volume)


def ewald_curvature(lattice,coords,charges,q,u,v,*,precision=1e-9,reference_lattice=None):
    """Bilinear ionic Hessian for (-q,u),(+q,v), without a supercell.

    q is fractional reciprocal; u,v are real Cartesian amplitudes. The
    complex result preserves the atomic cell-label convention even when
    reference atoms lie outside the primitive cell.
    """
    eta,images,integers=_ewald_setup(lattice if reference_lattice is None else reference_lattice,precision)
    reciprocal=2*jnp.pi*jnp.linalg.inv(lattice).T
    shift=jnp.floor(coords@jnp.linalg.inv(lattice));positions=coords-shift@lattice
    minus=u*jnp.exp(2j*jnp.pi*(shift@q))[:,None]
    plus=v*jnp.exp(-2j*jnp.pi*(shift@q))[:,None]
    delta=positions[:,None,None]-positions[None,:,None]+(images@lattice)[None,None]
    r2=jnp.sum(delta**2,axis=-1);r=jnp.sqrt(jnp.where(r2>1e-24,r2,1.))
    radial=lambda x:erfc(eta*x)/x
    first=jax.vmap(jax.grad(radial))(r.ravel()).reshape(r.shape)
    second=jax.vmap(jax.grad(jax.grad(radial)))(r.ravel()).reshape(r.shape)
    phase=jnp.exp(2j*jnp.pi*(images@q))
    du=minus[:,None,None]-phase[None,None,:,None]*minus[None,:,None]
    dv=plus[:,None,None]-phase.conj()[None,None,:,None]*plus[None,:,None]
    pair=(second-first/r)/r**2*jnp.sum(du*delta,axis=-1)*jnp.sum(dv*delta,axis=-1)
    pair+=first/r*jnp.sum(du*dv,axis=-1)
    real=.5*jnp.sum(charges[:,None,None]*charges[None,:,None]*jnp.where(r2>1e-24,pair,0.))
    gv=integers@reciprocal;wave=q@reciprocal
    def kernel(g):
        g2=jnp.sum(g*g,axis=-1);safe=jnp.where(g2>0,g2,1.)
        return jnp.where(g2>0,jnp.exp(-g2/(4*eta**2))/safe,0.)
    structure=jnp.exp(-1j*gv@coords.T)*charges
    mixed=-jnp.sum(structure*(gv@u.T)*(gv@v.T),axis=1)
    qp=gv+wave;phases=jnp.exp(-1j*qp@coords.T)*charges
    first_u=-1j*jnp.sum(phases*(qp@u.T),axis=1)
    first_v=-1j*jnp.sum(phases*(qp@v.T),axis=1)
    reciprocal_part=jnp.sum(kernel(gv)*(jnp.sum(structure,axis=1).conj()*mixed).real)
    reciprocal_part+=jnp.sum(kernel(qp)*first_u.conj()*first_v)
    return real+4*jnp.pi/jnp.linalg.det(lattice)*reciprocal_part


def coulomb_kernel(gvectors):
    g2=jnp.sum(gvectors*gvectors,axis=-1)
    return jnp.where(g2>1e-24,4*jnp.pi/jnp.where(g2>1e-24,g2,1.),0.)
