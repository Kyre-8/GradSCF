"""GTH local Fourier potential and normalized nonlocal Gaussian projectors."""
import math
import jax.numpy as jnp
from .coulomb import coulomb_kernel


def solid_harmonics(vectors,l):
    """Real, orthonormal r**l Y_lm; phases/order cancel in each GTH l block."""
    x,y,z=vectors.T;r2=x*x+y*y+z*z
    values=[]
    for m in range(l+1):
        poly=jnp.zeros_like(z)
        for k in range((l-m)//2+1):
            c=(-1)**k*math.factorial(2*l-2*k)/(2**l*math.factorial(k)*math.factorial(l-k)*math.factorial(l-2*k-m))
            poly=poly+c*z**(l-2*k-m)*r2**k
        norm=math.sqrt((2*l+1)/(4*math.pi)*math.factorial(l-m)/math.factorial(l+m))
        value=(-1)**m*norm*(x+1j*y)**m*poly
        values.extend([value.real] if m==0 else [math.sqrt(2)*value.real,math.sqrt(2)*value.imag])
    return jnp.stack(values,axis=1)


def projector(vectors,l,index,radius):
    q2=jnp.sum(vectors*vectors,axis=1)*radius**2
    previous=jnp.ones_like(q2);current=2*l+3-q2
    for i in range(1,index):
        previous,current=current,(4*i+2*l+3-q2)*current-2*i*(2*i+2*l+1)*previous
    polynomial=previous if index==0 else current
    norm=4*jnp.pi**1.5/math.sqrt(math.gamma(l+2*index+1.5))
    radial=norm*radius**(l+1.5)*polynomial*jnp.exp(-q2/2)
    return (-1j)**l*solid_harmonics(vectors,l)*radial[:,None]


def local_potential(gvectors,coords,pseudopotentials,*,boundary=None):
    g2=jnp.sum(gvectors*gvectors,axis=1);coul=coulomb_kernel(gvectors,boundary=boundary)
    potential=jnp.zeros(len(gvectors),dtype=complex)
    for pos,pp in zip(coords,pseudopotentials):
        charge=sum(pp[0]);radius=pp[1];coefs=pp[3];x=g2*radius**2
        polys=[jnp.ones_like(x),3-x,15-10*x+x*x,105-105*x+21*x*x-x**3]
        gaussian=(2*jnp.pi)**1.5*radius**3*jnp.exp(-x/2)*sum(c*p for c,p in zip(coefs,polys))
        ionic=-charge*coul*jnp.exp(-x/2)
        if boundary is None:
            ionic=jnp.where(g2>0,ionic,2*jnp.pi*charge*radius**2)
        potential=potential+jnp.exp(-1j*gvectors@pos)*(ionic+gaussian)
    return potential


def _projector_overlap(vectors,ao_fourier,pos,l,radius,count,volume):
    """Separable GTH projector/AO overlaps, also used by displacement AD."""
    beta=jnp.stack([projector(vectors,l,i,radius) for i in range(count)],axis=0)
    beta=beta*jnp.exp(-1j*vectors@pos)[None,:,None]
    return jnp.einsum('igm,gp->imp',beta.conj(),ao_fourier)/volume


def nonlocal_matrix(vectors,ao_fourier,coords,pseudopotentials,volume):
    result=jnp.zeros((ao_fourier.shape[1],)*2,dtype=complex)
    for pos,pp in zip(coords,pseudopotentials):
        for l,(radius,count,coupling) in enumerate(pp[5:]):
            if not count: continue
            overlap=_projector_overlap(vectors,ao_fourier,pos,l,radius,count,volume)
            result=result+jnp.einsum('imp,ij,jmq->pq',overlap.conj(),jnp.asarray(coupling),overlap)
    return result
