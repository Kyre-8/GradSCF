"""3D Ewald electrostatics with a uniform compensating background."""
import itertools
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ReducedBoundary:
    """Static periodicity/grid and a live lattice for reduced-D electrostatics."""
    lattice: object
    dimension: int
    mesh: tuple

    def tree_flatten(self):
        return (self.lattice,), (self.dimension, self.mesh)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0], *metadata)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ScreenedExchangeContext:
    """Static BvK geometry used to fix screened-Ewald AD truncation topology."""
    reference_lattice: tuple
    kmesh: tuple
    precision: float
    mesh: tuple = ()

    def tree_flatten(self):
        return (), (self.reference_lattice, self.kmesh, self.precision, self.mesh)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*metadata)


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


def coulomb_kernel(gvectors, *, boundary=None):
    if boundary is not None:
        from .truncated import slab_kernel, wire_kernel
        if boundary.dimension == 2:
            return slab_kernel(gvectors, boundary.lattice)
        if boundary.dimension != 1:
            raise ValueError('Reduced boundary dimension must be 1 or 2.')
        from .ao import reciprocal_grid
        if gvectors.shape != (int(np.prod(boundary.mesh)), 3):
            raise ValueError('Wire Coulomb action requires the complete FFT grid.')
        axis = boundary.lattice[0]/jnp.linalg.norm(boundary.lattice[0])
        shift = gvectors[0]
        wave = shift@axis
        valid = (jnp.max(abs(shift-wave*axis)) < 1e-10)
        valid &= jnp.max(abs(gvectors-shift-reciprocal_grid(boundary.lattice, boundary.mesh))) < 1e-10
        return wire_kernel(boundary.lattice, boundary.mesh, wave=wave).reshape(-1)*jnp.where(valid, 1., jnp.nan)
    g2=jnp.sum(gvectors*gvectors,axis=-1)
    return jnp.where(g2>1e-24,4*jnp.pi/jnp.where(g2>1e-24,g2,1.),0.)


def exchange_kernel(gvectors, *, lattice, nk, exxdiv='ewald', boundary=None,
                    omega=None, range_part='full'):
    """Exchange quadrature kernel; Hartree retains its own boundary.

    vcut_sph uses Rc=(3*Nk*V/(4*pi))**(1/3), with an analytic finite G=0
    limit. The sinc expression avoids subtraction and square-root derivatives
    at zero. This is the finite-cell spherical truncation used by PySCF
    (Phys. Rev. B 77, 193110); k convergence remains necessary.

    Screened 3D exchange uses erfc(omega*r)/r (short) or erf(omega*r)/r
    (long), omega in inverse Bohr. Reduced boundaries and screened spherical
    truncation are unsupported. Screened Ewald corrections are separate.
    """
    if range_part not in ('full','short','long'):
        raise ValueError('range_part must be full, short or long.')
    if range_part != 'full':
        if boundary is not None:
            raise NotImplementedError('Screened exchange currently requires 3D electrostatics.')
        if exxdiv not in ('ewald',None):
            raise NotImplementedError('Screened exchange does not support vcut_sph.')
        if omega is None:
            raise ValueError('Screened exchange requires a positive omega in inverse Bohr.')
        if isinstance(omega,(int,float,np.number)) and (not np.isfinite(omega) or omega<=0):
            raise ValueError('Screened exchange requires a positive finite omega.')
        omega=jnp.asarray(omega)
        g2=jnp.sum(gvectors*gvectors,axis=-1)
        x=g2/(4*omega**2)
        small=x<1e-6
        ratio=jnp.where(small,1-x/2+x*x/6-x**3/24,
                        -jnp.expm1(-x)/jnp.where(small,1.,x))
        short=jnp.pi/omega**2*ratio
        # Full Coulomb omits G=0. Retain the finite SR limit and put its
        # negative into LR, so the same regularized operator is SR+LR.
        value=short if range_part=='short' else coulomb_kernel(gvectors)-short
        return value*jnp.where(jnp.isfinite(omega)&(omega>0),1.,jnp.nan)
    if omega is not None:
        raise ValueError('omega requires range_part=short or long.')
    if exxdiv in ('ewald',None):
        return coulomb_kernel(gvectors,boundary=boundary)
    if exxdiv!='vcut_sph':
        raise NotImplementedError('Use exxdiv=ewald, vcut_sph or None.')
    if boundary is not None:
        raise NotImplementedError('Spherical exchange quadrature currently requires a 3D cell.')
    radius=(3*nk*jnp.linalg.det(lattice)/(4*jnp.pi))**(1/3)
    square=jnp.sum(gvectors**2,axis=-1)*radius**2
    # sinc(sqrt(x)/(2*pi))**2 = (1-cos(sqrt(x)))/(x/2).
    small=square<1e-8
    argument=jnp.sqrt(jnp.where(small,1.,square))/(2*jnp.pi)
    factor=jnp.where(small,1-square/12+square**2/360-square**3/20160,jnp.sinc(argument)**2)
    return 2*jnp.pi*radius**2*factor


def screened_exchange_correction(lattice,full_madelung,context,omega,range_part,
                                 *,reference_omega=None):
    """Screened probe-charge correction relative to the finite SR/LR G=0.

    The Gaussian reciprocal sum is the attenuated Ewald probe charge used by
    PySCF. Its static bounds use reference geometry and omega; live lattice
    and omega retain derivatives. A traced omega requires reference_omega.
    The two corrections sum exactly to the full Coulomb Madelung constant.
    """
    if context is None:
        raise ValueError('Screened Ewald exchange requires the BvK reference context.')
    if reference_omega is None:
        if not isinstance(omega,(int,float,np.number)):
            raise ValueError('A traced screened-Ewald omega requires reference_omega for fixed topology.')
        reference_omega=float(omega)
    if not np.isfinite(reference_omega) or reference_omega<=0:
        raise ValueError('reference_omega must be positive and finite.')
    reference=np.asarray(context.kmesh)[:,None]*np.asarray(context.reference_lattice)
    reciprocal=2*np.pi*np.linalg.inv(reference).T
    cutoff=2*reference_omega*np.sqrt(-np.log(context.precision*1e-3))
    ng=int(np.ceil(cutoff/np.linalg.svd(reciprocal)[1].min()))+2
    indices=_integer_box(ng)
    supercell=jnp.asarray(context.kmesh)[:,None]*lattice
    gv=indices@(2*jnp.pi*jnp.linalg.inv(supercell).T)
    g2=jnp.sum(gv*gv,axis=-1)
    volume=jnp.linalg.det(supercell)
    omega=jnp.asarray(omega)
    reciprocal_sum=jnp.sum(coulomb_kernel(gv)*jnp.exp(-g2/(4*omega**2)))/volume
    long_probe=2*omega/jnp.sqrt(jnp.pi)-reciprocal_sum
    long=long_probe+jnp.pi/(omega**2*volume)
    return full_madelung-long if range_part=='short' else long
