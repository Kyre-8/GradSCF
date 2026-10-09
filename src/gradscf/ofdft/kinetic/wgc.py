"""WGC99 second-order Taylor kinetic kernel (ATLAS parameters).

Wang, Govind & Carter, Phys. Rev. B 60, 16350 (1999),
https://doi.org/10.1103/PhysRevB.60.16350 . Kernel conventions were cross-checked
against libKEDF (BSD-3-Clause; see LICENSE.libKEDF and WGC.md).

A dimensionless ODE is solved once on the host. Only its fixed interpolation
coefficients are cached. Density, reference density, wavevectors and lattice
remain JAX values. alpha/beta/gamma are the documented fixed parameter set;
this module does not advertise gradients of those constants.
"""
from functools import lru_cache
import math
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from scipy.interpolate import BPoly, PPoly

from .semilocal import TF_CONSTANT
from .nonlocal_kernel import inverse_lindhard_minus_tf_vw

ALPHA=(5+math.sqrt(5))/6
BETA=(5-math.sqrt(5))/6
GAMMA=2.7
_MIN_ETA=1e-6
_MAX_ETA=200.


def _response_numpy(eta):
    if eta<1e-3:
        return -(8/3)*eta**2+(8/45)*eta**4
    if eta>2:
        # Evaluate the convergent large-eta Lindhard series without subtracting
        # two O(eta²) numbers. Cancellation noise otherwise stalls adaptive ODE
        # integration when a tight tolerance is requested.
        z=eta**-2
        coefficients=[1/((2*k+3)*(2*k+5)) for k in range(32)]
        h=np.polynomial.polynomial.polyval(z,coefficients)
        return -1-3*h/(1/3+z*h)
    if eta==1:
        return -2.
    return 1/(.5+(1-eta**2)/(4*eta)*np.log(abs((1+eta)/(1-eta))))-1-3*eta**2


@lru_cache(maxsize=1)
def _kernel_table():
    """Quintic Hermite table of w(log eta), independent of physical inputs."""
    # In t=log(eta), the defining Euler ODE has constant coefficients:
    # w_tt + (gamma-10) w_t + 36 alpha beta w = 20 R(exp(t)).
    coefficient=36*ALPHA*BETA
    def rhs(t,y):
        return [y[1],20*_response_numpy(np.exp(t))-(GAMMA-10)*y[1]-coefficient*y[0]]
    # Use the asymptotic particular solution, avoiding a spurious finite-start
    # homogeneous component from approximating the derivative by zero.
    a2=20*(-24/175)/(4-2*(GAMMA-10)+coefficient)
    a4=20*(-8/125)/(16-4*(GAMMA-10)+coefficient)
    w0=-32/coefficient+a2/_MAX_ETA**2+a4/_MAX_ETA**4
    d0=-2*a2/_MAX_ETA**2-4*a4/_MAX_ETA**4
    solution=solve_ivp(rhs,(math.log(_MAX_ETA),math.log(_MIN_ETA)),[w0,d0],
        method='DOP853',rtol=2e-12,atol=2e-14,max_step=.025,dense_output=True)
    if not solution.success:
        raise RuntimeError('WGC dimensionless kernel ODE failed: '+solution.message)
    knots=np.r_[np.linspace(math.log(_MIN_ETA),0,6001),np.linspace(0,math.log(_MAX_ETA),4001)[1:]]
    values=solution.sol(knots)
    second=np.array([rhs(t,y)[1] for t,y in zip(knots,values.T)])
    spline=PPoly.from_bernstein_basis(BPoly.from_derivatives(knots,
        np.stack([values[0],values[1],second],axis=1)))
    return knots,spline.c


def kernel_values(eta):
    """Return w, d w/d log eta, d² w/d(log eta)².

    The second derivative is reconstructed from the defining ODE, enforcing the
    exact uniform-density response despite interpolation error. Its derivatives
    inherit the explicit Kohn-anomaly guard of the Lindhard response.
    """
    eta=jnp.asarray(eta)
    shape=eta.shape
    q=eta.reshape(-1)
    knots,coefficients=_kernel_table()
    t=jnp.log(jnp.clip(q,_MIN_ETA,_MAX_ETA))
    knots=jnp.asarray(knots);coefficients=jnp.asarray(coefficients)
    index=jnp.clip(jnp.searchsorted(knots,t,side='right')-1,0,knots.size-2)
    offset=t-knots[index]
    c=coefficients[:,index]
    w=jnp.polyval(c,offset)
    d1=jnp.polyval(c[:-1]*jnp.arange(5,0,-1)[:,None],offset)
    # Small-eta homogeneous terms vanish faster than eta²; this limit is only
    # used below 1e-6, well below the wavevectors in molecular/crystal examples.
    b2=20*(-8/3)/(4+2*(GAMMA-10)+36*ALPHA*BETA)
    b4=20*(8/45)/(16+4*(GAMMA-10)+36*ALPHA*BETA)
    w=jnp.where(q<_MIN_ETA,b2*q**2+b4*q**4,w)
    d1=jnp.where(q<_MIN_ETA,2*b2*q**2+4*b4*q**4,d1)
    a2=20*(-24/175)/(4-2*(GAMMA-10)+36*ALPHA*BETA)
    a4=20*(-8/125)/(16-4*(GAMMA-10)+36*ALPHA*BETA)
    safe=jnp.maximum(q,_MAX_ETA)
    w=jnp.where(q>_MAX_ETA,-32/(36*ALPHA*BETA)+a2/safe**2+a4/safe**4,w)
    d1=jnp.where(q>_MAX_ETA,-2*a2/safe**2-4*a4/safe**4,d1)
    d2=20*inverse_lindhard_minus_tf_vw(q)-(GAMMA-10)*d1-36*ALPHA*BETA*w
    return tuple(a.reshape(shape) for a in (w,d1,d2))


def wgc_kernels(gvectors,reference_density):
    """Four Fourier kernels K0,K1,K2,K12; no C_TF prefactor included."""
    reference_density = jnp.asarray(reference_density)
    if reference_density.shape != ():
        raise ValueError('WGC reference_density must be a scalar.')
    if not isinstance(reference_density, jax.core.Tracer):
        value = np.asarray(reference_density)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('WGC reference_density must be finite and positive.')
    g2=jnp.sum(gvectors**2,axis=-1)
    q=jnp.where(g2>0,jnp.sqrt(jnp.where(g2>0,g2,1.)),0.)
    eta=q/(2*(3*jnp.pi**2*reference_density)**(1/3))
    w,d1,d2=kernel_values(eta)
    return (w,-d1/(6*reference_density),
            (d2+(6-GAMMA)*d1)/(36*reference_density**2),
            (d2+GAMMA*d1)/(36*reference_density**2))


def wang_govind_carter(features,reference_density=None):
    """Second-order nonlocal term; standard WGC also contains TF and full vW.

    K(r,r') = K0 + K1(theta+theta') + K2/2(theta²+theta'²)
              + K12 theta theta', theta=n-n_ref. Reference density defaults
    to cell average; pass it explicitly to follow a fixed-reference convention.
    """
    if not features.mesh:
        raise ValueError('WGC requires a periodic FFT grid.')
    n=jnp.maximum(features.rho,1e-18)
    ref=jnp.sum(features.weights*features.rho)/features.volume if reference_density is None else reference_density
    k0,k1,k2,k12=(k.reshape(features.mesh) for k in wgc_kernels(features.gvectors,ref))
    theta=(n-ref).reshape(features.mesh)
    a=n.reshape(features.mesh)**ALPHA
    b=n.reshape(features.mesh)**BETA
    fa=jnp.fft.fftn(a);fta=jnp.fft.fftn(theta*a);ftta=jnp.fft.fftn(.5*theta**2*a)
    first=jnp.fft.ifftn(k0*fa+k1*fta+k2*ftta).real
    second=jnp.fft.ifftn(k1*fa+k12*fta).real
    third=jnp.fft.ifftn(k2*fa).real
    integrand=b*(first+theta*second+.5*theta**2*third)
    return TF_CONSTANT*jnp.sum(features.weights*integrand.reshape(-1))
