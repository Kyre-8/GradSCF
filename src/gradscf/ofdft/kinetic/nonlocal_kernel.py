"""Wang--Teter periodic convolution (alpha=beta=5/6).

Reference: Wang & Teter, Phys. Rev. B 45, 13196 (1992),
https://doi.org/10.1103/PhysRevB.45.13196 . DFTpy is a numerical oracle;
this JAX implementation differentiates a scalar energy, not separate potentials.
"""
import jax
import jax.numpy as jnp
from .semilocal import TF_CONSTANT


def _inverse_response(eta):
    """1/L(eta) - 1 - 3 eta^2, with stable small/large-wavevector limits.

    The value at eta=1 is continuous; lattice/kernel response exactly at this
    Kohn anomaly is not defined (logarithmically divergent derivative).
    """
    small, large = eta < 1e-3, eta > 30.
    # Keep inactive branches finite for higher-order AD.
    z = jnp.where(small | large, .5, eta)
    distance = jnp.abs(1-z)
    safe = jnp.where(distance == 0, 1., distance)
    logarithm = jnp.log((1+z)/safe)
    lindhard = .5 + (1-z*z)/(4*z)*logarithm
    normal = 1/lindhard - 1 - 3*z*z
    low = -(8/3)*eta**2 + (8/45)*eta**4
    inv = 1/jnp.where(large, eta, 30.)**2
    high = -8/5 - 24/175*inv - 8/125*inv**2
    return jnp.where(small, low, jnp.where(large, high, normal))


@jax.custom_jvp
def inverse_lindhard_minus_tf_vw(eta):
    """Finite kernel value; response at the exact Kohn anomaly is rejected."""
    return _inverse_response(eta)


@inverse_lindhard_minus_tf_vw.defjvp
def _inverse_response_jvp(primals, tangents):
    eta, = primals
    direction, = tangents
    value, slope = jax.jvp(_inverse_response, (eta,), (jnp.ones_like(eta),))
    slope = slope*jnp.where(jnp.asarray(eta)==1., jnp.nan, 1.)
    return value, slope*direction


def wang_teter(features):
    """Nonlocal contribution only; TF and full vW are added by the wrapper."""
    if not features.mesh:
        raise ValueError('WT requires a periodic FFT grid.')
    rho = features.rho
    rho0 = jnp.sum(features.weights*rho) / features.volume
    g2 = jnp.sum(features.gvectors**2, axis=-1)
    q = jnp.where(g2 > 0, jnp.sqrt(jnp.where(g2 > 0, g2, 1.)), 0.)
    eta = q/(2*(3*jnp.pi**2*rho0)**(1/3))
    alpha = 5/6
    kernel = TF_CONSTANT*5/(9*alpha**2)*inverse_lindhard_minus_tf_vw(eta)
    # A declared low-density floor is necessary for fractional-power Hessians.
    power = jnp.maximum(rho, 1e-18)**alpha
    convolution = jnp.fft.ifftn(jnp.fft.fftn(power.reshape(features.mesh))*
                                kernel.reshape(features.mesh)).real.reshape(-1)
    return jnp.sum(features.weights*power*convolution)
