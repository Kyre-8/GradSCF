"""Factored local omegaB97 form used only by the omegaB97X-D adapter.

Chai and Head-Gordon, JCP 128, 084106 (2008), Eqs. 6--7 and 11--26:
https://doi.org/10.1063/1.2834918. Coefficients remain supplied by the upstream
omegaB97X-D factory (PCCP 10, 6615, https://doi.org/10.1039/B810189B).

The generated jax_xc expression expands powers of the bounded B97 variable
into separate large density reciprocals. Their third derivatives overflow
at active molecular quadrature points. Horner evaluation below is the same
polynomial, without these intermediate powers. The SR-LDA attenuation uses
the same 1.35 branch and eight-term asymptotic series as LibXC/jax_xc.
"""
import jax
import jax.numpy as jnp


def _polynomial(coefficients, u):
    value = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        value = coefficient + u*value
    return value


def _bounded_gradient(q):
    # The second form avoids cancellation between O(1/q) terms in the
    # derivative of q/(1+q) when the reduced gradient is large.
    return jnp.where(q > 1., 1-1/(1+q), q/(1+q))


def _pw92_component(rs, a, alpha1, beta1, beta2, beta3, beta4):
    """PW92 Eq. 10, using the parameters present in LibXC's WB97 kernel."""
    root = jnp.sqrt(rs)
    denominator = 2*a*root*(beta1 + root*(beta2 + root*(beta3 + root*beta4)))
    return -2*a*(1 + alpha1*rs)*jnp.log1p(1/denominator)


def _pw92_polarized(rs):
    return _pw92_component(rs, .015545, .20548, 14.1189, 6.1977, 3.3662, .62517)


def _pw92_total(rs, zeta, zeta_threshold):
    unpolarized = _pw92_component(rs, .031091, .21370, 7.5957, 3.5876, 1.6382, .49294)
    polarized = _pw92_polarized(rs)
    stiffness = -_pw92_component(rs, .016887, .11125, 10.357, 3.6231, .88026, .49671)
    spin_factors = jnp.maximum(jnp.asarray([1+zeta, 1-zeta]), zeta_threshold)
    f = (jnp.sum(spin_factors*jnp.cbrt(spin_factors))-2)/(2**(4/3)-2)
    # Preserve the PW92/LibXC parameter, including its published rounding.
    # Replacing it by the analytic limit would change this named functional.
    f_second_zero = 1.709921
    zeta4 = zeta**4
    return unpolarized + f*(zeta4*(polarized-unpolarized)
                            + (1-zeta4)*stiffness/f_second_zero)


def _sr_attenuation(a):
    # Keep both branches in-domain under vmap and higher-order differentiation.
    # At equality the selected series gets its right-branch derivative. The
    # upstream nested (> then >=) clamps accidentally gave zero AD there.
    small = jnp.where(a < 1.35, a, 1.35)
    large = jnp.where(a >= 1.35, a, 1.35)
    inverse_square = (1/large)**2
    series = inverse_square*_polynomial(
        (1/36, -1/960, 1/26880, -1/829440, 1/28385280,
         -1/1073479680, 1/44590694400, -1/2021444812800), inverse_square)
    exponential = jnp.exp(-1/(4*small**2))
    direct = 1-(8/3)*small*(jnp.sqrt(jnp.pi)*jax.lax.erf(1/(2*small))
                            + 2*small*(exponential-1.5-2*small**2*(exponential-1)))
    return jnp.where(a >= 1.35, series, direct)


def _polarized_epsilon(p, density, sigma):
    total = jnp.sum(density)
    spin_factors = 2*density/total
    active = (density > p.dens_threshold) & (spin_factors > p.zeta_threshold)
    # Retain every positive spin density, including values below the masking
    # threshold. Only an exactly absent spin needs a finite unused branch.
    safe_density = jnp.where(density > 0., density, total/2)
    root = jnp.cbrt(safe_density)
    reduced_squared = sigma/(safe_density*root)**2
    u_x = _bounded_gradient(.004*reduced_squared)
    u_ss = _bounded_gradient(.2*reduced_squared)
    u_ab_numerator = .003*jnp.sum(reduced_squared)
    u_ab = _bounded_gradient(u_ab_numerator)
    a = p.cam_omega/(2*jnp.cbrt(6*jnp.pi**2)*root)
    exchange = (-1.5*jnp.cbrt(3/(4*jnp.pi))*safe_density*root
                *_sr_attenuation(a)*_polynomial(p.params.c_x, u_x))
    rs_spin = jnp.cbrt(3/(4*jnp.pi))/root
    same_spin = safe_density*_pw92_polarized(rs_spin)
    exchange = jnp.where(active, exchange, 0.)
    same_spin = jnp.where(active, same_spin, 0.)
    rs = jnp.cbrt(3/(4*jnp.pi*total))
    zeta = (density[0]-density[1])/total
    opposite_spin = total*_pw92_total(rs, zeta, p.zeta_threshold)-jnp.sum(same_spin)
    correlation = (jnp.sum(same_spin*_polynomial(p.params.c_ss, u_ss))
                   + opposite_spin*_polynomial(p.params.c_ab, u_ab))
    return (jnp.sum(exchange)+correlation)/total


def invoke(p, rho, r, mo=None):
    """Retain the upstream density-function interface and density threshold."""
    from jax_xc.impl.utils import rho_to_arguments

    density, sigma = rho_to_arguments(p, rho, r, mo)
    if p.nspin == 1:
        density = jnp.asarray([density/2, density/2])
        sigma = jnp.asarray([sigma/4, sigma/4])
    else:
        sigma = jnp.asarray([sigma[0], sigma[2]])
    return jax.lax.cond(jnp.sum(density) < p.dens_threshold,
                        lambda: jnp.asarray(0., dtype=density.dtype),
                        lambda: _polarized_epsilon(p, density, sigma))
