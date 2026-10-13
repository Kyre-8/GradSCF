"""Independent, handwritten PBE oracle for DFPT tests (atomic units).

PBE: doi:10.1103/PhysRevLett.77.3865; PW92: doi:10.1103/PhysRevB.45.13244.
Constants follow the authors' EXCHPBE/CORPBE routines (not PBEsol):
https://dft.uci.edu/pubs/PBEsol.html
This test helper is not a production XC backend. Both spin densities must be
positive; the density floor regularizes molecular vacuum quadrature points.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _pw92(rs, a, alpha, b1, b2, b3, b4):
    denominator = 2*a*(b1*jnp.sqrt(rs)+b2*rs+b3*rs**1.5+b4*rs**2)
    return -2*a*(1+alpha*rs)*jnp.log1p(1/denominator)


def pbe_energy_density(rho, gradient):
    """Energy per volume, rho (2,...), gradient (2,3,...), spin polarized."""
    rho = jnp.maximum(rho, 1e-20)
    sigma = jnp.sum(gradient**2, axis=1)
    reduced_square = sigma/(4*(6*jnp.pi**2)**(2/3)*rho**(8/3))
    enhancement = 1+.804-.804/(1+.2195149727645171/.804*reduced_square)
    exchange = -.75*(6/jnp.pi)**(1/3)*jnp.sum(rho**(4/3)*enhancement, axis=0)

    total = jnp.sum(rho, axis=0)
    zeta = (rho[0]-rho[1])/total
    rs = (3/(4*jnp.pi*total))**(1/3)
    eu = _pw92(rs, .0310907, .21370, 7.5957, 3.5876, 1.6382, .49294)
    ep = _pw92(rs, .01554535, .20548, 14.1189, 6.1977, 3.3662, .62517)
    am = _pw92(rs, .0168869, .11125, 10.357, 3.6231, .88026, .49671)
    spin_factor = ((1+zeta)**(4/3)+(1-zeta)**(4/3)-2)/(2**(4/3)-2)
    fzz = 8/(9*(2**(4/3)-2))
    ec = eu+spin_factor*(zeta**4*(ep-eu)-am*(1-zeta**4)/fzz)
    phi = ((1+zeta)**(2/3)+(1-zeta)**(2/3))/2
    gamma, beta = (1-jnp.log(2))/jnp.pi**2, .06672455060314922
    ks_square = 4*(3*jnp.pi**2*total)**(1/3)/jnp.pi
    t_square = jnp.sum(jnp.sum(gradient, axis=0)**2, axis=0)/(4*phi**2*ks_square*total**2)
    a = (beta/gamma)/jnp.expm1(-ec/(gamma*phi**3))
    at = a*t_square
    h = gamma*phi**3*jnp.log1p((beta/gamma)*t_square*(1+at)/(1+at+at**2))
    return exchange+total*(ec+h)


def pbe_xc_energy(scale, molecule, density):
    """External energy callback using the same AO inputs as the SCF."""
    ao, dao = molecule['ao'], molecule['ao_deriv1'][1:4]
    rho = jnp.einsum('gp,spq,gq->sg', ao, density, ao)
    gradient = (jnp.einsum('xgp,spq,gq->sxg', dao, density, ao)
                + jnp.einsum('gp,spq,xgq->sxg', ao, density, dao))
    return scale*jnp.sum(molecule['weights']*pbe_energy_density(rho, gradient))


def test_pbe_energy_potential_and_kernel_against_libxc():
    from pyscf.dft import libxc
    rng = np.random.default_rng(48)
    rho = rng.normal(size=(2,4,12))*.08
    rho[:,0] = rng.uniform(.03, 1.4, size=(2,12))
    rho[:,1:,0] = 0.  # Uniform gas, including gradient Hessian at zero gradient.
    rho[1,:,1] = rho[0,:,1]  # Unpolarized limit.
    expected = [libxc.eval_xc_eff('PBE', rho, deriv=n) for n in range(3)]
    point = lambda r: pbe_energy_density(r[:,0], r[:,1:])
    value, potential = jax.vmap(jax.value_and_grad(point), in_axes=2, out_axes=(0,2))(jnp.asarray(rho))
    kernel = jax.vmap(jax.hessian(point), in_axes=2, out_axes=-1)(jnp.asarray(rho))
    np.testing.assert_allclose(value, expected[0]*rho[:,0].sum(axis=0), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(potential, expected[1], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(kernel, expected[2], rtol=1e-9, atol=1e-10)
