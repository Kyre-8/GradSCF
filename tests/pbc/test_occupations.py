"""Finite-temperature particle conservation and degenerate density response."""
import jax
import jax.numpy as jnp
import numpy as np


def test_weighted_fermi_occupations_and_chemical_potential_derivative():
    from gradscf.pbc.occupations import fermi_occupations
    energies = jnp.array([[[-1., .2, .8], [-.4, .4, 1.2]]])
    weights = jnp.array([.3, .7])
    result = jax.jit(lambda e:fermi_occupations(e, jnp.array([1.3]), .12, weights))(energies)
    assert result.converged
    np.testing.assert_allclose(jnp.einsum('k,skn->s', weights, result.occupations), [1.3], atol=1e-12)
    derivative = jax.grad(lambda shift:fermi_occupations(energies+shift, jnp.array([1.3]), .12, weights).chemical_potential.sum())(0.)
    np.testing.assert_allclose(derivative, 1., atol=1e-11)
    charge = lambda x:jnp.einsum('k,skn->', weights,
        fermi_occupations(energies.at[0,0,0].add(x), jnp.array([1.3]), .12, weights).occupations)
    np.testing.assert_allclose(jax.grad(charge)(0.), 0., atol=1e-11)


def test_degenerate_fermi_density_jvp_and_vjp():
    from gradscf.pbc.occupations import fermi_density
    fock = jnp.zeros((1,1,2,2), dtype=complex)
    b = jnp.array([[[[.2, .3j], [-.3j, -.2]]]])
    evaluate = lambda f:fermi_density(f, jnp.eye(2, dtype=complex)[None],
                                      jnp.array([1.]), .2, jnp.ones(1))[0]
    value, derivative = jax.jit(lambda:jax.jvp(evaluate, (fock,), (b,)))()
    np.testing.assert_allclose(value, .5*jnp.eye(2)[None,None], atol=1e-12)
    np.testing.assert_allclose(derivative, -b/.8, atol=1e-11)
    probe = lambda t:jnp.real(jnp.sum(evaluate(fock+t*b)*b.conj()))
    np.testing.assert_allclose(jax.grad(probe)(0.), (probe(1e-5)-probe(-1e-5))/2e-5, atol=1e-9)


def test_empty_and_full_spin_channels():
    from gradscf.pbc.occupations import fermi_density
    fock = jnp.broadcast_to(jnp.diag(jnp.array([-1., 1.])), (2,1,2,2))
    density, result = fermi_density(fock, jnp.eye(2)[None], jnp.array([0.,2.]), .1, jnp.ones(1))
    assert result.converged
    np.testing.assert_allclose(density[0], 0., atol=1e-12)
    np.testing.assert_allclose(density[1,0], jnp.eye(2), atol=1e-12)
    np.testing.assert_allclose(result.entropy, 0., atol=1e-12)


def test_temperature_and_charge_guard():
    from gradscf.pbc.occupations import fermi_occupations,fermi_density
    result = fermi_occupations(jnp.zeros((1,1,2)), jnp.array([3.]), .1, jnp.ones(1))
    assert not result.converged
    assert jnp.isnan(result.occupations).all()
    density,status=fermi_density(jnp.zeros((1,1,2,2)),jnp.eye(2)[None],jnp.array([3.]),.1,jnp.ones(1))
    assert not status.converged
    assert jnp.isnan(density).all()


def test_low_temperature_gapped_chemical_potential_is_resolved():
    from gradscf.pbc.occupations import fermi_occupations
    e=jnp.array([[[-1.,1.]]])
    out=fermi_occupations(e,jnp.array([1.]),.01,jnp.ones(1))
    assert out.converged
    np.testing.assert_allclose(out.chemical_potential,0.,atol=1e-9)
    expected_entropy=2*(jnp.log1p(jnp.exp(-100.))+100*jax.nn.sigmoid(-100.))
    np.testing.assert_allclose(out.entropy,expected_entropy,rtol=1e-12,atol=0.)
    slope=jax.grad(lambda shift:fermi_occupations(e+shift,jnp.array([1.]),.01,jnp.ones(1)).chemical_potential.sum())(0.)
    np.testing.assert_allclose(slope,1.,atol=1e-9)
