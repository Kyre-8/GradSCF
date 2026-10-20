"""One custom-XC port for explicit recipes and external energy models."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft


def _factory():
    assert callable(getattr(dft, 'custom_xc', None)), 'dft.custom_xc must be a public constructor'
    return dft.custom_xc


def test_custom_traditional_recipe_keeps_its_exact_exchange_and_name():
    functional = _factory()('.3*HF + .7*gga_x_pbe + gga_c_pbe', name='my_pbe_hybrid')
    assert isinstance(functional, dft.TraditionalXCFunctional)
    assert functional.name == 'my_pbe_hybrid'
    assert functional.recipe.exact_exchange.full_range == pytest.approx(.3)
    assert functional.response_kind == 'GGA'


@pytest.mark.parametrize('network', [False, True])
def test_external_formula_and_network_use_the_existing_full_energy_derivatives(network):
    constructor = _factory()
    ao = jnp.asarray([[.8, .2], [.3, .7], [-.2, .6]])
    molecule = SimpleNamespace(ao=ao, overlap_matrix=jnp.eye(2),
        grid=SimpleNamespace(weights=jnp.asarray([.4, .5, .3]), coords=jnp.zeros((3, 3))))
    density = jnp.asarray([[.9, .1], [.1, .4]])
    if network:
        features = lambda fields: fields.total_density_matrix.reshape(-1)
        def energy(params, inputs):
            value = inputs
            for weight, bias in params[:-1]:
                value = jnp.tanh(value @ weight+bias)
            weight, bias = params[-1]
            return jnp.sum(value @ weight+bias)
        params = [(jnp.full((4, 3), .1), jnp.zeros(3)),
                  (jnp.full((3, 3), .2), jnp.zeros(3)),
                  (jnp.full((3, 1), -.1), jnp.zeros(1))]
    else:
        features = lambda fields: dict(rho=fields.rho, weights=fields.weights)
        energy = lambda params, inputs: params['a']*jnp.dot(inputs['weights'], inputs['rho']**2)
        params = dict(a=jnp.asarray(.02))
    functional = constructor(input_fn=features, energy_fn=energy, name='external')
    assert type(functional) is dft.Functional
    assert functional.name == 'external'
    value = functional.energy_for_density(params, molecule, density)
    potential = functional.potential(params, molecule, density)
    kernel = functional.kernel_action(params, molecule, density, jnp.eye(2))
    assert np.isfinite(value) and np.isfinite(potential).all() and np.isfinite(kernel).all()
    gradient = jax.grad(lambda p: functional.energy_for_density(p, molecule, density))(params)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(gradient))


def test_custom_xc_rejects_ambiguous_recipe_plus_energy_callback():
    with pytest.raises(TypeError, match='recipe|spec'):
        _factory()('pbe', input_fn=lambda fields: fields.rho,
                   energy_fn=lambda params, rho: jnp.sum(rho))
