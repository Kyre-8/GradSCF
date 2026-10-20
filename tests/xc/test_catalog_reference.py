"""Every registered reference recipe's local energy/potential against Libxc.

This tests semilocal channels only, including those of methods that require
separate dispersion/NLC/PT2 operators. It is not a complete-method certificate.
"""
import json
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import xc
from test_libxc_reference import _libxc_energy_gradient


_CATALOG = json.loads(files('gradscf.dft.xc').joinpath('_catalog.json').read_text())
_CASES = list(_CATALOG['names'].items())


@pytest.mark.parametrize('reference,record', _CASES, ids=[name for name, _ in _CASES])
def test_registered_local_energy_and_potential_match_libxc(reference, record):
    if record['needs_laplacian']:
        pytest.skip('This evaluator has no Laplacian field; complete consumers reject it.')
    recipe = xc.parse_recipe(reference)
    if not xc.available(reference):
        pytest.skip('No enabled local JAX factory for this reference recipe; see xc.info.')
    jax.config.update('jax_enable_x64', True)
    size = {'LDA': 2, 'GGA': 8, 'MGGA': 10}[recipe.kind]
    fields = np.asarray([.31, .23, .12, -.04, .03, -.02, .11, .05, .25, .20])[:size]
    expected_energy, expected_potential = _libxc_energy_gradient(reference, fields, recipe.kind)
    energy, potential = xc.grid_value_and_grad(reference, recipe.kind)(jnp.asarray(fields[None]))
    np.testing.assert_allclose(energy[0], expected_energy, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(potential[0], expected_potential, rtol=2e-8, atol=2e-10)
