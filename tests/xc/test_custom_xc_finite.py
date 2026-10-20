"""Invalid external contributions cannot become a different finite functional."""
import jax.numpy as jnp
import pytest

from gradscf.dft.derivatives import xc_energy_and_potential_from_density


@pytest.mark.parametrize('field', ['exchange', 'extra_fock'])
def test_invalid_external_contributions_are_reported(field):
    kwargs = dict(exact_exchange_fraction=jnp.nan) if field == 'exchange' else dict(
        extra_fock_matrix=jnp.full((2, 2), jnp.nan))
    with pytest.raises(FloatingPointError, match='finite'):
        xc_energy_and_potential_from_density({}, molecule=None, density=jnp.eye(2),
            xc_energy_fn=lambda params, molecule, density: jnp.sum(density**2), **kwargs)


def test_explicit_custom_exchange_coefficient_is_not_silently_clipped():
    result = xc_energy_and_potential_from_density({}, molecule=None, density=jnp.eye(2),
        xc_energy_fn=lambda params, molecule, density: jnp.sum(density**2),
        exact_exchange_fraction=1.1)
    assert float(result.exact_exchange_fraction) == pytest.approx(1.1)
