"""Dense-capacity fallback preserves every requested Taylor coefficient."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.mp.integrals import _reference_data
from test_tensor_action import model


def test_packed_capacity_keeps_the_complete_required_space():
    from gradscf.mp.series import make_series_space
    config = mp.MPConfig(order=4, algorithm="tensor", max_tensor_elements=80)
    dense = make_series_space(4, 2, config=replace(config, max_tensor_elements=10000))
    packed = make_series_space(4, 2, config=config)
    assert packed.size == dense.size
    assert packed.ranks == dense.ranks
    assert any(item is None for maps in packed.maps for item in maps)
    with pytest.raises(ValueError, match="tensor"):
        make_series_space(4, 2, config=replace(config, max_tensor_elements=1))


@pytest.mark.parametrize("unrestricted", [False, True])
def test_bounded_packed_series_preserves_history_and_response(unrestricted):
    h, g, nocc = model(unrestricted)
    hs, _, focks, _ = _reference_data(h, g, nocc)
    hs = tuple(a - (f - jnp.diag(jnp.diag(f))) for a, f in zip(hs, focks))
    h = hs if unrestricted else hs[0]
    config = mp.MPConfig(order=4, algorithm="tensor", with_coefficients=True)
    bounded = replace(config, max_tensor_elements=80)

    def values(parameter, config):
        # Keep a canonical Fock while varying the physical interaction. This
        # exercises the full raw h/g response chart and every retained wave.
        scaled = jax.tree.map(lambda x: parameter * x, g)
        _, _, fs, _ = _reference_data(h, scaled, nocc)
        adjusted = tuple(a + jnp.diag(jnp.diag(f0)) - f
                         for a, f0, f in zip(hs, focks, fs))
        return mp.run_mp(adjusted if unrestricted else adjusted[0], scaled,
                          nocc=nocc, config=config)

    def evaluate(config):
        def scalar(x):
            result = values(x, config)
            return result.correlation_energy, result
        def bundle(x):
            ((_, result), gradient), (_, hvp) = jax.jvp(
                jax.value_and_grad(scalar, has_aux=True), (x,), (.2,))
            return result, gradient, hvp
        return jax.jit(bundle)(1.)

    expected, actual = evaluate(config), evaluate(bounded)
    assert bool(actual[0].valid)
    assert actual[0].wavefunction_coefficients is not None
    assert actual[0].t2 is not None
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-10)
    energy_only = values(1., replace(bounded, with_t2=False, with_coefficients=False))
    assert energy_only.t2 is None and energy_only.wavefunction_coefficients is None
    np.testing.assert_allclose(energy_only.corrections, actual[0].corrections,
                               atol=3e-12, rtol=3e-10)
