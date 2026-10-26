"""MO-transformation roundoff must not invalidate physical static screening."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.screened import build_static_screening


@pytest.mark.parametrize('asymmetry, expected', [(1e-15, True), (4.2e-13, False), (1e-7, False)])
def test_standalone_screening_retains_strict_validation(asymmetry, expected):
    # Representative of the 61-MO aug-cc-pVDZ methane transformation. The
    # physical factors are symmetric; a long contraction adds rounding error.
    nmo = 61
    factors = np.ones((2, nmo, nmo))
    factors[0, 3, 8] += asymmetry
    energy = jnp.linspace(-1., 2., nmo)
    state = build_static_screening(
        energy, jnp.asarray(factors), occupied=range(5), virtual=range(5, nmo),
    )
    assert bool(state.valid) is expected


def test_bse_roundoff_projection_forward_and_response():
    import jax
    from gradscf import bse

    rng = np.random.default_rng(91)
    raw = rng.normal(size=(3, 61, 61)) * .08
    factors = jnp.asarray((raw + raw.transpose(0, 2, 1)) / 2)
    factors = factors.at[0, 0, 1].add(4.2e-13)
    symmetric = (factors + factors.swapaxes(1, 2)) / 2
    energy = jnp.concatenate((jnp.array([-.7, -.4]), jnp.linspace(.2, 3., 59)))
    space = bse.make_bse_space(61, 2, virtual=(2, 3, 4))
    cfg = bse.BSEConfig(tda=False, solver='dense', nroots=2)

    def solve(l):
        return bse.run_bse(energy, energy, l, space, config=cfg)

    out, reference = solve(factors), solve(symmetric)
    assert np.all(out.converged & out.stable)
    np.testing.assert_allclose(out.excitation_energies, reference.excitation_energies, atol=2e-12)
    loss = lambda l: solve(l).excitation_energies.sum()
    gradient = jax.jit(jax.grad(loss))(factors)
    # One input boundary defines both screening and kernel factors; reverse
    # response must respect that same symmetric physical representation.
    np.testing.assert_allclose(gradient, gradient.swapaxes(1, 2), atol=1e-12)
    direction = symmetric * .2
    tangent = jax.jvp(loss, (factors,), (direction,))[1]
    step = 1e-4
    finite_difference = (loss(factors + step * direction) - loss(factors - step * direction)) / (2 * step)
    np.testing.assert_allclose(tangent, jnp.sum(gradient * direction), atol=1e-11)
    np.testing.assert_allclose(tangent, finite_difference, atol=2e-8, rtol=2e-6)

    invalid = solve(factors.at[0, 0, 1].add(1e-5))
    assert not bool(invalid.screening_valid)
    assert not np.any(invalid.converged)
