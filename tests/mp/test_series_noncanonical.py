"""MP coefficients and their response in a smooth, noncanonical HF frame."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _model(parameter):
    rng = np.random.default_rng(810)
    factors = rng.normal(size=(4, 4, 4)) * .06
    factors = factors + factors.transpose(0, 2, 1)
    eri = jnp.asarray(np.einsum("Lpq,Lrs->pqrs", factors, factors))
    fock = jnp.diag(jnp.array([-1., -1., .5, .5]))
    direction = jnp.array([[.03, .12, 0., 0.], [.12, -.02, 0., 0.],
                           [0., 0., -.04, -.09], [0., 0., -.09, .01]])
    fock = fock + parameter * direction
    mean_field = 2*jnp.einsum("pqii->pq", eri[:, :, :2, :2])
    mean_field -= jnp.einsum("piiq->pq", eri[:, :2, :2, :])
    return fock - mean_field, eri, fock


def _reference(parameter, order):
    """Independent canonical-frame PySCF FCI actions and explicit RS recursion."""
    from test_series import reference_series
    h, g, f = map(np.asarray, _model(parameter))
    eo, co = np.linalg.eigh(f[:2, :2])
    ev, cv = np.linalg.eigh(f[2:, 2:])
    c = np.zeros((4, 4))
    c[:2, :2], c[2:, 2:] = co, cv
    rotated = np.einsum("pqrs,pi,qj,rk,sl->ijkl", g, c, c, c, c, optimize=True)
    coefficients, _, _ = reference_series(c.T @ h @ c, rotated,
                                          np.r_[eo, ev], 2, order)
    return coefficients[2:]


@pytest.mark.parametrize("algorithm", ["series", "tensor"])
@pytest.mark.parametrize("order", [4, 5, 6])
def test_high_orders_match_independent_reference_in_rotated_frame(algorithm, order):
    from gradscf import mp
    h, g, _ = _model(.3)
    cfg = mp.MPConfig(order=order, algorithm=algorithm, orbital_basis="noncanonical",
                      with_t2=False)
    result = jax.jit(lambda h, g: mp.run_mp(h, g, nocc=2, config=cfg))(h, g)
    assert result.valid
    np.testing.assert_allclose(result.corrections, _reference(.3, order), atol=2e-11, rtol=1e-8)


@pytest.mark.parametrize("algorithm", ["series", "tensor"])
@pytest.mark.parametrize("order", [4, 6])
def test_high_order_gradient_and_hvp_at_exact_internal_degeneracy(algorithm, order):
    from gradscf import mp
    cfg = mp.MPConfig(order=order, algorithm=algorithm, orbital_basis="noncanonical",
                      with_t2=False)

    def coefficients(x):
        h, g, _ = _model(x)
        return mp.run_mp(h, g, nocc=2, config=cfg).corrections

    energy = lambda x: jnp.sum(coefficients(x))
    values, derivatives = jax.jit(lambda x: jax.jvp(coefficients, (x,), (1.,)))(0.)
    value, derivative = jnp.sum(values), jnp.sum(derivatives)
    curvature = jax.jit(lambda x: jax.jvp(jax.grad(energy), (x,), (1.,))[1])(0.)
    assert np.isfinite([value, derivative, curvature]).all()
    np.testing.assert_allclose(value, _reference(0., order).sum(), atol=2e-11)
    for step in (1e-3, 3e-4):
        coefficient_samples = [_reference(x, order) for x in (step, 0., -step)]
        np.testing.assert_allclose(derivatives,
            (coefficient_samples[0]-coefficient_samples[2])/(2*step), atol=2e-12, rtol=2e-5)
        plus, zero, minus = [sample.sum() for sample in coefficient_samples]
        np.testing.assert_allclose(derivative, (plus-minus)/(2*step), atol=2e-9, rtol=2e-6)
        np.testing.assert_allclose(curvature, (plus-2*zero+minus)/step**2,
                                   atol=2e-8, rtol=2e-4)


def test_noncanonical_series_still_rejects_zero_excitation_denominator():
    from gradscf import mp
    cfg = mp.MPConfig(order=4, algorithm="series", orbital_basis="noncanonical",
                      with_t2=False)
    def energy(x):
        return mp.run_mp(jnp.zeros((2, 2)), x*jnp.zeros((2,)*4), nocc=1,
                         config=cfg).correlation_energy
    assert np.isnan(energy(1.))
    assert np.isnan(jax.grad(energy)(1.))


@pytest.mark.parametrize("algorithm", ["series", "tensor"])
def test_true_zero_spectrum_is_rejected_when_coordinate_gaps_are_nonzero(algorithm):
    from gradscf import mp

    occupied = jnp.array([[-.75, .25], [.25, -.75]])  # eigenvalues -1, -.5
    virtual = jnp.array([[0., .5], [.5, 0.]])        # eigenvalues -.5, .5
    h = jnp.zeros((4, 4)).at[:2, :2].set(occupied).at[2:, 2:].set(virtual)
    g = jnp.zeros((4,) * 4)
    diagonal_gaps = jnp.diag(h)[None, 2:] - jnp.diag(h)[:2, None]
    np.testing.assert_allclose(diagonal_gaps, .75, atol=0., rtol=0.)
    # Explicit algorithms use the generic series engine even at order two.
    config = mp.MPConfig(order=2, algorithm=algorithm, orbital_basis="noncanonical",
                          with_t2=False)
    result = mp.run_mp(h, g, nocc=2, config=config)
    assert not result.valid
    assert result.min_abs_denominator < config.denominator_tol
    assert np.isnan(result.correlation_energy)

    def energy(shift):
        # A uniform one-electron shift preserves the hidden spectral pole.
        return mp.run_mp(h + shift * jnp.eye(4), g, nocc=2,
                         config=config).correlation_energy

    value, derivative = jax.jit(jax.value_and_grad(energy))(0.)
    tangent = jax.jit(lambda x: jax.jvp(energy, (x,), (1.,))[1])(0.)
    assert np.isnan(value) and np.isnan(derivative) and np.isnan(tangent)
