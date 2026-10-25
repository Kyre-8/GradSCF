"""Pure occupied/virtual preparation keeps coefficient and Fock response live."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _problem(unrestricted):
    rng = np.random.default_rng(2505)
    factors = rng.normal(scale=.1, size=(4, 5, 5))
    factors += factors.transpose(0, 2, 1)
    eri = jnp.asarray(np.einsum("Qpq,Qrs->pqrs", factors, factors))
    coefficients = tuple(jnp.asarray(np.eye(5) + rng.normal(scale=.05, size=(5, 5)))
                         for _ in range(2 if unrestricted else 1))
    focks = (jnp.diag(jnp.array([-1.5, -.9, .2, .7, 1.1])),)
    if unrestricted:
        focks += (jnp.diag(jnp.array([-1.3, -.6, .4, .8, 1.2])),)
    return coefficients, focks, eri, (2, 1) if unrestricted else 2


def _full_oracle(coefficients, focks, eri, nocc, frozen, *, with_t2=False):
    from gradscf.mp import MPConfig, run_mp

    unrestricted = isinstance(nocc, tuple)
    pairs = ((0, 0), (0, 1), (1, 1)) if unrestricted else ((0, 0),)
    gs = tuple(jnp.einsum("pqrs,pi,qj,rk,sl->ijkl", eri,
                         coefficients[a], coefficients[a], coefficients[b], coefficients[b],
                         precision="highest") for a, b in pairs)
    # Fix the supplied full MO Fock partition while varying transformed ERIs.
    # Reconstruct the one-body test oracle so run_mp obtains those same Focks.
    if unrestricted:
        na, nb = nocc
        aa, ab, bb = gs
        va = (jnp.einsum("pqii->pq", aa[:, :, :na, :na])
              - jnp.einsum("piiq->pq", aa[:, :na, :na, :])
              + jnp.einsum("pqii->pq", ab[:, :, :nb, :nb]))
        vb = (jnp.einsum("pqii->pq", bb[:, :, :nb, :nb])
              - jnp.einsum("piiq->pq", bb[:, :nb, :nb, :])
              + jnp.einsum("iipq->pq", ab[:na, :na]))
        h, g = (focks[0] - va, focks[1] - vb), gs
    else:
        g = gs[0]
        potential = (2*jnp.einsum("pqii->pq", g[:, :, :nocc, :nocc])
                     - jnp.einsum("piiq->pq", g[:, :nocc, :nocc, :]))
        h = focks[0] - potential
    return run_mp(h, g, nocc=nocc, frozen=frozen,
                  config=MPConfig(with_t2=with_t2))


@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
def test_orbital_preparation_jit_coefficient_gradient_and_hvp(unrestricted):
    from gradscf.mp import MPConfig
    from gradscf.mp.integrals import from_orbitals
    from gradscf.mp.mp2 import evaluate_mp2

    cs, focks, eri, nocc = _problem(unrestricted)
    frozen = ([0], [4]) if unrestricted else [0, 4]
    config = MPConfig(with_t2=False)

    def selected(coefficients):
        ints = from_orbitals(coefficients if unrestricted else coefficients[0],
                             focks, nocc, -7., frozen=frozen, eri=eri, max_memory=.1)
        return evaluate_mp2(ints, config).e2

    full = lambda coefficients: _full_oracle(coefficients, focks, eri, nocc, frozen).e2
    direction = tuple(jnp.full_like(c, .03) for c in cs)
    actual = jax.jit(jax.value_and_grad(selected))(cs)
    expected = jax.value_and_grad(full)(cs)
    actual_hvp = jax.jit(lambda coefficients: jax.jvp(
        jax.grad(selected), (coefficients,), (direction,))[1])(cs)
    expected_hvp = jax.jvp(jax.grad(full), (cs,), (direction,))[1]
    for a, e in zip(jax.tree.leaves((actual, actual_hvp)),
                    jax.tree.leaves((expected, expected_hvp))):
        np.testing.assert_allclose(a, e, atol=2e-12, rtol=2e-10)
    assert sum(float(jnp.linalg.norm(g)) for g in actual[1]) > 1e-8


@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
def test_orbital_preparation_preserves_noncanonical_occupied_virtual_rotation(unrestricted):
    from gradscf.mp import MPConfig
    from gradscf.mp.integrals import from_orbitals
    from gradscf.mp.mp2 import evaluate_mp2

    cs, focks, eri, nocc = _problem(unrestricted)
    config = MPConfig(with_t2=False, orbital_basis="noncanonical")

    def value(angle):
        rotated_cs, rotated_fs = [], []
        for spin, (c, f) in enumerate(zip(cs, focks)):
            p, q = (1, 2) if spin else (0, 1)
            rotation = jnp.eye(c.shape[1])
            rotation = rotation.at[p, p].set(jnp.cos(angle)).at[q, q].set(jnp.cos(angle))
            rotation = rotation.at[p, q].set(-jnp.sin(angle)).at[q, p].set(jnp.sin(angle))
            rotated_cs.append(c @ rotation)
            rotated_fs.append(rotation.T @ f @ rotation)
        ints = from_orbitals(tuple(rotated_cs) if unrestricted else rotated_cs[0],
                             tuple(rotated_fs), nocc, -7., eri=eri, max_memory=.1)
        return evaluate_mp2(ints, config)

    reference, rotated = value(0.), value(.29)
    assert reference.valid and rotated.valid
    np.testing.assert_allclose(rotated.e2, reference.e2, atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["all-float32", "integrals-float64"])
def test_orbital_preparation_uses_integral_dtype_for_float32_focks(unrestricted, dtype):
    from gradscf.mp import MPConfig
    from gradscf.mp.integrals import from_orbitals
    from gradscf.mp.mp2 import evaluate_mp2

    cs, focks, eri, nocc = _problem(unrestricted)
    cs, eri = tuple(c.astype(dtype) for c in cs), eri.astype(dtype)
    focks = tuple(f.astype(jnp.float32) for f in focks)
    ints = from_orbitals(cs if unrestricted else cs[0], focks, nocc,
                         jnp.asarray(-7., dtype=jnp.float32), eri=eri, max_memory=.1)
    actual = evaluate_mp2(ints, MPConfig(with_t2=True))
    # Keep the oracle's Fock reconstruction in float64 so float32 cancellation
    # does not become a separate stationarity-tolerance test.
    expected = _full_oracle(tuple(c.astype(jnp.float64) for c in cs),
                            tuple(f.astype(jnp.float64) for f in focks),
                            eri.astype(jnp.float64), nocc, None, with_t2=True)
    assert actual.valid and expected.valid
    assert all(e.dtype == dtype for e in ints.energies)
    assert all(f.dtype == dtype for f in ints.focks)
    actual_t2 = actual.t2 if unrestricted else (actual.t2,)
    expected_t2 = expected.t2 if unrestricted else (expected.t2,)
    tolerance = 2e-7 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(actual.e2, expected.e2, atol=tolerance, rtol=tolerance)
    for a, e in zip(actual_t2, expected_t2):
        assert a.dtype == dtype
        np.testing.assert_allclose(a, e, atol=tolerance, rtol=tolerance)
