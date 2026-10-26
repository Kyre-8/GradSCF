"""Real CI RDMs: independent FCI density oracle, frozen electrons and AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import json


@pytest.mark.parametrize("unrestricted,frozen", [(False, None), (False, [0, 3]),
                                                (True, None), (True, ([0], [3]))])
def test_densities_match_fci_for_arbitrary_ci_vector(unrestricted, frozen):
    from gradscf import ci
    from pyscf import fci
    n, occ = 4, (2, 1) if unrestricted else (2, 2)
    space = (ci.make_uci_space(n, occ, frozen=frozen) if unrestricted else
             ci.make_ci_space(n, occ[0], frozen=frozen))
    vector = np.random.default_rng(71).normal(size=space.size)
    vector /= np.linalg.norm(vector)
    strings = [list(map(int, fci.cistring.make_strings(range(n), no))) for no in occ]
    full = np.zeros(tuple(map(len, strings)))
    for c, d in zip(vector, space.determinants):
        full[strings[0].index(d & ((1 << n)-1)), strings[1].index(d >> n)] = c
    expected = (fci.direct_spin1.make_rdm12s(full, n, occ) if unrestricted else
                fci.direct_spin1.make_rdm12(full, n, occ))
    actual = jax.jit(lambda c: ci.make_rdm12(c, space))(jnp.asarray(vector))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=0)
    dm1, dm2 = actual
    if unrestricted:
        np.testing.assert_allclose([np.trace(a) for a in dm1], occ, atol=1e-12)
        dm1 = dm1[0]+dm1[1]
        dm2 = dm2[0]+dm2[1]+dm2[1].transpose(2, 3, 0, 1)+dm2[2]
    np.testing.assert_allclose(np.einsum('pqrr->pq', dm2), (sum(occ)-1)*dm1,
                               atol=2e-12, rtol=0)
    direction = jnp.asarray(np.random.default_rng(73).normal(size=space.size))
    def value(x):
        densities = ci.make_rdm12(jnp.asarray(vector)+x*direction, space)
        return sum(jnp.sum(a*a) for a in jax.tree.leaves(densities))
    np.testing.assert_allclose(jax.jit(jax.grad(value))(0.),
        (value(1e-5)-value(-1e-5))/2e-5, atol=2e-6, rtol=0)
    derivative = jax.jit(jax.grad(value))
    second = jax.jvp(derivative,(0.,),(1.,))[1]
    steps = (3e-5, 1e-5)
    finite = [(derivative(step)-derivative(-step))/(2*step) for step in steps]
    print(json.dumps(dict(coefficient_hessian=float(second),steps=steps,
        absolute_errors=[float(jnp.abs(second-x)) for x in finite])),flush=True)
    for estimate in finite:
        np.testing.assert_allclose(second,estimate,atol=2e-5,rtol=2e-6)


def test_density_facade_and_invalid_requests(radical):
    from gradscf import ci
    mf, h, g = radical
    obj = ci.UCISD(ci.UnrestrictedReference(h, g, (2, 1)), solver="dense")
    with pytest.raises(RuntimeError, match="CI"):
        obj.make_rdm1()
    obj.run()
    # The default dependence cutoff drops the last small residual too early
    # for this density comparison, even after tightening the energy tolerance.
    reference = mf.CISD().run(conv_tol=1e-16, lindep=1e-20, max_cycle=200)
    for actual, expected in zip(obj.make_rdm1(), reference.make_rdm1()):
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)
    for actual, expected in zip(obj.make_rdm2(), reference.make_rdm2()):
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)
    with pytest.raises(ValueError, match="root"):
        obj.make_rdm1(root=1)
    with pytest.raises(ValueError, match="shape"):
        ci.make_rdm1(jnp.zeros(2), obj.space)
    with pytest.raises(NotImplementedError, match="real"):
        ci.make_rdm1(jnp.ones(obj.space.size, dtype=complex), obj.space)
    assert np.isnan(ci.make_rdm1(jnp.zeros(obj.space.size), obj.space)[0]).all()


@pytest.mark.parametrize("change", ["frozen", "rank", "h1", "eri", "source"])
def test_density_rejects_stale_ci_state(change):
    from gradscf import ci
    h, g = np.diag([-1., .2, .6]), np.zeros((3,)*4)
    obj = ci.CISD(ci.CIReference(h, g, 1), solver="dense").run()
    if change == "frozen":
        obj.frozen = [0]
    elif change == "rank":
        obj.max_excitation = 1
    elif change == "h1":
        h[0, 0] += .01
    elif change == "eri":
        g[0, 0, 0, 0] += .01
    else:
        obj.mf = ci.CIReference(h+.01*np.eye(3), g, 1)
    with pytest.raises(RuntimeError, match="changed.*kernel"):
        obj.make_rdm1()
