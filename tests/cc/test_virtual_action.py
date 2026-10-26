"""Restricted virtual actions retain the original residual and raw-integral AD.

The explicit W and CC2 term oracles below preserve the pre-WF10a formulas,
adapted from PySCF 2.9.0 under Apache-2.0. The large-virtual
production path contracts the dressing before it becomes a four-virtual tensor.
"""
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc import _equations, _intermediates as imd, rccsd
from gradscf.cc.integrals import prepare_integrals


einsum = partial(jnp.einsum, precision="highest")
MODELS = ("ccs", "ccd", "lccd", "lccsd", "cc2", "ccsd", "qcisd")


def _explicit_contract(t1, eris, x):
    ovvv = jnp.asarray(eris.get_ovvv())
    dressed = einsum("kdac,kb->abcd", ovvv, -t1)
    dressed -= einsum("kcbd,ka->abcd", ovvv, t1)
    dressed += jnp.asarray(eris.vvvv).transpose(0, 2, 1, 3)
    return einsum("abcd,...cd->...ab", dressed, x)


def _explicit_cc2(t1, eris):
    dressed = einsum("kcbd,ka->abcd", eris.get_ovvv(), -t1)
    dressed = dressed+dressed.transpose(1, 0, 3, 2)
    dressed += eris.vvvv.transpose(0, 2, 1, 3)
    return einsum("abcd,ic,jd->ijab", dressed, t1, t1)


def _inputs(nocc=2, nvir=3, dtype=jnp.float64):
    rng = np.random.default_rng(2310)
    n = nocc+nvir
    # Raw h/g deliberately lack permutation symmetry: preserve their AD chart.
    h = jnp.asarray(rng.normal(size=(n, n))*.1, dtype=dtype)
    g = jnp.asarray(rng.normal(size=(n,)*4)*.05, dtype=dtype)
    t1 = jnp.asarray(rng.normal(size=(nocc, nvir))*.03, dtype=dtype)
    t2 = jnp.asarray(rng.normal(size=(nocc, nocc, nvir, nvir))*.04, dtype=dtype)
    return h, g, t1, t2


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("leading", [(), (8,), (2, 4)])
@pytest.mark.parametrize("nocc,nvir", [(2, 3), (4, 2)])
def test_virtual_action_matches_explicit_dressed_tensor(dtype, leading, nocc, nvir):
    h, g, t1, _ = _inputs(nocc, nvir, dtype=dtype)
    ints = prepare_integrals(h, g, nocc=nocc)
    rng = np.random.default_rng(2311)
    x = jnp.asarray(rng.normal(size=leading+(nvir, nvir)), dtype=dtype)
    actual = jax.jit(lambda a, b: imd.contract_Wvvvv(a, ints, b))(t1, x)
    expected = _explicit_contract(t1, ints, x)
    tolerance = 3e-6 if dtype == jnp.float32 else 2e-12
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("nocc,nvir", [(0, 3), (3, 0)])
def test_virtual_action_empty_spaces(nocc, nvir):
    h, g, t1, t2 = _inputs(nocc, nvir)
    ints = prepare_integrals(h, g, nocc=nocc)
    actual = jax.jit(lambda a, x: imd.contract_Wvvvv(a, ints, x))(t1, t2)
    expected = _explicit_contract(t1, ints, t2)
    assert actual.shape == expected.shape == t2.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("model", MODELS)
def test_large_virtual_models_do_not_construct_dressed_tensor(monkeypatch, model):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit dressed four-virtual tensor requested")

    def checked(original):
        def call(expression, *args, **kwargs):
            if expression.partition("->")[2] == "abcd":
                forbidden()
            return original(expression, *args, **kwargs)
        return call

    monkeypatch.setattr(imd, "cc_Wvvvv", forbidden, raising=False)
    monkeypatch.setattr(_equations, "einsum", checked(_equations.einsum))
    monkeypatch.setattr(imd, "einsum", checked(imd.einsum))
    h, g, t1, t2 = _inputs()
    ints = prepare_integrals(h, g, nocc=2)
    jax.block_until_ready(jax.jit(lambda a, b: rccsd.residual(a, b, ints, model=model))(t1, t2))


def test_small_virtual_action_uses_cheaper_transient(monkeypatch):
    h, g, t1, t2 = _inputs(4, 2)
    ints = prepare_integrals(h, g, nocc=4)
    original = imd.einsum
    requests = []

    def record(expression, *args, **kwargs):
        if expression.partition("->")[2] == "abcd":
            requests.append(expression)
        return original(expression, *args, **kwargs)

    monkeypatch.setattr(imd, "einsum", record)
    actual = imd.contract_Wvvvv(t1, ints, t2)
    np.testing.assert_allclose(actual, _explicit_contract(t1, ints, t2),
                               atol=2e-12, rtol=2e-12)
    assert requests, "Small virtual space unnecessarily constructed the larger B*o*v scratch"


@pytest.mark.parametrize("nocc,nvir", [(2, 3), (4, 2)])
def test_cc2_virtual_term_matches_original_inline(nocc, nvir):
    h, g, t1, _ = _inputs(nocc, nvir)
    ints = prepare_integrals(h, g, nocc=nocc)
    pair = einsum("ic,jd->ijcd", t1, t1)
    actual = imd.contract_Wvvvv(t1, ints, pair)
    np.testing.assert_allclose(actual, _explicit_cc2(t1, ints), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("model", ["ccsd", "cc2", "qcisd"])
@pytest.mark.parametrize("nocc,nvir", [(2, 3), (4, 2)])
def test_ground_callers_pass_their_model_amplitudes(monkeypatch, model, nocc, nvir):
    h, g, t1, t2 = _inputs(nocc, nvir)
    ints = prepare_integrals(h, g, nocc=nocc)
    original = imd.contract_Wvvvv
    received = []

    def record(singles, eris, x):
        received.append((singles, x))
        return original(singles, eris, x)

    monkeypatch.setattr(imd, "contract_Wvvvv", record)
    rccsd.residual(t1, t2, ints, model=model)
    pair = einsum("ia,jb->ijab", t1, t1)
    expected = {"ccsd": (t1, t2+pair), "cc2": (t1, pair),
                "qcisd": (jnp.zeros_like(t1), t2)}[model]
    assert len(received) == 1
    np.testing.assert_array_equal(received[0][0], expected[0])
    if model == "cc2":
        assert isinstance(received[0][1], tuple), "CC2 expanded its singles pair before the action"
        for factor in received[0][1]:
            np.testing.assert_array_equal(factor, t1)
    else:
        np.testing.assert_array_equal(received[0][1], expected[1])


def test_small_virtual_cc2_does_not_materialize_pair(monkeypatch):
    h, g, t1, t2 = _inputs(4, 2)
    ints = prepare_integrals(h, g, nocc=4)

    def checked(original):
        def call(expression, *args, **kwargs):
            inputs, _, output = expression.partition("->")
            if (len(args) == 2 and all(a.shape == t1.shape for a in args)
                    and len(output) == 4 and len(set(inputs.replace(",", ""))) == 4):
                raise AssertionError("Small-virtual CC2 materialized the T1 outer product")
            return original(expression, *args, **kwargs)
        return call

    monkeypatch.setattr(_equations, "einsum", checked(_equations.einsum))
    monkeypatch.setattr(imd, "einsum", checked(imd.einsum))
    rccsd.residual(t1, t2, ints, model="cc2")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("nocc,nvir", [(2, 3), (4, 2)])
def test_independent_factored_pair_matches_dense_action(dtype, nocc, nvir):
    h, g, t1, _ = _inputs(nocc, nvir, dtype)
    ints = prepare_integrals(h, g, nocc=nocc)
    rng = np.random.default_rng(2315)
    left = jnp.asarray(rng.normal(size=(3, nvir))*.04, dtype)
    right = jnp.asarray(rng.normal(size=(4, nvir))*.03, dtype)
    expected = _explicit_contract(t1, ints, einsum("ic,jd->ijcd", left, right))
    actual = jax.jit(lambda a, l, r: imd.contract_Wvvvv(a, ints, (l, r)))(t1, left, right)
    tolerance = 3e-6 if dtype == jnp.float32 else 2e-12
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("nocc,nvir,nleft,nright", [
    (2, 3, 0, 4), (2, 3, 3, 0), (0, 3, 3, 4), (3, 0, 3, 4),
])
def test_factored_pair_empty_spaces(nocc, nvir, nleft, nright):
    h, g, t1, _ = _inputs(nocc, nvir)
    ints = prepare_integrals(h, g, nocc=nocc)
    left, right = jnp.ones((nleft, nvir)), jnp.ones((nright, nvir))
    actual = jax.jit(lambda l, r: imd.contract_Wvvvv(t1, ints, (l, r)))(left, right)
    expected = _explicit_contract(t1, ints, einsum("ic,jd->ijcd", left, right))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("nocc,nvir", [(2, 3), (4, 2)])
def test_factored_pair_raw_jvp_vjp_hvp(nocc, nvir):
    rng = np.random.default_rng(2316)
    # Neither ovvv nor the separate L_ic/R_jd factors have permutation symmetry.
    variables = tuple(jnp.asarray(rng.normal(size=s))*.04 for s in (
        (nocc, nvir), (nocc, nvir, nvir, nvir), (nvir,)*4, (3, nvir), (4, nvir)))
    direction = tuple(jnp.asarray(rng.normal(size=x.shape))*.02 for x in variables)
    probe = jnp.asarray(rng.normal(size=(3, 4, nvir, nvir)))

    def bundle(v, dense):
        def function(values):
            t1, ovvv, vvvv, left, right = values
            ints = SimpleNamespace(vvvv=vvvv, get_ovvv=lambda: ovvv)
            return (_explicit_contract(t1, ints, einsum("ic,jd->ijcd", left, right))
                    if dense else imd.contract_Wvvvv(t1, ints, (left, right)))
        value, jvp = jax.jvp(function, (v,), (direction,))
        _, pullback = jax.vjp(function, v)
        vjp, = pullback(probe)
        loss = lambda x: jnp.sum(function(x)**2)
        hvp = jax.jvp(jax.grad(loss), (v,), (direction,))[1]
        return value, jvp, vjp, hvp

    actual = jax.jit(lambda v: bundle(v, False))(variables)
    expected = jax.jit(lambda v: bundle(v, True))(variables)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, reference, atol=2e-11, rtol=2e-12)


def test_cc2_water_small_virtual_matches_pyscf():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from pyscf.cc import rccsd as pyscf_rccsd
    from gradscf import cc

    mol = pyscf.gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
                      basis="sto-3g", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    h = c.T@mf.get_hcore()@c
    g = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    reference = pyscf_rccsd.RCCSD(mf).set(cc2=True, conv_tol=1e-12,
        conv_tol_normt=1e-10, max_cycle=150).run()
    actual = cc.run_cc(h, g, nocc=5, nuclear_repulsion=mol.energy_nuc(),
        config=cc.CCConfig(method="cc2", conv_tol=1e-12, residual_tol=1e-10,
                           max_cycle=150))
    assert reference.converged and actual.converged
    np.testing.assert_allclose(actual.total_energy, reference.e_tot, atol=2e-9, rtol=0)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("nocc,nvir", [(2, 3), (4, 2)])
def test_raw_residual_energy_jvp_vjp_hvp_match_dense_oracle(monkeypatch, model, nocc, nvir):
    variables = _inputs(nocc, nvir)
    rng = np.random.default_rng(2312)
    direction = tuple(jnp.asarray(rng.normal(size=x.shape))*.02 for x in variables)
    cotangent = tuple(jnp.asarray(rng.normal(size=s))
                      for s in (variables[2].shape, variables[3].shape, ()))

    def evaluate():
        def data(v):
            h, g, t1, t2 = v
            ints = prepare_integrals(h, g, nocc=nocc)
            r1, r2 = rccsd.residual(t1, t2, ints, model=model)
            energy = ints.reference_energy+rccsd.correlation_energy(t1, t2, ints, model=model)
            return r1, r2, energy

        def bundle(v):
            value, jvp = jax.jvp(data, (v,), (direction,))
            _, pullback = jax.vjp(data, v)
            vjp, = pullback(cotangent)
            loss = lambda x: sum(jnp.sum(a*a) for a in data(x))
            hvp = jax.jvp(jax.grad(loss), (v,), (direction,))[1]
            return value, jvp, vjp, hvp

        # Each call traces a fresh wrapper so a previous JIT cannot conceal
        # the test-only substitution of the shared virtual action.
        return jax.jit(bundle)(variables)

    actual = evaluate()
    reference_calls = []

    def reference(singles, eris, x):
        reference_calls.append(None)
        return (_explicit_cc2(singles, eris) if model == "cc2"
                else _explicit_contract(singles, eris, x))

    with monkeypatch.context() as patch:
        # CC2's original inline term ignores x: this independently protects
        # its singles-pair selection in addition to the caller checks above.
        patch.setattr(imd, "contract_Wvvvv", reference)
        expected = evaluate()
    assert reference_calls, "The dense oracle substitution was hidden by a stale JIT"
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, reference, atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize("nocc,nvir", [(2, 3), (4, 2)])
def test_virtual_action_raw_hvp_matches_finite_difference(nocc, nvir):
    rng = np.random.default_rng(2313)
    variables = tuple(jnp.asarray(rng.normal(size=s))*.03
                      for s in ((nocc, nvir), (nocc, nvir, nvir, nvir),
                                (nvir,)*4, (nocc, nocc, nvir, nvir)))
    direction = tuple(jnp.asarray(rng.normal(size=x.shape))*.1 for x in variables)
    def loss(v):
        t1, ovvv, vvvv, x = v
        ints = SimpleNamespace(vvvv=vvvv, get_ovvv=lambda: ovvv)
        return jnp.sum(imd.contract_Wvvvv(t1, ints, x)**2)
    gradient = jax.jit(jax.grad(loss))
    actual = jax.jit(lambda v: jax.jvp(jax.grad(loss), (v,), (direction,))[1])(variables)
    for step in (3e-4, 1e-4):
        # A fourth-order stencil removes the observed O(step**2) truncation
        # of a two-point difference for this sixth-degree polynomial loss.
        plus = gradient(jax.tree.map(lambda a, d: a+step*d, variables, direction))
        minus = gradient(jax.tree.map(lambda a, d: a-step*d, variables, direction))
        plus2 = gradient(jax.tree.map(lambda a, d: a+2*step*d, variables, direction))
        minus2 = gradient(jax.tree.map(lambda a, d: a-2*step*d, variables, direction))
        for result, p, m, p2, m2 in zip(actual, plus, minus, plus2, minus2):
            np.testing.assert_allclose(result, (-p2+8*p-8*m+m2)/(12*step),
                                       atol=2e-9, rtol=2e-6)
