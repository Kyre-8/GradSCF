"""Restricted direct actions retain non-symmetric densities and bounded batches."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
from gradscf.tddft.response import build_restricted_tda_operator, build_restricted_tdhf_operator


class IntegralAction:
    def __init__(self, eri, width=None):
        self.eri, self.width = eri, width

    def get_jk(self, density):
        if self.width is not None:
            assert density.shape[0] <= self.width
        return (jnp.einsum("pqrs,nrs->npq", self.eri, density),
                jnp.einsum("prqs,nrs->npq", self.eri, density))


def reference(eri, *, direct, width=None):
    n = eri.shape[0]
    return SimpleNamespace(
        mo_coeff=jnp.eye(n), mo_occ=jnp.array([2., 2., 0., 0.]),
        mo_energy=jnp.array([-1.3, -.8, .4, .9]), nocc=2,
        ao=jnp.zeros((1, n)), grid=SimpleNamespace(weights=jnp.ones(1)),
        rdm1=jnp.stack((jnp.diag(jnp.array([1., 1., 0., 0.])),) * 2),
        rep_tensor=jnp.zeros((0, 0, 0, 0)) if direct else eri,
        eri_pair_matrix=None, df_factors=None,
        direct_basis=IntegralAction(eri, width) if direct else None)


@pytest.fixture
def eri():
    rng = np.random.default_rng(902)
    factors = rng.normal(size=(5, 4, 4)) * .03
    factors += factors.transpose(0, 2, 1)
    return jnp.asarray(np.einsum("Ppq,Prs->pqrs", factors, factors))


@pytest.mark.parametrize("full_td", [False, True])
def test_direct_response_action_matches_array_oracle_with_input_ad(eri, full_td):
    builder = build_restricted_tdhf_operator if full_td else build_restricted_tda_operator
    x = jnp.asarray(np.random.default_rng(903).normal(size=(3, 8 if full_td else 4)))
    functional = SemilocalResponseFunctional("hf")

    def action(scale, values, direct):
        out = builder(reference(scale * eri, direct=direct), functional,
                      response_kernel_options={"two_electron_mode": "direct"})
        return (out if full_td else out[0])(values)

    expected = action(1., x, False)
    actual = jax.jit(lambda s, x: action(s, x, True))(1., x)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    loss = lambda s, x, direct: jnp.sum(action(s, x, direct)**2)
    for direct in (False, True):
        derivative = jax.jit(jax.grad(lambda s, x: loss(s, x, direct), (0, 1)))(1., x)
        if direct:
            for observed, wanted in zip(derivative, baseline):
                np.testing.assert_allclose(observed, wanted, atol=2e-11, rtol=2e-11)
        else:
            baseline = derivative


def test_direct_action_honors_budget_for_nondivisible_vector_batches(eri):
    from gradscf.tddft.response_options import plan_response_memory

    plan = plan_response_memory(4, 4, 1, np.float64, .02)
    values = jnp.asarray(np.random.default_rng(904).normal(size=(2 * plan.batch_size + 1, 4)))
    functional = SemilocalResponseFunctional("hf")
    action, _, _ = build_restricted_tda_operator(
        reference(eri, direct=True, width=plan.batch_size), functional,
        response_kernel_options={"two_electron_mode": "direct", "max_memory": .02})
    oracle, _, _ = build_restricted_tda_operator(reference(eri, direct=False), functional)
    np.testing.assert_allclose(jax.jit(action)(values), oracle(values), atol=2e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda x: jnp.sum(action(x)**2)))(values),
                               jax.grad(lambda x: jnp.sum(oracle(x)**2))(values), atol=2e-11)


def test_memory_plan_rejects_budget_below_minimum_without_changing_roots():
    from gradscf.tddft.response_options import plan_response_memory

    with pytest.raises(ValueError, match="memory|Memory"):
        plan_response_memory(1000, 10000, 20, np.float64, .01)


def test_large_transition_space_has_no_quartic_or_full_response_workspace():
    from gradscf.tddft.response_options import plan_response_memory

    # DABNA-sized dimensions only: this is an allocation-structure test,
    # not an electronic-structure result for that molecule.
    nao, no = 1060, 194
    nv = nao - no
    dim = no * nv
    plan = plan_response_memory(nao, dim, 20, np.float64, 2000.)
    assert plan.max_subspace < dim and plan.batch_size < 42

    class DensityAction:
        def get_jk(self, density):
            assert density.shape[0] <= plan.batch_size
            return density, .5 * density

    mol = SimpleNamespace(
        mo_coeff=jnp.eye(nao), mo_occ=jnp.concatenate((jnp.full(no, 2.), jnp.zeros(nv))),
        mo_energy=jnp.linspace(-2., 2., nao), nocc=no,
        ao=jnp.zeros((1, nao)), grid=SimpleNamespace(weights=jnp.ones(1)),
        direct_basis=DensityAction(), df_factors=None, eri_pair_matrix=None, rep_tensor=None)
    action, _, _ = build_restricted_tda_operator(mol, SemilocalResponseFunctional("hf"))
    graph = jax.make_jaxpr(action)(jax.ShapeDtypeStruct((42, dim), jnp.float64))

    def shapes(value):
        if hasattr(value, "jaxpr"):
            return shapes(value.jaxpr)
        found = {v.aval.shape for v in (*value.constvars, *value.invars, *value.outvars)
                 if hasattr(v.aval, "shape")}
        for equation in value.eqns:
            found.update(v.aval.shape for v in equation.outvars if hasattr(v.aval, "shape"))
            for parameter in jax.tree.leaves(equation.params):
                if hasattr(parameter, "jaxpr") or hasattr(parameter, "eqns"):
                    found.update(shapes(parameter))
        return found

    allocated = shapes(graph)
    assert (dim, dim) not in allocated and (nao, nao, nao, nao) not in allocated
    assert (42, nao, nao) not in allocated
