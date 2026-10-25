"""Public FCI solver/action budgets share the existing CPU float64 tiles."""
import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import fci
from gradscf.fci import hamiltonian as ham
from gradscf.solvers import EigenSolverConfig


def _problem(norb=4, nelec=(2, 1)):
    rng = np.random.default_rng(2701)
    h = np.diag(np.linspace(-1.4, .9, norb)) + rng.normal(scale=.12, size=(norb, norb))
    h = (h+h.T)*.5
    factors = rng.normal(scale=.08, size=(norb+1, norb, norb))
    factors = (factors+factors.transpose(0, 2, 1))*.5
    g = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    return fci.make_fci_space(norb, nelec), jnp.asarray(h), jnp.asarray(g)


def _full_matrix(h, g, space):
    """Tiny independent former full-excitation formula, only for test oracles."""
    pairs = space.norb**2
    identity = jnp.eye(space.size, dtype=jnp.result_type(h, g))
    vectors = identity.reshape(space.shape+(space.size,))
    excitation = ham.excitation_actions(vectors, space)
    effective = ham.absorb_h1e(h, g, space, fac=.5)
    weighted = (effective.reshape(pairs, pairs)
                @ excitation.reshape(pairs, space.size**2)).reshape(excitation.shape)
    return ham._excitation_transpose(weighted, space).reshape(space.size, space.size)


@pytest.mark.parametrize("method", ("dense", "davidson"))
def test_integral_and_explicit_reference_facades_accept_small_action_tiles(method):
    space, h, g = _problem()
    cap = space.norb**2*3
    assert cap < space.norb**2*space.size and cap < space.norb**4
    expected = np.linalg.eigvalsh(np.asarray(_full_matrix(h, g, space)))[0]+.4
    options = dict(solver=method, max_space=space.size, conv_tol=1e-11,
                   max_workspace_elements=cap, max_memory=4.)
    integral = fci.FCISolver(**options).run(h, g, space.norb, space.nelec, ecore=.4)
    explicit = fci.FCI(fci.FCIReference(h, g, space.nelec, .4), **options).run()
    for result in (integral, explicit):
        assert result.converged
        np.testing.assert_allclose(result.e_tot, expected, atol=3e-10, rtol=0)
        assert np.max(result.result.residual_norms) < 3e-9
    np.testing.assert_allclose(abs(jnp.vdot(integral.ci, explicit.ci)), 1., atol=3e-9, rtol=0)


@pytest.fixture(scope="module")
def native_rhf():
    from gradscf import gto, scf

    source = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True),
                     integral_backend="native", execution_device="cpu",
                     jk_backend="direct", direct_scf_tol=0., max_memory=128.,
                     conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    assert source.converged
    return source


@pytest.mark.parametrize("method", ("dense", "davidson"))
@pytest.mark.parametrize("budget", (None, 3.5), ids=("inherited", "override"))
def test_molecular_facade_small_tiles_share_source_or_override_budget(
        native_rhf, monkeypatch, method, budget):
    api = importlib.import_module("gradscf.fci.api")
    solver = importlib.import_module("gradscf.fci.solver")
    source = native_rhf
    expected = fci.FCI(source, solver="dense").run().e_tot
    active_reference, build_hamiltonian = api.active_reference, solver.build_hamiltonian
    observed = []

    def prepare(*args, **kwargs):
        observed.append(("active", kwargs.get("max_memory")))
        return active_reference(*args, **kwargs)

    def build(*args, **kwargs):
        observed.append(("operator", kwargs.get("max_memory")))
        return build_hamiltonian(*args, **kwargs)

    monkeypatch.setattr(api, "active_reference", prepare)
    monkeypatch.setattr(solver, "build_hamiltonian", build)
    result = fci.FCI(source, solver=method, max_space=4, max_workspace_elements=8,
                     max_memory=budget).run()
    assert result.converged
    resolved = source.max_memory if budget is None else budget
    assert result.max_memory == resolved
    assert observed == [("active", resolved), ("operator", resolved)]
    np.testing.assert_allclose(result.e_tot, expected, atol=3e-11, rtol=0)
    result.max_memory = 2.5
    with pytest.raises(RuntimeError, match="changed"):
        result.make_rdm1()


@pytest.mark.parametrize("budget", (.000001, 0., -1., np.nan, True))
def test_low_level_budget_rejected_before_operator_builder(monkeypatch, budget):
    solver = importlib.import_module("gradscf.fci.solver")
    space, h, g = _problem()

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid or insufficient FCI solver budget reached the operator builder")

    monkeypatch.setattr(solver, "build_hamiltonian", forbidden)
    with pytest.raises(ValueError, match="(?i)(memory|budget|resident|tile)"):
        fci.solve_fci(h, g, space, max_memory=budget)


@pytest.mark.parametrize("kind", ("explicit", "scf"))
def test_molecular_budget_rejected_before_active_reference(native_rhf, monkeypatch, kind):
    api = importlib.import_module("gradscf.fci.api")
    space, h, g = _problem()
    source = native_rhf if kind == "scf" else fci.FCIReference(h, g, space.nelec)

    def forbidden(*args, **kwargs):
        pytest.fail("An insufficient FCI action budget reached active-reference preparation")

    monkeypatch.setattr(api, "active_reference", forbidden)
    with pytest.raises(ValueError, match="(?i)(memory|budget|resident|tile)"):
        fci.FCI(source, max_memory=.000001).run()


@pytest.mark.parametrize("operation", ("contract_1e", "contract_2e", "energy"))
def test_integral_facade_forwards_budget_to_public_actions(monkeypatch, operation):
    space, h, g = _problem()
    coefficients = jnp.arange(space.size, dtype=h.dtype).reshape(space.shape)/17+.1
    engine = fci.FCISolver(max_memory=3.25, max_workspace_elements=space.norb**2*3)
    original = getattr(ham, operation)
    observed = []

    def action(*args, **kwargs):
        observed.append((kwargs.get("max_memory"), kwargs.get("max_workspace_elements")))
        return original(*args, **kwargs)

    arguments = ((h, coefficients) if operation == "contract_1e"
                 else (g, coefficients) if operation == "contract_2e"
                 else (h, g, coefficients))
    expected = original(*arguments, space)
    monkeypatch.setattr(ham, operation, action)
    actual = getattr(engine, operation)(*arguments, space.norb, space.nelec)
    assert observed == [(3.25, space.norb**2*3)]
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=2e-12)


def test_integral_facade_budget_default_and_stale_snapshot():
    space, h, g = _problem()
    assert fci.FCISolver().max_memory == 60000.
    result = fci.FCISolver(solver="dense", max_memory=4.).run(h, g, space.norb, space.nelec)
    assert result.converged
    result.max_memory = 3.
    with pytest.raises(RuntimeError, match="changed"):
        result.make_rdm1()


@pytest.mark.parametrize("method", ("dense", "davidson"))
def test_low_level_small_tile_energy_gradient_and_hvp_match_dense_oracle(method):
    space, h, g = _problem(3, (1, 1))
    direction = jnp.array([[.1, -.03, .02], [-.03, -.2, .04], [.02, .04, .07]])
    config = EigenSolverConfig(method=method, max_subspace=space.size,
                                maxiter=150, atol=1e-11)

    def selected(t):
        result = fci.solve_fci(h+t*direction, g*(1+.2*t), space, ecore=.31*t,
                               config=config, max_workspace_elements=18, max_memory=4.)
        return result.total_energies[0]

    def oracle(t):
        return jnp.linalg.eigvalsh(_full_matrix(h+t*direction, g*(1+.2*t), space))[0]+.31*t

    point = jnp.asarray(.13)
    actual = jax.jit(jax.value_and_grad(selected))(point)
    expected = jax.value_and_grad(oracle)(point)
    actual_hvp = jax.jit(lambda t: jax.jvp(jax.grad(selected), (t,), (jnp.ones_like(t),))[1])(point)
    expected_hvp = jax.jvp(jax.grad(oracle), (point,), (jnp.ones_like(point),))[1]
    for observed, reference in zip(jax.tree.leaves((actual, actual_hvp)),
                                    jax.tree.leaves((expected, expected_hvp))):
        assert np.isfinite(np.asarray(observed)).all()
        assert np.isfinite(np.asarray(reference)).all()
        if os.environ.get('GRADSCF_TEST_GPU') == '1':
            assert observed.devices() == {jax.devices('gpu')[0]}
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-8, equal_nan=False)
    np.testing.assert_allclose(actual_hvp, expected_hvp, atol=3e-8, rtol=3e-7, equal_nan=False)
    step = 1e-4
    finite_difference = (selected(point+step)-selected(point-step))/(2*step)
    assert np.isfinite(np.asarray(finite_difference)).all()
    np.testing.assert_allclose(actual[1], finite_difference, atol=3e-8, rtol=3e-7,
                               equal_nan=False)


@pytest.mark.parametrize("method", ["make_rdm1", "make_rdm12", "trans_rdm12", "spin_square"])
def test_public_properties_obey_the_same_memory_budget(method):
    solver = fci.FCISolver(max_memory=.0001)
    c = jnp.ones((6, 6), dtype=jnp.float64)
    with pytest.raises(ValueError, match="(?i)(memory|budget|resident)"):
        if method.startswith("trans_"):
            getattr(solver, method)(c, c, 4, (2, 2))
        else:
            getattr(solver, method)(ci=c, norb=4, nelec=(2, 2))
