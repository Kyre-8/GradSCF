"""Streaming BSE kernels and matrix-free auxiliary screening."""

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_full import model, independent_blocks

from gradscf import bse
from gradscf.gw.screened import build_static_screening, apply_static_screening, solve_static_screening
from gradscf.solvers import LinearSolverConfig


GMRES = LinearSolverConfig(method="gmres", rtol=1e-12, atol=1e-14, restart=4)


@pytest.mark.parametrize("method", ["direct", "gmres"])
@pytest.mark.parametrize("singlet", [True, False])
def test_screened_actions_diagonal_and_transpose(method, singlet):
    qp, e, l, _ = model()
    cfg = replace(GMRES, method=method)
    space = bse.make_bse_space(5, 2)
    state = build_static_screening(e, l, occupied=(0, 1), virtual=(2, 3, 4), config=cfg)
    if method == "gmres":
        assert state.dielectric is None
    a, b = bse.build_bse_operators(qp, l, space, state, singlet=singlet, block_size=1)
    rng = np.random.default_rng(23)
    x = jnp.asarray(rng.normal(size=(6, 2)))
    for op, expected in zip((a, b), independent_blocks(qp, e, l, singlet)):
        np.testing.assert_allclose(jax.jit(op.apply)(x), expected @ x, atol=2e-12)
        np.testing.assert_allclose(op.diagonal, np.diag(expected), atol=2e-12)
        transposed = jax.linear_transpose(op.matvec, jnp.zeros(6))(x[:, 0])[0]
        np.testing.assert_allclose(transposed, expected.T @ x[:, 0], atol=2e-12)
        block_transpose = jax.linear_transpose(op.apply, jnp.zeros_like(x))(x)[0]
        np.testing.assert_allclose(block_transpose, expected.T @ x, atol=2e-12)


@pytest.mark.parametrize("tda", [True, False])
@pytest.mark.parametrize("solver", ["dense", "davidson"])
def test_iterative_screening_optical_response(tda, solver):
    qp, e, l, d = model()
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(nroots=2, tda=tda, solver=solver, block_size=1,
                        screening_config=GMRES)

    def loss(t, config):
        out = bse.run_bse(qp * (1 + .03*t), e * (1 - .05*t),
                          l * (1 + .2*t), space, config=config)
        strengths = bse.oscillator_strengths(out, d * (1 + .1*t), space)
        return jnp.sum(out.excitation_energies * jnp.array([.3, .8]) + strengths)

    fun = lambda t: loss(t, cfg)
    direct = lambda t: loss(t, replace(cfg, screening_config=None))
    value, grad = jax.jit(jax.value_and_grad(fun))(0.)
    expected, reference_grad = jax.jit(jax.value_and_grad(direct))(0.)
    np.testing.assert_allclose(value, expected, atol=2e-11)
    np.testing.assert_allclose(grad, reference_grad, atol=2e-9)
    np.testing.assert_allclose(jax.jvp(fun, (0.,), (1.,))[1], grad, atol=2e-9)
    np.testing.assert_allclose(grad, (fun(1e-4)-fun(-1e-4))/2e-4, atol=2e-8)


def test_screening_failure_and_empty_space():
    _, e, l, _ = model()
    bad = build_static_screening(e, l, occupied=(0, 1), virtual=(2, 3, 4),
                                config=replace(GMRES, restart=1, maxiter=1))
    rhs = jnp.arange(l.shape[0], dtype=float) + 1
    fun = lambda t: jnp.sum(apply_static_screening(bad, rhs*t))
    assert np.isnan(fun(1.))
    assert np.isnan(jax.grad(fun)(1.))
    empty = build_static_screening(e, l, occupied=(), virtual=(), config=GMRES)
    np.testing.assert_allclose(apply_static_screening(empty, rhs), rhs, atol=1e-13)
    zero = build_static_screening(e, l[:0], occupied=(0, 1), virtual=(2, 3, 4), config=GMRES)
    assert apply_static_screening(zero, jnp.zeros((0, 3))).shape == (0, 3)


def test_failed_screening_invalidates_bse():
    qp, e, l, d = model()
    ref = bse.BSEReference(qp, e, l, 2, d)
    calc = bse.BSE(ref, nroots=1, solver="dense",
                   screening_config=replace(GMRES, restart=1, maxiter=1)).run()
    assert not np.any(calc.result.converged | calc.result.response_valid)
    with pytest.raises(RuntimeError, match="stable"):
        calc.oscillator_strength()


@pytest.mark.parametrize("nvec", [1, 4])
def test_auxiliary_and_screened_pair_storage_forward_and_reverse(nvec):
    naux, no, nv, block = 17, 3, 7, 2
    rng = np.random.default_rng(73)
    l = jnp.asarray(rng.normal(size=(naux, no + nv, no + nv))) * .02
    l = (l + l.swapaxes(1, 2)) * .5
    e = jnp.concatenate((-jnp.arange(no, 0, -1.), jnp.arange(1., nv + 1)))
    space = bse.make_bse_space(no + nv, no)
    probe = jnp.asarray(rng.normal(size=(space.size, nvec)))

    def objective(factors):
        state = build_static_screening(e, factors, occupied=space.occupied,
                                       virtual=space.virtual, config=GMRES)
        a, b = bse.build_bse_operators(e, factors, space, state, block_size=block)
        return jnp.sum((a.apply(probe) + b.apply(probe))**2)

    def inspect(value):
        if hasattr(value, "jaxpr"):
            inspect(value.jaxpr)
        elif hasattr(value, "eqns"):
            for eqn in value.eqns:
                for v in eqn.outvars:
                    shape = getattr(getattr(v, "aval", None), "shape", ())
                    assert shape not in {(naux, naux), (naux, nv, nv)}
                    if eqn.primitive.name == "custom_linear_solve" and len(shape) == 2:
                        assert shape[0] == naux and shape[1] <= min(no * nvec, nv) * block
                inspect(eqn.params)
        elif isinstance(value, dict):
            for child in value.values():
                inspect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                inspect(child)

    inspect(jax.make_jaxpr(objective)(l))
    inspect(jax.make_jaxpr(jax.grad(objective))(l))


def test_large_auxiliary_screening_checks_physical_column_residuals():
    # A left-diagonal-preconditioned stopping test prematurely accepted one
    # of these B-kernel RHS columns at 1.052 times the physical tolerance.
    naux, no, nv = 384, 6, 64
    rng = np.random.default_rng(701)
    l = rng.normal(size=(naux, no + nv, no + nv)) * .03
    l = jnp.asarray((l + l.transpose(0, 2, 1)) * .5)
    e = jnp.concatenate((-jnp.linspace(2., .2, no), jnp.linspace(.2, 3., nv)))
    x = jnp.asarray(rng.normal(size=no*nv)).reshape(no, nv)
    rhs = jnp.einsum("Pib,jb->Pij", l[:, :no, no:], x).reshape(naux, -1)
    cfg = LinearSolverConfig(method="gmres", rtol=1e-11, atol=1e-13, restart=12)
    state = build_static_screening(e, l, occupied=tuple(range(no)),
                                   virtual=tuple(range(no, no+nv)), config=cfg)
    out = jax.jit(solve_static_screening)(state, rhs)
    assert out.converged and np.isfinite(out.solution).all()
    residuals = jnp.linalg.norm(state.operator().apply(out.solution) - rhs, axis=0)
    limits = cfg.atol + cfg.rtol*jnp.linalg.norm(rhs, axis=0)
    assert np.all(residuals <= limits)


def test_cached_screening_rejects_an_inaccurate_factor():
    _, e, l, _ = model()
    state = build_static_screening(e, l, occupied=(0, 1), virtual=(2, 3, 4))
    state = replace(state, cholesky=state.cholesky*1.01)
    rhs = jnp.arange(l.shape[0], dtype=float) + 1
    out = solve_static_screening(state, rhs)
    assert not out.converged and np.isnan(out.solution).all()
    loss = lambda t: jnp.sum(solve_static_screening(state, rhs*t).solution)
    assert np.isnan(jax.grad(loss)(1.))
    transpose = jax.linear_transpose(lambda b: solve_static_screening(state, b).solution, rhs)
    assert np.isnan(transpose(rhs)[0]).all()
