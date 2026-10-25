"""True-residual regression for restarted JAX GMRES's early-inner-exit path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers import solve_linear, LinearSolverConfig


@pytest.mark.parametrize('scale',[1.,1e-10])
def test_gmres_early_exit_retains_requested_accuracy(scale):
    diagonal = jnp.linspace(1.,229.,343)
    rhs = scale*jnp.sin(jnp.arange(343))
    cfg = LinearSolverConfig(rtol=1e-10,atol=0.,restart=40,maxiter=20)
    result = solve_linear(jnp.diag(diagonal),rhs,config=cfg)
    assert result.converged, result.residual_norm
    np.testing.assert_allclose(result.solution,rhs/diagonal,rtol=1e-8,atol=scale*1e-11)
    # Transpose/adjoint uses the same numerical retry through custom_linear_solve.
    derivative = jax.grad(lambda shift:jnp.sum(solve_linear(
        jnp.diag(diagonal+shift),rhs,config=cfg).solution))(0.)
    np.testing.assert_allclose(derivative,-jnp.sum(rhs/diagonal**2),rtol=1e-8,atol=scale*1e-10)


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
@pytest.mark.parametrize("tolerance", ["relative", "absolute"])
def test_gmres_true_residual_has_margin_below_acceptance(scale, tolerance):
    # A well-conditioned deterministic SPD system avoids a random or nearly
    # singular fixture. With no internal margin GMRES stops at ~0.97 of the
    # requested threshold, even though its caller's acceptance check passes.
    diagonal = jnp.linspace(1., 10., 64)
    index = jnp.arange(64)
    rhs = scale * (jnp.cos(.7 * index) + jnp.sin(.3 * index))
    rtol = 1e-8 if tolerance == "relative" else 0.
    atol = 0. if tolerance == "relative" else float(1e-8 * jnp.linalg.norm(rhs))
    cfg = LinearSolverConfig(rtol=rtol, atol=atol, restart=20, maxiter=50)

    result = solve_linear(jnp.diag(diagonal), rhs, config=cfg)

    assert result.converged, result.residual_norm
    true_residual = jnp.linalg.norm(diagonal * result.solution - rhs)
    acceptance = cfg.atol + cfg.rtol * jnp.linalg.norm(rhs)
    assert true_residual <= .5 * acceptance, true_residual / acceptance
    np.testing.assert_allclose(result.residual_norm, true_residual, rtol=1e-6)


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_gmres_margin_preserves_implicit_first_and_second_derivatives(scale):
    diagonal = jnp.linspace(1., 10., 64)
    index = jnp.arange(64)
    rhs = scale * (jnp.cos(.7 * index) + jnp.sin(.3 * index))
    cfg = LinearSolverConfig(rtol=1e-8, atol=0., restart=20, maxiter=50)

    def observable(shift):
        return jnp.sum(solve_linear(jnp.diag(diagonal + shift), rhs,
            config=cfg).solution)

    first = jax.grad(observable)(0.)
    second = jax.jvp(jax.grad(observable), (0.,), (1.,))[1]
    np.testing.assert_allclose(first, -jnp.sum(rhs / diagonal**2),
        rtol=5e-8, atol=scale * 1e-9)
    np.testing.assert_allclose(second, 2 * jnp.sum(rhs / diagonal**3),
        rtol=5e-8, atol=scale * 1e-9)


@pytest.mark.parametrize("tolerance", ["relative", "absolute"])
def test_gmres_batched_retry_has_true_residual_margin(monkeypatch, tolerance):
    from importlib import import_module

    gmres_module = import_module("gradscf.solvers.linear.gmres")
    upstream_gmres = gmres_module.gmres

    def failed_incremental(operator, value, **kwargs):
        if kwargs["solve_method"] == "incremental":
            # Force the upstream failure without relying on a JAX-version
            # specific Arnoldi bug. The retry itself is a real numerical solve.
            return jnp.full_like(value, jnp.nan), 0
        return upstream_gmres(operator, value, **kwargs)

    monkeypatch.setattr(gmres_module, "gmres", failed_incremental)
    diagonal = jnp.linspace(1., 100., 64)
    index = jnp.arange(64)
    rhs = jnp.cos(.7 * index) + jnp.sin(.3 * index)
    rtol = 1e-8 if tolerance == "relative" else 0.
    atol = 0. if tolerance == "relative" else float(1e-8 * jnp.linalg.norm(rhs))
    cfg = LinearSolverConfig(rtol=rtol, atol=atol, restart=5, maxiter=100)

    result = solve_linear(jnp.diag(diagonal), rhs, config=cfg)

    assert result.converged, result.residual_norm
    true_residual = jnp.linalg.norm(diagonal * result.solution - rhs)
    acceptance = cfg.atol + cfg.rtol * jnp.linalg.norm(rhs)
    assert true_residual <= .5 * acceptance, true_residual / acceptance


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
@pytest.mark.parametrize("tolerance", ["relative", "absolute"])
def test_gmres_retry_corrects_a_finite_candidate_with_a_small_krylov_budget(
        monkeypatch, scale, tolerance):
    from importlib import import_module

    gmres_module = import_module("gradscf.solvers.linear.gmres")
    upstream_gmres = gmres_module.gmres
    diagonal = jnp.linspace(1., 100., 64)

    def inaccurate_incremental(operator, value, **kwargs):
        if kwargs["solve_method"] == "incremental":
            # Control only the inaccurate primary result. Its remaining error
            # occupies one eigendirection; the retry is a real GMRES solve.
            candidate = (value / diagonal).at[0].add(1e-6)
            return candidate, 0
        return upstream_gmres(operator, value, **kwargs)

    monkeypatch.setattr(gmres_module, "gmres", inaccurate_incremental)
    rhs = scale * (jnp.cos(.7 * jnp.arange(64)) + jnp.sin(.3 * jnp.arange(64)))
    rtol = 1e-10 if tolerance == "relative" else 0.
    atol = 0. if tolerance == "relative" else float(1e-10 * jnp.linalg.norm(rhs))
    # Three fresh Krylov vectors cannot resolve this 64-eigenvalue system.
    # They suffice to correct the finite candidate's one-direction residual.
    config = LinearSolverConfig(rtol=rtol, atol=atol, restart=3, maxiter=1)
    result = solve_linear(jnp.diag(diagonal), rhs, config=config)

    assert result.converged, result.residual_norm
    actual_residual = jnp.linalg.norm(diagonal * result.solution-rhs)
    assert actual_residual <= .5 * (atol+rtol*jnp.linalg.norm(rhs))
    np.testing.assert_allclose(result.solution, rhs/diagonal, atol=scale*1e-12, rtol=1e-11)


def test_gmres_stagnating_primary_has_bounded_probe_work(monkeypatch):
    from importlib import import_module

    gmres_module = import_module("gradscf.solvers.linear.gmres")
    upstream_gmres = gmres_module.gmres
    diagonal = jnp.linspace(1., 100., 64)
    work = []

    def stagnating_incremental(operator, value, **kwargs):
        if kwargs["solve_method"] == "incremental":
            # Model a stalled primary without depending on an upstream Arnoldi
            # bug. Each simulated cycle performs an actual operator action.
            def cycle(index, candidate):
                norm = jnp.linalg.norm(operator(jnp.roll(value, index)))
                jax.debug.callback(lambda x: work.append(float(x)), norm, ordered=True)
                return candidate
            candidate = jax.lax.fori_loop(0, kwargs["maxiter"], cycle, value/diagonal)
            return candidate.at[0].add(1e-6), 0
        return upstream_gmres(operator, value, **kwargs)

    monkeypatch.setattr(gmres_module, "gmres", stagnating_incremental)
    rhs = jnp.cos(.7*jnp.arange(64))+jnp.sin(.3*jnp.arange(64))
    result = jax.block_until_ready(solve_linear(jnp.diag(diagonal), rhs,
        config=LinearSolverConfig(rtol=1e-10, atol=0., restart=3, maxiter=20)))
    jax.effects_barrier()
    assert result.converged, result.residual_norm
    assert len(work) <= 2
    np.testing.assert_allclose(result.solution, rhs/diagonal, atol=1e-12, rtol=1e-11)


def test_gmres_full_incremental_recovers_after_batched_failure(monkeypatch):
    from importlib import import_module

    gmres_module = import_module("gradscf.solvers.linear.gmres")
    upstream_gmres = gmres_module.gmres
    failed_batches = []

    def failed_batched(operator, value, **kwargs):
        if kwargs["solve_method"] == "batched":
            jax.debug.callback(lambda: failed_batches.append(True), ordered=True)
            return jnp.full_like(value, jnp.nan), 0
        return upstream_gmres(operator, value, **kwargs)

    monkeypatch.setattr(gmres_module, "gmres", failed_batched)
    diagonal = jnp.linspace(1., 10., 12)
    rhs = jnp.sin(jnp.arange(12))+jnp.cos(.3*jnp.arange(12))
    result = jax.block_until_ready(solve_linear(jnp.diag(diagonal), rhs,
        config=LinearSolverConfig(rtol=1e-10, atol=0., restart=2, maxiter=60)))
    jax.effects_barrier()
    assert failed_batches  # The short probe could not solve the full system.
    assert result.converged, result.residual_norm
    np.testing.assert_allclose(result.solution, rhs/diagonal, atol=1e-10, rtol=1e-9)
