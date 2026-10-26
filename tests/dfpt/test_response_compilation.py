"""The finite-q action must not embed grid-sized snapshots in compiled HLO."""
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto, dft
from gradscf.dfpt.force_constants import primitive_force_response


def _reference(mesh):
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6,
        unit='Bohr', mesh=(mesh,)*3)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((3,1,1)), xc='pbe').run(
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10)
    assert mf.converged
    return mf


def test_primitive_action_does_not_capture_large_grid_constants():
    option = 'jax_captured_constants_warn_bytes'
    if not hasattr(jax.config, option):
        pytest.skip('This JAX version has no captured-constant diagnostic')
    mf = _reference(15)
    direction = jnp.array([[.3,-.2,.1],[-.4,.1,.2]])
    previous = getattr(jax.config, option)
    try:
        jax.config.update(option, 1_000_000)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            _, action = primitive_force_response(mf, np.array([1/3,0.,0.]))
            column, residual, converged = action(direction)
            column.block_until_ready()
        assert bool(converged) and float(residual)<1e-8
        assert np.isfinite(column).all()
        large = [str(w.message) for w in captured if 'constants were captured' in str(w.message)]
        assert not large, '\n'.join(large)
    finally:
        jax.config.update(option, previous)


def test_primitive_action_retains_its_reference_snapshot():
    mf = _reference(7);cell = mf.cell
    q = np.array([1/3,0.,0.])
    direction = jnp.array([[.3,-.2,.1],[-.4,.1,.2]])
    _, first = primitive_force_response(mf, q)
    _, pending = primitive_force_response(mf, q)
    expected = first(direction)[0]
    # The second action has not been traced yet. Its arrays and metadata must
    # still describe one coherent snapshot after the original objects change.
    mf.xc = 'hf'
    cell.mesh = (9,)*3
    q[0] = -1/3
    actual, residual, converged = pending(direction)
    assert bool(converged) and float(residual)<1e-8
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_primitive_action_directional_derivative_matches_linear_map():
    mf = _reference(7)
    _, action = primitive_force_response(mf, np.array([1/3,0.,0.]))
    direction = jnp.array([[.3,-.2,.1],[-.4,.1,.2]])
    tangent = jnp.array([[-.2,.4,.1],[.3,-.1,.2]])
    actual = jax.jvp(lambda v:action(v)[0], (direction,), (tangent,))[1]
    expected, _, converged = action(tangent)
    assert bool(converged)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
    weights = jnp.arange(6.)+1j*jnp.arange(6.)[::-1]
    gradient = jax.grad(lambda v:jnp.vdot(weights,action(v)[0]).real)(direction)
    np.testing.assert_allclose(jnp.sum(gradient*tangent),jnp.vdot(weights,expected).real,
                               atol=1e-9,rtol=1e-9)
