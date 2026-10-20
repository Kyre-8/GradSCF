"""Complete CIS clusters retain operator and moving-dipole response."""
from dataclasses import replace
from types import SimpleNamespace
import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.molecules import RestrictedMolecule, QuadratureGrid
from gradscf.solvers import EigenSolverConfig, LinearSolverConfig


def api():
    assert importlib.util.find_spec("gradscf_tools.cis_cluster") is not None, "CISCluster is not implemented"
    return importlib.import_module("gradscf_tools.cis_cluster")


def reference(t=0.):
    fock = jnp.diag(jnp.array([-1., 0., 0., 0., 1.]))
    perturb = jnp.array([[.1, .2, 0., .1], [.2, -.1, .03, -.2],
                         [0., .03, .2, .15], [.1, -.2, .15, -.05]])
    fock = fock.at[1:, 1:].add(t*perturb)
    probe = jnp.array([[1., .2, .1], [.1, .8, -.1], [0., .3, 1.1], [.4, -.2, .3]])
    probe = probe+t*jnp.arange(12).reshape(4, 3)*.007
    dipole = jnp.zeros((3, 5, 5)).at[:, 0, 1:].set(probe.T)
    dipole = dipole.at[:, 1:, 0].set(probe.T)
    return RestrictedMolecule(ao=jnp.zeros((0, 5)), grid=QuadratureGrid(jnp.zeros(0)),
        dipole_integrals=dipole, rep_tensor=jnp.zeros((5,)*4), mo_coeff=jnp.eye(5),
        mo_occ=jnp.array([2., 0., 0., 0., 0.]), mo_energy=jnp.diag(fock),
        rdm1=jnp.stack([jnp.diag(jnp.array([1., 0., 0., 0., 0.]))]*2),
        h1e=fock, nuclear_repulsion=0., nocc=1, fock_matrix=fock)


@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_complete_cluster_jvp_vjp_includes_moving_probes(method):
    module = api()
    config = EigenSolverConfig(method=method, nroots=3, max_subspace=4, max_dense=4, atol=1e-11)
    linear = LinearSolverConfig(method="direct", max_dense=12, rtol=1e-11)
    weights = jnp.array([[1., .2, 0.], [.2, -.7, .1], [0., .1, .3]])
    def loss(t):
        return jnp.sum(weights*module.cluster_properties(reference(t), (0, 3),
            config=config, response_config=linear).transition_tensor)
    def oracle(t):
        ref = reference(t)
        a = np.asarray(ref.fock_matrix)[1:, 1:]+np.eye(4)
        _, vectors = np.linalg.eigh(a)
        d = -np.sqrt(2)*np.asarray(ref.dipole_integrals)[:, 0, 1:].T
        projected = vectors[:, :3].T@d
        return np.sum(np.asarray(weights)*(projected.T@projected))
    out = module.cluster_properties(reference(), (0, 3), config=config, response_config=linear)
    assert bool(out.response_valid)
    derivative = jax.jit(jax.grad(loss))(0.)
    tangent = jax.jit(lambda t: jax.jvp(loss, (t,), (jnp.ones_like(t),))[1])(0.)
    finite = (oracle(1e-4)-oracle(-1e-4))/2e-4
    assert abs(float(derivative)) > 1e-4
    np.testing.assert_allclose([derivative, tangent], finite, atol=2e-8, rtol=0)


def test_reference_frame_rotation_preserves_tensor():
    module = api()
    ref = reference(.07)
    q = np.linalg.qr(np.random.default_rng(41).normal(size=(4, 4)))[0]
    frame = np.eye(5)
    frame[1:, 1:] = q
    moved = replace(ref, mo_coeff=jnp.asarray(frame), mo_energy=jnp.diag(frame.T@ref.fock_matrix@frame))
    actual = module.cluster_properties(moved, (0, 3)).transition_tensor
    expected = module.cluster_properties(ref, (0, 3)).transition_tensor
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


def test_cut_cluster_and_nonstationary_reference_reject_derivative():
    module = api()
    for bounds, stationary in [((0, 2), True), ((0, 3), False)]:
        out = module.cluster_properties(reference(), bounds, stationary=stationary)
        assert not bool(out.response_valid)
        derivative = jax.grad(lambda t: jnp.sum(module.cluster_properties(
            reference(t), bounds, stationary=stationary).transition_tensor))(0.)
        assert not np.isfinite(derivative)


def test_requires_noncanonical_fock_payload_and_valid_bounds():
    module = api()
    with pytest.raises(ValueError, match="fock_matrix"):
        module.cluster_properties(replace(reference(), fock_matrix=None), (0, 3))
    for bounds in [(0, 0), (-1, 3), (0, 5), (0, 2.5)]:
        with pytest.raises(ValueError, match="bounds"):
            module.cluster_properties(reference(), bounds)


@pytest.mark.parametrize("failure", ["nonstationary", "ritz", "cut_cluster", "nonfinite"])
def test_geometry_response_checks_primal_certificate_before_backward(failure):
    module = api()
    bounds = (0, 2) if failure == "cut_cluster" else (0, 3)
    config = EigenSolverConfig(method="dense", nroots=3,
        atol=1e-30 if failure == "ritz" else 1e-10)
    # The array fixture isolates orchestration from another native-integral
    # calculation; it uses actual shared spectral convergence/validity flags.
    model = SimpleNamespace(properties=lambda r, frame=None: module.cluster_properties(
        reference(jnp.sqrt(r[0]) if failure == "nonfinite" else r[0]), bounds,
        config=config, stationary=failure != "nonstationary"))
    point = jnp.array([0. if failure in {"cut_cluster", "nonfinite"} else .07])
    error = ArithmeticError if failure in {"cut_cluster", "nonfinite"} else RuntimeError
    with pytest.raises(error, match="stationary|converge|boundary|finite"):
        module.CISCluster.geometry_response(model, point, jnp.eye(3), jnp.ones(1))
