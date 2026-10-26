"""Polar subtraction/interpolation/restoration on manufactured force constants."""
import importlib
import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def model(cutoff=5, scale=1.):
    api = importlib.import_module('gradscf.pbc.polar_ifc')
    lattice = np.diag([5., 6., 7.])
    positions = np.array([[.2, .3, .4], [.7, 1., 1.2]])
    charge = jnp.array([[1.1, .2, -.1], [.3, 1.4, .2], [.1, -.2, .9]])*scale
    born = jnp.stack((charge, -charge))
    dielectric = np.array([[2.4, .1, .2], [.1, 3.1, -.1], [.2, -.1, 2.8]])
    cell = SimpleNamespace(lattice=lattice, coords=positions)
    correction = api.DipoleCorrection(lattice, positions, born, dielectric,
                                      smearing=.35, cutoff=cutoff)
    return api, cell, correction


def test_fixed_sum_symmetry_periodicity_cutoff_and_gamma_limit():
    from gradscf.pbc.vibrational import nonanalytic_force_constants
    _, cell, correction = model()
    q = np.array([.13, -.17, .21])
    value = np.asarray(correction.matrix(q))
    np.testing.assert_allclose(value, value.conj().T, atol=1e-13)
    assert np.linalg.eigvalsh(value).min() > -1e-13
    np.testing.assert_allclose(correction.matrix(-q), value.conj(), atol=1e-13)
    np.testing.assert_allclose(correction.matrix(q+[1., -1., 0.]), value, atol=1e-11)
    np.testing.assert_allclose(correction.matrix([0., 0., .5]).imag, 0., atol=1e-12)
    _, _, smaller = model(cutoff=4)
    np.testing.assert_allclose(smaller.matrix(q), value, atol=1e-11)
    qcart = q@(2*np.pi*np.linalg.inv(cell.lattice).T)
    limit = correction.gamma()+nonanalytic_force_constants(
        qcart, correction.born_charges, correction.dielectric, np.linalg.det(cell.lattice))
    np.testing.assert_allclose(correction.gamma(qcart), limit, atol=1e-13)
    errors = [float(np.max(abs(correction.matrix(t*q)-limit))) for t in (1e-3, 1e-4, 1e-5)]
    assert errors[-1] < 1e-6 and errors[-1] < errors[0]/50


def test_joint_rotation_and_atomic_reimaging_covariance():
    api, cell, correction = model()
    q = np.array([.13, -.17, .21])
    value = correction.matrix(q)
    rotation = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))[0]
    rotation[:, 0] *= np.linalg.det(rotation)
    rotated = api.DipoleCorrection(cell.lattice@rotation.T, cell.coords@rotation.T,
        np.einsum('ij,ajk,lk->ail', rotation, correction.born_charges, rotation),
        rotation@correction.dielectric@rotation.T, smearing=.35, cutoff=5)
    transform = np.kron(np.eye(2), rotation)
    np.testing.assert_allclose(rotated.matrix(q), transform@value@transform.T, atol=2e-13)
    shifts = np.array([[1, -1, 0], [0, 0, 1]])
    reimaged = api.DipoleCorrection(cell.lattice, cell.coords+shifts@cell.lattice,
        correction.born_charges, correction.dielectric, smearing=.35, cutoff=5)
    gauge = np.repeat(np.exp(2j*np.pi*(shifts@q)), 3)
    np.testing.assert_allclose(reimaged.matrix(q), gauge[:, None]*value*gauge.conj()[None, :], atol=2e-13)


def test_q_and_charge_response_without_gamma_q_derivative():
    _, _, correction = model()
    q = jnp.array([.13, -.17, .21])
    direction = jnp.array([.2, -.1, .3])
    function = jax.jit(lambda k:correction.matrix(k).real)
    _, tangent = jax.jvp(function, (q,), (direction,))
    weight = jnp.arange(36.).reshape(6, 6)/36
    _, reverse = jax.vjp(function, q)
    np.testing.assert_allclose(jnp.sum(weight*tangent), reverse(weight)[0]@direction, atol=1e-12)
    finite = (function(q+1e-5*direction)-function(q-1e-5*direction))/2e-5
    np.testing.assert_allclose(tangent, finite, atol=1e-9)
    gamma = jax.jit(lambda scale:jnp.sum(model(scale=scale)[2].gamma().real))
    # The gamma parameter derivative is well-defined; a q derivative there is not.
    np.testing.assert_allclose(jax.grad(gamma)(1.), (gamma(1.+1e-4)-gamma(1.-1e-4))/2e-4, atol=1e-10)
    for point in [jnp.zeros(3), jnp.array([1., 0., 0.])]:
        with pytest.raises(Exception, match='Gamma|gamma'):
            jax.block_until_ready(function(point))


def test_reconstruct_short_range_plus_dipoles_and_keep_total_raw_asr():
    api, cell, correction = model()
    qpoints = np.array(list(itertools.product([0., 1/3, -1/3], repeat=3)))
    spring = .2*np.block([[np.eye(3), -np.eye(3)], [-np.eye(3), np.eye(3)]])
    # A real symmetric onsite convention can be shifted between the two parts.
    onsite = np.zeros((6, 6))
    onsite[:3, :3] = np.diag([.1, .2, .3])
    onsite[3:, 3:] = np.diag([.3, .4, .5])

    def short(q):
        return spring-onsite+.03*np.sum(1-np.cos(2*np.pi*q))*np.eye(6)

    def total(q):
        long = correction.gamma() if np.all(q == 0) else correction.matrix(q)
        return short(q)+long

    matrices = np.array([total(q) for q in qpoints])
    polar = api.PolarForceConstants.from_qmesh(cell, qpoints, matrices,
        masses=[7., 1.], correction=correction, tolerance=1e-9)
    for q, expected in zip(qpoints, matrices):
        actual = polar.gamma() if np.all(q == 0) else polar.matrix(q)
        np.testing.assert_allclose(actual, expected, atol=1e-11)
    target = np.array([.12, -.09, .17])
    np.testing.assert_allclose(polar.matrix(target), total(target), atol=1e-11)
    translations = np.tile(np.eye(3), (2, 1))
    np.testing.assert_allclose(polar.raw_asr_residual, np.linalg.norm(matrices[0]@translations), atol=1e-12)
    assert abs(float(polar.raw_asr_residual-polar.short_range.raw_asr_residual)) > 1e-4
    assert polar.sample_reconstruction_error < 1e-11
    # Existing interpolation preserves an onsite atom-block diagonal constant.
    from gradscf.pbc.ifc import ForceConstants
    constant = ForceConstants.from_qmesh(cell, qpoints, np.broadcast_to(onsite, matrices.shape), masses=[7., 1.])
    np.testing.assert_allclose(constant.matrix(target), onsite, atol=1e-12)

    class ShiftedCorrection(api.DipoleCorrection):
        def matrix(self, q):
            return super().matrix(q)+onsite

        def gamma(self, direction=None):
            return super().gamma(direction)+onsite

    shifted = ShiftedCorrection(correction.lattice, correction.positions,
        correction.born_charges, correction.dielectric, smearing=.35, cutoff=5)
    equivalent = api.PolarForceConstants.from_qmesh(cell, qpoints, matrices,
        masses=[7., 1.], correction=shifted, tolerance=1e-9)
    np.testing.assert_allclose(equivalent.matrix(target), polar.matrix(target), atol=1e-11)
    np.testing.assert_allclose(equivalent.gamma([1., 0., 0.]), polar.gamma([1., 0., 0.]), atol=1e-11)


@pytest.mark.parametrize('invalid', ['volume', 'epsilon', 'smearing', 'nonfinite'])
@pytest.mark.parametrize('transform', ['value', 'jvp', 'vjp'])
def test_invalid_dynamic_model_rejected(invalid, transform):
    api, cell, correction = model()

    def observable(parameter):
        lattice = jnp.asarray(cell.lattice)
        epsilon, smearing = correction.dielectric, jnp.array(.35)
        if invalid == 'volume':
            lattice = lattice.at[0, 0].set(parameter)
        elif invalid == 'epsilon':
            epsilon = epsilon.at[0, 0].set(parameter)
        elif invalid == 'smearing':
            smearing = parameter
        else:
            epsilon = epsilon.at[0, 0].set(parameter*jnp.nan)
        value = api.DipoleCorrection(lattice, cell.coords, correction.born_charges,
                                     epsilon, smearing=smearing, cutoff=3)
        return jnp.sum(value.matrix(jnp.array([.1, .2, .3])).real)

    if transform == 'jvp':
        function = lambda t:jax.jvp(observable, (t,), (jnp.ones_like(t),))[1]
    elif transform == 'vjp':
        function = jax.grad(observable)
    else:
        function = observable
    with pytest.raises(Exception, match='positive|finite'):
        jax.block_until_ready(jax.jit(function)(-1.))


def test_smearing_sensitivity_decreases_with_q_mesh():
    api, cell, original = model(cutoff=5)
    alternative = api.DipoleCorrection(cell.lattice, cell.coords, original.born_charges,
        original.dielectric, smearing=.45, cutoff=5)
    target = np.array([.12, -.09, .17])
    spring = .2*np.block([[np.eye(3), -np.eye(3)], [-np.eye(3), np.eye(3)]])

    def total(q):
        long = original.gamma() if np.all(q == 0) else original.matrix(q)
        return spring+.03*np.sum(1-np.cos(2*np.pi*q))*np.eye(6)+long

    expected = total(target)
    errors = []
    for size in (3, 5, 7):
        axis = np.arange(-(size//2), size//2+1)/size
        qpoints = np.array(list(itertools.product(axis, repeat=3)))
        matrices = np.array([total(q) for q in qpoints])
        polar = api.PolarForceConstants.from_qmesh(cell, qpoints, matrices, masses=[7., 1.],
                                                  correction=alternative, tolerance=1e-9)
        errors.append(float(np.max(abs(polar.matrix(target)-expected))))
    assert errors[-1] < 2e-6 and errors[-1] < errors[0]/10, errors


def test_insufficient_reciprocal_cutoff_is_visible_before_interpolation():
    api, cell, resolved = model()
    unresolved = api.DipoleCorrection(cell.lattice, cell.coords, resolved.born_charges,
        resolved.dielectric, smearing=1.2, cutoff=1)
    # At a self-inverse q, a physical lattice-gauge matrix must be real.
    # A small fixed G cube lacks the shifted partner terms and visibly fails.
    assert np.max(abs(unresolved.matrix([0., 0., .5]).imag)) > 1e-5
    qpoints = np.array(list(itertools.product([0., .5], repeat=3)))
    matrices = np.array([resolved.gamma() if np.all(q == 0) else resolved.matrix(q)
                         for q in qpoints])
    with pytest.raises(ValueError, match='conjugacy|real-space'):
        api.PolarForceConstants.from_qmesh(cell, qpoints, matrices, masses=[7., 1.],
                                          correction=unresolved, tolerance=1e-9)
