"""Exact signed-permutation response reduction, native RHF in float64/bohr."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def methane():
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF('C 0 0 0; H 1.2 1.2 1.2; H 1.2 -1.2 -1.2; '
                      'H -1.2 1.2 -1.2; H -1.2 -1.2 1.2', unit='Bohr')
    reference, _ = model.evaluate(model.coordinates, gradient=False)
    return model, reference


def test_methane_group_and_coordinate_orbits(methane):
    from gradscf_tools.molecular_symmetry import signed_permutation_symmetry, coordinate_orbits

    model, reference = methane
    operations = signed_permutation_symmetry(model, model.coordinates, reference)
    leaders, mapping = coordinate_orbits(operations)
    assert len(operations) == 24
    assert leaders == [0, 3]
    for j, (i, operation, sign) in enumerate(mapping):
        t, q = operations[operation]
        np.testing.assert_array_equal(t[:, i], sign*np.eye(15)[:, j])
        np.testing.assert_allclose(q @ q.T, np.eye(9), atol=1e-11)


@pytest.mark.parametrize('field,value', [('angular_momenta', (2,)),
                                        ('contraction_counts', (2,))])
def test_reject_unsupported_shells(methane, field, value):
    from gradscf_tools.molecular_symmetry import signed_permutation_symmetry

    model, reference = methane
    original = getattr(model.topology, field)
    topology = replace(model.topology, **{field: value + original[1:]})
    changed = SimpleNamespace(**vars(model), unsupported=True)
    changed.topology = topology
    with pytest.raises(ValueError, match='shell|contraction'):
        signed_permutation_symmetry(changed, model.coordinates, reference)


def test_reject_mismatched_atom_basis(methane):
    from gradscf_tools.molecular_symmetry import signed_permutation_symmetry

    model, reference = methane
    changed = SimpleNamespace(**vars(model))
    exponents = list(model.parameters.exponents)
    exponents[-1] = exponents[-1]*1.01
    changed.parameters = replace(model.parameters, exponents=tuple(exponents))
    with pytest.raises(ValueError, match='basis'):
        signed_permutation_symmetry(changed, model.coordinates, reference)


def test_atom_order_translation_and_distorted_geometry(methane):
    from gradscf_tools.molecular_ep import NativeRHF
    from gradscf_tools.molecular_symmetry import signed_permutation_symmetry, coordinate_orbits

    model, _ = methane
    order = [3, 0, 4, 1, 2]
    coordinates = model.coordinates[order]+[.1, -.2, .3]
    atom = '; '.join(f'{"C" if i == 0 else "H"} {r[0]} {r[1]} {r[2]}'
                     for i, r in zip(order, coordinates))
    reordered = NativeRHF(atom, unit='Bohr')
    reference, _ = reordered.evaluate(coordinates, gradient=False)
    operations = signed_permutation_symmetry(reordered, coordinates, reference)
    assert len(operations) == 24
    assert coordinate_orbits(operations)[0] == [0, 3]
    distorted = coordinates + np.random.default_rng(17).normal(size=coordinates.shape)*.01
    reference, _ = reordered.evaluate(distorted, gradient=False)
    operations = signed_permutation_symmetry(reordered, distorted, reference)
    assert len(operations) == 1
    assert coordinate_orbits(operations)[0] == list(range(coordinates.size))


def test_orbital_gauge_covariance_and_invalid_reference(methane):
    from gradscf_tools.molecular_symmetry import signed_permutation_symmetry

    model, reference = methane
    operations = signed_permutation_symmetry(model, model.coordinates, reference)
    rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(9, 9)))
    rotated = SimpleNamespace(mo_coeff=reference.mo_coeff @ rotation,
                              overlap_matrix=reference.overlap_matrix,
                              fock_matrix=reference.fock_matrix,
                              density_matrix=reference.density_matrix,
                              converged=True)
    other = signed_permutation_symmetry(model, model.coordinates, rotated)
    for (t, q), (t_other, q_other) in zip(operations, other):
        np.testing.assert_array_equal(t, t_other)
        np.testing.assert_allclose(q_other, rotation.T @ q @ rotation, atol=1e-12)
    rotated.fock_matrix = np.array(reference.fock_matrix)
    rotated.fock_matrix[-1, -1] += .1
    with pytest.raises(ValueError, match='invariance'):
        signed_permutation_symmetry(model, model.coordinates, rotated)


@pytest.mark.parametrize('field', ['coordinates', 'mo_coeff', 'overlap_matrix',
                                  'fock_matrix', 'density_matrix'])
def test_reject_complex_inputs(methane, field):
    from gradscf_tools.molecular_symmetry import signed_permutation_symmetry

    model, reference = methane
    coordinates = model.coordinates
    changed = SimpleNamespace(**{name: getattr(reference, name) for name in
                              ('mo_coeff', 'overlap_matrix', 'fock_matrix',
                               'density_matrix', 'converged')})
    if field == 'coordinates':
        coordinates = coordinates.astype(complex)
    else:
        setattr(changed, field, np.asarray(getattr(changed, field)).astype(complex))
    with pytest.raises(ValueError, match='real'):
        signed_permutation_symmetry(model, coordinates, changed)


def test_analytic_response_reconstruction_and_transport(methane):
    from gradscf_tools.molecular_symmetry import signed_permutation_symmetry, coordinate_orbits

    model, reference = methane
    coordinates = model.coordinates
    operations = signed_permutation_symmetry(model, coordinates, reference)
    leaders, mapping = coordinate_orbits(operations)
    response = model.response(coordinates, reference)
    basis = np.eye(coordinates.size).reshape((-1,) + coordinates.shape)
    columns = {i: response(basis[i]) for i in leaders}
    hessian = np.empty((coordinates.size, coordinates.size))
    fock = np.empty((coordinates.size, model.topology.nao, model.topology.nao))
    for j, (i, operation, sign) in enumerate(mapping):
        t, q = operations[operation]
        h, f = columns[i]
        hessian[:, j] = sign*t @ h.ravel()
        fock[j] = sign*q @ f @ q.T
    for j in (1, 7, 14):
        h, f = response(basis[j])
        np.testing.assert_allclose(hessian[:, j], h.ravel(), atol=2e-8)
        np.testing.assert_allclose(fock[j], f, atol=2e-8)
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-8)
    step = 2e-4
    j = 7
    plus, gp = model.evaluate(coordinates+step*basis[j], reference.density_matrix)
    minus, gm = model.evaluate(coordinates-step*basis[j], reference.density_matrix)
    fp, _ = model.transport(plus, reference.mo_coeff, coordinates, coordinates+step*basis[j])
    fm, _ = model.transport(minus, reference.mo_coeff, coordinates, coordinates-step*basis[j])
    np.testing.assert_allclose(hessian[:, j], ((gp-gm)/(2*step)).ravel(), atol=2e-6)
    np.testing.assert_allclose(fock[j], (fp-fm)/(2*step), atol=2e-6)
