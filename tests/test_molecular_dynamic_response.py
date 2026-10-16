"""Full-space restricted dynamic response, including noncanonical orbitals."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule
from gradscf.solvers import LinearSolverConfig
from gradscf.tddft import gen_tdhf_vind
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional


def toy_reference():
    rng = np.random.default_rng(305)
    factors = rng.normal(size=(4, 5, 5))*.05
    factors = (factors+factors.transpose(0, 2, 1))/2
    eri = np.einsum('Ppq,Prs->pqrs', factors, factors)
    dipole = rng.normal(size=(3, 5, 5))*.1
    dipole = (dipole+dipole.transpose(0, 2, 1))/2
    energies = np.array([-1.2, -.7, .2, .5, .9])
    reference = RestrictedMolecule(
        ao=jnp.zeros((0, 5)), grid=QuadratureGrid(jnp.zeros(0)),
        dipole_integrals=jnp.asarray(dipole), rep_tensor=jnp.asarray(eri),
        mo_coeff=jnp.eye(5), mo_occ=jnp.array([2., 2., 0., 0., 0.]),
        mo_energy=jnp.asarray(energies), rdm1=jnp.stack([jnp.diag(jnp.array([1., 1., 0., 0., 0.]))]*2),
        h1e=jnp.diag(jnp.asarray(energies)), nuclear_repulsion=0., nocc=2)
    return reference, jnp.diag(jnp.asarray(energies))


def rotation():
    rng = np.random.default_rng(806)
    occupied = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    virtual = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    full = np.zeros((5, 5))
    full[:2, :2], full[2:, 2:] = occupied, virtual
    transition = np.kron(occupied, virtual)
    doubled = np.zeros((12, 12))
    doubled[:6, :6], doubled[6:, 6:] = transition, transition
    return jnp.asarray(full), jnp.asarray(doubled)


def test_noncanonical_fock_preserves_full_operator_under_occupied_virtual_rotations():
    reference, fock = toy_reference()
    rotate, transition = rotation()
    # An explicit molecule-like payload makes the old implementation fail on
    # physics (discarded off-diagonal Fock blocks), rather than a missing class.
    moved = SimpleNamespace(**reference.__dict__)
    moved.mo_coeff = rotate
    moved.mo_energy = jnp.diag(rotate.T@fock@rotate)
    moved.fock_matrix = fock
    functional = SemilocalResponseFunctional('hf')
    original = gen_tdhf_vind(reference, functional)(jnp.eye(12)).T
    changed = gen_tdhf_vind(moved, functional)(jnp.eye(12)).T
    np.testing.assert_allclose(changed, transition.T@original@transition, atol=3e-14, rtol=0)


def test_fock_payload_is_dynamic_and_default_canonical_path_is_unchanged():
    reference, fock = toy_reference()
    full = replace(reference, fock_matrix=fock)
    functional = SemilocalResponseFunctional('hf')
    expected = gen_tdhf_vind(reference, functional)(jnp.eye(12))
    np.testing.assert_array_equal(gen_tdhf_vind(full, functional)(jnp.eye(12)), expected)
    leaves, structure = jax.tree_util.tree_flatten(full)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_array_equal(rebuilt.fock_matrix, fock)
    probe = jnp.arange(12., dtype=jnp.float64).reshape(1, 12)/12
    direction = jnp.zeros((5, 5)).at[2, 3].set(.17).at[3, 2].set(.17)
    action = jax.jit(lambda record: gen_tdhf_vind(record, functional)(probe))
    tangent_record = jax.tree.map(jnp.zeros_like, full)
    tangent_record = replace(tangent_record, fock_matrix=direction)
    _, tangent = jax.jvp(action, (full,), (tangent_record,))
    step = 1e-4
    finite = (action(replace(full, fock_matrix=fock+step*direction))
              - action(replace(full, fock_matrix=fock-step*direction)))/(2*step)
    assert np.linalg.norm(tangent) > .01
    np.testing.assert_allclose(tangent, finite, atol=2e-12, rtol=2e-10)


def test_dynamic_resolvent_preserves_complex_parts_and_frequency_conjugation():
    from gradscf_tools.dynamic_response import dynamic_polarizability
    from scipy.linalg import solve

    reference, fock = toy_reference()
    reference = replace(reference, fock_matrix=fock)
    functional = SemilocalResponseFunctional('hf')
    matrix = np.asarray(gen_tdhf_vind(reference, functional)(jnp.eye(12))).T
    dipole = np.asarray(reference.dipole_integrals[:, :2, 2:]).reshape(3, 6).T
    rhs = np.concatenate([dipole, -dipole])
    config = LinearSolverConfig(method='direct', rtol=1e-12, atol=1e-14, max_dense=24)
    for frequency in [0., .14, -.14]:
        actual = dynamic_polarizability(reference, functional, frequency, damping=.027, config=config)
        independent = solve(matrix-(frequency+.027j)*np.eye(12), rhs)
        expected = 2*dipole.T@(independent[:6]+independent[6:])
        assert bool(actual.converged) and float(actual.residual_norm) < 1e-12
        np.testing.assert_allclose(actual.polarizability, expected, atol=2e-13, rtol=2e-12)
    plus = dynamic_polarizability(reference, functional, .14, damping=.027, config=config)
    minus = dynamic_polarizability(reference, functional, -.14, damping=.027, config=config)
    assert np.linalg.norm(np.asarray(plus.polarizability).imag) > 1e-5
    np.testing.assert_allclose(minus.polarizability, plus.polarizability.conj(), atol=2e-13, rtol=0)


def test_dynamic_response_is_invariant_to_noncanonical_orbital_gauge():
    from gradscf_tools.dynamic_response import dynamic_polarizability

    reference, fock = toy_reference()
    rotate, _ = rotation()
    reference = replace(reference, fock_matrix=fock)
    rotated = replace(reference, mo_coeff=rotate, mo_energy=jnp.diag(rotate.T@fock@rotate))
    functional = SemilocalResponseFunctional('hf')
    for omega, eta in [(0., 0.), (.15, .02)]:
        original = dynamic_polarizability(reference, functional, omega, damping=eta)
        changed = dynamic_polarizability(rotated, functional, omega, damping=eta)
        assert original.converged & changed.converged
        np.testing.assert_allclose(changed.polarizability, original.polarizability, atol=2e-12, rtol=2e-11)


def pyscf_response(coordinates, symbols, basis, frequency, damping, xc='hf'):
    from pyscf import dft, gto, scf
    from pyscf.tdscf.rhf import get_ab
    from scipy.linalg import solve

    molecule = gto.M(atom=list(zip(symbols, coordinates)), unit='Bohr', basis=basis, cart=True, verbose=0)
    mean_field = scf.RHF(molecule) if xc == 'hf' else dft.RKS(molecule, xc=xc)
    if xc != 'hf':
        mean_field.grids.level = 0
    mean_field.conv_tol, mean_field.conv_tol_grad = 1e-13, 1e-10
    mean_field.kernel()
    assert mean_field.converged
    a, b = get_ab(mean_field)
    nocc, nvir = a.shape[:2]
    n = nocc*nvir
    a, b = a.reshape(n, n), b.reshape(n, n)
    matrix = np.block([[a, b], [-b, -a]])
    c = mean_field.mo_coeff
    position = molecule.intor('int1e_r', comp=3)
    d = np.einsum('pi,xpq,qa->xia', c[:, :nocc], position, c[:, nocc:]).reshape(3, n).T
    solution = solve(matrix-(frequency+1j*damping)*np.eye(2*n), np.concatenate([d, -d]))
    return 2*d.T@(solution[:n]+solution[n:]), mean_field


@pytest.mark.parametrize('molecule,xc', [('h2', 'hf'), ('water', 'hf'), ('h2', 'pbe'), ('h2', 'pbe0')])
def test_full_molecular_response_static_energy_limit_and_geometry_jvp(molecule, xc):
    from gradscf import dft, gto, scf
    from gradscf_tools.dynamic_response import MolecularPolarizability

    if molecule == 'h2':
        symbols, coordinates, basis = ['H', 'H'], np.array([[0., 0., 0.], [0., 0., 1.43]]), '6-31g'
        direction = np.array([[0., 0., -.5], [0., 0., .5]])
    else:
        symbols, basis = ['O', 'H', 'H'], 'sto-3g'
        coordinates = np.array([[0., 0., 0.], [0., 1.44, 1.12], [0., -1.44, 1.12]])
        direction = np.array([[0., 0., -.2], [0., .3, .1], [0., -.3, .1]])
    frequency, damping = .11, .015
    expected, independent = pyscf_response(coordinates, symbols, basis, frequency, damping, xc)
    mol = gto.M(atom=list(zip(symbols, coordinates)), unit='Bohr', basis=basis, cart=True)
    options = dict(conv_tol=1e-12, conv_tol_grad=1e-10, init_guess=independent.make_rdm1())
    seed = (scf.RHF(mol, **options) if xc == 'hf' else
            dft.RKS(mol, xc=xc, grids_level=0, **options)).run()
    assert seed.converged
    model = MolecularPolarizability(seed, gradient_tolerance=1e-10)
    point, tangent = jnp.asarray(coordinates), jnp.asarray(direction)
    evaluate = jax.jit(lambda r: model(r, frequency, damping=damping))
    actual = evaluate(point)
    assert actual.converged and actual.residual_norm < 1e-10
    np.testing.assert_allclose(actual.polarizability, expected, atol=3e-8, rtol=2e-8)
    static = jax.jit(lambda r: model(r, 0., damping=0.).polarizability)(point)
    alpha_energy = -jax.jit(jax.hessian(lambda f: model.energy(point, f)))(jnp.zeros(3))
    np.testing.assert_allclose(static.real, alpha_energy, atol=3e-8, rtol=2e-8)
    np.testing.assert_allclose(static.imag, 0., atol=1e-14, rtol=0)
    alpha = lambda r: evaluate(r).polarizability
    response = jax.jit(lambda r, v: jax.jvp(alpha, (r,), (v,))[1])(point, tangent)
    errors = []
    for step in [1e-3, 3e-4]:
        finite = (alpha(point+step*tangent)-alpha(point-step*tangent))/(2*step)
        independent_plus, _ = pyscf_response(coordinates+step*direction, symbols, basis, frequency, damping, xc)
        independent_minus, _ = pyscf_response(coordinates-step*direction, symbols, basis, frequency, damping, xc)
        errors.append(float(np.max(abs(np.asarray(response)-(independent_plus-independent_minus)/(2*step)))))
        np.testing.assert_allclose(response, finite, atol=3e-6, rtol=3e-6)
    assert errors[-1] < 3e-6
    print(molecule, xc, 'dynamic/static/JVP independent max errors', float(np.max(abs(actual.polarizability-expected))),
          float(np.max(abs(static.real-alpha_energy))), errors)
