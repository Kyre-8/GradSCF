"""Stationary R/U HF response with direct J/K and no resident AO ERIs.

Coordinates are in bohr, energies in Hartree, and every solve uses float64.
The full native ERI calculation is only a small-system storage oracle.
"""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import gto, integrals, scf
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan
from gradscf_tools.molecular_response import MolecularEnergy


@pytest.fixture(scope="module", params=["rhf", "uhf"])
def native_problem(request):
    unrestricted = request.param == "uhf"
    atom = "N 0 0 0; H 0 0 1.95" if unrestricted else "H 0 0 0; H 0 0 1.4"
    basis = "sto-3g" if unrestricted else "3-21g"
    mol = gto.M(atom=atom, basis=basis, spin=2 if unrestricted else 0,
                unit="Bohr", cart=True)
    driver = scf.UHF if unrestricted else scf.RHF
    reference = driver(mol, integral_backend="native", max_cycle=200,
                       conv_tol=1e-12, conv_tol_density=1e-11,
                       conv_tol_grad=1e-10).run()
    assert reference.converged
    topology, parameters = integrals.prepare_basis(mol.to_spec(), basis, cart=True)
    plan = integrals.make_plan(topology, backend="native")
    coordinates = jnp.asarray(parameters.nuclear_coords)
    owners = np.argmin(np.linalg.norm(np.asarray(parameters.centers)[:, None]
                                     - np.asarray(coordinates)[None], axis=-1), axis=1)
    coefficients = np.asarray(reference.mo_coeff)
    occupations = np.asarray(reference.mo_occ)
    if not unrestricted:
        occupations = np.stack((occupations / 2, occupations / 2))
    minimize = scf.minimize_uks_from_integrals if unrestricted else scf.minimize_roks_from_integrals
    differentiation = scf.SCFDifferentiationConfig(
        mode="implicit", tolerance=1e-10, max_iter=100, restart=40)

    def calculate(r, *, direct, perturbation=0.):
        moving = replace(parameters, nuclear_coords=r, centers=r[owners])
        nao = topology.nao
        core = plan.evaluate("kinetic", moving) + plan.evaluate("nuclear", moving)
        potential = jnp.diag(jnp.linspace(-.13, .19, nao))
        potential = potential.at[0, -1].set(.07).at[-1, 0].set(.07)
        options = (dict(direct_basis=NativeDirectBasis(plan, moving),
                        direct_scf_tol=0., direct_scf_workers=1) if direct else {})
        return minimize(
            overlap=plan.evaluate("overlap", moving), hcore=core + perturbation * potential,
            eri=None if direct else plan.evaluate("eri", moving),
            nuclear_repulsion=scf.nuclear_repulsion_energy(
                r, jnp.asarray(topology.nuclear_charges)),
            ao=jnp.zeros((0, nao)), ao_deriv1=jnp.zeros((4, 0, nao)),
            grid_weights=jnp.zeros(0), mo_coeff=coefficients, mo_occ=occupations,
            xc_spec="hf", max_iterations=300, gradient_tolerance=1e-9,
            orthonormalize_initial=True, differentiation=differentiation, **options)

    return reference, coordinates, plan, parameters, owners, calculate


def _forbid_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("The direct stationary path must not assemble AO ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)


def test_direct_stationary_hf_matches_full_state(native_problem, monkeypatch):
    reference, coordinates, plan, parameters, _, calculate = native_problem
    expected = calculate(coordinates, direct=False)
    assert expected.stationary
    _forbid_eri(monkeypatch)
    observed = jax.jit(lambda r: calculate(r, direct=True))(coordinates)
    assert observed.stationary
    np.testing.assert_allclose(observed.total_energy, expected.total_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(observed.total_energy, reference.e_tot, atol=2e-9, rtol=0)
    np.testing.assert_allclose(observed.density_matrix, expected.density_matrix, atol=2e-8, rtol=0)
    np.testing.assert_allclose(observed.fock_matrix, expected.fock_matrix, atol=2e-9, rtol=0)
    overlap = plan.evaluate("overlap", parameters)
    electrons = jnp.einsum("spq,qp->s", observed.density_matrix, overlap)
    np.testing.assert_allclose(electrons, jnp.sum(observed.mo_occ, axis=-1), atol=2e-10, rtol=0)


def test_direct_density_observable_retains_implicit_response(native_problem, monkeypatch):
    _, coordinates, _, _, _, calculate = native_problem
    nao = native_problem[2].topology.nao
    weight = jnp.cos(jnp.arange(nao * nao, dtype=jnp.float64)).reshape(nao, nao)
    weight = .5 * (weight + weight.T)

    def observable(theta, *, direct):
        state = calculate(coordinates, direct=direct, perturbation=theta)
        return jnp.sum(state.density_matrix[0] * weight)

    theta = .02
    expected = jax.jit(jax.grad(lambda t: observable(t, direct=False)))(theta)
    _forbid_eri(monkeypatch)
    value = jax.jit(lambda t: observable(t, direct=True))
    observed_vjp = jax.jit(jax.grad(lambda t: observable(t, direct=True)))(theta)
    observed_jvp = jax.jit(lambda t: jax.jvp(
        lambda q: observable(q, direct=True), (t,), (jnp.ones_like(t),))[1])(theta)
    assert abs(float(expected)) > 1e-7
    np.testing.assert_allclose(observed_vjp, expected, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(observed_jvp, observed_vjp, atol=2e-9, rtol=2e-8)
    for step in (3e-4, 1e-4):
        finite_difference = (value(theta + step) - value(theta - step)) / (2 * step)
        np.testing.assert_allclose(observed_vjp, finite_difference, atol=3e-6, rtol=2e-5)


def test_direct_geometry_gradient_and_hvp_match_full(native_problem, monkeypatch):
    _, coordinates, _, _, _, calculate = native_problem
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    full_energy = lambda r: calculate(r, direct=False).total_energy
    full_gradient = jax.jit(jax.grad(full_energy))
    expected_gradient = full_gradient(coordinates)
    expected_hvp = jax.jit(lambda r: jax.jvp(
        full_gradient, (r,), (direction,))[1])(coordinates)
    _forbid_eri(monkeypatch)
    energy = lambda r: calculate(r, direct=True).total_energy
    value = jax.jit(energy)
    gradient = jax.jit(jax.grad(energy))
    observed_gradient = gradient(coordinates)
    observed_hvp = jax.jit(lambda r: jax.jvp(
        gradient, (r,), (direction,))[1])(coordinates)
    np.testing.assert_allclose(observed_gradient, expected_gradient, atol=3e-8, rtol=2e-7)
    np.testing.assert_allclose(observed_gradient.sum(axis=0), 0., atol=2e-9, rtol=0)
    np.testing.assert_allclose(observed_hvp, expected_hvp, atol=3e-7, rtol=2e-6)
    for step in (3e-4, 1e-4):
        energy_difference = (value(coordinates + step * direction)
                             - value(coordinates - step * direction)) / (2 * step)
        np.testing.assert_allclose(jnp.sum(observed_gradient * direction), energy_difference,
                                   atol=2e-6, rtol=1e-5)
        finite_difference = (gradient(coordinates + step * direction)
                             - gradient(coordinates - step * direction)) / (2 * step)
        np.testing.assert_allclose(observed_hvp, finite_difference, atol=4e-6, rtol=2e-5)


def test_molecular_energy_direct_does_not_assemble_eri(native_problem, monkeypatch):
    reference, coordinates, _, _, _, _ = native_problem
    full = MolecularEnergy(reference, gradient_tolerance=1e-9)
    expected = jax.jit(full.solve)(coordinates)
    expected_gradient = jax.jit(jax.grad(full))(coordinates)
    _forbid_eri(monkeypatch)
    direct = MolecularEnergy(reference, jk_backend="direct", gradient_tolerance=1e-9)
    inputs = direct.inputs(coordinates)
    assert inputs["eri"] is None
    assert isinstance(inputs["direct_basis"], NativeDirectBasis)
    observed = jax.jit(direct.solve)(coordinates)
    observed_gradient = jax.jit(jax.grad(direct))(coordinates)
    assert observed.stationary
    np.testing.assert_allclose(observed.total_energy, expected.total_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(observed.density_matrix, expected.density_matrix, atol=2e-8, rtol=0)
    np.testing.assert_allclose(observed_gradient, expected_gradient, atol=3e-8, rtol=2e-7)


def test_molecular_energy_retains_seed_direct_workers(native_problem, monkeypatch):
    reference = copy(native_problem[0])
    reference.direct_scf_workers = 3
    energy = MolecularEnergy(reference, jk_backend="direct")
    supplied = {"eri": None, "direct_basis": object()}
    monkeypatch.setattr(energy, "inputs", lambda *args, **kwargs: supplied)
    captured = {}
    state = object()

    def minimize(**kwargs):
        captured.update(kwargs)
        return state

    name = "minimize_uks_from_integrals" if energy.unrestricted else "minimize_roks_from_integrals"
    monkeypatch.setattr(scf, name, minimize)
    observed, inputs = energy.solve_with_inputs(energy.coordinates)
    assert observed is state and inputs is supplied
    assert captured.get("direct_scf_workers") == 3
    assert captured.get("direct_scf_tol", 0.) == 0.


def test_molecular_energy_default_keeps_full_eri_inputs(native_problem):
    reference, coordinates, _, _, _, _ = native_problem
    inputs = MolecularEnergy(reference).inputs(coordinates)
    nao = np.asarray(reference.mo_coeff).shape[-1]
    assert inputs["eri"].shape == (nao, nao, nao, nao)
    assert inputs.get("direct_basis") is None


def test_molecular_energy_direct_seed_requires_explicit_opt_in(native_problem):
    reference = copy(native_problem[0])
    reference.jk_backend = "direct"
    with pytest.raises(NotImplementedError, match="(?i)full"):
        MolecularEnergy(reference)
    direct = MolecularEnergy(reference, jk_backend="direct")
    assert direct.inputs(direct.coordinates)["eri"] is None


@pytest.mark.parametrize("xc", ["pbe", "wb97xd"])
def test_molecular_energy_direct_is_explicitly_hf_only(native_problem, xc):
    reference = copy(native_problem[0])
    reference.xc = xc
    with pytest.raises(NotImplementedError, match="(?i)(HF|Hartree.Fock)"):
        MolecularEnergy(reference, jk_backend="direct")


@pytest.mark.parametrize("unrestricted", [False, True], ids=["rhf", "uhf"])
@pytest.mark.parametrize("case", ["missing", "provider", "eri", "dimensions"])
def test_direct_orbital_input_validation(unrestricted, case):
    """Invalid storage/provider combinations fail before any integral action."""
    provider = SimpleNamespace(nao=2, get_jk=lambda density: None)
    options = {}
    eri = None
    error, message = ValueError, "direct_basis"
    if case == "provider":
        options["direct_basis"] = object()
        error, message = TypeError, "get_jk"
    elif case == "eri":
        options["direct_basis"] = provider
        eri = np.zeros((2, 2, 2, 2))
        message = "eri=None"
    elif case == "dimensions":
        provider.nao = 3
        options["direct_basis"] = provider
        message = "AO dimensions"
    minimize = scf.minimize_uks_from_integrals if unrestricted else scf.minimize_roks_from_integrals
    coefficient = np.eye(2)
    with pytest.raises(error, match=message):
        minimize(overlap=np.eye(2), hcore=np.diag([-1., .5]), eri=eri,
                 nuclear_repulsion=0., ao=np.zeros((0, 2)),
                 ao_deriv1=np.zeros((4, 0, 2)), grid_weights=np.zeros(0),
                 mo_coeff=np.stack((coefficient, coefficient)) if unrestricted else coefficient,
                 mo_occ=np.array([[1., 0.], [1., 0.]]), **options)


def test_molecular_energy_rejects_unknown_jk_backend():
    with pytest.raises(ValueError, match="jk_backend"):
        MolecularEnergy(None, jk_backend="df")
