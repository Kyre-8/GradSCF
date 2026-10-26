"""H2 gates for relaxed CIS density and dipole from explicit one-body fields."""
from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry
from gradscf_tools.molecular_response import MolecularEnergy


@pytest.fixture(scope="module")
def hydrogen():
    coordinates = np.array([[.2, -.1, -.6], [.2, -.1, .8]])
    mol = gto.M(atom=list(zip(("H", "H"), coordinates)), basis="6-31g*", unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()
    assert mf.converged
    policy = EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", max_dense=8, rtol=1e-10))
    model = CISGeometry(mf, nroots=2, response=policy)
    return coordinates, mf, model


def test_explicit_ao_potential_has_symmetric_coordinate_convention(hydrogen):
    coordinates, mf, _ = hydrogen
    engine = MolecularEnergy(mf)
    field = jnp.array([.001, -.002, .003])
    n = mf.mo_coeff.shape[0]
    potential = jnp.arange(n*n, dtype=jnp.float64).reshape(n, n)*1e-3
    baseline = engine.inputs(coordinates, field)
    changed = engine.inputs(coordinates, field, potential=potential)
    np.testing.assert_allclose(changed["hcore"]-baseline["hcore"], .5*(potential+potential.T), atol=1e-14)
    np.testing.assert_allclose(changed["nuclear_repulsion"], baseline["nuclear_repulsion"], atol=1e-14)
    antisymmetric = engine.inputs(coordinates, field, potential=potential-potential.T)
    np.testing.assert_allclose(antisymmetric["hcore"], baseline["hcore"], atol=1e-14)


@pytest.mark.parametrize("kind", ["shape", "complex"])
def test_explicit_ao_potential_rejects_invalid_shape_or_complex_values(hydrogen, kind):
    coordinates, mf, _ = hydrogen
    potential = (jnp.zeros((2, 2)) if kind == "shape"
                 else jnp.eye(mf.mo_coeff.shape[0], dtype=jnp.complex128))
    with pytest.raises(ValueError, match="potential"):
        MolecularEnergy(mf).inputs(coordinates, potential=potential)


def test_cis_geometry_rejects_unrestricted_orbital_arrays(hydrogen):
    _, mf, _ = hydrogen
    invalid = copy(mf)
    invalid.mo_coeff = np.stack([mf.mo_coeff]*2)
    invalid.mo_occ = np.stack([np.asarray(mf.mo_occ)/2]*2)
    with pytest.raises(NotImplementedError, match="RHF"):
        CISGeometry(invalid, nroots=2)


def _pyscf_energies(coordinates, field, potential=None):
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf
    from scipy.linalg import eigh

    mol = pgto.M(atom=list(zip(("H", "H"), coordinates)), basis="6-31g*",
                 unit="Bohr", cart=True, verbose=0)
    mf = pscf.RHF(mol)
    core = mf.get_hcore()+np.einsum("x,xpq->pq", field, mol.intor("int1e_r", comp=3))
    if potential is not None:
        core = core+.5*(potential+potential.T)
    nuclear = mol.energy_nuc()-np.asarray(field)@(mol.atom_charges()@coordinates)
    mf.get_hcore = lambda mol=None: core
    mf.energy_nuc = lambda *args, **kwargs: nuclear
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-12, 180
    mf.kernel()
    assert mf.converged
    a, _ = mf.TDA().get_ab()
    roots = eigh(a.reshape(a.shape[0]*a.shape[1], -1), eigvals_only=True)
    return np.array([mf.e_tot, mf.e_tot+roots[0]])


def test_relaxed_cis_density_has_particle_count_and_dipole_identity(hydrogen):
    coordinates, mf, model = hydrogen
    field = np.array([0., 0., .002])
    density, dipole, state = model.density_and_dipole(coordinates, 0, field=field)
    overlap = model.energy.inputs(coordinates, field)["overlap"]
    position = model.energy.plan.evaluate("dipole", model.energy.parameters, origin=jnp.zeros(3))
    coefficients = np.asarray(state.coefficients[:, :1])
    ground_density = 2*coefficients@coefficients.T
    np.testing.assert_allclose(density, density.T, atol=1e-11)
    np.testing.assert_allclose(np.einsum("pq,qp", density, overlap), 2., atol=1e-8)
    np.testing.assert_allclose(np.einsum("pq,qp", density-ground_density, overlap), 0., atol=1e-8)
    expected_dipole = coordinates.sum(axis=0)-np.einsum("pq,xqp->x", density, position)
    np.testing.assert_allclose(dipole, expected_dipole, atol=1e-9)
    step = 1e-4
    energies, excitations = [], []
    for sign in (1, -1):
        moved = model.state(coordinates, field=field, potential=sign*step*overlap)
        energies.append(float(moved.ground_energy+moved.excitation_energies[0]))
        excitations.append(float(moved.excitation_energies[0]))
    np.testing.assert_allclose((energies[0]-energies[1])/(2*step), 2., atol=2e-8)
    np.testing.assert_allclose((excitations[0]-excitations[1])/(2*step), 0., atol=2e-8)


def test_relaxed_cis_density_and_dipole_match_independent_finite_perturbations(hydrogen):
    coordinates, _, model = hydrogen
    field = np.array([0., 0., .002])
    density, dipole, _ = model.density_and_dipole(coordinates, 0, field=field)
    step = 3e-5
    reference_dipole = np.array([-(
        _pyscf_energies(coordinates, field+step*axis)[1]
        -_pyscf_energies(coordinates, field-step*axis)[1])/(2*step) for axis in np.eye(3)])
    np.testing.assert_allclose(dipole, reference_dipole, atol=2e-6, rtol=0)
    rng = np.random.default_rng(104)
    for _ in range(2):
        direction = rng.normal(size=density.shape)
        direction = .5*(direction+direction.T)
        direction /= np.linalg.norm(direction)
        expected = (_pyscf_energies(coordinates, field, step*direction)[1]
                    -_pyscf_energies(coordinates, field, -step*direction)[1])/(2*step)
        np.testing.assert_allclose(np.sum(density*direction), expected, atol=2e-6, rtol=0)
