"""Clamped-nuclei total excited-state field response before formaldehyde X08."""
import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry


def test_h2_total_excited_polarizability_matches_independent_finite_fields():
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf
    from scipy.linalg import eigh

    coordinates = np.array([[0., -.4, -.4], [.4, .2, 1.]])
    mol = gto.M(atom=list(zip(("H", "H"), coordinates)), basis="6-31g*", unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-11).run()
    assert mf.converged
    model = CISGeometry(mf, nroots=2, gradient_tolerance=1e-11,
        response=EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", max_dense=8, rtol=1e-11)))
    field = np.array([.0002, -.0001, .0004])
    dipole, alpha, state = model.electric_response(coordinates, 0, field=field)
    np.testing.assert_allclose(alpha, alpha.T, atol=2e-9, rtol=0)
    assert np.linalg.norm(alpha) > .1
    step = 3e-4
    native_fd = np.column_stack([(model.density_and_dipole(coordinates, 0, field=field+step*axis)[1]
        -model.density_and_dipole(coordinates, 0, field=field-step*axis)[1])/(2*step) for axis in np.eye(3)])
    np.testing.assert_allclose(alpha, native_fd, atol=2e-4, rtol=0)

    def energy(current):
        reference_mol = pgto.M(atom=list(zip(("H", "H"), coordinates)), basis="6-31g*", unit="Bohr", cart=True, verbose=0)
        reference = pscf.RHF(reference_mol)
        position = reference_mol.intor("int1e_r", comp=3)
        hcore = reference.get_hcore()+np.einsum("x,xpq->pq", current, position)
        enuc = reference_mol.energy_nuc()-current@(reference_mol.atom_charges()@coordinates)
        reference.get_hcore = lambda mol=None: hcore
        reference.energy_nuc = lambda *args, **kwargs: enuc
        reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-12
        reference.kernel()
        assert reference.converged
        matrix, _ = reference.TDA().get_ab()
        values = eigh(matrix.reshape(3, 3), eigvals_only=True)
        assert values[1]-values[0] > .01
        return reference.e_tot+values[0]

    e0 = energy(field)
    np.testing.assert_allclose(e0, state.ground_energy+state.excitation_energies[0], atol=1e-8, rtol=0)
    moment_fd = np.array([-(energy(field+1e-5*v)-energy(field-1e-5*v))/(2e-5) for v in np.eye(3)])
    np.testing.assert_allclose(dipole, moment_fd, atol=2e-6, rtol=0)
    reference_alpha = np.zeros((3, 3))
    for i, axis_i in enumerate(np.eye(3)):
        reference_alpha[i, i] = -(energy(field+step*axis_i)+energy(field-step*axis_i)-2*e0)/step**2
        for j, axis_j in enumerate(np.eye(3)[:i]):
            mixed = sum(si*sj*energy(field+step*(si*axis_i+sj*axis_j)) for si in (1, -1) for sj in (1, -1))
            reference_alpha[i, j] = reference_alpha[j, i] = -mixed/(4*step**2)
    np.testing.assert_allclose(alpha, reference_alpha, atol=2e-4, rtol=0)
