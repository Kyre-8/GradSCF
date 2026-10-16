"""Original fixed-omega RKS singlet response against PySCF full RSH."""
from types import SimpleNamespace
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto, tdscf
from gradscf.tddft.response import _restricted_ao_response_action


def test_rsh_ao_action_keeps_full_coulomb_and_both_exchange_channels():
    rng = np.random.default_rng(173)
    eri = jnp.asarray(rng.normal(size=(3, 3, 3, 3)))
    eri_lr = jnp.asarray(rng.normal(size=(3, 3, 3, 3)))
    density = jnp.asarray(rng.normal(size=(2, 3, 3)))
    sr, lr = .222036, 1.

    def action(scale):
        molecule = SimpleNamespace(rep_tensor=eri, eri_lr=scale*eri_lr,
            dispersion_energy=jnp.asarray(0.), range_separated_coefficients=(.2, lr, sr))
        return _restricted_ao_response_action(molecule, sr, include_exchange=True,
                                              dtype=jnp.float64)(density)

    coulomb = jnp.einsum("pqrs,nrs->npq", eri, density)
    exchange = jnp.einsum("prqs,nrs->npq", sr*eri+(lr-sr)*eri_lr, density)
    np.testing.assert_allclose(action(1.), coulomb-.5*exchange, rtol=2e-14, atol=2e-14)
    tangent = jax.jvp(action, (jnp.asarray(1.),), (jnp.asarray(1.),))[1]
    expected = -.5*(lr-sr)*jnp.einsum("prqs,nrs->npq", eri_lr, density)
    np.testing.assert_allclose(tangent, expected, rtol=2e-14, atol=2e-14)


@pytest.fixture(scope="module")
def water():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft as pyscf_dft
    atom = "O 0 0 0; H 0 .76 .58; H 0 -.76 .58"
    reference_molecule = pyscf.gto.M(atom=atom, basis="sto-3g", cart=True, verbose=0)
    reference = pyscf_dft.RKS(reference_molecule)
    reference.xc = "wb97xd"
    reference.grids.level = 0
    reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-9
    reference.kernel()
    assert reference.converged
    molecule = gto.M(atom=atom, basis="sto-3g")
    mean_field = dft.RKS(molecule, xc="wb97xd", grids_level=0,
                         conv_tol=1e-11, conv_tol_grad=1e-8,
                         init_guess=reference.make_rdm1()).run()
    assert mean_field.converged
    return mean_field, reference


def test_wb97xd_singlet_a_b_matrices_match_pyscf(water):
    mean_field, reference = water
    expected_a, expected_b = reference.TDDFT().get_ab()
    dimension = int(np.prod(expected_a.shape[:2]))
    solver = tdscf.TDDFT(mean_field)._build_solver()
    matrix = np.asarray(solver.gen_tdhf_vind()(jnp.eye(2*dimension))).T
    a, b = matrix[:dimension, :dimension], matrix[:dimension, dimension:]
    # Independent SCF diagonalizations may choose opposite real MO phases.
    # Transform the complete A/B matrices to the reference MO gauge, rather
    # than comparing absolute matrix elements and losing relative signs.
    overlap = reference.mo_coeff.T @ reference.get_ovlp() @ np.asarray(mean_field.mo_coeff)
    signs = np.sign(np.diag(overlap))
    np.testing.assert_allclose(overlap, np.diag(signs), atol=2e-8, rtol=0)
    nocc = expected_a.shape[0]
    phase = (signs[:nocc, None]*signs[None, nocc:]).reshape(-1)
    a, b = phase[:, None]*a*phase[None, :], phase[:, None]*b*phase[None, :]
    a_error = np.max(np.abs(a-expected_a.reshape(dimension, dimension)))
    b_error = np.max(np.abs(b-expected_b.reshape(dimension, dimension)))
    print("wb97xd singlet A/B max errors Ha", a_error, b_error)
    np.testing.assert_allclose(a, expected_a.reshape(dimension, dimension), rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(b, expected_b.reshape(dimension, dimension), rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("method", ["TDA", "TDDFT"])
def test_wb97xd_singlet_roots_and_oscillators_match_pyscf(water, method):
    mean_field, reference = water
    expected = getattr(reference, method)()
    expected.nstates, expected.conv_tol = 3, 1e-10
    expected.kernel()
    assert np.asarray(expected.converged).all()
    actual = getattr(tdscf, method)(mean_field, nstates=3, davidson_tol=1e-10,
                                   davidson_max_iter=100)
    actual.kernel()
    assert actual.converged
    print("wb97xd", method, "roots Ha", np.asarray(actual.e).tolist(),
          "oscillators", np.asarray(actual.oscillator_strength()).tolist())
    np.testing.assert_allclose(actual.e, expected.e, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(actual.oscillator_strength(), expected.oscillator_strength(),
                               atol=2e-7, rtol=2e-6)


def test_rsh_td_reference_keeps_other_response_boundaries_closed(water):
    from gradscf.scf.reference import as_reference, reference_state_signature
    from gradscf.tddft.unrestricted import build_unrestricted_tda_operator

    mean_field, _ = water
    solved_state = mean_field.scf_result
    reference = tdscf.TDA(mean_field).molecule
    assert mean_field.scf_result is solved_state
    for operation in (mean_field.to_reference, lambda: as_reference(reference),
                      lambda: reference_state_signature(mean_field),
                      lambda: build_unrestricted_tda_operator(reference)):
        with pytest.raises(NotImplementedError, match="(?i)range.separated"):
            operation()
    triplet = tdscf.TDA(mean_field)
    triplet.singlet = False
    with pytest.raises(NotImplementedError, match="triplet"):
        triplet.kernel()
    with pytest.raises(NotImplementedError, match="complete wb97xd"):
        tdscf.TDA(mean_field, xc_functional="hyb_gga_xc_wb97x_d").kernel()


@pytest.mark.parametrize("mode", ["df", "ris"])
def test_rsh_td_rejects_unimplemented_two_electron_backends(water, mode):
    with pytest.raises(NotImplementedError, match="full AO ERIs"):
        tdscf.TDA(water[0], response_kernel_options={"two_electron_mode": mode}).kernel()


@pytest.mark.parametrize("field", ["eri_lr", "dispersion_energy"])
def test_rsh_td_rejects_incomplete_array_reference(water, field):
    reference = tdscf.TDA(water[0]).molecule
    with pytest.raises(ValueError, match=field):
        tdscf.TDA(replace(reference, **{field: None})).kernel()


def test_rsh_local_response_cutoff_and_zero_gradient_are_finite(water):
    from gradscf.tddft._semilocal_response import SemilocalResponseFunctional

    reference = tdscf.TDA(water[0]).molecule
    functional = SemilocalResponseFunctional("wb97xd")
    inactive = replace(reference, rdm1=jnp.zeros_like(reference.rdm1))
    np.testing.assert_array_equal(functional.grid_response_tensor(inactive), 0.)
    constant_ao = jnp.ones_like(reference.ao)
    derivative = jnp.zeros_like(reference.ao_deriv1).at[0].set(constant_ao)
    constant = replace(reference, ao=constant_ao, ao_deriv1=derivative)
    tensor = functional.grid_response_tensor(constant)
    assert np.isfinite(tensor).all()
    assert float(jnp.max(jnp.abs(tensor))) > 1e-7


@pytest.mark.parametrize("method", ["TDA", "TDDFT"])
def test_rsh_td_root_keeps_lr_tensor_differentiable(water, method):
    reference = tdscf.TDA(water[0]).molecule

    def energy(scale):
        state = replace(reference, eri_lr=scale*reference.eri_lr)
        return getattr(tdscf, method)(state, nstates=1,
                                      davidson_tol=1e-10).kernel().excitation_energies[0]

    derivative = jax.jit(jax.grad(energy))(jnp.asarray(1.))
    compiled = jax.jit(energy)
    step = 2e-4
    finite = (compiled(1.+step)-compiled(1.-step))/(2*step)
    assert abs(float(finite)) > 1e-6
    print("wb97xd", method, "LR root derivative and AD-FD", float(derivative), float(derivative-finite))
    np.testing.assert_allclose(derivative, finite, rtol=2e-5, atol=2e-7)
