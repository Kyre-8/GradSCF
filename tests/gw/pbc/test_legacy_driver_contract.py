"""Periodic legacy drivers must preserve the shared QP validity contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.pbc.krgw import g0w0_cd_gamma, evgw_cd_gamma
from gradscf.gw.pbc.kugw import g0w0_cd_gamma_unrestricted
from gradscf.gw.pbc import ksigma
from gradscf.gw.pbc import krgw
from gradscf.pbc import gto, scf


@pytest.mark.parametrize('backend', ['dense', 'transition'])
def test_single_k_endpoint_matches_gamma_static_contour(backend, monkeypatch):
    from test_kpoint_invariants import _toy
    from gradscf.gw.qp import QPRootResult

    data = _toy(nk=1, paired=True)
    # Give this contour integration test a nonzero own-G-pole coupling.
    # The cosine/sine basis otherwise hides the boundary through parity.
    # All three factor paths use exactly the same paired real RPA model.
    factors = jnp.zeros((9, 2, 2), dtype=jnp.complex128).at[1].set(
        jnp.array([[.27, .18], [.18, .31]]))
    monkeypatch.setattr(krgw, 'gamma_product_factors', lambda *a, **k: factors)
    monkeypatch.setattr(ksigma, 'kpoint_product_factors', lambda *a, **k: [factors[None]])
    def builder(*a, **k):
        def build(q, *, pairs, k_indices=None):
            indices = [0] if k_indices is None else k_indices
            return jnp.stack([factors[:, pairs[:, 0], pairs[:, 1]] for _ in indices])
        return build
    monkeypatch.setattr(ksigma, 'kpoint_product_factor_builder', builder)
    def at_reference(function, reference, **kwargs):
        residual = function(reference)
        invalid = jnp.zeros(reference.shape, dtype=bool)
        return QPRootResult(reference, residual, invalid,
            jnp.full_like(reference, jnp.nan), reference-residual,
            jnp.full_like(reference, jnp.nan), jnp.zeros(reference.shape, dtype=jnp.int32), invalid)
    monkeypatch.setattr(ksigma, 'solve_quasiparticle_roots', at_reference)
    result = ksigma.g0w0_cd_kpoints(**data, eta=0., fc=False, screening_backend=backend)
    gamma = g0w0_cd_gamma(inputs=data['inputs'], mo_energy=data['mo_energy_k'][0],
        mo_coeff=data['mo_coeff_k'][0], nocc=1, fock_matrix=data['fock_k'][0],
        hcore_matrix=data['hcore_k'][0], density_spin=data['density_spin'][:, 0],
        mesh=data['mesh'], nw=data['nw'], eta=0., fc=False, evaluate_only=True)
    assert not bool(result.converged)
    np.testing.assert_array_equal(result.mo_energy, data['mo_energy_k'])
    # This HF toy has delta_v=0, so F(e_p)=-Re Sigma(e_p).
    np.testing.assert_allclose(-result.qp_residual[0], gamma.sigma_qp.real, atol=2e-12, rtol=0)


@pytest.fixture(scope="module")
def reference():
    cell = gto.M(atom="H 3 3 2.3; H 3 3 3.7", a=np.eye(3)*6., unit="Bohr",
                 basis="gth-szv", pseudo="gth-pade", mesh=(7, 7, 7), precision=1e-10)
    mf = scf.KRHF(cell, kpts=np.zeros((1, 3))).run()
    assert mf.converged
    result = mf.result
    return dict(inputs=mf.inputs, kpts_frac=np.zeros((1, 3)),
                mo_energy_k=result.mo_energy_spin[0], mo_coeff_k=result.mo_coeff_spin[0],
                nocc=1, fock_k=result.fock_spin[0], hcore_k=mf.inputs.hcore,
                density_spin=result.density_spin, mesh=(7, 7, 7), nw=8)


def test_gamma_result_retains_weights_coverage_and_screening(reference):
    data = reference
    result = g0w0_cd_gamma(inputs=data["inputs"], mo_energy=data["mo_energy_k"][0],
        mo_coeff=data["mo_coeff_k"][0], nocc=1, fock_matrix=data["fock_k"][0],
        hcore_matrix=data["hcore_k"][0], density_spin=data["density_spin"][:, 0],
        mesh=data["mesh"], nw=8, fc=False, orbs=(0,))
    assert result.qp_weight is not None
    assert result.qp_branch_valid is not None
    assert result.qp_candidate_count is not None
    np.testing.assert_array_equal(result.qp_computed_mask, [True, False])
    np.testing.assert_array_equal(result.screening_energy, data["mo_energy_k"][0])
    assert jnp.isnan(result.qp_weight[1])


def test_unrestricted_gamma_result_retains_weights_coverage_and_screening(reference):
    data = reference
    energy, coefficient = data["mo_energy_k"][0], data["mo_coeff_k"][0]
    result = g0w0_cd_gamma_unrestricted(inputs=data["inputs"], mo_energy=(energy, energy),
        mo_coeff=(coefficient, coefficient), nocc=(1, 1),
        fock_matrix=(data["fock_k"][0], data["fock_k"][0]),
        hcore_matrix=data["hcore_k"][0], density_spin=data["density_spin"][:, 0],
        mesh=data["mesh"], nw=8, orbs=(0,))
    assert result.qp_weight is not None
    assert result.qp_branch_valid is not None
    np.testing.assert_array_equal(result.qp_computed_mask, [[True, False], [True, False]])
    np.testing.assert_array_equal(result.screening_energy, jnp.stack((energy, energy)))
    assert jnp.isnan(result.qp_weight[:, 1]).all()


@pytest.mark.parametrize("occupations", [(1, 0), (2, 1)])
def test_unrestricted_gamma_preserves_empty_and_full_spin_contours(reference, occupations, monkeypatch):
    from gradscf.integrals.periodic.fft import get_jk
    from gradscf.gw.pbc import kugw

    data = reference
    coefficient = data["mo_coeff_k"][0]
    density = jnp.stack([coefficient[:, :count] @ coefficient[:, :count].conj().T
                         for count in occupations])
    hartree, exchange = get_jk(data["inputs"], density, exxdiv="ewald", with_k=True)
    fock = data["hcore_k"][0]+hartree-exchange
    projected = coefficient.conj().T @ fock @ coefficient
    # Parity keeps these new spin Fock matrices diagonal in the same two MOs.
    energies = jnp.diagonal(projected, axis1=-2, axis2=-1).real
    np.testing.assert_allclose(projected, jax.vmap(jnp.diag)(energies), atol=1e-9, rtol=0)
    original, recorded = kugw._qp_loop, []

    def record(**kwargs):
        recorded.append((kwargs["mo_energy"], kwargs["ef"], kwargs["nocc"]))
        return original(**kwargs)

    monkeypatch.setattr(kugw, "_qp_loop", record)
    result = g0w0_cd_gamma_unrestricted(inputs=data["inputs"],
        mo_energy=(energies[0], energies[1]), mo_coeff=(coefficient, coefficient),
        nocc=occupations, fock_matrix=(fock[0], fock[1]), hcore_matrix=data["hcore_k"][0],
        density_spin=density, mesh=data["mesh"], nw=8, orbs=(0,))
    assert jnp.isfinite(result.mo_energy).all()
    for energy, fermi, occupied in recorded:
        if occupied == 0:
            assert fermi < jnp.min(energy)
        elif occupied == len(energy):
            assert fermi > jnp.max(energy)


def test_evgw_gamma_records_its_outer_method_and_slope_certificate(reference):
    data = reference
    result = evgw_cd_gamma(inputs=data["inputs"], mo_energy=data["mo_energy_k"][0],
        mo_coeff=data["mo_coeff_k"][0], nocc=1, fock_matrix=data["fock_k"][0],
        hcore_matrix=data["hcore_k"][0], density_spin=data["density_spin"][:, 0],
        mesh=data["mesh"], nw=8, fc=False, max_iter=30, tol=1e-7)
    assert result.converged
    assert result.method == "periodic_gamma_evgw"
    assert jnp.all(result.qp_weight > 0)
    np.testing.assert_array_equal(result.qp_candidate_count, [0, 0])
    np.testing.assert_array_equal(result.screening_energy, result.mo_energy)


def test_gamma_eigenvalue_update_keeps_the_mean_field_q0_vertex(reference, monkeypatch):
    data = reference
    energy = data["mo_energy_k"][0]
    original = krgw.transition_moments_gamma
    recorded = []

    def record(inputs, spectrum, coefficient, occupied):
        recorded.append(np.asarray(spectrum))
        return original(inputs, spectrum, coefficient, occupied)

    monkeypatch.setattr(krgw, "transition_moments_gamma", record)
    g0w0_cd_gamma(inputs=data["inputs"], mo_energy=energy,
        mo_energy_poles=energy+jnp.array([-.03, .02]),
        mo_coeff=data["mo_coeff_k"][0], nocc=1, fock_matrix=data["fock_k"][0],
        hcore_matrix=data["hcore_k"][0], density_spin=data["density_spin"][:, 0],
        mesh=data["mesh"], nw=2, fc=True, evaluate_only=True, orbs=(0,))
    assert len(recorded) == 1
    np.testing.assert_array_equal(recorded[0], energy)


def test_kpoint_result_rejects_a_negative_fixed_gw_weight(reference, monkeypatch):
    # This controlled self-energy has a regular zero with F'=-1, Z=-1.
    monkeypatch.setattr(ksigma, "sigma_imag_part", lambda omega, *args, **kwargs: 2*omega)
    monkeypatch.setattr(ksigma, "_sigma_residue_kpoint", lambda *args, **kwargs: 0.)
    result = ksigma.g0w0_cd_kpoints(**reference, fc=False, orbs=(0,), screening_backend="dense")
    assert not bool(result.converged)
    assert not bool(result.converged_mask[0, 0])
    assert result.qp_weight is not None
    assert result.qp_weight[0, 0] < 0
    assert not bool(result.qp_branch_valid[0, 0])


def test_gamma_real_spectrum_response_preserves_the_paired_manifold(reference):
    data = reference
    coefficient, energy = data["mo_coeff_k"][0], data["mo_energy_k"][0]
    overlap = jnp.asarray(data["inputs"].overlap[0])
    direction = jnp.array([.03, -.02])
    perturbation = overlap @ coefficient @ jnp.diag(direction) @ coefficient.conj().T @ overlap

    def solve(t):
        result = g0w0_cd_gamma(inputs=data["inputs"], mo_energy=energy+t*direction,
            mo_coeff=coefficient, nocc=1, fock_matrix=data["fock_k"][0]+t*perturbation,
            hcore_matrix=data["hcore_k"][0]+t*perturbation,
            density_spin=data["density_spin"][:, 0], mesh=data["mesh"], nw=8, fc=False)
        return result.mo_energy

    step = 1e-4
    finite_difference = (solve(step)-solve(-step))/(2*step)
    value, tangent = jax.jvp(solve, (jnp.array(0.),), (jnp.array(1.),))
    assert jnp.isfinite(value).all() and jnp.isfinite(tangent).all()
    np.testing.assert_allclose(tangent, finite_difference, atol=2e-8, rtol=3e-5)


@pytest.mark.parametrize("driver", ["gamma", "kpoints", "ugamma"])
def test_legacy_driver_rejects_unpaired_complex_transition_covariances(reference, driver):
    # A complex occupied/virtual mixture changes their real-space projectors;
    # the VO and OV transition covariances need not coincide even at Gamma.
    angle = .3
    rotation = jnp.array([[jnp.cos(angle), 1j*jnp.sin(angle)],
                          [1j*jnp.sin(angle), jnp.cos(angle)]])
    coefficients = reference["mo_coeff_k"] @ rotation
    # Keep the supplied HF eigenproblem consistent in the nonorthogonal AO
    # metric, with a complex one-electron Hamiltonian breaking time reversal.
    from gradscf.integrals.periodic.fft import get_kpoint_jk
    density = coefficients[:, :, :1] @ coefficients[:, :, :1].conj().transpose(0, 2, 1)
    density = jnp.stack((density, density))
    hartree, exchange = get_kpoint_jk(reference["inputs"], density,
                                    mesh=reference["mesh"], exxdiv="ewald", with_k=True)
    overlap = jnp.asarray(reference["inputs"].overlap)
    fock = overlap @ coefficients @ jnp.diag(reference["mo_energy_k"][0])
    fock = fock @ coefficients.conj().transpose(0, 2, 1) @ overlap
    core = fock-hartree+exchange[0]
    with pytest.raises(NotImplementedError, match="paired|Lehmann|full.*query"):
        if driver == "gamma":
            g0w0_cd_gamma(inputs=reference["inputs"], mo_energy=reference["mo_energy_k"][0],
                mo_coeff=coefficients[0], nocc=1, fock_matrix=fock[0], hcore_matrix=core[0],
                density_spin=density[:, 0], mesh=reference["mesh"], nw=8, fc=False, orbs=(0,))
        elif driver == "ugamma":
            energy = reference["mo_energy_k"][0]
            g0w0_cd_gamma_unrestricted(inputs=reference["inputs"], mo_energy=(energy, energy),
                mo_coeff=(coefficients[0], coefficients[0]), nocc=(1, 1),
                fock_matrix=(fock[0], fock[0]), hcore_matrix=core[0], density_spin=density[:, 0],
                mesh=reference["mesh"], nw=8, orbs=(0,))
        else:
            ksigma.g0w0_cd_kpoints(**{**reference, "mo_coeff_k": coefficients,
                                     "fock_k": fock, "hcore_k": core, "density_spin": density},
                                  fc=False, orbs=(0,), screening_backend="transition")
