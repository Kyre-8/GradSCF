"""Single-q factors and exact streamed periodic screening against dense controls.

The paired synthetic model checks exact dense/transition algebra and
certified roots. Non-TR rejection and negative-weight roots are covered by
the legacy-driver contract tests; a separate native control checks full chi.
"""
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def modules():
    product = importlib.import_module("gradscf.gw.pbc.product_basis")
    driver = importlib.import_module("gradscf.gw.pbc.ksigma")
    assert hasattr(product, "kpoint_product_factor_builder"), "Single-q selected-pair builder is not implemented"
    assert "screening_backend" in inspect.signature(driver.g0w0_cd_kpoints).parameters, "Streamed periodic screening is not implemented"
    return product, driver


def toy(nk=3, *, paired=False):
    path = Path(__file__).with_name("test_kpoint_invariants.py")
    spec = importlib.util.spec_from_file_location("kpoint_streaming_fixture", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    value = helper._toy(nk=nk, paired=paired)
    value["nw"] = 8
    return value


def test_selected_pairs_and_k_indices_match_existing_full_factors():
    product, _ = modules()
    from gradscf.pbc.momentum import momentum_transfer_table
    data = toy()
    table = momentum_transfer_table(data["kpts_frac"])
    full = product.kpoint_product_factors(data["inputs"], data["mo_coeff_k"], mesh=data["mesh"], momentum_table=table)
    build = product.kpoint_product_factor_builder(data["inputs"], data["mo_coeff_k"],
        mesh=data["mesh"], momentum_table=table)
    pairs, indices = [(0, 1), (1, 0), (1, 1)], [2, 0]
    actual = build(1, pairs=pairs, k_indices=indices)
    expected = np.stack([np.stack([np.asarray(full[1])[ki, :, m, n] for m, n in pairs], axis=-1)
                         for ki in indices])
    assert actual.shape == (2, 9, 3)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)


def test_pair_byte_guard_runs_before_fft(monkeypatch):
    product, _ = modules()
    from gradscf.pbc.momentum import momentum_transfer_table
    data = toy()
    build = product.kpoint_product_factor_builder(data["inputs"], data["mo_coeff_k"], mesh=data["mesh"],
        momentum_table=momentum_transfer_table(data["kpts_frac"]), max_block_bytes=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("FFT was evaluated before its array-byte guard")
    monkeypatch.setattr(jnp.fft, "fftn", forbidden)
    with pytest.raises(ValueError, match="byte"):
        build(0, pairs=[(0, 1)], k_indices=[0])


def test_shared_transition_block_keeps_full_k_ov_order():
    product, _ = modules()
    from gradscf.pbc.momentum import momentum_transfer_table
    data = toy()
    table = momentum_transfer_table(data["kpts_frac"])
    build = product.kpoint_product_factor_builder(data["inputs"], data["mo_coeff_k"],
        mesh=data["mesh"], momentum_table=table)
    energy = np.array([[-.7, .6], [-.6, .8], [-.8, .5]])
    block = product.kpoint_transition_block(build, energy, 1, table, 1)
    full = product.kpoint_product_factors(data["inputs"], data["mo_coeff_k"], mesh=data["mesh"], momentum_table=table)
    expected = np.concatenate([np.asarray(full[1])[ki, :, :1, 1:].reshape(9, -1) for ki in range(3)], axis=1)
    np.testing.assert_allclose(block["factors"], expected, atol=2e-13)
    np.testing.assert_allclose(block["gram"], expected.conj().T@expected, atol=2e-13)
    np.testing.assert_allclose(block["differences"], [energy[ki, 0]-energy[table[ki, 1], 1] for ki in range(3)], atol=0, rtol=0)


def test_driver_cache_guard_runs_before_exchange_or_product_build(monkeypatch):
    _, driver = modules()
    data = toy()
    def forbidden(*args, **kwargs):
        raise AssertionError("A tensor builder ran before the memory guard")
    monkeypatch.setattr(driver, "get_kpoint_jk", forbidden)
    monkeypatch.setattr(driver, "kpoint_product_factor_builder", forbidden)
    with pytest.raises(ValueError, match="memory|byte"):
        driver.g0w0_cd_kpoints(**data, screening_backend="transition", max_memory_bytes=1)


@pytest.fixture(scope="module", params=[False, True], ids=["body", "head_wing"])
def toy_comparison(request):
    _, driver = modules()
    data = toy(paired=True)
    dense = driver.g0w0_cd_kpoints(**data, fc=request.param, screening_backend="dense")
    streamed = driver.g0w0_cd_kpoints(**data, fc=request.param, screening_backend="transition")
    return data, request.param, dense, streamed


def test_streamed_toy_matches_dense_roots_and_status(toy_comparison):
    _, _, dense, streamed = toy_comparison
    assert bool(dense.converged) and bool(streamed.converged)
    np.testing.assert_array_equal(streamed.converged_mask, dense.converged_mask)
    assert np.max(abs(dense.qp_residual)) < 2e-8
    assert np.max(abs(streamed.qp_residual)) < 2e-8
    np.testing.assert_allclose(streamed.mo_energy, dense.mo_energy, atol=2e-9, rtol=0)
    np.testing.assert_allclose(streamed.qp_residual, dense.qp_residual, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(streamed.qp_computed_mask, np.ones((3, 2), dtype=bool))
    np.testing.assert_array_equal(streamed.screening_energy, dense.screening_energy)
    assert np.all(np.isfinite(dense.qp_weight)) and np.all(np.asarray(dense.qp_weight) > 0)
    assert np.all(np.isfinite(streamed.qp_weight)) and np.all(np.asarray(streamed.qp_weight) > 0)
    np.testing.assert_allclose(streamed.qp_weight, dense.qp_weight, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(streamed.qp_branch_valid, np.ones((3, 2), dtype=bool))
    np.testing.assert_array_equal(streamed.qp_branch_valid, dense.qp_branch_valid)
    np.testing.assert_array_equal(streamed.qp_candidate_count, dense.qp_candidate_count)


def test_qp_window_retains_full_internal_and_screening_spaces(toy_comparison, monkeypatch):
    _, driver = modules()
    data, fc, _, full = toy_comparison
    actual_builder = driver.kpoint_product_factor_builder
    calls = []
    def observed_builder(*args, **kwargs):
        build = actual_builder(*args, **kwargs)
        def selected(q, *, pairs, k_indices=None):
            calls.append((q, tuple(map(tuple, pairs)), tuple(k_indices)))
            return build(q, pairs=pairs, k_indices=k_indices)
        return selected
    def forbidden(*args, **kwargs):
        raise AssertionError("The streaming path called the all-q full-pair builder")
    monkeypatch.setattr(driver, "kpoint_product_factor_builder", observed_builder)
    monkeypatch.setattr(driver, "kpoint_product_factors", forbidden)
    window = driver.g0w0_cd_kpoints(**data, fc=fc, orbs=[1], qp_kpoints=[1], screening_backend="transition")
    assert bool(window.converged) and bool(window.converged_mask[1, 1])
    assert np.asarray(window.qp_branch_valid)[1, 1] and np.asarray(window.qp_weight)[1, 1] > 0
    assert abs(float(window.qp_residual[1, 1])) < 2e-8
    expected_mask = np.zeros((3, 2), dtype=bool)
    expected_mask[1, 1] = True
    np.testing.assert_array_equal(window.qp_computed_mask, expected_mask)
    np.testing.assert_allclose(window.mo_energy[1, 1], full.mo_energy[1, 1], atol=2e-9, rtol=0)
    np.testing.assert_allclose(np.asarray(window.mo_energy)[~expected_mask], np.asarray(data["mo_energy_k"])[~expected_mask], atol=0, rtol=0)
    assert window.g_orbitals == (0, 1)
    assert window.screening_occupied == (0,) and window.screening_virtual == (1,)
    assert {q for q, _, _ in calls} == {0, 1, 2}
    assert all(len(indices) == 1 and len(pairs) <= 2 for _, pairs, indices in calls)


def test_streamed_driver_preserves_complex_band_phases(toy_comparison):
    _, driver = modules()
    data, fc, _, baseline = toy_comparison
    phases = np.exp(1j*np.array([[.3, -.7], [.8, .2], [-.4, .1]]))
    changed = dict(data, mo_coeff_k=data["mo_coeff_k"]*phases[:, None, :])
    actual = driver.g0w0_cd_kpoints(**changed, fc=fc, screening_backend="transition")
    np.testing.assert_array_equal(actual.converged_mask, baseline.converged_mask)
    np.testing.assert_allclose(actual.mo_energy, baseline.mo_energy, atol=2e-9, rtol=0)


@pytest.mark.parametrize("name", ["gamma_product_factors", "kpoint_product_factors"])
def test_old_product_paths_reject_reduced_dimensional_input_before_arrays(name):
    product = importlib.import_module("gradscf.gw.pbc.product_basis")
    arguments = dict(mesh=(3, 3, 3))
    if name == "kpoint_product_factors":
        arguments["momentum_table"] = None
    with pytest.raises(NotImplementedError, match="three-dimensional"):
        getattr(product, name)(SimpleNamespace(boundary=object()), None, **arguments)


def test_builder_rejects_open_direction_kpoints_and_array_driver_requires_3d():
    product, driver = modules()
    from gradscf.integrals.periodic.coulomb import ReducedBoundary
    from gradscf.pbc.momentum import momentum_transfer_table

    data = toy()
    data["inputs"].boundary = ReducedBoundary(data["inputs"].lattice, 2, data["mesh"])
    data["inputs"].kpoints = data["inputs"].kpoints.at[0, 2].set(.01)
    with pytest.raises(ValueError, match="nonperiodic directions"):
        product.kpoint_product_factor_builder(data["inputs"], data["mo_coeff_k"],
            mesh=data["mesh"], momentum_table=momentum_transfer_table(data["kpts_frac"]))
    with pytest.raises(NotImplementedError, match="three-dimensional"):
        driver.g0w0_cd_kpoints(**data)


@pytest.fixture(scope="module")
def real_small_periodic_reference():
    modules()  # Missing implementation fails before a real SCF in the RED.
    from gradscf.pbc import gto, scf
    cell = gto.M(atom="H 3 3 2.3; H 3 3 3.7", a=np.eye(3)*6., unit="Bohr",
        basis="gth-szv", pseudo="gth-pade", mesh=(7, 7, 7), precision=1e-10)
    mean_field = scf.KRHF(cell, kpts=cell.make_kpts([2, 1, 1])).run()
    assert mean_field.converged
    result = mean_field.result
    data = dict(inputs=mean_field.inputs,
        kpts_frac=np.asarray(mean_field.kpts)@np.asarray(cell.lattice).T/(2*np.pi),
        mo_energy_k=result.mo_energy_spin[0], mo_coeff_k=result.mo_coeff_spin[0], nocc=1,
        fock_k=result.fock_spin[0], hcore_k=mean_field.inputs.hcore,
        density_spin=result.density_spin, mesh=(7, 7, 7), nw=8)
    return data, mean_field


@pytest.fixture(scope="module")
def real_small_periodic_pbe_reference(real_small_periodic_reference):
    from gradscf.pbc import dft
    _, hf = real_small_periodic_reference
    # This is a separate PBE SCF on exactly the HF cell/grid/k mesh. The
    # public full-response query currently supports pure LDA/GGA references.
    mean_field = dft.KRKS(hf.cell, kpts=hf.kpts, xc="pbe",
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150).run()
    assert mean_field.converged
    return mean_field


@pytest.mark.parametrize("fc", [False, True])
def test_small_real_periodic_unpaired_reference_uses_full_query(
        real_small_periodic_reference, real_small_periodic_pbe_reference, fc):
    from gradscf.pbc.momentum import momentum_transfer_table
    product, driver = modules()
    data, _ = real_small_periodic_reference
    # The finite shifted FFT cutoff breaks exact TR pairing in this native
    # HF model. Neither legacy backend may certify an even-W approximation.
    for backend in ("dense", "transition"):
        with pytest.raises(NotImplementedError, match="paired OV/VO"):
            driver.g0w0_cd_kpoints(**data, fc=fc, screening_backend=backend)

    table = np.asarray(momentum_transfer_table(data["kpts_frac"]))
    factors = product.kpoint_product_factors(data["inputs"], data["mo_coeff_k"],
                                            mesh=data["mesh"], momentum_table=table)
    energy = np.asarray(data["mo_energy_k"])
    nocc, nk, nmo = data["nocc"], len(energy), energy.shape[1]
    responses = []
    for frequency in (.37, -.37):
        expected = np.zeros((factors[0].shape[1],)*2, dtype=complex)
        for k, partner in enumerate(table[:, 0]):
            for m in range(nmo):
                for n in range(nmo):
                    sign = int(m < nocc)-int(n < nocc)
                    if sign:
                        vertex = np.asarray(factors[0][k, :, m, n])
                        weight = 2*sign/(nk*(1j*frequency+energy[k, m]-energy[partner, n]))
                        expected += weight*np.outer(vertex, vertex.conj())
        actual = driver._response_iw_kpoint(frequency, 0, factors, energy, nocc, table, nk)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
        responses.append(expected)
    np.testing.assert_allclose(responses[0], responses[1].conj().T, atol=2e-13, rtol=2e-12)
    assert not np.allclose(responses[0], responses[1], atol=1e-10, rtol=1e-10)

    from gradscf.gw.pbc import g0w0_bands
    from gradscf.solvers.nonlinear import ScalarRootConfig
    pbe = real_small_periodic_pbe_reference
    result = g0w0_bands(pbe, pbe.kpts, (0, 1), nw=8, pole_subtraction=True,
                       fc=fc, q0_order=2, self_energy="diagonal",
                       root_config=ScalarRootConfig(method="hybrid", maxiter=120,
                                                    ftol=1e-10, xtol=1e-10))
    assert result.converged
    assert np.all(np.asarray(result.qp_branch_valid))
    assert np.all(np.asarray(result.qp_weight) > 0)
    assert np.max(abs(np.asarray(result.qp_residual))) < 2e-8
