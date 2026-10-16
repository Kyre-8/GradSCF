"""Single-q factors and exact streamed periodic screening against dense controls.

The degenerate synthetic spectrum is a negative QP-convergence control for
the declared finite-eta contour prescription. It must retain failure flags;
strict converged-QP accuracy is checked separately on a real periodic HF.
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


def toy(nk=3):
    path = Path(__file__).with_name("test_kpoint_invariants.py")
    spec = importlib.util.spec_from_file_location("kpoint_streaming_fixture", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    value = helper._toy(nk=nk)
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
    data = toy()
    dense = driver.g0w0_cd_kpoints(**data, fc=request.param, screening_backend="dense")
    streamed = driver.g0w0_cd_kpoints(**data, fc=request.param, screening_backend="transition")
    return data, request.param, dense, streamed


def test_streamed_toy_preserves_dense_nonconvergence(toy_comparison):
    _, _, dense, streamed = toy_comparison
    assert not bool(dense.converged) and not bool(streamed.converged)
    np.testing.assert_array_equal(streamed.converged_mask, dense.converged_mask)
    assert np.max(abs(dense.qp_residual)) > 2e-8
    np.testing.assert_allclose(streamed.mo_energy, dense.mo_energy, atol=2e-9, rtol=0)
    np.testing.assert_allclose(streamed.qp_residual, dense.qp_residual, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(streamed.qp_computed_mask, np.ones((3, 2), dtype=bool))
    np.testing.assert_array_equal(streamed.screening_energy, dense.screening_energy)


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
    assert not bool(window.converged) and not bool(window.converged_mask[1, 1])
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


def test_new_builder_and_array_driver_reject_reduced_dimensional_inputs():
    product, driver = modules()
    with pytest.raises(NotImplementedError, match="three-dimensional"):
        product.kpoint_product_factor_builder(SimpleNamespace(boundary=object()), None,
            mesh=(3, 3, 3), momentum_table=None)
    data = toy()
    data["inputs"].boundary = object()
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
    return dict(inputs=mean_field.inputs,
        kpts_frac=np.asarray(mean_field.kpts)@np.asarray(cell.lattice).T/(2*np.pi),
        mo_energy_k=result.mo_energy_spin[0], mo_coeff_k=result.mo_coeff_spin[0], nocc=1,
        fock_k=result.fock_spin[0], hcore_k=mean_field.inputs.hcore,
        density_spin=result.density_spin, mesh=(7, 7, 7), nw=8)


@pytest.mark.parametrize("fc", [False, True])
def test_small_real_periodic_qp_and_residual_match_dense(real_small_periodic_reference, fc):
    _, driver = modules()
    data = real_small_periodic_reference
    dense = driver.g0w0_cd_kpoints(**data, fc=fc, screening_backend="dense")
    actual = driver.g0w0_cd_kpoints(**data, fc=fc, screening_backend="transition")
    assert bool(dense.converged) and bool(actual.converged)
    np.testing.assert_allclose(actual.mo_energy, dense.mo_energy, atol=2e-8, rtol=0)
    np.testing.assert_allclose(actual.qp_residual, dense.qp_residual, atol=2e-8, rtol=0)
    assert np.max(abs(actual.qp_residual)) < 2e-8
