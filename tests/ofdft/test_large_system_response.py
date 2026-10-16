"""Fixed-resolution supercells and nontrivial WGC density-response gates."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('ase', reason='The Al supercell example uses ASE inputs')


def example():
    path = Path(__file__).resolve().parents[2]/'examples/ofdft/large_system_response.py'
    assert path.is_file(), 'The large-system OFDFT example must exist'
    spec = importlib.util.spec_from_file_location('examples.ofdft.large_system_response', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supercell_preserves_voxel_and_probe_normalization():
    module = example()
    for repeat in (1, 3):
        atoms, mesh, probes = module.crystal_probes(repeat)
        assert len(atoms) == repeat**3 and mesh == (17*repeat,)*3
        np.testing.assert_allclose(atoms.get_volume()/np.prod(mesh), 18.435/17**3,
                                   atol=1e-14, rtol=0)
        np.testing.assert_allclose(probes.mean(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(np.mean(probes**2, axis=0), 1., atol=1e-12)
        if repeat > 1:
            field = probes[:, 0].reshape(mesh)
            assert np.linalg.norm(field-np.roll(field, 17, axis=1)) > 1.


def test_stationarity_reports_energy_and_constraint_separately():
    module = example()
    data, probes = module.make_inputs(1, base_mesh=(7,)*3)
    solve = module.solver(data, probes)
    theta = jnp.zeros(2)
    ground = jax.jit(solve)(theta)
    metrics = module.stationarity(data, ground)
    assert ground.converged and metrics['euler_rms_hartree'] < 1.01e-9
    assert abs(metrics['sphere_constraint']) < 2e-13
    shifted = module.stationarity(data, ground._replace(chemical_potential=ground.chemical_potential+.02))
    np.testing.assert_allclose(shifted['euler_rms_hartree'], .02, atol=2e-9)
    np.testing.assert_allclose(shifted['full_kkt_residual'], 2*data.nelectron*.02, atol=2e-8)
    density = lambda t: solve(t).density
    direction = jnp.asarray([.6, .8])
    response = jax.jit(lambda t: jax.jvp(density, (t,), (direction,))[1])(theta)
    assert np.isfinite(response).all()
    assert float(jnp.linalg.norm(response)) > 1e-4
    np.testing.assert_allclose(jnp.dot(data.weights, response), 0., atol=1e-9)
    cotangent = data.weights*jnp.asarray(probes[:, 0])/data.nelectron
    reverse = jax.jit(jax.grad(lambda t: jnp.dot(cotangent, density(t))))(theta)
    np.testing.assert_allclose(reverse@direction, cotangent@response, atol=2e-9, rtol=2e-7)


def test_ks_reference_keeps_electrons_after_overlap_rank_reduction(monkeypatch):
    pytest.importorskip('pyscf')
    folder = Path(__file__).resolve().parents[2]/'examples/ofdft'
    monkeypatch.syspath_prepend(str(folder))
    from atlas_ks_pyscf import calculate
    info, density = calculate('Al', spacing=.18, kmesh=(1, 1, 1), basis_level=3,
                             sigma=.005, max_cycle=100, verbose=0)
    assert info['overlap_min_eigenvalue'] < info['overlap_cutoff']
    assert info['converged'] and np.isfinite(density).all()
    np.testing.assert_allclose(info['orbital_electrons'], 3., atol=1e-8)
    np.testing.assert_allclose(info['grid_electrons'], 3., atol=1e-8)
