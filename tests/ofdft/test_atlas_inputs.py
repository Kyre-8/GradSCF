"""Independent PZ-LDA energy/potential baseline for ATLAS-style examples."""
import importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("ase", reason="ATLAS solid examples use ASE structures and units")


def test_pz_lda_matches_libxc_energy_and_potential():
    pytest.importorskip('pyscf')
    from pyscf.dft import libxc
    path=Path(__file__).resolve().parents[2]/'examples/ofdft/atlas_inputs.py'
    spec=importlib.util.spec_from_file_location('atlas_inputs',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rho=jnp.array([1e-5,.001,.01,.1,.3,1.,10.])
    expected, vxc, *_=libxc.eval_xc('LDA_X,LDA_C_PZ',np.asarray(rho))
    energy=module.pz_lda_energy_density(rho)
    potential=jax.grad(lambda r:module.pz_lda_energy_density(r).sum())(rho)
    np.testing.assert_allclose(energy,rho*expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(potential,vxc[0],rtol=1e-12,atol=1e-12)


def test_al_oepp_minimum_reaches_stationarity():
    from gradscf import ofdft
    path=Path(__file__).resolve().parents[2]/'examples/ofdft/atlas_inputs.py'
    spec=importlib.util.spec_from_file_location('atlas_inputs',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    atoms,mesh=module.crystal('Al',.32)
    data=module.gaussian_free_inputs(atoms,mesh)
    result=ofdft.run_ofdft(data,xc_energy_fn=module.pz_lda_energy,
        config=ofdft.OFDFTConfig(xc=None,maxiter=1000,tolerance=1e-8))
    assert result.converged, result.residual_norm


def test_ks_reference_preserves_odd_electron_count(monkeypatch):
    pytest.importorskip('pyscf')
    folder = Path(__file__).resolve().parents[2] / 'examples/ofdft'
    monkeypatch.syspath_prepend(str(folder))
    from atlas_ks_pyscf import fermi_occupations
    energies = [np.array([-.4, -.1, .2, .8]), np.array([-.3, .0, .3, .9])]
    occupations, entropy, chemical_potential = fermi_occupations(energies, 3, .005)
    np.testing.assert_allclose(np.sum(occupations) / 2, 3, atol=1e-12)
    assert all(np.all((o >= 0) & (o <= 2)) for o in occupations)
    assert np.isfinite(chemical_potential) and entropy >= 0
    gamma_occupations, _, _ = fermi_occupations(energies[:1], 3, .005)
    np.testing.assert_allclose(np.sum(gamma_occupations), 3, atol=1e-12)
