"""Native-direct molecular excitations against independent PySCF and MolGW.

MolGW 3.4/b831818d reference inputs, binary identity and rounded outputs are
stored with the numerical fixture. No MolGW installation is needed for pytest.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest


@pytest.fixture(scope="module")
def water():
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf, tdscf as ptdscf
    from gradscf import gto, scf

    gold = json.loads((Path(__file__).parent / "data/molgw_water_hf_response.json").read_text())
    atoms = [(a[0], a[1:]) for a in gold["geometry_angstrom"]]
    started = perf_counter()
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Angstrom", cart=True)
    actual = scf.RHF(mol, integral_backend="native", jk_backend="direct", grids_level=0,
                     conv_tol=1e-13, conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    actual_seconds = perf_counter() - started
    started = perf_counter()
    pmol = pgto.M(atom=atoms, basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
    reference = pscf.RHF(pmol).run(conv_tol=1e-13, conv_tol_grad=1e-11)
    reference_seconds = perf_counter() - started
    assert actual.converged and reference.converged
    assert actual._scf_inputs.eri is None and actual._scf_inputs.eri_pair_matrix is None
    np.testing.assert_allclose(actual.e_tot, reference.e_tot, atol=2e-10, rtol=0)
    return actual, reference, ptdscf, gold, (actual_seconds, reference_seconds)


@pytest.mark.parametrize("method", ["tda_singlet", "tda_triplet", "tdhf_singlet"])
def test_direct_hf_response_matches_pyscf_and_molgw(water, method):
    from gradscf import ci, tdscf

    mf, pmf, ptdscf, gold, scf_times = water
    full = method == "tdhf_singlet"
    singlet = method != "tda_triplet"
    started = perf_counter()
    if not singlet:
        solver = ci.CIS(mf, singlet=False, nroots=10, conv_tol=1e-10, max_memory=64.).run()
        energies, strengths = np.asarray(solver.e), np.zeros(10)
    else:
        solver = (tdscf.TDDFT if full else tdscf.TDA)(
            mf, nstates=10, davidson_tol=1e-10, max_memory=64.)
        result = solver.kernel()
        energies = np.asarray(result.excitation_energies)
        strengths = np.asarray(solver.oscillator_strength())
    gradscf_seconds = perf_counter() - started
    assert bool(np.all(solver.converged))

    started = perf_counter()
    reference = (ptdscf.TDHF if full else ptdscf.TDA)(pmf)
    reference.singlet, reference.nstates, reference.conv_tol = singlet, 10, 1e-11
    reference.kernel()
    pyscf_seconds = perf_counter() - started
    assert np.all(reference.converged)
    expected = np.asarray(reference.e)
    oscillator = np.asarray(reference.oscillator_strength())
    np.testing.assert_allclose(energies, expected, atol=2e-9, rtol=0)
    np.testing.assert_allclose(strengths, oscillator, atol=2e-8, rtol=0)

    molgw = gold["cases"][method]
    # MolGW's YAML rounds the two core roots to only six eV decimals.
    np.testing.assert_allclose(energies, molgw["excitation_energy_hartree"], atol=5e-8, rtol=0)
    np.testing.assert_allclose(strengths, molgw["oscillator_strength"], atol=2e-7, rtol=0)
    if method == "tda_singlet":
        cis = ci.CIS(mf, nroots=10, conv_tol=1e-10, max_memory=64.).run()
        np.testing.assert_allclose(cis.e, energies, atol=2e-9, rtol=0)
        np.testing.assert_allclose(np.sum(np.asarray(cis.amplitudes)**2, axis=(1, 2)), 1., atol=2e-12)
    print("REFERENCE_RESULT " + json.dumps(dict(
        method=method, roots=len(energies), energy_hartree=energies.tolist(),
        oscillator_strength=strengths.tolist(),
        pyscf_energy_max_error=float(np.max(np.abs(energies-expected))),
        molgw_energy_max_error=float(np.max(np.abs(energies-molgw["excitation_energy_hartree"]))),
        pyscf_oscillator_max_error=float(np.max(np.abs(strengths-oscillator))),
        molgw_oscillator_max_error=float(np.max(np.abs(strengths-molgw["oscillator_strength"]))),
        gradscf_first_driver_seconds=gradscf_seconds, pyscf_driver_seconds=pyscf_seconds,
        hf_driver_seconds=dict(gradscf=scf_times[0], pyscf=scf_times[1])), sort_keys=True))
