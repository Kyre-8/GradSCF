"""Offline molecular reference from an executed, pinned MolGW binary."""
from pathlib import Path
import json
import numpy as np
import pytest
from gradscf import gto, scf, gw, bse

CASES=[]
for filename in ('molgw_molecular.json','molgw_expanded.json'):
    CASES.extend(json.loads((Path(__file__).parent/'data'/filename).read_text())['cases'])


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_molecular_gw_bse_matches_molgw(case):
    mf=scf.RHF(gto.M(atom=case['atom'],basis=case['basis']),conv_tol=1e-12)
    if case['auxbasis']:
        mf.density_fit(case['auxbasis'])
        mf.df_tol=1e-6
    if case.get('scf_amplitudes'):
        mf=mf.multistart(amplitudes=case['scf_amplitudes'],seed=case['scf_seed']).selected
        assert mf is not None
    else:
        mf.run()
    np.testing.assert_allclose(mf.e_tot,case['hf_hartree'],rtol=0,atol=1e-7)
    obj=gw.GW(mf,nw=case.get('nw',200),eta=1e-5,method=case['method'],max_cycle=100,conv_tol=1e-10,
              g_orbitals=case['g_orbitals'],screening_occupied=case['screening_occupied'],
              screening_virtual=case['screening_virtual']).run(orbs=case['qp_targets'])
    indices=np.array(case['qp_targets'])
    assert obj.converged
    np.testing.assert_allclose(np.asarray(obj.mo_energy)[indices],np.array(case['qp_hartree'])[indices],rtol=0,atol=2e-6)
    if 'qp_weight' in case:
        np.testing.assert_allclose(np.asarray(obj.result.qp_weight)[indices],case['qp_weight'],rtol=0,atol=5e-5)
    for optical in case['optical']:
        response=bse.BSE(obj,occupied=case['screening_occupied'],virtual=case['screening_virtual'],
            tda=optical['tda'],singlet=optical['singlet'],nroots=len(optical['energies_hartree']),solver='dense').run()
        assert response.converged.all()
        np.testing.assert_allclose(response.e,optical['energies_hartree'],rtol=0,atol=3e-6)
        np.testing.assert_allclose(response.oscillator_strength(),optical['oscillator_strengths'],rtol=0,atol=2e-5)
        if optical['singlet']:
            np.testing.assert_allclose(response.polarizability(),optical['static_polarizability_au'],rtol=0,atol=3e-4)
            np.testing.assert_allclose(response.absorption_cross_section(np.array(optical['omega_hartree']),eta=.01),
                                       optical['cross_section_au'],rtol=0,atol=2e-4)
