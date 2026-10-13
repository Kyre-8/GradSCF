"""Offline periodic basis coverage, valence pairing, and numerical oracles."""
import numpy as np
import pytest
from gradscf.pbc import gto


def test_periodic_basis_catalog_and_independent_copies():
    assert 'gth-dzvp-molopt-sr' in gto.basis_names()
    assert {'Si', 'Fe', 'Cu'} <= set(gto.basis_elements('gth-dzvp-molopt-sr'))
    first = gto.load_basis('DZVP-MOLOPT-SR-GTH', 'Si')
    expected = gto.load_basis('GTH_DZVP_MOLOPT_SR', 'Si')
    assert first == expected
    first[0][1][0] = -1.
    assert gto.load_basis('gth-dzvp-molopt-sr', 'Si') == expected


def test_molopt_cell_and_valence_mismatch():
    cell = gto.M(atom='Si 0 0 0', a=np.eye(3)*6., unit='Bohr',
        basis='DZVP-MOLOPT-SR-GTH', pseudo='gth-pbe', mesh=(7,)*3)
    assert cell.nelectron == 4
    with pytest.raises(ValueError, match='valence'):
        gto.M(atom='Rh 0 0 0', a=np.eye(3)*6., unit='Bohr', spin=1,
            basis='gth-dzvp-molopt-sr-q9', pseudo='gth-pbe', mesh=(7,)*3)
    rhodium = gto.M(atom='Rh 0 0 0', a=np.eye(3)*6., unit='Bohr', spin=1,
        basis='DZVP-MOLOPT-SR-GTH-q9', pseudo='GTH-PBE-q9', mesh=(7,)*3)
    assert rhodium.nelectron == 9


def test_missing_element_reports_available_coverage():
    with pytest.raises(ValueError, match='available'):
        gto.load_basis('gth-tzv2px-molopt', 'Fe')
    with pytest.raises(ValueError, match='available'):
        gto.load_pseudo('gth-pbe-q99', 'Si')


def test_basis_loading_without_pyscf_runtime():
    import os
    from pathlib import Path
    import subprocess
    import sys
    script = '''
import sys
from importlib.abc import MetaPathFinder
class BlockPySCF(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pyscf', 'gpu4pyscf'):
            raise RuntimeError('External chemistry runtime dependency: '+fullname)
sys.meta_path.insert(0, BlockPySCF())
import jax
import numpy as np
jax.config.update('jax_enable_x64', True)
from gradscf.pbc import gto
cell = gto.M(atom='Si 0 0 0', a=np.eye(3)*6., unit='Bohr',
             basis='DZVP-MOLOPT-SR-GTH', pseudo='GTH-PBE-q4', mesh=(7,)*3)
assert cell.nelectron == 4
'''
    source = str(Path(__file__).resolve().parents[2]/'src')
    env = dict(os.environ, PYTHONPATH=source+os.pathsep+os.environ.get('PYTHONPATH', ''),
               JAX_PLATFORMS='cpu')
    subprocess.run([sys.executable, '-c', script], env=env, check=True, capture_output=True, text=True)


def test_all_exported_data_match_pinned_pyscf():
    from pathlib import Path
    pyscf = pytest.importorskip('pyscf')
    from pyscf.pbc.gto import basis, pseudo
    from gradscf.integrals.periodic.library import _data
    if pyscf.__version__ != '2.9.0':
        pytest.skip('Exact source oracle is pinned to PySCF 2.9.0')
    root = Path(pyscf.__file__).parent/'pbc/gto/basis'
    for family in gto.basis_names():
        for symbol in gto.basis_elements(family):
            assert gto.load_basis(family, symbol) == basis.load(str(root/f'{family}.dat'), symbol)
    data = _data()
    for alias, elements in data['aliases'].items():
        for symbol, family in elements.items():
            assert gto.load_basis(alias, symbol) == gto.load_basis(family, symbol)
    for family, elements in data['pseudo_variants'].items():
        for symbol, variants in elements.items():
            for q, values in variants.items():
                assert gto.load_pseudo(f'{family}-q{q}', symbol) == pseudo.load(f'{family}-q{q}', symbol) == values


@pytest.mark.parametrize('family', ['gth-dzvp-molopt-sr', 'gth-tzv2px-molopt'])
def test_silicon_molopt_integrals_match_pyscf(family):
    pyscf = pytest.importorskip('pyscf')
    from gradscf.integrals.periodic.fft import build_inputs
    from pyscf.pbc import gto as pgto, scf as pscf
    from pyscf.pbc.gto import basis
    from pathlib import Path
    # Include general contractions and f shells. Use identical Cartesian AOs,
    # GTH-PBE potential, and FFT grid; no comparison of different basis limits.
    data = basis.load(str(Path(pyscf.__file__).parent/f'pbc/gto/basis/{family}.dat'), 'Si')
    kwargs = dict(atom='Si .3 .4 .5', a=np.eye(3)*5., unit='Bohr',
                  pseudo='gth-pbe', mesh=(31,)*3, precision=1e-10)
    actual = build_inputs(gto.M(**kwargs, basis=family))
    reference = pscf.RHF(pgto.M(**kwargs, basis={'Si':data}, cart=True, verbose=0))
    np.testing.assert_allclose(actual.overlap[0], reference.get_ovlp(), rtol=0, atol=2e-7)
    np.testing.assert_allclose(actual.hcore[0], reference.get_hcore(), rtol=0, atol=3e-6)


def test_molopt_hydrogen_scf_matches_pyscf():
    pytest.importorskip('pyscf')
    from gradscf.pbc import scf
    from pyscf.pbc import gto as pgto, scf as pscf
    kwargs = dict(atom='H 0 0 0; H 0 0 1.4', a=np.eye(3)*6., unit='Bohr',
                  basis='gth-szv-molopt-sr', pseudo='gth-pade', mesh=(41,)*3, precision=1e-10)
    actual = scf.RHF(gto.M(**kwargs)).run(conv_tol=1e-11, conv_tol_density=1e-10)
    reference = pscf.RHF(pgto.M(**kwargs, cart=True, verbose=0))
    reference.conv_tol = 1e-11
    reference.kernel()
    assert actual.converged and reference.converged
    np.testing.assert_allclose(actual.e_tot, reference.e_tot, rtol=0, atol=1e-6)
