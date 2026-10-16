import numpy as np
import pytest
from gradscf.pbc import gto,dft,scf


def test_off_mesh_hybrid_bands_require_a_matched_cutoff_reference():
    cell=gto.M(atom='He 0 0 0',a=np.eye(3)*5,unit='Bohr',mesh=(15,)*3)
    mf=scf.RHF(cell).run()
    with pytest.raises(NotImplementedError,match='vcut_sph'):
        mf.get_bands(np.array([[.13,0.,0.]]))


def test_non_scf_bands_match_pyscf_and_preserve_reference():
    pytest.importorskip('jax_xc');pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,dft as pdft
    kw=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6,unit='Bohr',
            basis='gth-szv',pseudo='gth-pbe',mesh=(31,)*3)
    cell=gto.M(**kw);kpts=cell.make_kpts((2,1,1))
    mf=dft.KRKS(cell,kpts=kpts,xc='pbe').run()
    saved=mf.result
    query=np.array([[.11,.03,-.07],[.21,-.09,.02]])
    energies,coefficients=mf.get_bands(query,chunk_size=1)
    assert mf.result is saved
    ref=pdft.KRKS(pgto.M(**kw,cart=True,verbose=0),kpts=np.asarray(kpts))
    ref.xc='pbe';ref.conv_tol=1e-11;ref.kernel()
    expected,_=ref.get_bands(query)
    np.testing.assert_allclose(energies,expected,atol=2e-6,rtol=0)
    assert coefficients.shape==(2,2,2)
    on_mesh,_=mf.get_bands(kpts)
    np.testing.assert_allclose(on_mesh,mf.mo_energy,atol=2e-7,rtol=0)
