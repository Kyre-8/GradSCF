"""Complete screened-hybrid periodic SCF/TD against independent PySCF."""
import numpy as np
import pytest


@pytest.mark.parametrize('xc',['hse06','cam-b3lyp'])
@pytest.mark.parametrize('nk',[1,2])
def test_periodic_rsh_scf_and_td_match_pyscf(xc,nk):
    pytest.importorskip('jax_xc');pytest.importorskip('pyscf')
    from gradscf.pbc import gto,dft,tdscf
    from pyscf.pbc import gto as pgto,dft as pdft,tdscf as ptd
    options=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',
                 basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,precision=1e-10)
    cell=gto.M(**options);points=cell.make_kpts((nk,1,1))
    ours=(dft.RKS if nk==1 else dft.KRKS)(cell,kpts=points,xc=xc).run(
        conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,max_cycle=100,damping=.1)
    reference_cell=pgto.M(**options,cart=True,verbose=0)
    reference=pdft.RKS(reference_cell) if nk==1 else pdft.KRKS(reference_cell,kpts=np.asarray(points))
    reference.xc=xc;reference.conv_tol=1e-12;reference.conv_tol_grad=1e-10
    reference.kernel()
    assert ours.converged and reference.converged
    np.testing.assert_allclose(ours.e_tot,reference.e_tot,atol=2e-6,rtol=0)
    for method,reference_method in ((tdscf.TDA,ptd.TDA),(tdscf.TDDFT,ptd.TDDFT)):
        actual=method(ours,nstates=1).run()
        expected=reference_method(reference)
        expected.nstates=1;expected.conv_tol=1e-9;expected.kernel()
        assert np.all(actual.converged) and np.all(expected.converged)
        np.testing.assert_allclose(actual.e,np.asarray(expected.e).reshape(-1)[:1],atol=1e-5,rtol=0)
