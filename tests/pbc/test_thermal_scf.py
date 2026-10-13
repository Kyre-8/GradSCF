import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto,scf


def test_odd_electron_restricted_cell_uses_fractional_spin_counts():
    cell=gto.M(atom='Al 0 0 0',a=np.eye(3)*7.,unit='Bohr',mesh=(13,)*3)
    assert cell.nelec == (1.5,1.5)
    with pytest.raises(ValueError,match='integer'):
        scf.RHF(cell).run()
    mf=scf.RHF(cell,temperature=.08,exxdiv=None).run(max_cycle=150,damping=.3,
        conv_tol=1e-10,conv_tol_density=1e-9,conv_tol_grad=1e-8)
    assert mf.converged
    np.testing.assert_allclose(mf.mo_occ.sum(),3.,atol=1e-11)
    np.testing.assert_allclose(mf.e_free,mf.e_tot-.08*mf.entropy,atol=1e-12)
    from gradscf import dfpt
    b=jnp.diag(jnp.array([.1,-.2,.05,.05]))[None]
    response=dfpt.DFPT(mf).response(potential=b)
    assert response.converged
    np.testing.assert_allclose(response.values['electron_number'],0.,atol=1e-9)
    from gradscf.pbc._kpoint import run_kpoint_scf
    def perturb(t):
        return run_kpoint_scf(mf.inputs._replace(hcore=mf.inputs.hcore+t*b),mesh=cell.mesh,
            nelec=cell.nelec,temperature=.08,exxdiv=None,max_cycle=150,damping=.3,
            conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    plus,minus=perturb(1e-4),perturb(-1e-4)
    assert plus.converged and minus.converged
    np.testing.assert_allclose(response.density,(plus.density_spin-minus.density_spin)/2e-4,atol=2e-6)
    with pytest.raises(NotImplementedError,match='Taylor'):
        dfpt.DFPT(mf).series(potential=b,order=2)


@pytest.mark.parametrize('mode',['explicit','implicit'])
def test_thermal_weighted_scf_free_energy_gradient(mode):
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.pbc._kpoint import run_kpoint_scf
    from gradscf.scf import SCFDifferentiationConfig
    cell=gto.M(atom='H 0 0 0; H 1.4 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(11,)*3)
    data=build_inputs(cell,kpts=cell.make_kpts((2,1,1)),kweights=[.3,.7])
    cfg=SCFDifferentiationConfig(mode=mode,tolerance=1e-11,max_iter=80)
    def run(temp):
        return run_kpoint_scf(data,mesh=cell.mesh,nelec=cell.nelec,temperature=temp,
            exxdiv=None,differentiation=cfg,max_cycle=120,damping=.1,
            conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10)
    out=run(.12)
    assert out.converged
    np.testing.assert_allclose(jnp.einsum('k,skn->s',data.kweights,out.mo_occ_spin),[1.,1.],atol=1e-11)
    derivative=jax.jit(jax.grad(lambda t:run(t).free_energy))(.12)
    finite_difference=(run(.12001).free_energy-run(.11999).free_energy)/2e-5
    np.testing.assert_allclose(derivative,finite_difference,atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(derivative,-out.entropy,atol=2e-7)


@pytest.mark.parametrize('unrestricted',[False,True])
def test_thermal_hf_matches_pyscf_smearing(unrestricted):
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf
    from pyscf.pbc.scf import addons
    kw=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',
            basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,precision=1e-10)
    cell=gto.M(**kw);kpts=cell.make_kpts((2,1,1))
    cls=scf.KUHF if unrestricted else scf.KRHF
    ours=cls(cell,kpts=kpts,temperature=.12).run(conv_tol=1e-12,
        conv_tol_density=1e-10,conv_tol_grad=1e-9,max_cycle=120,damping=.1)
    refcell=pgto.M(**kw,cart=True,verbose=0)
    reference=(pscf.KUHF if unrestricted else pscf.KRHF)(refcell,kpts=np.asarray(kpts))
    reference=addons.smearing_(reference,sigma=.12,method='fermi',fix_spin=unrestricted)
    reference.conv_tol=1e-12;reference.conv_tol_grad=1e-9;reference.max_cycle=120
    reference.kernel()
    assert ours.converged and reference.converged
    np.testing.assert_allclose(ours.e_tot,reference.e_tot,atol=1e-6,rtol=0)
    np.testing.assert_allclose(ours.e_free,reference.e_free,atol=1e-6,rtol=0)
    np.testing.assert_allclose(ours.entropy,reference.entropy,atol=2e-6,rtol=0)
    np.testing.assert_allclose(ours.mo_occ,reference.mo_occ,atol=2e-6,rtol=0)


def test_consumers_reject_unsupported_reference_domains():
    from gradscf.pbc import tdscf
    from gradscf.gw.pbc import KRGW
    cell=gto.M(atom='H 0 0 0; H 1.4 0 0',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    mf=scf.RHF(cell,exxdiv=None).run(temperature=.1)
    assert mf.converged
    with pytest.raises(NotImplementedError,match='integer'):
        tdscf.TDA(mf).kernel()
    with pytest.raises(NotImplementedError,match='integer'):
        KRGW(mf).kernel()
    mf.temperature=.2
    with pytest.raises(ValueError,match='again'):
        scf.validate_reference(mf,integer=False)
    with pytest.raises(NotImplementedError,match='equal k weights'):
        scf.KRHF(cell,kpts=cell.make_kpts((2,1,1)),kweights=[.3,.7]).run()
