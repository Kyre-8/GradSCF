"""Integral solver protocol, frozen-core embedding and source freshness."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import fci
from gradscf.solvers import EigenSolverConfig, EigenResponseConfig
from test_response import model


def test_integral_solver_protocol_and_warm_start():
    h,g=model();solver=fci.FCISolver(nroots=2,max_space=16,gradient_mode='implicit_eigenvector')
    e,c=solver.kernel(h,g,4,(2,1),ecore=.7)
    assert np.all(solver.converged) and c.shape==(2,6,4)
    d1,d2=solver.make_rdm12(c[0],4,(2,1))
    np.testing.assert_allclose(jnp.sum(h*d1.T)+.5*jnp.sum(g*d2)+.7,e[0],atol=1e-10)
    effective=solver.absorb_h1e(h,g,4,(2,1),fac=.5)
    np.testing.assert_allclose(solver.contract_2e(effective,c[0],4,(2,1)),(e[0]-.7)*c[0],atol=1e-9)
    warm=fci.FCISolver(max_space=16).kernel(h,g,4,(2,1),ci0=c[0],ecore=.7)
    np.testing.assert_allclose(warm[0],e[0],atol=1e-10)
    dense=fci.FCISolver(solver='dense').kernel(h,g,4,(2,1),ci0=c[0],ecore=.7)
    np.testing.assert_allclose(dense[0],e[0],atol=1e-10)
    solver.nroots=1
    with pytest.raises(RuntimeError,match='changed'):solver.make_rdm12()
    e1,c1=fci.kernel(h,g,4,3,solver='dense')
    assert c1.shape==(6,4)
    np.testing.assert_allclose(e1+.7,e[0],atol=1e-10)


def test_core_folding_against_projected_full_fci():
    from pyscf.fci import direct_spin1,cistring
    h,g=map(np.asarray,model(4));ecore=.4
    core=(2,);active=(3,0,1)
    ha,ga,ea=fci.fold_core(h,g,core=core,active=active,ecore=ecore)
    out=fci.FCI(fci.FCIReference(h,g,(2,2),ecore),core=core,active=active,solver='dense').run()
    n=4;strings=cistring.make_strings(range(n),2)
    count=len(strings)**2;eye=np.eye(count)
    effective=direct_spin1.absorb_h1e(h,g,n,(2,2),.5)
    matrix=np.stack([np.asarray(direct_spin1.contract_2e(effective,x.reshape(len(strings),-1),n,(2,2))).reshape(-1) for x in eye],axis=1)
    keep=[i*len(strings)+j for i,a in enumerate(strings) for j,b in enumerate(strings) if a & (1<<2) and b & (1<<2)]
    expected=np.linalg.eigvalsh(matrix[np.ix_(keep,keep)])[0]+ecore
    np.testing.assert_allclose(out.e_tot,expected,atol=1e-12)
    assert out.space.nelec==(1,1) and out.ci.shape==(3,3)
    d1,d2=out.make_rdm12()
    np.testing.assert_allclose(jnp.sum(ha*d1.T)+.5*jnp.sum(ga*d2)+ea,expected,atol=1e-12)
    all_core=fci.FCI(fci.FCIReference(h,g,(2,2),ecore),core=(0,2),active=(),solver='dense').run()
    assert all_core.ci.shape==(1,1) and all_core.make_rdm1().shape==(0,0)


def test_native_hf_fci_and_active_space():
    from gradscf import gto,dft
    from pyscf import gto as pgto, scf as pscf, fci as pfci
    mol=gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g')
    unrun=dft.RKS(mol,xc='hf')
    with pytest.raises(RuntimeError,match='converge'):fci.FCI(unrun).run()
    mf=unrun.run();obj=fci.FCI(mf,solver='dense').run()
    pmf=pscf.RHF(pgto.M(atom=mol.atom,basis=mol.basis,verbose=0)).run(conv_tol=1e-12)
    expected=pfci.FCI(pmf).kernel()[0]
    np.testing.assert_allclose(obj.e_tot,expected,atol=1e-9)
    np.testing.assert_allclose(np.trace(obj.make_rdm1()),2,atol=1e-12)
    np.testing.assert_allclose(obj.spin_square(),(0,1),atol=1e-12)
    mf.conv_tol=1e-7
    with pytest.raises(RuntimeError,match='changed'):obj.make_rdm1()


def test_capacity_before_transform_and_failed_rerun(monkeypatch):
    import gradscf.fci.api as api
    h,g=map(np.array,model(6));ref=fci.FCIReference(h,g,(3,3))
    def should_not_transform(*args):raise AssertionError('transformation should not run')
    monkeypatch.setattr(api,'active_reference',should_not_transform)
    with pytest.raises(ValueError,match='determinants'):
        fci.FCI(ref,max_determinants=10).run()
    with pytest.raises(ValueError,match='workspace'):
        fci.FCI(ref,max_workspace_elements=1).run()
    obj=fci.FCISolver(solver='dense').run(h,g,6,(1,1))
    ref.h1[0,0] += .1
    with pytest.raises(RuntimeError,match='changed'):obj.make_rdm1()
    with pytest.raises(ValueError):obj.kernel(h,g,6,(8,1))
    with pytest.raises(RuntimeError,match='Run FCI'):obj.make_rdm1()


def test_limits_and_invalid_topology():
    with pytest.raises(ValueError,match='determinants'):fci.make_fci_space(20,(10,10))
    with pytest.raises(ValueError,match='link'):fci.make_fci_space(8,(4,4),max_link_elements=10)
    for nelec in [True,(-1,1),(4,0),(1.,1),2.5]:
        with pytest.raises(ValueError):fci.make_fci_space(3,nelec)
    h,g=model();space=fci.make_fci_space(4,(2,1))
    with pytest.raises(ValueError,match='workspace'):
        fci.build_hamiltonian(h,g,space,max_workspace_elements=1)
    with pytest.raises(ValueError,match='disjoint'):fci.fold_core(h,g,core=(0,),active=(0,2))
    assert fci.make_fci_space(4,4,spin=2).nelec==(3,1)
    with pytest.raises(ValueError):fci.make_fci_space(4,(2,1),spin=True)
    np.testing.assert_array_equal(fci.make_strings(4,2),(3,5,6,9,10,12))


@pytest.mark.parametrize('representation',['eri','eri_pair_matrix','df_factors'])
def test_ao_core_folding_transforms_only_active_orbitals(representation,monkeypatch):
    from types import SimpleNamespace
    from gradscf.fci import reference
    rng=np.random.default_rng(901);n=8
    h=np.asarray(model(n)[0]);c=np.linalg.qr(rng.normal(size=(n,n)))[0]
    factors=rng.normal(size=(5,n,n))*.1;factors=(factors+factors.transpose(0,2,1))*.5
    g=np.einsum('Lpq,Lrs->pqrs',factors,factors)
    rows,cols=np.tril_indices(n)
    value={'eri':g,'eri_pair_matrix':g[rows[:,None],cols[:,None],rows[None,:],cols[None,:]],'df_factors':factors}[representation]
    inputs=SimpleNamespace(hcore=h,overlap=np.eye(n),nuclear_repulsion=.4,
                           **{name:value if name==representation else None for name in ('eri','eri_pair_matrix','df_factors')})
    source=SimpleNamespace(mo_coeff=c,_scf_inputs=inputs)
    core=(0,1,2,3,4);active=(7,5)
    original=reference.transform_integrals
    seen=[]
    def transform(hcore,coeff,**kwargs):
        seen.append(coeff.shape)
        return original(hcore,coeff,**kwargs)
    monkeypatch.setattr(reference,'transform_integrals',transform)
    got=reference.active_reference(source,core,active)
    hmo,gmo=original(h,c,eri=g)
    expected=fci.fold_core(hmo,gmo,core=core,active=active,ecore=.4)
    assert seen==[(n,len(active))]
    for a,b in zip(got,expected):np.testing.assert_allclose(a,b,atol=2e-12)


def test_native_rohf_odd_electrons():
    from gradscf import gto,scf
    from pyscf import gto as pgto,scf as pscf,fci as pfci
    mf=scf.ROHF(gto.M(atom='Li 0 0 0',basis='sto-3g',spin=1)).run()
    assert mf.converged
    result=fci.FCI(mf,solver='dense').run()
    pmf=pscf.ROHF(pgto.M(atom='Li 0 0 0',basis='sto-3g',spin=1,verbose=0)).run()
    expected=pfci.FCI(pmf).kernel()[0]
    np.testing.assert_allclose(result.e_tot,expected,atol=1e-8)
    np.testing.assert_allclose(result.spin_square(),(.75,2.),atol=1e-9)


def test_absorbed_integral_convention_matches_pyscf():
    from pyscf.fci import direct_spin1
    from pyscf import ao2mo
    h,g=map(np.asarray,model(3));space=fci.make_fci_space(3,(1,1))
    np.testing.assert_allclose(fci.absorb_h1e(h,g,space),
                              ao2mo.restore(1,direct_spin1.absorb_h1e(h,g,3,(1,1)),3),atol=1e-14)
    np.testing.assert_allclose(fci.absorb_h1e(h,g,space,fac=.5),
                              ao2mo.restore(1,direct_spin1.absorb_h1e(h,g,3,(1,1),fac=.5),3),atol=1e-14)


def test_zero_ci_norm_spin_is_invalid_even_in_empty_orbital_space():
    space=fci.make_fci_space(0,0)
    assert np.isnan(fci.spin_square(jnp.zeros((1,1)),space)[0])
