"""All molecular ground-state families accept native compressed ERIs."""
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals,scf


@pytest.mark.parametrize('method',['RHF','UHF','ROHF','GHF','RKS','UKS','ROKS','GKS'])
@pytest.mark.parametrize('aosym',['s4','s8'])
def test_ground_state_matches_full_eri(method,aosym):
    top,p=integrals.prepare_basis('H 0 0 0; H 0 0 .8','6-31g',cart=False)
    plan=integrals.make_plan(top);n=top.nao
    common=dict(overlap=plan.evaluate('overlap',p),
                hcore=plan.evaluate('kinetic',p)+plan.evaluate('nuclear',p),
                nuclear_repulsion=scf.nuclear_repulsion_energy(p.nuclear_coords,jnp.array(top.nuclear_charges)))
    cfg=dict(max_cycle=100,conv_tol=1e-11,conv_tol_density=1e-9)
    is_dft=method.endswith('KS')
    if is_dft:
        pytest.importorskip('jax_xc');pytest.importorskip('pyscf')
        from pyscf import gto,dft
        mol=gto.M(atom='H 0 0 0; H 0 0 .8',basis='6-31g',cart=False,verbose=0)
        grid=dft.gen_grid.Grids(mol);grid.level=0;grid.build()
        ao=dft.numint.eval_ao(mol,grid.coords,deriv=1)
        common.update(ao=ao[0],ao_deriv1=ao,grid_weights=grid.weights)
        cfg['xc_spec']='pbe0'
    if method in {'UHF','ROHF','UKS','ROKS'}:common.update(nalpha=1,nbeta=1)
    else:common['nelectron']=2
    fn=getattr(scf,'run_'+method.lower()+'_from_integrals')
    config=getattr(scf,method+'Config')(**cfg)
    expected=fn(**common,eri=plan.evaluate('eri',p),config=config)
    actual=fn(**common,eri=plan.evaluate('eri',p,aosym=aosym),config=config)
    assert actual.converged and expected.converged
    np.testing.assert_allclose(actual.total_energy,expected.total_energy,atol=1e-9,rtol=0)


def test_rks_native_direct_uses_no_eri_array():
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    top,p=integrals.prepare_basis('H 0 0 0; H 0 0 .8','sto-3g',cart=False)
    plan=integrals.make_plan(top);n=top.nao
    kwargs=dict(overlap=plan.evaluate('overlap',p),hcore=plan.evaluate('kinetic',p)+plan.evaluate('nuclear',p),
                nelectron=2,nuclear_repulsion=scf.nuclear_repulsion_energy(p.nuclear_coords,jnp.array(top.nuclear_charges)),
                ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
    expected=scf.run_rks_from_integrals(**kwargs,eri=plan.evaluate('eri',p),config=scf.RKSConfig(xc_spec='hf'))
    actual=scf.run_rks_from_integrals(**kwargs,eri=None,direct_basis=NativeDirectBasis(plan,p),
                                    config=scf.RKSConfig(xc_spec='hf',jk_backend='direct'))
    assert actual.converged
    np.testing.assert_allclose(actual.total_energy,expected.total_energy,atol=1e-10)
