"""A converged external periodic XC reference reaches the q=0 TD facade."""
import jax.numpy as jnp
import numpy as np
import pytest


def test_external_slater_scf_tda_and_tddft_match_named_recipe():
    pytest.importorskip('jax_xc')
    from gradscf.dft import Functional
    from gradscf.pbc import gto,dft,tdscf
    cell=gto.M(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    coefficient=-.75*(6/jnp.pi)**(1/3)
    model=Functional(lambda x:x,lambda p,x:p*coefficient*jnp.sum(
        x.weights[None]*jnp.maximum(x.rho_spin,0.)**(4/3)))
    controls=dict(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-10,max_cycle=100,damping=.1)
    named=dft.RKS(cell,xc='slater').run(**controls)
    external=dft.RKS(cell,functional=model,params=1.).run(**controls)
    assert named.converged and external.converged
    np.testing.assert_allclose(external.e_tot,named.e_tot,atol=2e-11,rtol=0)
    np.testing.assert_allclose(external.make_rdm1(),named.make_rdm1(),atol=2e-10,rtol=0)
    for method in (tdscf.TDA,tdscf.TDDFT):
        reference=method(named,nstates=1).run()
        actual=method(external,nstates=1).run()
        assert np.all(reference.converged) and np.all(actual.converged)
        np.testing.assert_allclose(actual.get_ab(),reference.get_ab(),atol=2e-10,rtol=0)
        np.testing.assert_allclose(actual.e,reference.e,atol=2e-10,rtol=0)
    external.params=1.1
    with pytest.raises(ValueError,match='functional parameters'):
        actual.get_ab()
