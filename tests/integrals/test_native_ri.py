"""Two-/three-center RI tensors and differentiable coefficient projection."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals


@pytest.mark.parametrize('cart',[False,True])
def test_native_ri_integrals_match_pyscf(cart):
    from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan
    pyscf=pytest.importorskip('pyscf')
    from pyscf import gto,df
    atom='O 0 0 0; H 0 .75 .58; H 0 -.75 .58'
    top,p=integrals.prepare_basis(atom,'6-31g(d)',cart=cart)
    at,ap=integrals.prepare_basis(atom,'def2-universal-jkfit',cart=cart)
    plan=make_auxiliary_plan(top,at)
    metric,three=plan.evaluate(p,ap)
    mol=gto.M(atom=atom,basis='6-31g(d)',cart=cart,verbose=0)
    aux=gto.M(atom=atom,basis='def2-universal-jkfit',cart=cart,verbose=0)
    np.testing.assert_allclose(metric,aux.intor('int2c2e'),atol=2e-10,rtol=1e-11)
    reference=df.incore.aux_e2(mol,aux,aosym='s2ij').T
    np.testing.assert_allclose(three,reference,atol=2e-10,rtol=1e-11)


def test_ri_coefficient_gradient_matches_finite_difference():
    from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan, project_factors
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix
    from dataclasses import replace
    atom='H 0 0 0; H 0 0 .8'
    top,p=integrals.prepare_basis(atom,'3-21g',cart=False)
    pt,pp=primitive_basis(top,p)
    at,ap=integrals.prepare_basis(atom,'def2-universal-jkfit',cart=False)
    factors=make_auxiliary_plan(pt,at).factors(pp,ap)
    def value(c):
        params=replace(p,coefficients=(c,*p.coefficients[1:]))
        b=project_factors(factors,contraction_matrix(top,params))
        return jnp.sum(b*b)
    c=p.coefficients[0];v=jnp.ones_like(c);h=1e-4
    ad=jax.jvp(value,(c,),(v,))[1]
    fd=(value(c+h*v)-value(c-h*v))/(2*h)
    np.testing.assert_allclose(ad,fd,atol=1e-6,rtol=1e-6)
    assert jnp.all(jnp.isfinite(jax.jacfwd(jax.grad(value))(c)))


def test_auxiliary_assembly_never_requests_four_center_eri(monkeypatch):
    from gradscf.integrals.plan import IntegralPlan
    from gradscf.scf import RKSConfig
    original=IntegralPlan.evaluate
    def checked(self,operator,*args,**kwargs):
        assert operator!='eri','RI assembly must build only 2c/3c Coulomb integrals'
        return original(self,operator,*args,**kwargs)
    monkeypatch.setattr(IntegralPlan,'evaluate',checked)
    data=integrals.build_rks_integral_inputs(atom='H 0 0 0; H 0 0 .8',basis='3-21g',
        config=RKSConfig(xc_spec='hf',jk_backend='df',auxbasis='def2-universal-jkfit'))
    assert data.eri is None and data.eri_pair_matrix is None
    assert data.df_factors.shape[1:]==(4,4)


def test_unrestricted_facade_density_fit_matches_reference():
    pytest.importorskip('pyscf')
    from pyscf import gto as pgto,scf as pscf
    from gradscf import gto,scf
    atom='N 0 0 0; H 0 .8 .6; H 0 -.8 .6'
    mol=gto.M(atom=atom,basis='sto-3g',spin=1)
    mf=scf.UHF(mol).density_fit('def2-universal-jkfit')
    mf.conv_tol=1e-11;mf.max_cycle=150;energy=mf.kernel()
    ref=pscf.UHF(pgto.M(atom=atom,basis='sto-3g',spin=1,cart=True,verbose=0)).density_fit('def2-universal-jkfit')
    ref.init_guess='1e';ref.conv_tol=1e-12;ref.max_cycle=150;expected=ref.kernel()
    assert mf.converged and ref.converged
    np.testing.assert_allclose(energy,expected,atol=1e-8,rtol=0)
