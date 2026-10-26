"""NNAO optimization uses direct J/K or RI factors, never dense ERIs."""
from importlib import import_module
import inspect
import runpy
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('jk_backend',['direct','df'])
def test_methane_gradient_without_full_eri(monkeypatch,jk_backend,tmp_path):
    from gradscf.integrals.plan import IntegralPlan
    original=IntegralPlan.evaluate
    def checked(self,op,*args,**kwargs):
        assert op!='eri','NNAO coefficient optimization must not allocate four-center ERIs'
        return original(self,op,*args,**kwargs)
    monkeypatch.setattr(IntegralPlan,'evaluate',checked)
    cls=vars(import_module("gradscf.model.nnao.training.objective"))['BasisRHF']
    assert 'jk_backend' in inspect.signature(cls).parameters
    assert 'eri_backend' not in inspect.signature(cls).parameters
    assert 'max_primitive_eri_gib' not in inspect.signature(cls).parameters
    ex=cls(basis_family='szp3_direct',jk_backend=jk_backend)
    if jk_backend=='direct':
        assert ex.rep is None
    else:
        assert ex.rep.ndim==2 and ex.rep.size < ex.primitive_topology.nao**4/10
        cache=ex.write_integral_cache(tmp_path/'integrals.npz')
        suffixless=ex.write_integral_cache(tmp_path/'suffixless')
        assert suffixless.is_file(), 'Returned cache path must point to the written file'
        def forbidden(*args,**kwargs):pytest.fail('A loaded DF cache must skip native integral evaluation')
        monkeypatch.setattr(IntegralPlan,'evaluate',forbidden)
        cached=cls(basis_family='szp3_direct',jk_backend='df',integral_cache=cache)
        np.testing.assert_array_equal(cached.ps,ex.ps)
        np.testing.assert_array_equal(cached.ph,ex.ph)
        np.testing.assert_array_equal(cached.rep,ex.rep)
    x=ex.layout.reference_outputs();energy,g,info=ex.evaluate(x)
    assert info['converged'] and info['reconstruction_error']<1e-9
    direction=jnp.zeros_like(x).at[0,0,2].set(.2).at[0,1,2].set(-.3).at[1:,1,1].set(.1)
    step=1e-4;e=lambda h:ex.evaluate(x+h*direction)[0]
    fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
    np.testing.assert_allclose(jnp.sum(g*direction),fd,atol=2e-6,rtol=2e-5)
    pytest.importorskip('pyscf')
    from pyscf import gto,scf
    labels=[f'{s}{i}' for i,s in enumerate(ex.symbols)]
    mol=gto.M(atom=list(zip(labels,ex.coords)),basis=dict(zip(labels,ex.layout.atom_shells(ex.layout.bind(x)))),
              unit='Angstrom',cart=False,verbose=0)
    mf=scf.RHF(mol)
    if jk_backend=='df':mf=mf.density_fit(auxbasis=ex.auxbasis)
    mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=150
    expected=mf.kernel();assert mf.converged
    np.testing.assert_allclose(energy,expected,atol=1e-8,rtol=0)
