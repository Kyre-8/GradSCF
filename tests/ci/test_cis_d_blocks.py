"""Selected CIS(D) blocks reuse the original shared correction formula."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf.ci.types import CISResult
from test_ci import h4


@pytest.mark.parametrize('promoted', ['amplitudes', 'excitation_energies'])
def test_cis_d_reserves_the_promoted_amplitude_workspace(promoted, monkeypatch):
    module = importlib.import_module('gradscf.ci.corrections')
    c, eps, g = jnp.eye(4, dtype=jnp.float32), jnp.arange(4, dtype=jnp.float32), jnp.zeros((4,)*4, jnp.float32)
    energy_dtype = jnp.float64 if promoted == 'excitation_energies' else jnp.float32
    amplitude_dtype = jnp.float64 if promoted == 'amplitudes' else jnp.float32
    singles = CISResult(jnp.ones(1, energy_dtype), jnp.ones((1, 2, 2), amplitude_dtype)/2,
                        jnp.zeros(1, jnp.float32), jnp.ones(1, dtype=bool), True)
    captured = []

    class Prepared(Exception):
        pass

    def checked(*args, **kwargs):
        captured.append(kwargs['reserved_bytes'])
        raise Prepared

    monkeypatch.setattr(module, 'transform_eri_blocks', checked)
    with pytest.raises(Prepared):
        module.cis_d_from_orbitals(c, eps, singles, nocc=2, eri=g, max_memory=32.)
    inputs = module.array_storage_bytes(c, eps, *jax.tree.leaves(singles))
    workspace_elements = 3*16*(4*2**2*2**2+2*2**3+2**3*2)+80*2**2*2**2
    assert captured == [inputs+workspace_elements*jnp.dtype(jnp.float64).itemsize]


def test_selected_cis_d_matches_original_formula_and_response(h4, monkeypatch):
    mf,h,g,_,_ = h4
    module=importlib.import_module('gradscf.ci.corrections')
    function=getattr(module,'cis_d_from_orbitals',None)
    assert callable(function),'Missing selected-block CIS(D) preparation'
    cfg=ci.CIConfig(solver='dense',gradient_mode='implicit_eigenvector')
    singles=ci.solve_cis(h,g,nocc=2,config=cfg)
    expected=ci.cis_d_correction(g,mf.mo_energy,singles,nocc=2)
    original=module.transform_eri_blocks
    def selected(coefficients,requests,**kwargs):
        assert all(tuple(coefficients[i].shape[1] for i in r)!=(4,)*4 for r in requests)
        return original(coefficients,requests,**kwargs)
    monkeypatch.setattr(module,'transform_eri_blocks',selected)
    actual=function(jnp.eye(4),mf.mo_energy,singles,nocc=2,eri=g,max_memory=32.)
    for a,e in zip(actual,expected):np.testing.assert_allclose(a,e,atol=2e-12)
    def loss(t,prepared):
        gs=g*(1+.13*t)
        result=(function(jnp.eye(4),mf.mo_energy,singles,nocc=2,eri=gs,max_memory=32.)
                if prepared else ci.cis_d_correction(gs,mf.mo_energy,singles,nocc=2))
        return result.corrections[0]
    def bundle(t,prepared):
        f=lambda x:loss(x,prepared)
        return jax.value_and_grad(f)(t),jax.jvp(jax.grad(f),(t,),(1.,))[1]
    actual=jax.jit(lambda x:bundle(x,True))(0.)
    expected=jax.jit(lambda x:bundle(x,False))(0.)
    for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,e,atol=3e-11,rtol=3e-9)
