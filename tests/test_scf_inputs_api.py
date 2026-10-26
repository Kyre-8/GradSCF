"""Public standalone integral assembly contracts and independent comparisons."""
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.scf.inputs.assembly import build_rks_integral_inputs, build_uks_integral_inputs
from gradscf.scf import RKSConfig, UKSConfig, run_rks_from_integrals
from gradscf.scf.init_guess import restricted_initial_guess, unrestricted_initial_guess

ATOM = "H 0 0 0; H 0 0 .74"


@pytest.mark.parametrize("backend", ["native", "cpu", "libcint", "jax"])
def test_restricted_inputs_use_standalone_integrals(backend):
    data = build_rks_integral_inputs(atom=ATOM,basis="3-21g",xc_spec="hf",integral_backend=backend)
    assert data.nelectron == 2
    assert data.overlap.shape == (4,4)
    assert data.eri_pair_matrix.shape == (10,10)
    assert data.init_density is None
    assert data.ao.shape[1] == 4
    result=run_rks_from_integrals(**data.as_rks_kwargs(),config=RKSConfig(xc_spec="hf",conv_tol=1e-12))
    assert result.converged
    np.testing.assert_allclose(result.total_energy,-1.122940256848,atol=1e-10,rtol=0)


def test_native_unrestricted_inputs_electron_counts_and_density():
    data=build_uks_integral_inputs(atom="H 0 0 0",basis="sto-3g",spin=1,xc_spec="hf")
    assert (data.nalpha,data.nbeta,data.total_electrons)==(1,0,1)
    assert data.eri.shape==(1,1)  # Native assembly now directly emits s4.


def test_native_inputs_match_independent_reference_integrals():
    pyscf=pytest.importorskip("pyscf")
    data=build_rks_integral_inputs(atom=ATOM,basis="3-21g",xc_spec="hf")
    mol=pyscf.gto.M(atom=ATOM,basis="3-21g",cart=True,verbose=0)
    np.testing.assert_allclose(data.overlap,mol.intor("int1e_ovlp"),atol=1e-12)
    np.testing.assert_allclose(data.hcore,mol.intor("int1e_kin")+mol.intor("int1e_nuc"),atol=1e-12)
    np.testing.assert_allclose(data.eri_pair_matrix,mol.intor("int2e",aosym="s4"),atol=1e-12)


def test_native_spectral_df_preserves_full_jk():
    from gradscf.integrals.molecular.jk import build_jk_from_df
    data=build_rks_integral_inputs(atom=ATOM,basis="3-21g",config=RKSConfig(xc_spec="hf",jk_backend="df",df_tol=1e-12))
    top,p=integrals.prepare_basis(ATOM,"3-21g")
    eri=integrals.make_plan(top,backend="native").evaluate("eri",p)
    dm=jnp.eye(4)
    j,k=build_jk_from_df(data.df_factors,dm)
    np.testing.assert_allclose(j,jnp.einsum("pqrs,rs->pq",eri,dm),atol=1e-10)
    np.testing.assert_allclose(k,jnp.einsum("prqs,rs->pq",eri,dm),atol=1e-10)


@pytest.mark.parametrize("name", ["minao", "sap", "atom", "chkfile"])
def test_external_initial_guesses_fail_explicitly(name):
    with pytest.raises(ValueError,match="Unsupported initial guess"):
        restricted_initial_guess(init_guess=name,dtype=jnp.float64)


def test_initial_density_inputs_are_validated_and_symmetrized():
    dm=jnp.array([[1.,.2],[.4,0.]])
    np.testing.assert_allclose(restricted_initial_guess(init_guess=dm,dtype=jnp.float64).density,[[1.,.3],[.3,0.]])
    pair=unrestricted_initial_guess(init_guess=jnp.stack([dm,dm]),dtype=jnp.float64)
    np.testing.assert_allclose(pair.density_alpha,pair.density_beta)
    with pytest.raises(ValueError,match="square"):
        restricted_initial_guess(init_guess=jnp.ones((2,3)),dtype=jnp.float64)


@pytest.mark.parametrize("options,error", [({"integral_backend":"gpu"},ValueError),
    ({"grid_ao_backend":"cpu"},ValueError),({"cart":False},NotImplementedError),
    ({"chkfile":"old.chk"},NotImplementedError),({"ecp":"lanl2dz"},TypeError)])
def test_unsupported_external_options_are_not_silently_ignored(options,error):
    with pytest.raises(error):build_rks_integral_inputs(atom=ATOM,basis="3-21g",xc_spec="hf",**options)


def test_default_facades_run_without_external_chemistry_imports():
    script='''
import builtins, sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name=="pyscf" or name.startswith("pyscf.") or name.startswith("gpu4pyscf"):
        raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
from gradscf import gto, scf
for cls,atom,spin in [(scf.RKS,"H 0 0 0; H 0 0 .74",0),(scf.UHF,"H 0 0 0",1)]:
    mol=gto.M(atom=atom,basis="sto-3g",spin=spin)
    mf=cls(mol,xc="hf") if cls is scf.RKS else cls(mol)
    mf.kernel()
    assert mf.converged
assert not any(n=="pyscf" or n.startswith("pyscf.") for n in sys.modules)
'''
    env=dict(os.environ,PYTHONPATH=str(Path("src").resolve()),JAX_PLATFORMS="cpu",JAX_ENABLE_X64="1")
    result=subprocess.run([sys.executable,"-c",script],env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
