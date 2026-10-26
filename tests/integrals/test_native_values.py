"""CPU float64 native integrals versus PySCF, using identical libcint tables."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def native_evaluate():
    try:
        module = importlib.import_module("gradscf.integrals.backends.native")
    except ModuleNotFoundError:
        pytest.fail("native integral backend has not been implemented")
    return module.evaluate


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("operator,intor", [
    ("overlap", "int1e_ovlp"), ("kinetic", "int1e_kin"),
    ("nuclear", "int1e_nuc"), ("dipole", "int1e_r"), ("eri", "int2e"),
])
def test_native_matches_pyscf_and_jit(operator, intor, cart):
    pyscf = pytest.importorskip("pyscf")
    mol = pyscf.gto.M(atom="O 0 0 0; H 0 .76 .58; H 0 -.76 .58",
                      basis="6-31g*", cart=cart, verbose=0)
    evaluate = native_evaluate()
    env = jnp.asarray(mol._env)
    call = lambda e: evaluate(operator, mol._atm, mol._bas, e, mol.nao, cart=cart)
    expected = mol.intor(intor, aosym="s1")
    actual = call(env)
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-12)
    np.testing.assert_allclose(jax.jit(call)(env), expected, atol=3e-11, rtol=3e-12)
    # A changed coordinate must flow through the compiled function's dynamic env.
    changed = np.array(mol._env, copy=True)
    changed[mol._atm[1, 1] + 1] += 0.05
    ref = mol.copy()
    ref._env = changed
    np.testing.assert_allclose(jax.jit(call)(jnp.asarray(changed)),
                               ref.intor(intor, aosym="s1"), atol=3e-11, rtol=3e-12)


def test_native_ad_is_explicitly_unsupported():
    pyscf = pytest.importorskip("pyscf")
    mol = pyscf.gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    evaluate = native_evaluate()
    call = lambda e: evaluate("overlap", mol._atm, mol._bas, e, mol.nao).sum()
    with pytest.raises((ValueError, NotImplementedError), match="(?i)(derivative|jvp|differentiation)"):
        jax.grad(call)(jnp.asarray(mol._env))


def _primitive_tables():
    # A single unnormalized primitive, phi = exp(-r^2) / sqrt(4*pi).
    atm = np.array([[2, 20, 1, 0, 0, 0]], dtype=np.int32)
    bas = np.array([[0, 0, 1, 1, 0, 23, 24, 0]], dtype=np.int32)
    env = np.zeros(25, dtype=np.float64)
    env[23:25] = 1.0
    return atm, bas, env


def test_native_works_without_importing_pyscf():
    script = r'''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pyscf" or name.startswith("pyscf."):
        raise AssertionError("Native runtime must not import PySCF")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.integrals.backends.native import evaluate
atm = np.array([[2, 20, 1, 0, 0, 0]], dtype=np.int32)
bas = np.array([[0, 0, 1, 1, 0, 23, 24, 0]], dtype=np.int32)
env = jnp.zeros(25, dtype=jnp.float64).at[23:25].set(1.)
actual = jax.jit(lambda e: evaluate("overlap", atm, bas, e, 1))(env)
np.testing.assert_allclose(actual, np.sqrt(np.pi)/(8*np.sqrt(2)), rtol=2e-14)
assert not any(m == "pyscf" or m.startswith("pyscf.") for m in sys.modules)
'''
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    completed = subprocess.run([sys.executable, "-c", script], env=env,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("case", ["pointer", "negative_exponent", "nonfinite", "range", "nuclei"])
def test_native_rejects_invalid_dynamic_or_static_tables(case):
    atm, bas, env = _primitive_tables()
    if case == "pointer":
        bas[0, 6] = 1000
    elif case == "negative_exponent":
        env[23] = -1.0
    elif case == "nonfinite":
        env[20] = np.nan
    elif case == "range":
        env[8] = 0.3
    elif case == "nuclei":
        atm[0, 2] = 2
    evaluate = native_evaluate()
    with pytest.raises(Exception, match="(?i)(invalid|positive|finite|range-separated|point nuclei)"):
        jax.jit(lambda e: evaluate("overlap", atm, bas, e, 1))(jnp.asarray(env)).block_until_ready()


def test_native_rejects_wrong_precision_and_ao_count():
    atm, bas, env = _primitive_tables()
    evaluate = native_evaluate()
    with pytest.raises(ValueError, match="float64"):
        evaluate("overlap", atm, bas, env.astype(np.float32), 1)
    with pytest.raises(ValueError, match="AO count"):
        evaluate("overlap", atm, bas, env, 2)


def test_vendor_checksums():
    from gradscf.integrals._native import build
    build.verify_vendor(Path(build.__file__).resolve().parent)


def test_eri_rejects_overflowing_shell_workspace_before_loading(monkeypatch):
    from gradscf.integrals.backends.native import ffi as native
    atm, bas, env = _primitive_tables()
    bas[0, 1], bas[0, 3] = 12, 3  # 273 AOs in one Cartesian shell.
    def unexpected_load():
        pytest.fail("Oversized ERI shell must be rejected before allocation/loading")
    monkeypatch.setattr(native, "register_integrals", unexpected_load)
    with pytest.raises(ValueError, match="shell workspace"):
        native.evaluate("eri", atm, bas, env, 273)


def test_spherical_eri_rejects_internal_cartesian_cache_overflow_before_loading(monkeypatch):
    from gradscf.integrals.backends.native import ffi as native
    atm, bas, env = _primitive_tables()
    bas[0, 1], bas[0, 3] = 12, 2
    # Output shell dimension is only 50, but libcint internally contracts 91
    # Cartesian components: its cache query returns zero (int32 overflow).
    # Block the loader so this RED regression cannot reach the unsafe C fill.
    def unexpected_load():
        pytest.fail("Internal Cartesian cache overflow must be rejected before native loading")
    monkeypatch.setattr(native, "register_integrals", unexpected_load)
    with pytest.raises(ValueError, match="Cartesian cache"):
        native.evaluate("eri", atm, bas, env, 50, cart=False)
