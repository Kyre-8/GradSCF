"""Complete N2 CCSD(T) nuclear response at exactly degenerate pi orbitals.

Native integrals, the implicit HF response and the implicit CC amplitudes are
inside the GradSCF differentiated map. The independent PySCF calculation uses
the (T)-corrected Lambda equations, and independently reconverges HF/CCSD at
each displaced geometry. Energies are Hartree; coordinates are bohr.
"""
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


COORDINATES = np.array([[0., 0., -1.04], [0., 0., 1.04]])
BOND_DIRECTION = np.array([[0., 0., -.5], [0., 0., .5]])


def test_ccsd_t_reuses_existing_correlated_adapter(monkeypatch):
    from gradscf import cc
    from gradscf_tools import correlated_response

    reference = SimpleNamespace(unrestricted=False, coordinates=COORDINATES)
    monkeypatch.setattr(correlated_response, "MolecularEnergy", lambda *a, **k: reference)
    mf = SimpleNamespace(xc="hf", mo_occ=np.array([2., 2., 0.]), mo_coeff=np.eye(3))
    model = correlated_response.CorrelatedEnergy(mf, "ccsd(t)")
    assert model.reference is reference
    assert isinstance(model.config, cc.CCConfig)
    assert model.config.method == "ccsd"


def _pyscf_sample(coordinates, *, gradient=False):
    from pyscf import cc, gto, scf
    from pyscf.cc import ccsd_t_lambda
    from pyscf.grad import ccsd_t

    mol = gto.M(atom=list(zip(("N", "N"), coordinates)), basis="sto-3g",
                unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 250
    mf.kernel(dm0=1.4 * np.eye(10))
    assert mf.converged
    assert np.min(np.abs(np.diff(mf.mo_energy))) < 1e-9
    correlated = cc.CCSD(mf)
    correlated.conv_tol, correlated.conv_tol_normt = 1e-12, 1e-11
    correlated.max_cycle = 200
    correlated.kernel()
    assert correlated.converged
    correction = correlated.ccsd_t()
    total = correlated.e_tot + correction
    if not gradient:
        return total
    eris = correlated.ao2mo()
    converged, l1, l2 = ccsd_t_lambda.kernel(
        correlated, eris, correlated.t1, correlated.t2, max_cycle=100, tol=1e-11,
        verbose=0)
    assert converged
    # Ordinary CCSD Lambda amplitudes do not give a CCSD(T) gradient.
    derivative = ccsd_t.Gradients(correlated).kernel(
        correlated.t1, correlated.t2, l1, l2, eris=eris)
    return total, derivative, correction


def _nitrogen_model():
    from gradscf import gto, scf
    from gradscf_tools.correlated_response import CorrelatedEnergy

    mol = gto.M(atom=list(zip(("N", "N"), COORDINATES)), basis="sto-3g",
                unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", init_guess=1.4 * np.eye(10),
                 conv_tol=1e-13, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                 max_cycle=250).run()
    assert mf.converged
    assert np.min(np.abs(np.diff(np.asarray(mf.mo_energy)))) < 1e-9
    return CorrelatedEnergy(mf, "ccsd(t)")


def test_nitrogen_ccsd_t_geometry_matches_independent_analytic_and_fd():
    pytest.importorskip("pyscf")
    model = _nitrogen_model()
    value, gradient = jax.jit(jax.value_and_grad(model))(model.coordinates)
    assert np.isfinite(value) and np.isfinite(np.asarray(gradient)).all()
    reference, analytic, triples = _pyscf_sample(COORDINATES, gradient=True)
    assert abs(triples) > 1e-6
    np.testing.assert_allclose(value, reference, atol=5e-9, rtol=0)
    np.testing.assert_allclose(gradient, analytic, atol=5e-7, rtol=0)
    derivative = jnp.sum(gradient * jnp.asarray(BOND_DIRECTION))
    forward = jax.jit(model)
    external_errors, internal_errors = [], []
    for step in (3e-4, 1e-4):
        plus = COORDINATES + step * BOND_DIRECTION
        minus = COORDINATES - step * BOND_DIRECTION
        reference_plus, reference_minus = _pyscf_sample(plus), _pyscf_sample(minus)
        actual_plus, actual_minus = forward(plus), forward(minus)
        external_errors.append(abs(float(derivative) - (reference_plus - reference_minus) / (2 * step)))
        internal_errors.append(abs(float(derivative) - float(actual_plus - actual_minus) / (2 * step)))
        np.testing.assert_allclose((actual_plus, actual_minus),
                                   (reference_plus, reference_minus), atol=5e-9, rtol=0)
        np.testing.assert_allclose(derivative, (reference_plus - reference_minus) / (2 * step),
                                   atol=5e-7, rtol=2e-5)
        np.testing.assert_allclose(derivative, (actual_plus - actual_minus) / (2 * step),
                                   atol=5e-7, rtol=2e-5)
    np.testing.assert_allclose(gradient[:, :2], 0., atol=2e-8, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8, rtol=0)
    print("N2 CCSD(T): E = %.12f Ha; dE/dR = %.10f Ha/bohr; "
          "PySCF (T)-Lambda dE/dR = %.10f Ha/bohr"
          % (value, derivative, np.sum(analytic * BOND_DIRECTION)))
    print("absolute errors: energy = %.3e Ha; analytic gradient = %.3e Ha/bohr; "
          "external FD = %.3e Ha/bohr; internal FD = %.3e Ha/bohr"
          % (abs(float(value) - reference), np.max(np.abs(np.asarray(gradient) - analytic)),
             max(external_errors), max(internal_errors)))


def test_nitrogen_ccsd_t_hvp_matches_independent_gradient_differences():
    pytest.importorskip("pyscf")
    overall_started = perf_counter()
    started = perf_counter()
    model = _nitrogen_model()
    print("N2 CCSD(T) HVP: native RHF setup = %.3f s"
          % (perf_counter() - started), flush=True)
    started = perf_counter()
    _, _, triples = _pyscf_sample(COORDINATES, gradient=True)
    assert abs(triples) > 1e-6
    print("N2 CCSD(T) HVP: PySCF base gradient = %.3f s; (T) = %.10e Ha"
          % (perf_counter() - started, triples), flush=True)

    mixed_direction = np.array([[.2, -.3, -.4], [-.1, .2, .5]])
    mixed_direction /= np.linalg.norm(mixed_direction)
    directions = (BOND_DIRECTION, mixed_direction)
    # Keep direction dynamic: the second call reuses the same executable.
    hvp = jax.jit(lambda coordinates, direction: jax.jvp(
        jax.grad(model), (coordinates,), (direction,))[1])
    products, errors = [], []
    for index, direction in enumerate(directions):
        label = "bond" if index == 0 else "mixed Cartesian"
        started = perf_counter()
        observed = np.asarray(hvp(model.coordinates, jnp.asarray(direction)))
        print("N2 CCSD(T) HVP: %s %s = %.3f s"
              % (label, "compile+execute" if index == 0 else "reuse execute",
                 perf_counter() - started), flush=True)
        assert np.isfinite(observed).all()
        products.append(observed)
        np.testing.assert_allclose(observed.sum(axis=0), 0., atol=2e-8, rtol=0)

        for step in (3e-4, 1e-4):
            started = perf_counter()
            plus = COORDINATES + step * direction
            minus = COORDINATES - step * direction
            _, plus_gradient, _ = _pyscf_sample(plus, gradient=True)
            _, minus_gradient, _ = _pyscf_sample(minus, gradient=True)
            finite = (plus_gradient - minus_gradient) / (2 * step)
            error = float(np.max(np.abs(observed - finite)))
            errors.append(error)
            print("N2 CCSD(T) HVP: %s step = %.1e bohr; "
                  "PySCF gradient pair = %.3f s; max error = %.3e Ha/bohr^2"
                  % (label, step, perf_counter() - started, error), flush=True)
            np.testing.assert_allclose(observed, finite, atol=3e-6, rtol=2e-5)

    mixed_left = np.sum(directions[0] * products[1])
    mixed_right = np.sum(directions[1] * products[0])
    symmetry_error = abs(mixed_left - mixed_right)
    print("N2 CCSD(T) HVP: max external FD error = %.3e Ha/bohr^2; "
          "mixed Hessian symmetry error = %.3e Ha/bohr^2; total = %.3f s"
          % (max(errors), symmetry_error, perf_counter() - overall_started), flush=True)
    np.testing.assert_allclose(mixed_left, mixed_right, atol=2e-8, rtol=0)
