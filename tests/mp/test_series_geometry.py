"""All-electron N2 MP4/MP6 nuclear response at exact internal MO degeneracy.

The independent oracle reconverges PySCF RHF at every displacement, then uses
its direct FCI action and ordinary NumPy Rayleigh-Schrodinger recursion. It
does not differentiate canonical eigenvectors or reuse GradSCF MP equations.
Energies are in Hartree, coordinates and bond displacements in bohr.
"""
import runpy
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


COORDINATES = np.array([[0., 0., -1.04], [0., 0., 1.04]])
BOND_DIRECTION = np.array([[0., 0., -.5], [0., 0., .5]])
STEPS = (3e-4, 1e-4)


def _pyscf_samples():
    """Independent HF and E2..E6 at the reference and four displaced geometries."""
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    mp_coefficients = runpy.run_path("examples/mp/_pyscf_reference.py")["mp_coefficients"]
    samples = {}
    for shift in (0., *[sign*step for step in STEPS for sign in (1., -1.)]):
        coordinates = COORDINATES+shift*BOND_DIRECTION
        mol = gto.M(atom=list(zip(("N", "N"), coordinates)), basis="sto-3g",
                    unit="Bohr", cart=True, verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 250
        # The same symmetry-preserving AO seed chooses the closed-shell branch
        # in each independent SCF. Tr[D S] = 14 for these normalized STO-3G AOs.
        mf.kernel(dm0=1.4*np.eye(10))
        assert mf.converged
        assert np.min(np.abs(np.diff(mf.mo_energy))) < 1e-9
        samples[shift] = dict(hf_energy=mf.e_tot, corrections=mp_coefficients(mf, 6),
                              orbital_energies=mf.mo_energy)
    return samples


def _reference_energy(sample, order):
    return sample["hf_energy"]+np.sum(sample["corrections"][:order-1])


@pytest.fixture(scope="module")
def nitrogen():
    from gradscf import gto, scf
    samples = _pyscf_samples()
    mol = gto.M(atom=list(zip(("N", "N"), COORDINATES)), basis="sto-3g",
                unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", init_guess=1.4*np.eye(10),
                 conv_tol=1e-13, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                 max_cycle=250).run()
    assert mf.converged
    assert np.min(np.abs(np.diff(np.asarray(mf.mo_energy)))) < 1e-9
    np.testing.assert_allclose(mf.e_tot, samples[0.]["hf_energy"], atol=2e-10, rtol=0)
    return mf, samples


@pytest.mark.parametrize("order", [4, 6])
def test_nitrogen_high_order_geometry_matches_independent_rs(nitrogen, order):
    from gradscf import mp
    from gradscf_tools.correlated_response import CorrelatedEnergy
    mf, samples = nitrogen
    config = mp.MPConfig(order=order, algorithm="series", orbital_basis="noncanonical",
                          max_determinants=20000, with_t2=False)
    model = CorrelatedEnergy(mf, "mp", config=config)
    print("N2 MP%d: lowering the energy gradient" % order, flush=True)
    started = time.perf_counter()
    lowered_gradient = jax.jit(jax.value_and_grad(model)).lower(model.coordinates)
    gradient_lower_seconds = time.perf_counter()-started
    print("N2 MP%d: gradient lowered in %.3f s; compiling" %
          (order, gradient_lower_seconds), flush=True)
    started = time.perf_counter()
    differentiated = lowered_gradient.compile()
    gradient_compile_seconds = time.perf_counter()-started
    print("N2 MP%d: gradient compiled in %.3f s; evaluating" %
          (order, gradient_compile_seconds), flush=True)
    started = time.perf_counter()
    value, gradient = differentiated(model.coordinates)
    jax.block_until_ready((value, gradient))
    gradient_evaluation_seconds = time.perf_counter()-started
    print("N2 MP%d: gradient evaluated in %.3f s" %
          (order, gradient_evaluation_seconds), flush=True)
    assert np.isfinite(value) and np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(value, _reference_energy(samples[0.], order), atol=1e-8, rtol=0)
    derivative = jnp.sum(gradient*jnp.asarray(BOND_DIRECTION))
    # Reuse the compiled gradient function for all four displaced energy
    # evaluations. These are gradient-function calls, not forward-only timings.
    print("N2 MP%d: evaluating FD energies with the compiled gradient function" % order, flush=True)
    for step in STEPS:
        plus, _ = differentiated(model.coordinates+step*BOND_DIRECTION)
        minus, _ = differentiated(model.coordinates-step*BOND_DIRECTION)
        reference_plus = _reference_energy(samples[step], order)
        reference_minus = _reference_energy(samples[-step], order)
        finite = (plus-minus)/(2*step)
        reference_finite = (reference_plus-reference_minus)/(2*step)
        print("N2 MP%d h=%.1e bohr: GradSCF FD=%.12f; PySCF RS FD=%.12f; "
              "AD-FD=%.3e; AD-reference FD=%.3e Ha/bohr; energy errors (+/-)=%.3e/%.3e Ha"
              % (order, step, finite, reference_finite, derivative-finite,
                 derivative-reference_finite, plus-reference_plus, minus-reference_minus), flush=True)
        np.testing.assert_allclose((plus, minus), (reference_plus, reference_minus),
                                   atol=1e-8, rtol=0)
        np.testing.assert_allclose(derivative, finite, atol=5e-7, rtol=2e-5)
        np.testing.assert_allclose(derivative, reference_finite,
                                   atol=5e-7, rtol=2e-5)
    np.testing.assert_allclose(gradient[:, :2], 0., atol=2e-8, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8, rtol=0)
    print("N2 MP%d: E = %.12f Ha; dE/dR = %.10f Ha/bohr; "
          "gradient lower/compile/evaluate = %.3f/%.3f/%.3f s"
          % (order, value, derivative, gradient_lower_seconds, gradient_compile_seconds,
             gradient_evaluation_seconds), flush=True)
