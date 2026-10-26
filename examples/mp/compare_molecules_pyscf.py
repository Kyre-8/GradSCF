"""Canonical molecular MP2--MP6 with Cartesian 6-31G*, RHF and UHF.

PySCF 2.9.0 supplies native MP2. Higher-order reference values use its full
Hamiltonian action and an independent NumPy RS recurrence. Every virtual MO
is retained. LiH/BeH freeze the 1s core; other systems correlate all electrons.
This is a correctness comparison, not a matched performance benchmark.
"""
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, mp
from pyscf import gto as pyscf_gto, scf as pyscf_scf, mp as pyscf_mp
from _pyscf_reference import mp_coefficients

order = 6
cases = [
    # name, geometry / Angstrom, 2S, charge, frozen-core count
    ("H2", "H 0 0 0; H 0 0 .74", 0, 0, 0),
    ("H4", "H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.8", 0, 0, 0),
    ("He", "He 0 0 0", 0, 0, 0),
    ("LiH", "Li 0 0 0; H 0 0 1.6", 0, 0, 1),
    ("H3", "H 0 0 0; H 0 0 .85; H 0 0 1.9", 1, 0, 0),
    ("H3 quartet", "H 0 0 0; H 0 0 .85; H 0 0 1.9", 3, 0, 0),
    ("H4+", "H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.8", 1, 1, 0),
    ("Li", "Li 0 0 0", 1, 0, 0),
    ("BeH", "Be 0 0 0; H 0 0 1.34", 1, 0, 1),
]
records = []

for name, atom, spin, charge, frozen in cases:
    print("\n%s / 6-31G*, %s, frozen=%d" % (name, "UHF" if spin else "RHF", frozen), flush=True)
    start = perf_counter()
    mol = gto.M(atom=atom, basis="6-31g*", cart=True, spin=spin, charge=charge)
    mf = (scf.UHF if spin else scf.RHF)(mol, conv_tol=1e-13,
        conv_tol_grad=1e-11, max_cycle=200).run()
    if not mf.converged:
        raise RuntimeError("%s GradSCF HF did not converge" % name)
    pt = mp.MP(mf, order=order, frozen=frozen, with_t2=False).run()
    gradscf_seconds = perf_counter()-start

    start = perf_counter()
    ref_mol = pyscf_gto.M(atom=atom, basis="6-31g*", cart=True,
        spin=spin, charge=charge, verbose=0)
    ref_mf = (pyscf_scf.UHF if spin else pyscf_scf.RHF)(ref_mol).run(
        conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=200, init_guess="hcore")
    if not ref_mf.converged:
        raise RuntimeError("%s PySCF HF did not converge" % name)
    reference = mp_coefficients(ref_mf, order, frozen=frozen)
    reference_seconds = perf_counter()-start
    native_mp2 = pyscf_mp.MP2(ref_mf, frozen=frozen).run().e_corr
    direct = mp.MP2(mf, frozen=frozen, with_t2=False).run().e_corr
    actual = np.asarray(pt.corrections)

    s2 = reference_s2 = 0.
    if spin:
        na, nb = ref_mol.nelec
        ca, cb = np.asarray(mf.mo_coeff)
        overlap = np.asarray(mf.reference.overlap_matrix)
        s2 = (spin/2)*(spin/2+1)+nb-np.sum((ca[:, :na].T @ overlap @ cb[:, :nb])**2)
        reference_s2 = ref_mf.spin_square()[0]
    np.testing.assert_allclose(mf.e_tot, ref_mf.e_tot, atol=3e-9, rtol=0.)
    np.testing.assert_allclose(s2, reference_s2, atol=2e-8, rtol=0.)
    np.testing.assert_allclose(actual, reference, atol=3e-9, rtol=0.)
    np.testing.assert_allclose(actual[0], native_mp2, atol=3e-9, rtol=0.)
    np.testing.assert_allclose(actual[0], direct, atol=2e-10, rtol=0.)

    print("HF / Ha: GradSCF = %.12f  PySCF = %.12f" % (mf.e_tot, ref_mf.e_tot))
    print("<S^2>: GradSCF = %.9f  PySCF = %.9f" % (s2, reference_s2))
    print("Retained determinants = %d; actual maximum rank = %d" %
          (pt.space.size, max(pt.space.ranks)))
    print("Order   GradSCF correction    PySCF-action correction   absolute difference / Ha")
    for degree, value, ref in zip(range(2, order+1), actual, reference):
        print("%3d     % .12f         % .12f           %.3e" %
              (degree, value, ref, abs(value-ref)))
    print("Native PySCF MP2 difference / Ha = %.3e" % abs(actual[0]-native_mp2))
    print("MP6 total / Ha = %.12f; response residual / Ha = %.3e" %
          (pt.e_tot, pt.result.residual_norm))
    print("Elapsed: GradSCF HF + Taylor = %.3f s; PySCF HF + RS = %.3f s" %
          (gradscf_seconds, reference_seconds), flush=True)
    records.append(dict(name=name, corrections=actual, reference=reference,
        hf_energy=float(mf.e_tot), hf_error=float(mf.e_tot-ref_mf.e_tot),
        total_energy=float(pt.e_tot), spin_squared=float(s2),
        spin_error=float(s2-reference_s2), native_mp2_error=float(actual[0]-native_mp2),
        determinant_count=pt.space.size, max_rank=max(pt.space.ranks),
        response_residual=float(pt.result.residual_norm),
        gradscf_seconds=gradscf_seconds, reference_seconds=reference_seconds))
    # Bound executable-cache accumulation during this multi-system CPU example.
    jax.clear_caches()

# Measured summary: E2, E3, E4, E5, E6 / Ha; maximum reference difference.
# macOS arm64 CPU, float64, JAX 0.8.1, PySCF 2.9.0; 2026-10-10.
# H2: (-0.017381257647, -0.005204191869, -0.001602365278, -0.000502183896, -0.000157958465)
#   max difference = 1.291e-15 Ha; <S^2> = 0.000000000
# H4: (-0.040167996392, -0.011771274490, -0.004139888850, -0.001554742032, -0.000613876365)
#   max difference = 5.855e-14 Ha; <S^2> = 0.000000000
# He: (-0.011200122910, -0.002865202141, -0.000713634432, -0.000172032007, -0.000039819557)
#   max difference = 6.418e-17 Ha; <S^2> = 0.000000000
# LiH: (-0.015051581287, -0.004236628394, -0.001449230526, -0.000574706162, -0.000267766816)
#   max difference = 1.588e-14 Ha; <S^2> = 0.000000000
# H3: (-0.018370186815, -0.005909831185, -0.002309585647, -0.001073361243, -0.000563031924)
#   max difference = 2.297e-13 Ha; <S^2> = 0.781153603
# H3 quartet: (-0.000735097569, -0.000217085363, -0.000064784443, -0.000019498408, -0.000005910235)
#   max difference = 2.208e-15 Ha; <S^2> = 3.750000000
# H4+: (-0.021799488576, -0.006322509907, -0.002471987967, -0.001171642546, -0.000630459968)
#   max difference = 4.429e-13 Ha; <S^2> = 0.792134899
# Li: (-0.000485952851, -0.000011528497, -0.000010195667, -0.000001242910, -0.000000391714)
#   max difference = 1.414e-15 Ha; <S^2> = 0.750002454
# BeH: (-0.020724496453, -0.005591509116, -0.001914211735, -0.000792180747, -0.000396398973)
#   max difference = 1.731e-13 Ha; <S^2> = 0.751791532
# Full settings and measured results: MOLECULAR_VALIDATION.md.
