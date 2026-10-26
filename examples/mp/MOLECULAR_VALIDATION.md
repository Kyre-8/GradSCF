# Molecular MP2--MP6 comparisons with 6-31G*

Date: 2026-10-10. Local macOS 26.6.2 arm64, CPU backend, Python 3.12.2,
JAX 0.8.1, PySCF 2.9.0, float64. Both programs independently compute their
integrals and HF references. All cases use Cartesian 6-31G*, integer
occupations, `conv_tol=1e-13`, `conv_tol_grad=1e-11`, `max_cycle=200`
and the core-Hamiltonian initial guess. Every virtual MO is retained.
Only LiH and BeH freeze one occupied spatial 1s core per spin.

## Command and independent reference

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python -u examples/mp/compare_molecules_pyscf.py
```

All nine cases completed with the example's assertions passing. GradSCF uses
one projected residual, Taylor lifting and the shared checked linear solver.
Each individual E2--E6 is compared with an independent NumPy
Rayleigh--Schrodinger recurrence using PySCF full-space Hamiltonian actions.
E2 is also checked against native PySCF MP2/UMP2 and the specialized GradSCF
MP2/UMP2 path. The higher-order reference is not a native PySCF MP6 API.
The example and water comparison reuse [_pyscf_reference.py](_pyscf_reference.py);
no PySCF import or reference algorithm is added to production MP.

Comparisons use absolute tolerances 3e-9 Ha for independent HF energies and
corrections, 2e-10 Ha for the two GradSCF E2 paths, and 2e-8 for spin squared,
all with zero relative tolerance. Maximum observed E2--E6 disagreement is
4.429e-13 Ha (H4+); maximum response residual is 1.024e-16 Ha (neutral H4).

## Systems and spin references

Coordinates are Angstrom. H2: z=(0,0.74); LiH: z=(0,1.6); BeH:
z=(0,1.34). H3 doublet/quartet: z=(0,0.85,1.9); H4/H4+:
z=(0,0.8,1.8,2.8), with x=y=0. Atoms He and Li are at the origin.
Neutral H2/H4/He/LiH use singlet RHF. H3/H4+/Li/BeH use doublet UHF;
the separate H3 quartet uses UHF with (Nalpha,Nbeta)=(3,0).
These are specified comparison geometries, not geometry-optimized benchmarks.

| System | Reference | Frozen count | Retained determinants | Actual maximum rank | GradSCF spin squared | Maximum E2--E6 difference / Ha |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H2 | RHF | 0 | 16 | 2 | 0.000000000 | 1.291e-15 |
| H4 | RHF | 0 | 784 | 4 | 0.000000000 | 5.855e-14 |
| He | RHF | 0 | 4 | 2 | 0.000000000 | 6.418e-17 |
| LiH | RHF | 1 | 256 | 2 | 0.000000000 | 1.588e-14 |
| H3 | UHF | 0 | 90 | 3 | 0.781153603 | 2.297e-13 |
| H3 quartet | UHF | 0 | 20 | 3 | 3.750000000 | 2.208e-15 |
| H4+ | UHF | 0 | 224 | 3 | 0.792134899 | 4.429e-13 |
| Li | UHF | 0 | 1575 | 3 | 0.750002454 | 1.414e-15 |
| BeH | UHF | 1 | 1920 | 3 | 0.751791532 | 1.731e-13 |

PySCF agrees with the displayed spin-squared values to nine decimal places.
The ideal doublet value is 0.75; the H3 and H4+ UHF references have modest spin
contamination. Their MP comparisons therefore validate the same unrestricted
reference, not a spin-projected or ROHF calculation. Quartet H3 has no beta
electrons and spin squared 3.75. Neutral H4 retains actual quadruple excitations,
testing a larger excitation rank than the two-/three-electron cases.

## Energy corrections

All numbers are Hartree. E2--E6 are individual corrections; the last column is
the HF energy plus their sum. The PySCF-action values agree within the errors
above; these partial sums are not asserted to be converged exact energies.

| System | E2 | E3 | E4 | E5 | E6 | MP6 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 | -0.017381257647 | -0.005204191869 | -0.001602365278 | -0.000502183896 | -0.000157958465 | -1.151603274351 |
| H4 | -0.040167996392 | -0.011771274490 | -0.004139888850 | -0.001554742032 | -0.000613876365 | -2.241376891126 |
| He | -0.011200122910 | -0.002865202141 | -0.000713634432 | -0.000172032007 | -0.000039819557 | -2.870151237201 |
| LiH | -0.015051581287 | -0.004236628394 | -0.001449230526 | -0.000574706162 | -0.000267766816 | -8.002290861444 |
| H3 | -0.018370186815 | -0.005909831185 | -0.002309585647 | -0.001073361243 | -0.000563031924 | -1.627150877535 |
| H3 quartet | -0.000735097569 | -0.000217085363 | -0.000064784443 | -0.000019498408 | -0.000005910235 | -1.159856820913 |
| H4+ | -0.021799488576 | -0.006322509907 | -0.002471987967 | -0.001171642546 | -0.000630459968 | -1.787029209746 |
| Li | -0.000485952851 | -0.000011528497 | -0.000010195667 | -0.000001242910 | -0.000000391714 | -7.431881643946 |
| BeH | -0.020724496453 | -0.005591509116 | -0.001914211735 | -0.000792180747 | -0.000396398973 | -15.176708545751 |

The generic representation can reach the whole finite excitation space for
these small correlated electron counts. This does not involve a production
FCI diagonalization. Frozen cores still contribute to the physical Fock and
reference energy in both implementations.

## Runtime and scope

Observed GradSCF HF + Taylor elapsed times from the completed nine-case run:

| System | Seconds |
| --- | ---: |
| H2 | 5.033 |
| H4 | 6.051 |
| He | 5.170 |
| LiH | 5.786 |
| H3 | 6.103 |
| H3 quartet | 6.099 |
| H4+ | 5.985 |
| Li | 7.852 |
| BeH | 7.801 |

The example clears JAX executable caches between systems. Times include
compilation/startup and overlap with a separate local MP regression run; they
are provenance for this run, not a fair speed comparison or warmed scaling
benchmark. PySCF's RS reference uses a different action/storage representation
and computes no AD graph. No GPU or peak-memory measurement was performed.

This extension measures forward energy corrections and reference spin values.
It does not extend the existing geometry-derivative coverage. No HF stability
audit or guarantee of MP-series convergence at lambda=1 is claimed.

For comparison, water/Cartesian 6-31G*, frozen=1, would require 5,446,687
determinants for the default MP6 state space, beyond the generic engine's
5,000-determinant and connection capacity controls. It was rejected during
static preflight, not computed with a smaller virtual space. The previous
water native-MP2 comparison remains documented in
[VALIDATION.md](../../src/gradscf/mp/VALIDATION.md).
