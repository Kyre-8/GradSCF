# MolGW molecular GW/BSE source review

Reviewed on 2026-09-30 against MolGW revision
`b831818d7a845c36f295d036dc8ceef59daf9991` (3.4). Tracked upstream sources were
unchanged. GradSCF comparison checkout: `test/molgw-expanded`, based on
`ae24d0e`, with the uncommitted comparison fixes. This is a focused source
review, not certification of all MolGW methods or distributed execution.

## Architecture and useful boundaries

| Responsibility | MolGW source | Implication for GradSCF |
| --- | --- | --- |
| Method selection and G/W iteration | `m_selfenergy_evaluation.f90` | Keep one self-energy implementation and separate G/W spectra. |
| Screened-interaction representation | `m_spectral_function.f90` | Distinguish pole, frequency-grid and static representations. Do not require all representations in the production interface. |
| Analytic pole self-energy | `m_gw_selfenergy_analytic.f90` | Useful independent small-system oracle for the CD implementation. |
| QP extraction | `m_selfenergy_tools.f90` | Root solving and physical branch selection are distinct operations. |
| Response orchestration | `m_linear_response.f90` | RPA/TDHF/TDDFT/BSE share response machinery. Avoid copying its large flag-driven orchestration routine. |
| Electron-hole kernel | `m_build_bse.f90` | Preserve exact matrix symmetries during construction. |
| Structured eigensolvers | `m_block_diago.f90` | Reuse GradSCF's shared stable-RPA solver. |
| Optical observables | `m_spectra.f90` | Use X+Y, both resonant and antiresonant poles, and explicit units. |

## Findings that affect the methane calculation

### 1. Converged QP root does not identify the dominant quasiparticle

`find_qp_energy_graphical` (lines 253-363) scans a finite frequency grid,
retains up to four candidate roots, and chooses the candidate with largest Z.
`find_fixed_point` (367-420) estimates Z from an adjacent-grid slope and
rejects Z below 1e-5. It does not enforce the upper bound Z<=1 mentioned in
its comment. The routine also has finite-window and grid-resolution limits.

A tiny executable probe compiled the unmodified `find_fixed_point` body:
for x=(-1,0,1), f(x)=0, the exact root at x=0 produced zero candidates because
the test is strictly `g(i)*g(i+1)<0`. For f(x)=0.1, it returned x=0.1, Z=1.
This demonstrates a grid-endpoint detection limitation, not failure of every
molecular calculation. Probe: `/private/tmp/molgw-design-review/root_probe.f90`.

GradSCF's secant/Newton/hybrid methods converge a locally selected root; its
current Z diagnostic does not perform dominant-root selection. Methane's high
virtual states can therefore have small residuals but low Z and different
roots within a nominally degenerate shell. This is consistent with branch
selection sensitivity; it does not establish that every low-Z root is wrong.

Borrow the separation of candidate detection, root refinement and physical
selection. Use AD slopes and the existing shared scalar solver for refinement.
A sign change across a self-energy pole is not sufficient evidence of a root:
require a small final Dyson residual. Do not copy fixed-grid finite differences,
the four-candidate cap, or implicit fallback energies.

### 2. MolGW evGW is not graphical multiroot selection inside every iteration

`m_inputparam.f90:182-189` selects EVSC for GNW0/GNWN. `se_init` sets a single
on-shell sample for EVSC. `m_selfenergy_evaluation.f90:579-581` invokes
`find_qp_energy_linearization` without the optional Z output, which takes the
unrenormalized update in `m_selfenergy_tools.f90:240-243`:

```
E_next = epsilon_MF + Re Sigma_c(E_current; G_current, W_current) + Delta_v.
```

This is the same fixed-point equation as GradSCF's current molecular evGW,
with optional damping in GradSCF. The inspected MolGW loop executes `nstep_gw`
iterations (default one); it does not exit based on a Dyson residual. It is
therefore incorrect to infer convergence simply from a completed MolGW run.

The local all-61-MO methane evGW0 attempt at damping 0.5 failed after 40
iterations with residual 1.522 Ha. MolGW's source design alone does not solve
this problem. Any acceleration must retain GradSCF's residual check. The
fixed-G/W partial derivative used for Z is not the full coupled evGW Jacobian;
changes in G and W contribute additional derivatives.

### 3. Symmetry belongs in construction, not a chain of relaxed checks

MolGW builds the lower triangle of A+B/A-B (`m_linear_response.f90:237-242`,
`m_build_bse.f90:640-643`), uses symmetric BLAS, and diagonalizes a Cholesky
reduction (`m_block_diago.f90:103-128`). It does not independently assemble
both triangles and test their agreement after long contractions.

Measured methane/aug-cc-pVDZ/Weigend-RI diagnostics in GradSCF:

- MO-factor maximum antisymmetry: 4.1477e-13; factor scale: 1.8714.
- A-block antisymmetric Frobenius norm: 5.3680e-13.
- Shared stable-RPA structure threshold: 4.2962e-13.
- Rebuilding with symmetrized factors gave A antisymmetry zero and B
  antisymmetry 7.76e-17. This probe checked matrix structure, not a finished
  three-method spectrum or backward validation.

The pending dimension-scaled screening tolerance alone is insufficient: it
passes screening validation but still fails the downstream structure check.
A minimal repair should validate the original factors, project only admissible
roundoff onto their known symmetry at a single BSE input boundary, and feed
the same factors to screening and both kernel blocks. Keep the shared solver's
structure/stability checks. Invalid asymmetric inputs must still be rejected.
This linear projection has a straightforward AD rule, but the complete changed
path still needs forward and derivative regression tests.

## Other design lessons

- **Reuse fixed W0:** MolGW skips rebuilding W after the first GnW0 iteration
  (`m_selfenergy_evaluation.f90:200-201`). GradSCF currently rebuilds fixed W0
  inside its shared driver. Invocation-local reuse is a concrete optimization;
  keys must include factors, screening energies/windows, grid and broadening.
  Avoid module-global or untracked disk caches and preserve parameter gradients.
- **Keep provenance checks:** MolGW's serial `SCREENED_COULOMB` reader
  (`m_spectral_function.f90:527-554`) loads dimensions, poles and residues,
  but does not validate geometry/basis/orbital fingerprints. Its BSE QP reader
  can fill missing energies from KS (`m_linear_response.f90:690-703`). GradSCF's
  recorded screening spectrum and computed/converged QP masks should remain.
- **Keep matrix-free BSE:** MolGW allocates distributed dense A+B/A-B even
  before its Davidson branch (`m_linear_response.f90:224-242,371-386`).
  ScaLAPACK distributes this storage; it does not remove quadratic global
  matrix storage. GradSCF should retain factorized operator actions and its
  bounded dense oracle rather than reproduce this storage layout.
- **Keep explicit window semantics:** MolGW separates G and W cutoffs but
  clips its QP target range to G's bounds (`m_selfenergy_tools.f90:79-114`).
  GradSCF's independent target/G/W/optical windows should not be collapsed.
- **Optical diagnostics:** MolGW reports oscillator-strength sums and static
  polarizability alongside cross sections (`m_spectra.f90:385-435`). GradSCF
  already provides the core observables. Add sums and selected-space metadata
  to the example, without introducing another spectrum implementation. A
  truncated basis/root sum is a diagnostic, not an exact TRK acceptance test.

## Minimal follow-up sequence

1. Resolve validated roundoff symmetry at one boundary; preserve solver gates.
2. Diagnose candidate QP roots and branch ambiguity using the existing residual
   and AD slope; do not equate this with evGW fixed-point acceleration.
3. Complete methane's explicitly documented windowed calculation and window/
   quadrature checks. Do not label it a full-space evGW spectrum.
4. Only after correctness, reuse fixed-W0 intermediates inside a run.

At the source-review checkpoint, no production-code changes were made and
the screening-tolerance experiment and methane spectrum were incomplete. No new full molecular MolGW run,
ScaLAPACK execution, GPU test, or complete repository regression was performed
for this review.

Source links at the reviewed revision:
[QP tools](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_selfenergy_tools.f90),
[GW driver](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_selfenergy_evaluation.f90),
[linear response](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_linear_response.f90),
[BSE construction](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_build_bse.f90),
[block diagonalization](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_block_diago.f90).


## Implemented follow-up

The subsequent methane run resolved the roundoff issue at the single BSE
functional boundary. The standalone screening-threshold experiment was
withdrawn. Forward/first-order response and affected BSE regressions passed
46 tests. All three windowed methane spectra completed; see the
[results and limitations](../../../reproducibility/gw_bse/methane_aug_cc_pvdz/README.md).
Global QP branch selection and full-space evGW convergence remain separate
work; neither is implied by this numerical repair. Invocation-local fixed-W0
reuse was subsequently implemented; see the
[resolvent reuse report](../../../reproducibility/gw_bse/cycloalkane_scaling/resolvent_reuse.md)
for its measured scope and limitations.
