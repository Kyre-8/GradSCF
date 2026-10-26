# Executed MolGW comparison for the methane spectrum

Executed 2026-10-01, Apple M4 Pro, CPU float64, one MolGW OpenMP thread,
MolGW 3.4 at `b831818d7a845c36f295d036dc8ceef59daf9991`. The external reference
used GNU Fortran 15.1, Accelerate, Homebrew LibXC and the previously built
libcint CINT2 interface. No GradSCF production source was changed in this step.

## What was compared

All three methods were run independently in MolGW: G0W0, GnW0 (evGW0),
and GnWn (evGW). Geometry, Cartesian aug-cc-pVDZ, Weigend RI, RHF, QP
indices 1-34 (Fortran), full 61-orbital G/W sums, and full singlet BSE were
matched to the native methane example. There are 145 optical transitions
(5 occupied times 29 virtual). BSE uses W0 for G0W0/evGW0 and updated W
for evGW. The broadening is the same 0.15 eV HWHM, without rescaling.

MolGW SCF tolerance was 1e-10; requesting 1e-12 initially stalled near the
numerical residual floor, so no unconverged SCF was accepted. GradSCF's
archived SCF tolerance remains 1e-12. The independently computed RHF energy
agrees to 4.34e-11 Ha when all RI modes are retained (MolGW stdout precision
also limits this energy comparison).

The GW reference uses MolGW's analytic RPA-pole self-energy. G0W0 uses its
graphical root scan: 4001 points, spacing 5e-4 Ha, +/-1 Ha around each MF
energy. GradSCF uses CD with 200 imaginary-axis quadrature points. Both use
GW eta=1e-5 Ha. No finite-difference Z accuracy claim is made for this grid.
MolGW evGW0/evGW ran 40 fixed-point updates, then one additional update to
verify stability: maximum changes were 0 and 2.31e-14 Ha in the matched
calculation. This checks convergence rather than assuming a fixed cycle
count implies convergence. GradSCF's archived tolerance is 1e-8 Ha.

## Two numerical reference configurations

### A. Default MolGW RI cutoff

Stock MolGW discards auxiliary Coulomb-metric eigenvalues below 1e-6.
For this methane geometry, its lowest eigenvalue is 2.451574e-7, so it keeps
108 of 109 directions. GradSCF's successful Cholesky whitening keeps all
109. Identical basis names therefore do not specify identical RI integrals.

The original MolGW executable was used for the GW stages. For BSE, the input
adapter described below separated the optical and screening windows.

| Method | Max QP error / Ha | Max BSE energy error / eV | Max oscillator-strength error |
| --- | ---: | ---: | ---: |
| G0W0 | 1.979e-4 | 0.005034 | 2.811e-4 |
| evGW0 | 1.979e-4 | 0.005037 | 2.817e-4 |
| evGW | 1.978e-4 | 0.005012 | 2.764e-4 |

These are real differences between the default RI approximations; they are
not reported as failures of identical equations or hidden by loose tolerances.

### B. Matched full auxiliary retention

A separate diagnostic executable lowers MolGW's fixed auxiliary cutoff to
1e-10, retaining the same 109-dimensional space as the existing GradSCF
calculation. `auxiliary_cutoff.patch` contains the two changed occurrences
(one numerical cutoff and its printed diagnostic). Only that numerical setting
and the BSE input adapter differ from the pinned sources. The threshold is
not fitted to excitation energies. Original sources and the original binary
remain unchanged; this is explicitly a reference variant, not stock defaults.

| Method | Max QP error / Ha | Max BSE energy error / eV | Max f error | Max sampled cross-section error / Mb |
| --- | ---: | ---: | ---: | ---: |
| G0W0 | 1.426e-8 | 5.330e-7 | 1.044e-8 | 8.626e-4 |
| evGW0 | 5.152e-9 | 6.354e-7 | 1.110e-9 | 1.090e-3 |
| evGW | 3.859e-9 | 5.306e-7 | 8.591e-10 | 9.614e-4 |

QP maxima cover all 34 requested states; BSE maxima cover all 145 roots.
Static polarizability differences are below 3.69e-8 a0^3. Cross sections
were evaluated with GradSCF's existing property API on MolGW's actual 1000-point
output frequency grid, avoiding interpolation of the native plotted curve.
The reported maximum includes that full output grid, which extends to 50 eV;
the comparison figure displays 8-24 eV. Printed reference precision and slightly
different atomic-unit constants limit comparisons at this level.

All three matched cases satisfy the existing molecular-reference tolerances:
QP 2e-6 Ha, BSE 3e-6 Ha, f 2e-5, static alpha 3e-4 a0^3 and cross section
2e-4 a0^2. No production tolerances were loosened for the comparison.

## BSE input adapter: why it is needed and what it changes

MolGW's stock BSE orchestration binds optical and screening windows. Its
RI route forbids writing SCREENED_COULOMB because the stored auxiliary frame
is runtime-dependent; this prohibition was not bypassed. Its stock static
screening fallback also uses the supplied MF spectrum, rather than implicitly
inheriting an evGW screening spectrum from a GW result object.

`bse_input_driver.patch` changes only the main program immediately before
calling the existing `polarizability` routine:

1. Initialize the full screening transition setup from the ordinary input.
2. Filter the output optical transition table to virtual indices <=34,
   leaving the input screening limit at all 61 orbitals.
3. For evGW only, explicitly load MolGW's own ENERGY_QP into the independent
   screening-energy argument. Orbitals and integrals remain those from its RHF.

The eigensolver, screening, kernel and property modules are unmodified.
No GradSCF energies, factors or transition moments are fed into the MolGW
reference. The adapter is enabled only by documented environment variables.

A control using the same 29-virtual optical/screening window for stock MolGW
and the adapter gave exactly identical printed energies, oscillator strengths
and all cross-section output columns. The control input, summarized differences
and output hashes are included here; raw control outputs remain local.
It tests the no-change route; the matched comparisons then
exercise the separate optical/screening windows and updated evGW screening.

## Interpretation

The ~16.5 eV strong narrow peak is reproduced independently by MolGW, with
both its default cutoff and the matched cutoff. Once numerical settings are
aligned, the two codes essentially overlay, while both disagree strongly
with the smooth experimental envelope. This supports implementation
consistency for this fixed-geometry, finite-space static BSE model; it is not
experimental validation or proof of basis/window/continuum convergence.
Do not infer that broadening alone will resolve the physical discrepancy.

## Artifacts and reproduction

- `../molgw_comparison.json`: reference arrays, per-method differences,
  fixed-point extra-step checks, control results and MolGW timings.
- `provenance.json`: binary/output hashes and exact execution commands.
  Raw stdout logs, QP files and control YAML/DAT outputs remain local and are
  not committed; their hashes preserve
  the identity of the recorded outputs, not their availability in this tree.
- `build.json`, `full_rank_build.json`: actual compile/link argument lists,
  including the original numerical object hashes. Paths record this local run.
- `default_cutoff/`, `matched_cutoff/`: exact input files. Full-precision QP
  energies, selected timings and diagnostics are retained in the comparison
  JSON; rerunning the documented commands regenerates stdout logs.
- The two `.patch` files fully describe the external-source adaptations.
  Apply them only to a separate reference checkout or copied source files;
  compile with the recorded flags and link against the pinned reference
  objects. Do not apply them to GradSCF or silently replace stock MolGW.

Runtime environment used:

```sh
DYLD_LIBRARY_PATH=/private/tmp/libcint-molgw-build OMP_NUM_THREADS=1 \
  /private/tmp/molgw-reference-20260930/molgw gw.in

DYLD_LIBRARY_PATH=/private/tmp/libcint-molgw-build OMP_NUM_THREADS=1 \
  MOLGW_COMPARE_OPTICAL_LAST=34 MOLGW_COMPARE_QP_SCREENING=no \
  /private/tmp/molgw-methane-20261001/driver/molgw_compare bse.in
```

Use `MOLGW_COMPARE_QP_SCREENING=yes` for evGW. For matched-cutoff calculations,
use `molgw_compare_full_rank` in both commands. Every method starts in its own
fresh directory; subsequent BSE/check stages intentionally reuse only that
method's local RHF restart and QP energies. No SCREENED_COULOMB file is used.

To recollect comparisons after reproducing that directory layout and rerunning
the native methane example to regenerate its NPZ snapshots:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python tests/comparisons/collect_methane_molgw.py \
  --runs /private/tmp/molgw-methane-20261001
python examples/bse/plot_methane_molgw.py
```

The matched MolGW GW stages reported 9.548 / 0.549 / 1.234 s and BSE
0.044 / 0.048 / 0.048 s for G0W0 / evGW0 / evGW. These are reference-run
wall times with different algorithms, JIT conventions and some concurrent
runs; they are not a controlled performance ranking against GradSCF.
GPU, distributed execution and evGW derivatives were not tested here.
