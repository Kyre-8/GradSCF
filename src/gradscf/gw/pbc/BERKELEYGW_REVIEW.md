# Periodic GW review against BerkeleyGW

## Direct source audit and implementation boundaries, 2026-10-18

GradSCF is an independent implementation. BerkeleyGW supplies external
algorithm and numerical references, not a runtime dependency or a required
public-interface convention. Different finite broadenings, quadratures or
update maps require controlled comparisons; their difference alone is not a
correctness defect. The installed 4.0 sources were read directly. The
original and finite-space comparison builds have identical hashes for
`chi_summation.f90`, `epsinv.f90`, `mtxel_cor.f90` and
`shiftenergy_dyn.f90`; the finite-space input guard patch is separate.

The following findings describe code, not inferred material accuracy:

- **QP candidates.** `Sigma/shiftenergy_dyn.f90:338--349` scans the supplied
  frequency table for positive-to-negative crossings of
  `epsilon_MF + Re(Sigma) - Vxc - omega`. It linearly interpolates each
  crossing and selects the real energy nearest Eqp0. GradSCF's shared
  `gw/qp.py::solve_quasiparticle_roots` refines six seeds, checks the original
  function residual and positive slope, then selects nearest on-shell or a
  supplied continuation target. Neither rule guarantees the largest
  spectral peak. Their candidate counts cover different searches, not all
  roots on the real axis. The BGW comment at line 307 is stronger than its
  actual crossing condition: the implemented segment test requires
  `d Re(Sigma)/d omega < 1`, not necessarily a nonpositive derivative.
- **Eigenvalue self-consistency.** When full-frequency Eqp1 is chosen as
  feedback, BGW freezes the current G/W while constructing a Sigma table,
  solves that table equation, then writes the next energy file
  (`shiftenergy_dyn.f90:246--270`, `sigma_main.f90:2099--2123`). Its
  `eqp.py` also permits Eqp0 feedback, so an inner nonlinear solve is not
  universal to every BGW eigenvalue workflow. GradSCF's periodic evGW
  instead updates the coupled residual
  `F(E) = E - epsilon_ref - Sigma_x + Vref - Re(Sigma_c(E;G(E),W(E)))`
  directly. The common fixed-point equation does not imply equal numerical
  trajectories or branch selection. A fixed-G/W AD frequency slope is only
  a numerical preconditioner, not the full coupled derivative dF/dE.
- **G versus W updates.** `Common/eqpcor.f90:145--156` replaces energies
  after checking their reference column; it does not repeatedly add the
  same correction. Updating the energy input to Sigma changes G. Updating W
  additionally requires recomputing Epsilon with the corrected transitions.
  The presence of a file named `eqp.dat` alone does not establish evGW.
- **Frequency conventions.** `Epsilon/inread.f90:579--603` uses an
  independently broadened real-frequency W and an unbroadened imaginary
  axis. `Sigma/mtxel_cor.f90:3130--3177` interpolates real-frequency W;
  GradSCF solves its response directly at residue frequencies. The current
  static-subtracted GradSCF imaginary-contour G kernel is also unbroadened.
  Its legacy finite-eta residue convention differs from the independently
  broadened W option. The latter is a general numerical control used in the
  matched external diagnostic, not a BerkeleyGW-specific physics model.
  Public API changes need an independent GradSCF use case and convergence
  evidence; they do not follow automatically from this comparison.
- **Meaning of scGW.** `Sigma/scGWtool.py:20--24,176--183,283--287` constructs
  and diagonalizes a static subspace QP Hamiltonian. Its default W stays
  fixed, and other mean-field terms are rotated rather than all rebuilt.
  This does not validate GradSCF's full-frequency Matsubara Dyson scGW or its
  analytic continuation. Imaginary-axis residual, particle number and
  causal-residue checks do not establish stable real-axis bands or DOS.
- **Bands and spectra.** `BSE/intwfn.f90:553--639` transports QP corrections
  using coarse/fine wavefunction overlaps and spatial interpolation weights,
  with optional overlap normalization. GradSCF currently interpolates MO
  operators through a common Bloch AO convention; that is a different
  finite-mesh approximation. More path samples alone cannot repair wrong
  source branches or establish source-grid convergence. In the inspected
  dynamic Sigma code, `spectrum.dat` contains self-energy components; the
  spectral-function calculation is commented out at
  `shiftenergy_dyn.f90:214--217,277--285`.

The measured evGW plateau has not been attributed to any one of these
differences. The measured scGW real-axis sensitivity likewise cannot be
resolved by copying BGW's table-root rule: the continuation itself needs a
separate frequency/rank/range stability study. Existing matched G0W0 checks
remain limited to their declared finite Hamiltonian and selected states.

## Same-orbital interface audit, 2026-10-18

The historical observations below describe the source and artifacts available
at that review checkpoint. Later implementation changes require their own
tests; they must not inherit an older comparison claim.

Read-only inspection of `jy_gpu_1` on 2026-10-18 found existing serial GNU/MKL
executables at `/home/yjiao/opt/berkeleygw-4.0/bin/epsilon.cplx.x` and
`sigma.cplx.x`. The accompanying `env.sh` sets the MKL library path needed by
these binaries. The current `arch.mk` does not enable HDF5. Finding binaries
does not establish that a numerical comparison has run.

The new `gradscf_tools.berkeleygw` interface writes the GNU Fortran binary
WFN format defined in BGW's `Common/wfn_rho_vxc_io_inc.f90`, together with
diagonal `vxc.dat`. It transforms the actual periodic Gaussian Bloch orbital
samples to normalized Fourier coefficients, preserves all supplied orbitals,
and rejects orthonormality failures instead of changing the orbitals. WFN
energies are in Ry; `vxc.dat` matrix elements are in eV, consistent with
`Sigma/sigma_main.f90` dividing the latter by `ryd` on input. A rectangular
orbital FFT and a larger product FFT do not establish equality with BGW's
spherical dielectric/bare cutoffs or its small-q cell integration. Those
controls and the frequency model must be reported in an executed comparison.
The [official WFN specification](https://manual.berkeleygw.org/4.0/wfn_h5_spec/)
documents the corresponding header and coefficient conventions.

There is a concrete finite-model interface restriction: the inspected 4.0
`Epsilon/input.f90` rejects `number_bands == WFN mnband` unconditionally,
before the optional degeneracy override, because its next check reads band
`number_bands+1`. Thus a complete 14-orbital Gaussian model cannot silently
be advertised as a full-14-band BGW screening calculation. A 13-band pilot
excludes the last empty orbital and must say so. A full-model interface
adaptation must explicitly handle the final-band lookahead and retain the
original BGW source and patch provenance; inventing a physical fifteenth
band is not an acceptable numerical comparison.

The available self-consistency routes also require precise labels:

- `eqp_corrections` changes the eigenvalues used in inner Green-function and
  screening sums. `eqp_outer_corrections` changes the external evaluation
  energies; neither keyword alone updates orbitals. See the official
  [Sigma overview](https://manual.berkeleygw.org/4.0/sigma-overview/) and
  [Epsilon keywords](https://manual.berkeleygw.org/4.0/epsilon-keywords/).
- The distributed `scGWtool.py` **does** diagonalize a subspace Hermitian
  quasiparticle Hamiltonian and rotate wavefunctions, energies and matrices.
  Its source states that W stays fixed by default, and that it does not
  reconstruct every mean-field term from the rotated orbitals. This is an
  orbital-updating quasiparticle/subspace workflow; it is not a solution of
  the full frequency-dependent Dyson equation with a self-consistent
  spectral Green function. A comparison to a Matsubara full-Dyson scGW
  implementation therefore cannot label both runs simply “scGW” and claim
  matching methods.

The initial interface inspection alone claimed no numerical agreement; the
later executed comparisons and their remaining differences are recorded below.

The isolated guard-patched serial build completed successfully on 2026-10-18
at 08:27:46 UTC, with affinity CPU64--79 and the existing `jax_scf` MKL
runtime. The original installation's three source hashes remain unchanged.
The patch is retained as `tools/berkeleygw_complete_finite_space.patch`;
it also avoids the nonexistent next-state argument in `assess_degeneracies`
while preserving comparisons between every actual state. This build result
does not constitute a numerical BGW run.

The preparation example exposes `head='body'` and `head='semiconductor'` in
separate output directories. The former uses `cell_average_cutoff 0.0` and
`skip_averagew` to remove the exact Gamma Coulomb head in Sigma. Source
inspection shows that semiconductor Gamma wings are already set to zero by
`fixwings_dyn`; the ordinary small-q epsilon inversion nevertheless retains
head/wing feedback in its body block. Removing the final self-energy head
does not undo this Schur feedback, so this setting alone does not establish
equality with GradSCF's response constructed with `v(G=0)=0` before inversion.
The installed `check_screening_trunc` also rejects exact zero q0 for ordinary
3D Coulomb input. Native occupied-state Ewald exchange adds a separate
`-madelung` constant relative to BGW's zero-head exchange; raw and converted
exchange terms must be reported separately. BGW's official
[`skip_averagew` documentation](https://manual.berkeleygw.org/4.0/sigma-keywords/#skip_averagew)
explicitly describes it as a debugging option.

The WFN interface was then exercised by the real Epsilon and Sigma programs.
Two rejected inputs were retained: differing source/shifted header cutoffs,
and a rectangular G subset that BGW's kinetic-sphere remapping interpreted
incorrectly. The exporter now gives WFN/WFNq one declared cutoff covering
both meshes, and pads the original rectangular Fourier coefficients with
exact zeros onto complete kinetic spheres. It does not truncate coefficients
or renormalize states. Six focused export tests passed remotely, including
the actual GaAs reference and independent record/sphere checks. The padded
GaAs WFN also passed BGW's own norm checks. The first executed 1-Ry Epsilon
and diagonal CD Sigma runs both finished normally; their narrow-window Eqp1
values extrapolate, so they remain interface diagnostics rather than QP-gap
results. Later cutoff/frequency cases must retain their own resource records
and flags.

### Imaginary-frequency discretizations are different

Matching `nw=16` does not match the two quadratures. The inspected BGW
`Epsilon/inread.f90:598--604` uses, for standard CD,

\[
z_i=(i-1)/N,\qquad \nu_i=E_p\,z_i/(1-z_i),\qquad i=1,\ldots,N.
\]

Thus it includes zero and ends at `(N-1)*E_p`; the example's
`plasma_freq=27.211386245988 eV` is `E_p=1 Ha`. For
`cd_integration_method=2`, `Sigma/mtxel_cor.f90` fits screened matrix elements
piecewise quadratically and integrates each segment analytically. The
optional Lorentzian tail following the final segment is commented out in
the inspected source.

GradSCF's current `gw/freq.py::scaled_legendre_grid` instead uses Legendre
roots and weights on `[-1,1]`, with default `x0=0.5 Ha`:

\[
\nu_j=x_0\frac{1+x_j}{1-x_j},\qquad
W_j=\frac{2x_0w_j^{GL}}{(1-x_j)^2}.
\]

The periodic frozen-source implementation calls this default. It is a
Gauss-Legendre quadrature of an integral over `[0,infinity)`, without a
zero-frequency node. Its points and weights are neither BGW's rational
uniform-z sampling nor BGW's piecewise integration rule. Agreement must be
tested under each method's own frequency convergence; an identical integer
point count is insufficient.

### Executed same-orbital GaAs cutoff study

The subsequent executed cases reuse **identical WFN, WFNq and Vxc hashes**:
18 electrons, 14 physical orbitals, PBE/GTH-SZV/GTH-PBE, source mesh
`(2,1,1)`, original orbital FFT `21^3`, and `q0=(0,0,0.001)`. Every source
k point and all 14 output bands were evaluated. The real screening grid was
extended to 200 eV after the fourteenth mean-field orbital was found near
135 eV; the earlier 60-eV run retains a frequency-range warning and is not
used for this table. The self-energy evaluation window is +/-20 eV around
each source eigenvalue, with 0.1-eV spacing. The imaginary grid remains the
BGW 16-point grid defined above.

The following are **on-shell correlation self-energies**, not QP gaps, at
Gamma bands 9/10 (one-based occupied/empty frontier indices). Units are eV.

| Dielectric cutoff / Ry | Bare cutoff / Ry | Re Cor, band 9 | Re Cor, band 10 |
|---:|---:|---:|---:|
| 1 | 16 | 0.589211 | -0.540095 |
| 1 | 64 | 0.589211 | -0.540095 |
| 1 | 399 | 0.589211 | -0.540095 |
| 8 | 399 | 0.365911 | 0.077467 |
| 16 | 399 | 0.366383 | 0.104934 |

At fixed W cutoff 1 Ry, increasing the bare cutoff 16->64 Ry changes the
Gamma band-9/10 exchange by -0.188398827/-0.067531622 eV; 64->399 Ry changes
it by another -0.055620252/-0.000576235 eV. Across all 28 source states the
largest exchange changes are 9.622630 and 1.178646 eV respectively. At bare
399 Ry, Gamma raw BGW exchange is -8.599873200/-6.444872098 eV. The native
occupied Ewald constant is -8.064732019 eV, giving -16.664605219 eV for the
converted occupied exchange; that declared boundary conversion is not a
fitted correction. Native and BGW product-space and screening differences
still remain.

Increasing the W cutoff 8->16 Ry changes Gamma frontier on-shell correlation
by +0.000472/+0.027467 eV. The largest change among all source states is
0.236608 eV, so these measurements do not establish full-space cutoff
convergence. Every completed case has 11,228 finite spectral-output rows and
a measured zero Gamma Coulomb head. The wider window eliminates extrapolated
Eqp1 flags, but most states still have multiple candidate roots; flags and
raw spectra remain part of the results. No unique physical QP branch or
converged bulk gap follows from these runs.

The 8-Ry Epsilon/Sigma commands took 9.535/24.881 s with peak RSS
737,068/1,026,596 KiB. The 16-Ry commands took 30.607/87.129 s with peak RSS
4,509,604/5,895,180 KiB. Each command ran only on CPU64--79, with 16 MKL
threads, a 600-s wall limit and a monitored 16-GiB RSS bound. Runtime records,
all components, QP flags and source/output hashes are retained in each
executed case; they are not inferred from an old path timing.
The host was verified as `c20`, Intel Xeon Platinum 8457C (192 logical CPUs,
two hardware threads per core); the allocated study used only its declared
16 logical CPUs.

Finally, BGW's `spectrum.dat` is a table of complex self-energy components.
The spectral-function code is commented out in the inspected
`shiftenergy_dyn.f90`. Raw imaginary components use that implementation's
convention and must not be relabelled as a retarded linewidth or an
already-normalized density of states.

### Native components, units, frequency convergence and the head block

`examples/pbc/band_structures/berkeleygw_gaas.py::compare_gaas` now compares
the actual BGW tables with the native frozen-source G0W0 model. The native
reference uses `exxdiv=None`, and its model Madelung term is verified to be
zero. No occupied-state constant is added in this comparison. It retains all
14 internal bands, fixes the screening sphere independently of the full-FFT
native bare exchange, and compares Gamma output bands 6--11 (zero-based).
These are local frontier-window diagnostics, not a complete band gap.

An inspected-build unit difference matters at the strict reference gate:
BGW 4.0 `Common/nrtype.f90` defines `RYD=13.60569253 eV`, so its printed eV
tables must be divided by `27.21138506 eV/Ha`. Using modern
`27.211386245988 eV/Ha` instead produces an apparent reference mismatch.
With the build conversion the selected mean-field energies agree within
`1.837e-11 Ha`; the original `1e-8 Ha` reference gate remains unchanged.
The exported Vxc used the modern physical conversion, so the installed BGW
actually reads a Vxc differing by approximately 0.6 micro-eV from the native
physical value. The example records both the actual-input and physical-Vxc
table equations separately. It does not fit or shift any measured self-energy.

The independent `read_sigma` and `read_eqp` parsers preserve the complex
self-energy components and the original MULT flags. The shared SRC QP
selector solves the BGW table equation using piecewise-linear real Sigma,
returns NaN outside the evaluated frequency interval, and retains its tested
candidate count, branch validity, weight and residual. Raw `eqp1.dat` is
retained as BGW output and is not treated as the unique nonlinear root.
Seven independent format/unit tests passed remotely; the two additional
Schur-copy fixtures described below also passed in the same focused run.

For W cutoff 8 Ry, native `nw=64 -> 128` changes Re Cor at the six selected
KS energies by at most `2.553e-8 eV` (the previous 16->64 change was about
1.31 meV). The controlled BGW family instead fixes `Ep=0.1 Ha`, using the
installed-build input `plasma_freq=2.721138506 eV`, and increases its rational
imaginary-frequency grid from 128 to 256 points. At fixed real grid
0--200 eV with 0.25-eV spacing, the largest selected on-shell change is
0.100 meV. These measured checks concern the selected states; equal integer
frequency counts were never used as an equivalence claim.

Even with those separately checked imaginary grids, Sigma-only Coulomb-head
removal leaves a correlation difference as large as 108.34 meV. The
small-q inverse dielectric matrix contains the head Schur feedback. For an
original inverse block matrix

\[
\epsilon^{-1}=\begin{pmatrix}A&B\\ C&E\end{pmatrix},
\qquad (\epsilon_{\mathrm{body,body}})^{-1}=E-CA^{-1}B.
\]

`copy_eps0_without_head_feedback` writes a **new** binary eps0mat with this
body block, head one and wings zero. It reads the frequency count and the
G=0 position from the file, preserves every header record, and transforms
retarded and advanced channels independently without adding a conjugation.
The original files remain unchanged. Its independent certificate inverts
the original sampled full matrix, extracts the dielectric body, inverts
that body and compares it with the written matrix over the complete body
Frobenius norm. The nonsymmetric complex fixture deliberately puts G=0 at
a nonfirst index and independently checks both channels.

For the actual W=8 Ry, Ep=0.1 Ha, 256-point case, the dynamic file header has
1057 frequencies and a 113-by-113 matrix. The written copy's independent
relative error at the first and last frequencies is `8.785e-16`. The
original eps0mat SHA256 is
`869a54ae5a85fd93012270d5730689e91df0f4c9a9dbc58673f26dac576c4694`;
the new copy is
`6741a3bd3a370d57e836d2367d08aed24ae0e5fff3c272f089b28b61d2dbd38d`.
The nonzero-q epsmat and both wavefunctions are reused byte-for-byte.
This is a controlled finite-body Hamiltonian diagnostic, **not standard
BerkeleyGW semiconductor output**, a q0 extrapolation or a fitted correction.

The actual new-copy Sigma finished in 9.15 s with 1,219,372 KiB peak RSS.
At the same KS energies, the measured native128-minus-BGW256 Re Cor
differences are as follows (meV; zero-based Gamma band labels):

| Band | Sigma-only head removal | Head Schur feedback also removed |
|---:|---:|---:|
| 6 | -45.349 | -6.369 |
| 7 | -26.553 | -5.203 |
| 8 | -33.683 | -5.611 |
| 9 | 108.340 | 13.070 |
| 10 | 18.848 | 19.968 |
| 11 | 19.219 | 19.089 |

The maximum falls to 19.97 meV, while exchange is unchanged. The remaining
native-minus-BGW bare-exchange differences for bands 6--11 are
`[9.420, 9.098, 10.079, 0.472, 2.368, 2.230] meV`. Native full-FFT bare
products, BGW's sphere, the finite small-q body, real-frequency interpolation
and nonlinear branch selection remain distinct checks. Removing the head
feedback alone does not certify QP-energy agreement: after the same shared
selector, band 6 differs by 50.77 meV and still has multiple roots in the
original BGW report. Occupied states can reorder (band 7 lies above band 8),
so subtracting the fixed KS frontier labels is not a QP gap calculation.

The next executed single-variable check changes the real screening spacing
from 0.25 to 0.05 eV, leaving the imaginary grid, Ep, broadening, cutoffs,
wavefunctions and Sigma evaluation table fixed. Both real cutoffs stay at
200 eV: in this BGW CD method, `Epsilon/inread.f90` forces the high cutoff to
equal the low cutoff, so input low=20/high=200 would silently truncate the
screening range near 20 eV. The finer file dynamically reports 4256 total
frequencies; its Schur copy passes the same independent `8.785e-16` matrix
certificate. Epsilon/raw Sigma/Schur Sigma and the native comparison all
finished within the 600-s/16-GiB bounds. Epsilon took 27.02 s (3,063,656 KiB)
and raw Sigma 15.68 s (4,098,208 KiB); the shared native/root comparison took
65.31 s (2,246,052 KiB).

For that finer Schur case, native-minus-BGW on-shell correlation differences
are `[-6.229,-5.043,-5.614,13.070,16.098,15.239] meV`, and the shared-selector
BGW-minus-native root differences are
`[14.072,-5.643,0.681,-5.401,-1.613,1.510] meV`. Both selected branches pass
their own residuals below `1e-13 eV`; the largest cross-equation residuals
remain 20.870 meV for native F at the BGW root and 42.803 meV for BGW's table
F at the native root. The maximum known Vxc input-unit root change is only
`4.309e-7 eV`. Thus neither the legacy conversion nor tiny own residuals
explain away the remaining model/branch differences. Original BGW flags
remain `MULT:[5,2,3,4,4,4]`, even though the shared tested-candidate counts
are `[2,1,1,1,2,2]` on each side. The local Gamma window gaps from bands
6--11 use the reordered occupied maximum, and are explicitly not a complete
or bulk gap.

The full plotted frequency window gives a much larger diagnostic:
actual native samples versus the piecewise-linear BGW Sigma table have
maximum `|Delta Re Sigma_c(E)|=1.064227 eV`, at Gamma band 7 (zero-based),
`E-epsilon_PBE=2.7 eV` (`E=10.7056973 eV`). This is **not** an on-shell
comparison, and the 16.098-meV on-shell result must not be advertised as
full-frequency agreement. The native sampled real-Sigma minimum there is
-2.170568 eV; its adjacent 2.6/2.8-eV samples are -1.367419/+1.439414 eV.
The nearest BGW sampled minimum is at relative energy 2.70000316 eV with
real Sigma -3.234850 eV, adjacent to -1.501660/+1.909700 eV. These observations
locate a rapidly varying **sampled extremum**, not an isolated mathematical
Sigma or W pole. Its 3.16-micro-eV relative sample-coordinate difference is
not evidence of a measured pole shift. Both Sigma- and W-pole distances are
recorded as undetermined/not isolated. Different FFT product closures,
finite q0, broadening, quadrature and table interpolation can affect this
region jointly; no single source of the deviation has been isolated.

A subsequent read-only check establishes a **known finite-broadening
protocol difference**, beyond the still-unisolated product/q0 effects.
In the installed BGW 4.0 source, `Epsilon/mtxel.f90:934` sets
`eden=epsilon_v-epsilon_c`. The ordinary full-frequency path in
`Epsilon/chi_summation.f90:1646--1650` contains

\[
-\tfrac12(f_v-f_c)\left[
\frac1{\mathrm{eden}-(\omega+i\delta)/\mathrm{RYD}}
+\frac1{\mathrm{eden}+(\omega+i\delta)/\mathrm{RYD}}\right].
\]

After the common overall susceptibility sign and conversion to Hartree,
the positive-frequency resonant and antiresonant Lehmann denominators are
`omega-Delta+i*delta_Ha` and `omega+Delta+i*delta_Ha`: both have the same
retarded shift. `Epsilon/inread.f90:582` sets the real-frequency imaginary
part directly from the input `broadening=0.027211386245988 eV`;
`add_w0_freq` is false in these actual inputs. Thus delta_Ha is
`0.027211386245988/27.21138506`, approximately 0.0010000000436 Ha.
The imaginary-frequency nodes replace this shift with their explicit
imaginary frequency at `inread.f90:603`.

For the actual CD path, `Sigma/mtxel_cor.f90:3101--3103` constructs the real
quantity `wx=E-epsilon_m`; lines 3128--3168 select `abs(wx)` and linearly
interpolate `I_epsR_array` at real tabulated frequencies. No additional
complex G-pole eta is added there. Its imaginary-axis CD integration also
uses the real `wxi` kernel rather than a separately broadened G pole.
The source explicitly excludes CD from `need_advanced`
(`Sigma/inread.f90:997`): `mtxel_cor.f90:853--888` states its folded-model
identity `epsR(q,-E)=epsA(q,E)`, so its negative real residue uses the
positive-frequency retarded table as the corresponding advanced branch.
For complex matrices the installed `epsinv.f90:524--538` constructs
`epsA=(epsR*v)^dagger/v`, i.e. `W_A=W_R^dagger`, with the same q/G-frame
index/phase mapping in Sigma. This is not elementwise conjugation. The
identity within BGW's folded CV construction does not automatically establish
the same finite-q closure for arbitrary native complex OV/VO vertices.

Native `product_basis.py::lehmann_weights` instead evaluates the
time-ordered response at the complex G pole. For transition sign s and
G-pole sign s_m its denominator is

\[
\omega+\Delta_{ia}+i\eta(s_m+s).
\]

For positive frequency, OV/VO shifts are `+2i*eta/0`; at negative frequency
they are `0/-2i*eta`. The nominal 0.001-Ha values therefore do **not** mean
matched finite broadening: the positive resonant native term has twice
BGW's shift, and the antiresonant term differs as well. Doubling only the
BGW broadening cannot match both terms. Neither implementation is silently
changed by this audit; a comparison of identical finite-eta protocols or
separately demonstrated eta->0 limits is still required. This known design
difference can affect the sharp sampled amplitudes, but its quantitative
share of the 1.064-eV discrepancy has not been isolated by a controlled run.
The original installation and guard-patched copy have identical hashes for
`chi_summation.f90` (`5aef5517...a7bb90ef`) and `mtxel_cor.f90`
(`8914dc00...5daf1dd`), so the finite-space guard patch did not cause this
frequency convention difference.

### Executed independent-G / response-broadening comparison

After 48 focused source/solver tests passed (including arbitrary nonpaired
complex positive/negative response, matrix-adjoint fixtures and default
compatibility), an explicit `screening_eta` comparison option was exercised.
This leaves the default time-ordered prescription unchanged. The controlled
case sets native `eta_g=0` and
`screening_eta=0.027211386245988/27.21138506 = 0.001000000043584257 Ha`,
matching BGW's real-G CD integral and independently shifted retarded/advanced
response definitions. It retains the same reference, all 14 internal states,
W=8 Ry, native N=128, and the unchanged BGW N=256/Ep=0.1-Ha fine-grid Schur
tables. Full OV/VO versus folded CV products, finite q0 and bare-product
closures still require their own checks.

The actual comparison, six-state curve sampling and rendering finished in
77.85 s with 2,514,252 KiB peak RSS on CPU80--95, BLAS1. At the *fixed old*
band-7 maximum point (relative energy 2.7 eV), the Sigma difference falls
from 1.064227 to 0.136225 eV; the native change is -0.928003 eV. That is a
measured effect of the finite-frequency prescription. It does not imply
uniform agreement: the maximum sampled-window discrepancy becomes
2.461957 eV at relative energy 2.8 eV, with the same 0.05-eV BGW W grid.
The on-shell maximum remains 16.0976 meV, and the selected-root maximum is
19.3033 meV. Own root residuals stay below 3.69e-15 eV while cross-equation
residuals reach 43.930/58.715 meV. Every original BGW MULT flag is retained.
The full curves and hashes, including this unfavorable shifted maximum,
are preserved separately from the default-prescription archive.

A bounded real-W spacing study first certifies its used domain. Across all
2406 actual output points for these six states and both source-k internal
sets, occupied-negative/empty-positive active residues require at most
`|E-epsilon_m|=23.59533993 eV`. A conservative 0.01-eV allowance gives
23.60534 eV, below a proposed 60-eV real grid. The unoccupied fourteenth
internal orbital still reaches 140.512 eV and remains in every polarizability
and imaginary-axis sum; this domain certificate concerns selected-state
real residues only, not the complete W spectrum or output band 14. Endpoint
conditions and the common Fermi threshold are recorded explicitly.

The first actual range-only control at fixed 0.05-eV spacing changes the
real cutoff 200->60 eV. All printed complex Sigma-total, Cor, residue and
imaginary-integral values at the 2406 points are exactly identical; all six
roots recomputed with the shared selector are also exactly identical.
Only after this strict control passed was the independent 0.01-eV W spacing
case started. Its Epsilon/Sigma commands completed in 37.58/18.12 s, with
4,419,020/5,902,044 KiB peak RSS. Refining W spacing again to 0.005 eV took
63.23/28.17 s and 8,467,708/11,298,536 KiB, still within the original 16-GiB
bound. On the same saved native sample set, the maximum plotted-window
difference changes from 2.461957 to 0.597711 to 0.598317 eV. Band 7's maximum
falls to 0.017194 eV, while the remaining maximum is near band 6's selected
root, between the 0.1-eV Sigma output nodes. The *full* +/-20-eV BGW table
still changes by up to 1.195590 eV in real correlation for W spacing
0.01->0.005 eV, including remote satellite regions; full-table convergence
has not been established.

The final independent control reuses precisely the same finest epsilon,
WFN/WFNq and inputs, changing only Sigma's evaluation spacing 0.1->0.01 eV.
All original 401 points per state are present in the 4001-point table and
are **exactly identical**, including complex correlation. New points expose
large errors of the former linear interpolant in satellite regions. Band
6's shared selected table root moves -19.148 meV. The actual dense Sigma
finished in 41.81 s with 11,300,472 KiB peak RSS. At iw>=1000 the BGW
format3000 joins adjacent I4 band/iw fields; two independent fixtures
reproduced the whitespace parser failure. The format-aware parser/Schur
suite then passed 11 tests, and the unchanged physical Sigma output was
processed successfully without a rerun.

Native curves were actually sampled at 0.01 eV across the displayed window
and 0.005 eV near the selected roots. Comparison/sampling/rendering took
105.55 s with 2,528,116 KiB peak RSS on CPU80--95, BLAS1. Final BGW-minus-native
selected-root differences for Gamma bands 6--11 are
`[-0.973,-6.304,-5.713,-5.309,-0.414,-0.292] meV`. The largest is 6.304 meV.
The native/BGW cross-equation residual maxima are 8.825/8.720 meV, while
own residuals stay below 3.69e-15 eV. Those numerical residuals do not
certify physical model equality.

The on-shell maximum correlation difference remains 16.008 meV. Within
0.10 eV of either selected root it is 16.027 meV; within 0.25 eV it is
146.813 meV; over the actual plotted native sample window it is 269.054 meV.
Native was **not** sampled over the full +/-20-eV BGW window. The final
plot uses denser native samples than the earlier plots, so the maxima of
different display sample sets are not an unqualified convergence bound.
Original BGW flags remain `MULT:[9,2,6,5,4,9]`; tested positive-slope candidate
counts are native `[3,2,2,1,2,2]` and BGW `[2,2,1,1,2,2]`. Neither is an
exhaustive mathematical root count.

All controlled steps and their full per-row source/input fingerprints are
recorded in the separate
[`matched_eta` data directory](../../../../reproducibility/ad_showcase/gaas_berkeleygw_matched_eta_20261018/README.md).
The two native source.py hashes differ only by one JIT decorator making k
dynamic rather than static. Whole-file SHA reconstruction proves that exact
text difference; 45 cross-k/root tests passed, and saved exchange, Vref,
on-shell correlation and selected native energies are unchanged. Both
fingerprints are retained, rather than labelling the code byte-identical.
The remaining finite-product/q0/CV-closure differences are reported without
fitting or an assertion of full-frequency or bulk-material equivalence.

The selected-state curves, compact CSV table, actual comparison arrays,
raw small-output archive, input hashes and resource records are retained in
[`reproducibility/ad_showcase/gaas_berkeleygw_20261018`](../../../../reproducibility/ad_showcase/gaas_berkeleygw_20261018/README.md).
The public example samples curves using only SRC's correlation kernel and
renders saved results; no physics implementation lives in that data folder.

This is the pre-repair review of the source fingerprint listed below. The
single-seed behaviour and missing diagnostics described here are historical;
subsequent repairs and measured limits are recorded in
[the repair validation](../../../../reproducibility/ad_showcase/gw_repairs_20261018/README.md).

Reviewed on 2026-10-18. This review concerns the arbitrary-k, restricted
`g0w0_bands` driver and the recent Si/GaAs/GaN/graphene examples. It does not
establish a defect in every molecular or periodic GW implementation.

The immediate finding is reproducible initial-guess dependence of the
nonlinear QP roots. Small residuals certify solutions of the implemented
equation; they do not certify the intended quasiparticle branch. Long-wave
screening and material convergence are separate unresolved requirements.

## Sources and evidence

The primary references are the [BerkeleyGW 4.0 manual](https://manual.berkeleygw.org/4.0/),
[Deslippe et al., 2012](https://arxiv.org/abs/1111.4429), and
[Ismail-Beigi, 2006](https://arxiv.org/abs/cond-mat/0603448).
Implementation details were read from a public mirror at commit
[`a13c40f12e2e4db96c2d6afffc0ed52899a55e69`](https://github.com/jwu277/BerkeleyGW-4.0/tree/a13c40f12e2e4db96c2d6afffc0ed52899a55e69).
Its version header says 4.0. The official archive could not be downloaded
because connections to its Box host timed out; equivalence of the mirror
to the official release has therefore not been verified. No BerkeleyGW
executable was run in this review.

GradSCF source inspected and executed has `query.py` SHA256
`fd8062efcaeeb6a98215d64e7a105c2353619f345a0a5199cb0317b631e71763`.
The [two-point diagnostic](../../../../reproducibility/ad_showcase/material_gw_bands/multiroot_audit/README.md)
contains the exact source/native fingerprints, measured roots, AD slopes,
resource log and verified result hashes. Numerical work ran only on
`jy_gpu_1`. No numerical implementation was changed during this review.

## QP equations, derivatives and root selection

For a fixed G/W model, GradSCF solves

\[
F_p(\omega)=\omega-\epsilon_p^{KS}-\Sigma^x_{pp}
             +V_{xc,pp}-\operatorname{Re}\Sigma^c_{pp}(\omega)=0.
\]

The exchange and XC compensation signs in `query.py` are consistent with
this equation. Differentiating with respect to frequency gives

\[
F'_p(\omega)=1-\partial_\omega\operatorname{Re}\Sigma^c_{pp}(\omega),
\qquad Z_p(\omega)=1/F'_p(\omega).
\]

A linearization about a specified energy \(E_o\) gives

\[
E_{\mathrm{lin}}=E_o-F_p(E_o)/F'_p(E_o).
\]

AD can evaluate this slope without a finite-difference interval. It cannot
by itself identify which of several roots represents the desired QP.
For an isolated smooth root and a parameter \(x\), implicit differentiation
would give \(dE/dx=-\partial_xF/F'\). Switching roots as \(x\) changes
invalidates that local derivative as a derivative of a consistently selected
state. This is a future periodic AD concern: the present arbitrary-k driver
explicitly excludes geometry/orbital AD and JIT.

BerkeleyGW recommends its off-shell, linearized/interpolated `eqp1` output
and reports the number of solutions in full-frequency calculations.
[Sigma overview](https://manual.berkeleygw.org/4.0/sigma-overview/)
In the inspected mirror, `get_eqp1` scans frequency intervals for crossings,
interpolates candidates, selects the one closest to the on-shell `eqp0`,
and distinguishes multiple candidates and extrapolation. It does not select
the largest Z or perform automatic continuation between k points.
[Full-frequency source](https://github.com/jwu277/BerkeleyGW-4.0/blob/a13c40f12e2e4db96c2d6afffc0ed52899a55e69/Sigma/shiftenergy_dyn.f90#L305)

Current GradSCF starts each scalar solve at \(\epsilon^{KS}\pm0.01\) Ha.
An optional hybrid retry uses the same original guess. Acceptance checks
residual convergence and finite self-energy, without enumerating roots.
`GWResult.qp_weight` exists, but this driver does not fill it. The stored
complex `sigma_qp` is the correlation contribution; it is not a separately
certified retarded linewidth.

### Reproduced multiroot behaviour

The graphene dense61 path was checked at zero-based points 6 and 7, band 0,
using the actual SRC equation and shared root solver. Each point has at
least three distinct converged roots under the tested seeds. Selected
measurements are:

| Point | Seed | Root / Ha | Formal AD Z | Absolute residual / Ha |
|---|---|---:|---:|---:|
| 6 | Original driver | -0.7499780905 | 0.206467 | 2.78e-16 |
| 6 | Neighbour's saved root | -1.1343110613 | 0.026601 | 4.00e-15 |
| 6 | KS energy | -0.7765950154 | -0.033829 | 6.38e-16 |
| 7 | Original driver | -1.1399767457 | 0.021834 | 1.11e-16 |
| 7 | Neighbour's saved root | -0.7479468278 | 0.306147 | 2.91e-16 |
| 7 | KS energy | -0.7666881945 | 0.381479 | 2.50e-16 |

The public queries reproduce the previously saved energies within
5e-15 Ha. Their large adjacent jump depends on which converged roots are
selected. An upper-energy candidate exists at both points with a much
smaller energy change. This establishes a selection problem, but does not
establish that the upper candidate is the physical QP. Negative or very
small formal Z warrants investigation; at finite eta these slopes alone
are not a proof of a causal spectral residue or an identification of a
satellite. Frequency/eta convergence and a consistent spectral convention
remain necessary. Four seeds and 25 sampled frequencies do not enumerate
all roots.

The largest stored Si and GaAs correction steps occur between singleton
KS clusters at both endpoints. They therefore cannot be attributed simply
to switching between the scalar and degenerate-matrix paths. Their root
selection has not yet been diagnosed with the same frequency probe.

## Degenerate subspaces and band identity

GradSCF rejects an output window that cuts a KS degenerate cluster. Within
a complete cluster it solves Hermitian projected Dyson branches; outside
clusters it uses diagonal self-energy. These are different approximations.
The cluster boundary is defined by a tolerance, and returned coefficients
remain KS coefficients rather than QP eigenvectors. There is no overlap
tracking between neighbouring path points.

In the inspected BerkeleyGW mirror, the default symmetrization averages
diagonal contributions over KS degeneracy groups. Explicit off-diagonal
calculation is a separate option, and the inspected off-diagonal driver
rejects full-frequency input. It is not evidence for a default full-frequency
nonlinear matrix Dyson implementation.
[Diagonal symmetrization](https://github.com/jwu277/BerkeleyGW-4.0/blob/a13c40f12e2e4db96c2d6afffc0ed52899a55e69/Sigma/shiftenergy.f90#L68),
[off-diagonal implementation](https://github.com/jwu277/BerkeleyGW-4.0/blob/a13c40f12e2e4db96c2d6afffc0ed52899a55e69/Sigma/offdiag.f90#L81)

A consistent matrix-window policy and QP vectors would allow checking
state identity without silently changing approximation at a near-degeneracy.
This is distinct from the multiple frequency roots of a scalar equation.

## Periodic screening and Coulomb integration

The query driver requires `fc=False`, excluding its correlation head/wing
correction. Bulk Ewald Madelung exchange is not a replacement for that
frequency-dependent screened contribution. Some other GradSCF periodic
routes have q0 support; that support must not be attributed to this query
path merely because the helper module exists.

BerkeleyGW separates small-q input for different screening regimes, exposes
a dielectric cutoff based on \(|q+G|^2\), and provides dimensional Coulomb
truncations. These controls support convergence of distinct physical
approximations. [Epsilon keywords](https://manual.berkeleygw.org/4.0/epsilon-keywords/)
The inspected source also averages the slab head over a mini Brillouin-zone
cell. [Coulomb generator](https://github.com/jwu277/BerkeleyGW-4.0/blob/a13c40f12e2e4db96c2d6afffc0ed52899a55e69/Common/vcoul_generator.f90#L620)

For in-plane small q, the slab interaction behaves as

\[
v^{2D}(q)=\frac{4\pi}{q^2}(1-e^{-qL_z/2})
        =\frac{2\pi L_z}{q}-\frac{\pi L_z^2}{2}+O(q).
\]

The exact-zero finite part used by GradSCF is the constant term. Restoring
that constant in bare exchange preserves the declared finite model; it
does not restore the singular cell integral or the screened-W average.
[Truncated Coulomb theory](https://arxiv.org/abs/cond-mat/0603448)
The current graphene source has a positive sampled gap and integer
screening occupations. Allowing a query Dirac contact is not a validated
gapless long-wave/intraband screening treatment.

The SZV basis, anisotropic source meshes, finite FFT boxes, 16 imaginary
frequencies and fixed eta have not been converged for material predictions.
A rectangular FFT set also differs from a kinetic-energy product cutoff.
These are explicit model limitations; they do not prove the cause of an
individual large QP jump.

## Algebra and workflow checks

Inspection found no simple error in the q sign, FFT phase, volume factor
or the exchange/XC compensation. With periodic orbital parts \(u_k\),
the implemented vertex transforms
\(u_L^*u_R e^{i(k_R-k_L+q)\cdot r}\) and contracts the interaction at
\(G-q\). This convention maps to the opposite sign of BerkeleyGW's q
label. The factors \(\sqrt{\Omega v_G}/N_{\rm grid}\), response spin factor
and separate k/q averages are consistent. This is an algebra audit, not an
independent numerical verification of every self-energy term.

Using a fixed q quadrature with shifted internal states from the same
frozen SCF potential is a valid discretization. On a finite grid its bare
exchange differs from an off-mesh Fock operator using a fixed source-k
density; continuum equivalence requires BZ convergence.

BerkeleyGW's `Inteqp` instead uses wavefunction projections to handle band
crossings and interpolates coarse QP corrections onto a fine grid.
[Inteqp overview](https://manual.berkeleygw.org/4.0/inteqp-overview/)
That is a separate numerical workflow from 61 independent GradSCF queries.
It must not be confused with graphical curve smoothing or with selecting
among several roots of one Dyson equation.

An incidental documentation error was found in `product_basis.py`: the
Deslippe DOI should be `10.1016/j.cpc.2011.12.006`, not `.023`.
This does not affect numerical results.

## Correction priorities

1. Reuse the existing self-energy evaluator and shared solvers to expose
   on-shell/linearized energies, AD Z, candidate roots and explicit ambiguous
   states. Establish a selection rule and test its stability; a smaller
   residual or a smoother curve is insufficient. Validate any reused
   molecular branch helper for this periodic finite-eta model.
2. Check selected Si jump endpoints and independently compare exchange,
   XC subtraction, chi/W and correlation before comparing final QP values.
   Converge frequency and eta first within one fixed spatial model.
3. Connect long-wave screening to arbitrary-k queries, with consistent
   three-dimensional and truncated low-dimensional cell integration.
   Validate graphene's screening regime before treating it as a material
   benchmark.
4. Establish basis, product-space, empty-state and BZ convergence on a small
   bulk reference. A matched BerkeleyGW run must state pseudopotentials,
   basis/cutoffs, q meshes, frequency model and root prescription; different
   representations cannot be compared as bitwise equivalents.
5. Add overlap-aware band projection/interpolation only after trustworthy
   QP inputs exist. Keep root ambiguity, matrix-policy changes and genuine
   band crossings separately observable.

No production fix, converged material benchmark or matched BerkeleyGW
forward comparison is claimed by this review. Existing path arrays and
figures are retained as diagnostic records.
