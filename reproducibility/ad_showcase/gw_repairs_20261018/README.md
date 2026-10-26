# GW repair validation — 2026-10-18

This record covers the existing molecular G0W0, evGW/evGW0, qsGW and
finite-temperature matrix scGW paths, together with periodic G0W0 queries.
The requested “qpGW” repair was applied to the repository's qsGW implementation.
Development used `feat/material-gw-bands`, based on `7c5b8c8`; results are
not a BerkeleyGW executable comparison or a converged bulk-material benchmark.

This is the initial repair-validation snapshot, with the source fingerprints
listed below. The later executed same-orbital BerkeleyGW comparison is recorded
separately in [the matched-frequency evidence](../gaas_berkeleygw_matched_eta_20261018/README.md).
New periodic self-consistent source drivers and continuation work must retain
their own validation; the 434-case count below does not include that work.

## Repairs and their mathematical scope

Nonlinear diagonal QP wrappers reuse the shared scalar solver. They test six
seeds, deduplicate converged finite positive-slope roots, and select the root
closest to the on-shell estimate or an explicit continuation target.
The reported candidate count is limited to those seeds. Distinct ties are
uncertified. Implicit response uses the selected local equation
`dE = -partial_parameter(F) / partial_frequency(F)`; failed, negative-slope,
singular and uncertified branches cannot supply a valid backward pass.
No global root enumeration or automatic main-peak identification is claimed.

G0W0, Gamma and paired k-mesh contour paths use matching zero-frequency
body/head/wing caches. With `delta=omega-epsilon_m`, `s=1-2*occupation_m`
and `W0=Wc(0)`, their zero-eta identity is

    Sigma_m = -integral[delta*(Wc(i*nu)-W0)/(delta**2+nu**2)]/pi
              + s*W0/2
              + (sign(delta)+s)*(Wc(abs(delta))-W0)/2.

The same identity applies to the matching q0 correction. Inactive G poles
are not inverted at unrelated bare-response singularities. Real-frequency W
still retains its chosen finite eta; the identity does not establish arbitrary
finite-eta boundary continuity or higher derivatives.

evGW validates fixed occupation sectors and local positive slopes, while its
outer Jacobian retains the changes in both G and W. qsGW retains its full
Hermitian mode-A Fock map. Both stop the numerical forward closures and their
captured parameters before attaching the shared converged-residual response.
This avoids reverse-mode differentiation through dynamic primal loops.

scGW requires a finite positive beta-weighted charge Schur complement in both
tangent and adjoint solves. Static reference density and G(tau) use shared
matrix-function Frechet response; G(i*omega) uses a direct resolvent.
The exact zero-interaction branch supports repeated static levels at first
order, while the Hartree/exchange quadratic dependence on DF factors remains.

Periodic full queries retain OV and VO Lehmann terms and both imaginary
frequency branches. The selected matrix window is used throughout the path;
a tolerance change cannot silently discard off-diagonal self-energy.
Gapped 3D `fc=True` replaces the correlation Gamma node with nonzero tensor
Gauss-cell samples, whose outer weights sum to 1/Nk. Actual shifted source
and internal bands supply these samples. Bare exchange and its original
Madelung term are retained.

A finite FFT model can break exact OV/VO covariance pairing at a zone edge.
For the native H2 two-k HF control, measured relative errors were 0.01119
at q=0 and 0.01698/0.00678 in the other paired channels. Legacy even-W
drivers reject such a model. The full query now supports exact HF at the
original source k points with `fc=False`, subtracting the actual
`F_SCF - hcore - J[density]` potential. The SCF exchange boundary is included.
Finite wrapped FFT products need not give exact Sigma_x/V_HF cancellation.
Pure LDA/GGA retain their arbitrary-k and gapped 3D cell paths.

## Remote validation

All numerical work ran on `jy_gpu_1`: Intel Xeon Platinum 8457C, JAX 0.10.0,
CPU, float64. Each process had a 16-logical-CPU affinity and a monitored
16 GiB RSS budget. Aggregate affinity never exceeded 96 logical CPUs;
each job was bounded below two hours. Local checks were AST, hashes and
Git diff checks only.

The completed focused groups cover **434 distinct pytest cases**. Repeated
runs and diagnostic scripts are not added to that count.

| Group | Distinct cases | Verification |
|---|---:|---|
| QP selection, controls and analytic contour | 104 | 100 original final-snapshot passes plus four corrected 320-node continuity cases; full 18-case contour file passed |
| evGW/qsGW outer response and model controls | 69 | Reconverged FD, JVP/VJP, frames, spin, W0 and composed BSE |
| scGW, Matsubara and coupling response | 50 | Passed in 465.68 s; relevant production files remained byte-identical thereafter |
| Full query, low-dimensional query, cell response and momentum | 100 | Final HF-compatible source passed in 177.53 s |
| Molecular differentiation, poles and transition screening | 38 | Passed in 90.69 s |
| Unrestricted molecular PySCF reference | 2 | Passed in 53.24 s |
| Restricted molecular reference and branch controls | 3 | Passed in 124.17 s |
| Periodic invariants and pole masks | 23 | Passed in 206.61 s |
| Streamed body/head and native full-response controls | 15 | Separate fresh processes: 10 + 3 + 2 |
| Reduced-dimensional interaction | 13 | Passed in 7.77 s |
| Gamma/U-Gamma/legacy metadata and endpoint controls | 13 | Union of 8 Gamma, 3 remaining and 3 static/negative-weight cases, with one repeated case excluded |
| Native HF source query and unsupported-domain guards | 4 | Passed in 34.97 s |

A combined periodic process encountered LLVM allocation/materialization errors
after several tests. Fresh independent processes passed all corresponding
groups with the same numerical tolerances and RSS budget. The cause was not
fully diagnosed. An earlier dense-facade water group reached its 600 s limit
without a final pytest result; it is not counted as passing. The restricted
water benchmark instead explicitly tests the existing cached array-driver
path. A first water diagnostic used an unsupported facade keyword and is
also excluded.

Full-suite, GPU and large-system performance validation were not performed.

## Numerical examples

The graphene diagnostic uses the original PBE/GTH-SZV finite slab model,
source mesh 2x2x1, FFT 25x25x97, nw=16, eta=0.001 Ha, fc=False and the
unchanged T=0 band-touching prescription. No energies were altered to smooth
the plot.

| Zero-based point, band 0 | Earlier selected root / Ha | Repaired selected root / Ha | Formal Z | Tested accepted roots |
|---|---:|---:|---:|---:|
| 6 | -0.749978090540 | -0.749978090540 | 0.206467 | 3 |
| 7 | -1.139976745725 | -0.766688194500 | 0.381479 | 2 |

These values demonstrate a changed selection rule in the declared finite
model. Other roots can remain undiscovered, and the rule can still change
branches along a path.

Water uses the original Cartesian def2-SVP RHF reference, nw=100,
eta=0.001 Ha and all original G/W bands. GradSCF uses spectral ERI factors
at tolerance 1e-12; PySCF uses its mp2fit representation. The default policy
and the PySCF-selected branch differ for orbital 11:

| Selection | Energy / Ha | GradSCF formal Z |
|---|---:|---:|
| GradSCF default | 1.201793674834 | 0.858278 |
| GradSCF explicit continuation | 1.278755685229 | 0.109949 |
| PySCF reference | 1.278671591342 | — |

Across the twelve **matched-branch** levels, the largest difference is
8.40939e-5 Ha, below the unchanged 3e-4 Ha tolerance. Reference energies
serve as numerical seeds; the GradSCF equation is still solved and certified.
Default roots are not asserted to equal an external program's independently
selected roots.

The head/wing endpoint oracle kept its 2e-12 Ha tolerance. Resolving
`|delta|=1e-5` required 320 quadrature nodes: maximum errors for
40/80/160/320 nodes were 9.30e-12, 8.12e-12, 3.98e-12 and 4.79e-13 Ha.
Only those four test cases use the higher order; production defaults were
not changed.

## Retained boundaries

- Multi-seed selection is local and incomplete. Positive formal Z alone
  does not certify a physical spectral peak or linewidth.
- qsGW response still rejects orbital degeneracy in its current static map.
  scGW interacting reference/RPA degeneracy and fourth-order correlation
  response at exactly zero coupling remain unsupported.
- Periodic full queries are eager forward calculations; their geometry/orbital
  AD and JIT are not implemented. Periodic/UHF scGW and unrestricted qsGW
  are not added by these repairs.
- HF full queries require source k points and fc=False. HF cell integration
  and off-mesh HF with the present Ewald band interface remain unsupported.
- Low-dimensional and gapless screened Gamma-cell integration, complete
  material convergence and scGW analytic continuation remain separate work.

## Evidence files

`validation.json` records exact commands, exits, CPU affinities and peak RSS.
`source_manifest.json` records the final relevant source/test fingerprints.
`graphene.json`, `water_branch.json`, `pairing_diagnostic.txt` and
`quadrature.json` preserve the measured controls. Execution scripts and
failed-attempt archives stay in ignored artifacts or remote job directories;
this reproducibility folder contains data and prose only.
