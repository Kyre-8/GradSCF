# A04 — Taylor MP coefficients and convergence

A04 reuses the existing [Taylor example](../../../examples/mp/taylor_series.py),
[independent molecular comparison](../../../examples/mp/compare_water_pyscf.py),
and [high-order convergence report](../../../examples/mp/CONVERGENCE.md).
Their roles are distinct: agreement of perturbation coefficients validates an
implementation; convergence of a physical MP series is a separate question.

## Current frozen-source checks

On `jy_gpu_1`, the H6/STO-3G Taylor MP2–MP6 example exited zero and checked E2/E3
against existing independent low-order kernels. It retained 400 determinants,
used maximum excitation rank 6, and reported response residual `4.963e-17 Eh`.
Its MP6 total is `-3.250531075592 Eh`. The full output is
[A04_taylor_small.log](../../../artifacts/ad_showcase/provenance/A04_taylor_mp/A04_taylor_small.log).

The water/STO-3G independent comparison also exited zero. GradSCF and PySCF
each constructed integrals and converged their own HF state; the higher-order
reference uses PySCF full-space Hamiltonian actions with the existing NumPy
Rayleigh–Schrödinger recurrence, not an assumed PySCF MP8 API. E2–E8 agreed
within `7.08e-14 Eh`; the MP8 total was `-75.009673431315 Eh`, compared with
FCI `-75.009700279727 Eh`. Water/Cartesian 6-31G*, frozen core 1, MP2 agreed
within `2.67e-14 Eh`. See [A04_water_pyscf.log](../../../artifacts/ad_showcase/provenance/A04_taylor_mp/A04_water_pyscf.log).
The water small-basis native and independent sections took 17.869 s and 0.096 s;
their AD/compilation work differs, so these are not a speed comparison.

Settings and execution hashes are in [the common runtime record](../A03_A07_RUNTIME.md)
and `verification_manifest.json`. These calculations do not establish large
system scaling or nuclear-coordinate derivatives.

## Reused high-order evidence and the incomplete rerun

The existing [He2 convergence figure](../../../examples/mp/he2_convergence.png)
and [Ne/F− figure](../../../examples/mp/ne_f_convergence.png) belong to the
2026-10-10 report packaged in commit
`1242c9346ab0ddc9fc6bc741efa2b0d3fc32ccf1`. Their file hashes are recorded in
[historical_artifacts.json](historical_artifacts.json). The available report
records Apple M4 Pro / Python 3.12.2 / JAX 0.8.1 / PySCF 2.9.0. Its original
runtime source-tree hash and ignored raw JSON/log bundle are not present in
this worktree; the figures and report are preserved without relabeling them
as new remote runs.

That report establishes native He2(2+) RMP/UMP2–30 in a fixed reconstructed
minimal basis, with independent coefficients and FCI. It shows both convergence
and growing stretched-bond corrections, and the deceptively small UMP3→UMP4
barrier change. It retains the unresolved UMP20 maximum-position discrepancy
against the old paper. Atomic Ne/F− high-order curves are explicitly external
PySCF/NumPy calculations; only atomic MP2/MP3 are native GradSCF checks.

A new attempt of the complete original He2 script printed its sampled-distance
rows and two optimized barriers, then exited **134** with
`LLVM ERROR: Unable to allocate section memory!`. See the unmodified
[failure log](../../../artifacts/ad_showcase/provenance/A04_taylor_mp/A04_native.log). No completed native MP30 JSON was produced and
no new full comparison is claimed. Repeated per-geometry compilation is a
candidate explanation from the script structure; memory/VMA measurements were
not collected, so the root cause remains undetermined. The accepted scope here
is reuse of the historical report plus the completed current small-system gates.
