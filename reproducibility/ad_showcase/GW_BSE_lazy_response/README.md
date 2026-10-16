# Lazy structured-RPA vector response: molecular acceptance

The structured Davidson solver now attaches its existing vector response only
when AD requests a derivative. A pure forward call returns the converged
vectors without performing zero-right-hand-side response solves. The physical
eigenproblem, response equations, stability checks, gap guards and shared
linear solver are unchanged.

The complete relevant regression group passed **67 tests in 1205.52 s** on
`jy_gpu_1`, CPU float64/JAX 0.10.0. Native water forward and complete geometry
response then passed separate molecular gates with direct and GMRES screening.
Only `src/gradscf/solvers/eigen/structured_rpa.py` differs between the frozen
pre-fix and post-fix Python source trees. Native sources and the compiled
integral library are checked separately in [native_runtime.json](native_runtime.json).

## Equations and scope

For real symmetric A and B, define

\[
H=\begin{pmatrix}A&B\\B&A\end{pmatrix},\quad
J=\begin{pmatrix}I&0\\0&-I\end{pmatrix},\quad
Hz=\omega Jz,\qquad z^T Jz=1.
\]

For an isolated positive root with constant J, the existing response uses

\[
K=H-\omega J+(Jz)(Jz)^T,\qquad
K\,dz=-dH\,z+Jz\,(z^T dH\,z).
\]

This enforces the metric gauge `z.T @ J @ dz = 0`. Previously the same
expression was evaluated with a zero primal perturbation to attach AD. That
caused linear solves even for ordinary forward calculations. The lazy custom
JVP keeps this formula in the derivative rule and closure-converts the physical
operator parameters using the existing shared spectral utility.

This is still a **first-order isolated-root RPA contract**. It does not add
higher-order RPA, support incomplete degenerate root sets, or bypass failed
stability/linear-response checks. The direct response capacity remains 2N for
the structured equation. Invalid responses remain nonfinite under AD.

## Same-input forward measurement

The frozen water RHF/Cartesian 6-31G* reference has 19 MOs, 70 transitions,
six requested roots, full G/W windows and `max_space=80`. Screening is GMRES
with rtol 1e-11/atol 1e-13, restart 40/maxiter 100; vector response stays enabled
with a bounded direct solve. Neither SCF nor GW is rerun in this timing probe.
Every timed call synchronizes the **complete result pytree**.

| Measurement | Seconds |
| --- | ---: |
| Previous whole-result warm forward | 54.8380 |
| Lazy forward warm samples | 4.5954, 4.5623, 4.5633 |
| Lazy forward warm median | 4.5633 |
| Lazy lowering/compilation | 43.6385 |
| Lazy first execution after compilation | 4.7909 |

Energies and oscillator strengths are identical to the baseline. This exact
shared-node case observes about a twelve-fold forward reduction; it is not an
asymptotic scaling result or an exclusive-device throughput measurement. The
original [profiling record](../GW_BSE_forward_gates/performance_breakdown.json)
remains unchanged. The response solve is deferred, not removed from actual AD.

## Complete molecular derivative checks

The molecular adapter moves the native AO integrals, AO metric, RHF solution,
canonical orbitals, all QP roots/residues, screening, BSE kernel and dipoles.
It reuses the exact geometry, normalized nuclear direction and continued
19-QP branch of the earlier
[independently stepped record](../X02_water_gw_bse_response/primary/g0w0_full_nw200_continued.json).

With both direct and GMRES screening, the new structured-solver derivatives
match the previous dense-solver derivatives within 1.37e-12 in the reported
units. Three fresh reconverged central-difference steps are retained. At the
final 1e-5 bohr step, the largest errors are 3.34e-6 for QP residues and
2.39e-8 for oscillator strengths; all predeclared gates pass. QP-residue
differences at larger steps remain visible rather than being omitted.

| Complete geometry map | Direct screening | GMRES screening |
| --- | ---: | ---: |
| Warm forward, s | 0.3764 | 4.7588 |
| Warm full JVP, s | 0.9290 | 54.3555 |
| Whole verification process, s | 69.36 | 323.74 |
| Process peak RSS, KiB | 2,422,156 | 3,716,204 |

The GMRES-screened response remains expensive; the forward optimization does
not establish a backward speedup. These derivatives are QP/optical responses,
not a variational GW+BSE excited-state force.

## Reproduction records

Results are [forward_gmres.json](forward_gmres.json),
[geometry_direct.json](geometry_direct.json), and
[geometry_gmres.json](geometry_gmres.json). Each records all Python source
hashes and relevant input/reference hashes. The exact acceptance driver,
changed solver and narrow regression are under [executed/](../../../artifacts/ad_showcase/provenance/GW_BSE_lazy_response/executed).
[Validation folders](validation/) contain commands, exit codes, raw logs,
wall time and RSS budgets. The larger regression includes the new lazy-response
checks, existing stable/matrix-free RPA, nested BSE JIT, response configuration,
full BSE and matrix-free BSE tests.

The [Python source archive checksum](source_archive_sha256.txt) identifies
`artifacts/ad_showcase/development/RPA_lazy_response/lazy-source.tar.gz` in the local ignored
workspace. This capsule is a focused acceptance record, not a passing full
GradSCF suite or a validation of every later worktree change.
