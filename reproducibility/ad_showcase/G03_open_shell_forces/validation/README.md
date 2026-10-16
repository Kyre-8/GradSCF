# G03 reference-response evidence

All numerical work here ran on `jy_gpu_1`, CPU float64, with
`OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2`. No GradSCF source was changed
to repair the external reference. All probes share the saved NO/UMP2 native
density as an orientation-selecting initial guess and independently reconverge
the same UHF branch. The native force gate remains `2e-6 Eh/bohr`.

| Record | Evidence and boundary |
| --- | --- |
| [Controlled tolerance probe](open_shell_ucphf_probe.json), [source](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open_shell_ucphf_probe.py), [log](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open-shell-ucphf-probe.log) | Default response; single-variable cycle/tolerance controls; paired Krylov dependence controls; one bounded public `x0` restart. The restart's relative residual still fails `1e-10`, despite force agreement. |
| [Initial probe](open_shell_ucphf_probe_initial.json), [source](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open_shell_ucphf_probe_initial.py), [log](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open-shell-ucphf-probe-initial.log) | Preserved initial default/tighter-response comparison. |
| [Precision probe](open_shell_ucphf_probe_precision.json), [source](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open_shell_ucphf_probe_precision.py), [log](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open-shell-ucphf-probe-precision.log) | Preserved paired tolerance/dependence sweep before the bounded restart. |
| [Nullspace diagnostic](open_shell_ucphf_nullspace_probe.json), [source](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open_shell_ucphf_nullspace_probe.py), [log](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open-shell-ucphf-nullspace.log) | Complete 337-dimensional original operator; one near-zero direction aligned with the physical pi generator. The unresolved iterative residual was mostly outside that nullspace. No solve was replaced in this diagnostic. |
| [Augmented-gauge probe](open_shell_ucphf_gauge_probe.json), [source](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open_shell_ucphf_gauge_probe.py), [log](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/open-shell-ucphf-gauge.log) | SciPy solve of the original operator with one physical-pi constraint. Full unprojected residual and force checks pass; `z+pi` and `z-pi` leave the nuclear gradient unchanged to `1.5e-14 Eh/bohr`. |
| [Executed four-case reference](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/compare_open_shell_forces_pyscf_executed.py), [log](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/correlated-references-gauge.log) | OH/NO times UMP2/UCCSD, all four independent gradients pass. The executed source SHA256 matches each final reference JSON; later measured-result comments in the public example do not change this preserved source. |
| [OH original MP2 reference](OH_mp2_pyscf_original.json), [OH original CCSD reference](OH_ccsd_pyscf_original.json) | Prior force comparisons, kept separately from the final augmented-gauge reference. Their force agreement alone did not establish a checked true UCPHF residual. |

The final reference JSON files in the parent directory contain the complete
response eigenvalues, matrix asymmetry, physical-pi/null overlap, RHS norm and
compatibility, absolute/relative/maximum unprojected residual, gauge residual,
versions, runtime and input/source hashes. The original default NO/MP2 force
failure is retained in the controlled probe rather than accepted as a result.

The four already completed native correlated runs are preserved in
[OH MP2](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/OH_MP2-complete-force.log), [OH CCSD](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/OH_CCSD-complete-force.log),
[NO MP2](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/NO_MP2-complete-force.log), and [NO CCSD](../../../../artifacts/ad_showcase/provenance/G03_open_shell_forces/validation/NO_CCSD-complete-force.log).
They were not rerun during reference repair. Their integration snapshot is
separate from the stationary snapshot used for the four mean-field cases;
see [the per-case provenance](../provenance.json).

The main reference implementation received an independent read-only review
of matrix/block ordering, physical pi normalization, the unprojected residual
and restoration of the scoped override. That review adds no numerical evidence.
