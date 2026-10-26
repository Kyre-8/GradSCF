# Si: narrow same-model scalar-root refinement

This data set copies the original 19-point, six-output-band native PBE/SZV
stable path and retries only its three rejected scalar QPs. All G bands and
all OV+VO screening transitions remain unchanged. The public SRC query uses
`nw=16`, `eta=0.001 Ha`, `fc=False`, `pole_subtraction=True` and the shared
hybrid root solver with `ftol=xtol=1e-9 Ha`, `maxiter=120`. The pinned query
source starts with SHA `d294e39a`; 22 source and 65 native fingerprints pass.

Two retries pass: point 5/band 1 has residual 1.26e-14 Ha and point 7/band 1
has residual -2.78e-16 Ha. Point 5/band 6 remains rejected, with residual
about -0.100668 Ha. Its original candidate is retained and masked from QP
plots. The resulting path is **113 accepted QPs out of 114 requests**; it
is not an all-root success.

A fresh public SCF seeded with the old density matches both source and all
19 query KS spectra to 7.77e-16 Ha, below the 1e-9 Ha gate. Original successful
cells and all unpatched arrays pass strict `np.array_equal` checks. The old
case is not modified. Original array/native SHA provenance and every retry
are recorded in [native.json](native.json).

Only the repair phase takes 40.113 s of GW work, 59.934 s including its SCF
preparation, and 63.293 s in the resource wrapper. It runs on `c20`, CPU0--15,
under 600 s / 16 GiB limits, with peak RSS 3,375,728 KiB (about 3.22 GiB).
These are not full-path runtimes. [refinement_budget.json](refinement_budget.json)
copies the new job's resource record; [original_budget.json](original_budget.json)
keeps the original 19-point job separately. The inherited `peak_rss_kib` field
inside native.json belongs to that original path; use the refinement budget
for the new phase. No executed native record or source hash is rewritten.

The PNG/PDF shows only directly computed and converged QP samples. This small
anisotropic two-q, body-only finite model does not establish material,
basis/k/FFT/frequency convergence or an experimental gap.
