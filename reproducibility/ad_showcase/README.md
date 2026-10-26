# AD showcase: numerical results and visualizations

The complete scope and verification contract are in
[the example plan](../../examples/AD_SHOWCASE_PLAN.md).

Each result directory will contain small numerical JSON/CSV files, PNG/PDF
figures, and a README stating the calculation settings, commands, provenance,
validation errors and limitations. Raw numerical data must accompany a figure.
Bulk arrays, compilation caches and temporary logs belong in `artifacts/`.

## Workspace layout

| Location | Contents |
| --- | --- |
| `examples/` | Maintained GradSCF calculations, separate comparisons and plotting scripts. |
| `reproducibility/ad_showcase/<case>/` | Scientific JSON/CSV/NPZ inputs and results, figures, summary metadata and README. |
| `artifacts/ad_showcase/provenance/<case>/` | Local executed-source snapshots and raw job records. |
| `artifacts/ad_showcase/development/` | Earlier diagnostics, transfer bundles, checkpoints and unvalidated drafts. |

The local archive is ignored by Git. Historical files retain their original
bytes and recorded execution paths; `artifacts/ad_showcase/organization.json`
maps the old paths to their archived locations. Scientific data and the
metadata consumed by public plotting scripts remain at their existing paths.
Use the public examples as the maintained entry points.

No Python, shell, notebook or other executable source files belong in this
result tree. Computational implementations belong in `src/`; demonstration
scripts belong in `examples/`. This directory contains data, figures and prose.

The first completed calculation is
[G01_water_rhf_gradient](G01_water_rhf_gradient/README.md): water/Cartesian
6-31G* native-integral AD with implicit RHF orbital response, central differences
and an independent PySCF analytic-gradient comparison. Maximum AD-reference
error is 4.52e-12 Hartree/Bohr. Other planned methods and properties remain
unvalidated by this result.

![Water RHF gradient checks](G01_water_rhf_gradient/gradient_checks.png)

The planned IR/Raman and electronic spectra will retain raw sticks and disclose
display broadening. Imposed broadening is not a predicted physical linewidth.

Additional measured components:

- [PBE moving-grid gradient](G01_water_pbe_gradient/README.md).
- [RHF geometry optimization](G04_water_rhf_optimization/README.md).
- [RHF Hessian, H2O/D2O shifts and animations](G05_water_rhf_hessian/README.md).
- [RHF dipole and static polarizability](R01_R02_water_electric/README.md).
- [RHF harmonic IR and static Raman](V01_V02_water_rhf_spectra/README.md).
- [Relaxed CIS density and permanent dipole](X04_formaldehyde_cis_density/README.md), distinguished from the unrelaxed wavefunction density and checked by independent field/AO-potential energies.
- [CIS transition densities and NTOs](X05_formaldehyde_cis_ntos/README.md), with independent normalization, reconstruction, weighted-projector and transition-moment checks.
- [Water CIS wavefunction sensitivities](X06_water_cis_probes/README.md), separating relaxed geometry response from normalized-amplitude and fixed-amplitude reference-orbital partials.
- [Tracked bright/dark oscillator-strength response](X07_formaldehyde_oscillator_response/README.md), with independently checked normal-mode derivatives and finite-amplitude intensity modulation.
- [Excited-state polarizability and Stark response](X08_formaldehyde_cis_stark/README.md), using total excited energy derivatives and independent finite-field tensor checks.

These entries validate the stated finite-basis components. The
[status inventory](../../examples/AD_SHOWCASE_STATUS.md) records remaining
methods and response prerequisites across the full collection.
