# Molecular vibrations and spectroscopy

After the RHF optimization and Hessian examples in `examples/geometry`, run
`water_ir_raman.py`, `compare_water_spectra_pyscf.py`, then
`plot_water_spectra.py` on `jy_gpu_1`. The first script uses JVPs of relaxed
dipole/polarizability derivatives; the second supplies independent PySCF
finite-difference validation. Neither calculation imports the other solver.

[Saved spectra and theory](../../reproducibility/ad_showcase/V01_V02_water_rhf_spectra/README.md)
specify mass units, intensity conversions, harmonic approximations and display
broadening. Raman activities are kept distinct from experimental intensities.
For normal-mode animations and isotope shifts, see the [Hessian results](../../reproducibility/ad_showcase/G05_water_rhf_hessian/README.md).

## Methods and saved calculations

All scripts use ordinary Python settings and functions; no CLI is required.
Each electronic method is optimized independently before its harmonic analysis.
Comparisons use the same geometry, basis and grid within each method pair.

| Workflow | Native example | Independent comparison | Saved results |
| --- | --- | --- | --- |
| PBE | `water_ir_raman.py` with the PBE Hessian/input | `compare_water_spectra_pyscf.py` | [PBE](../../reproducibility/ad_showcase/V01_V02_water_pbe_spectra/README.md) |
| PBE0, B3LYP, original omegaB97X-D | `water_hybrid_spectra.py` | `compare_water_hybrid_pyscf.py` | [PBE0](../../reproducibility/ad_showcase/M2_water_pbe0_spectra/README.md), [B3LYP](../../reproducibility/ad_showcase/M2_water_b3lyp_spectra/README.md), [omegaB97X-D](../../reproducibility/ad_showcase/M2_water_wb97xd_spectra/README.md) |
| MP2, CCSD | `water_correlated_spectra.py` | `compare_water_correlated_pyscf.py` | [MP2](../../reproducibility/ad_showcase/V05_water_mp2_spectra/README.md), [CCSD](../../reproducibility/ad_showcase/V05_water_ccsd_spectra/README.md) |
| Dynamic TDHF / Placzek | `water_dynamic_raman.py` | `compare_water_dynamic_raman_pyscf.py` | [Complete-space frequency/damping and full mode response](../../reproducibility/ad_showcase/V06_water_dynamic_raman/README.md) |
| Dynamic full BSE / Placzek | `water_bse_dynamic_raman.py` | `compare_water_bse_resolvent.py` | [All 70 states, three relaxed mode derivatives, frequency/damping checks](../../reproducibility/ad_showcase/V06_water_bse_dynamic_raman/README.md); the independent SciPy oracle uses the same native kernel |
| Single-state vibronic KHD Raman | `formaldehyde_khd_raman.py` | `compare_khd_resolvent.py` | [HCHO harmonic/linear-HT model, both time orderings and state convergence](../../reproducibility/ad_showcase/V07_formaldehyde_khd/README.md) |
| CO2 selection rules | `co2_selection_rules.py` | `compare_co2_hessian_pyscf.py` | [IR/Raman](../../reproducibility/ad_showcase/V03_co2_selection_rules/README.md) |
| Carbon isotope shift | `co2_carbon_isotope.py` | Saved independent force constants | [12CO2/13CO2](../../reproducibility/ad_showcase/V04_co2_carbon_isotope/README.md) |
| Quartic anharmonic models | `h2_quartic_levels.py`, `water_stretch_anharmonic.py` | Independent electronic derivatives and sinc DVR | [H2](../../reproducibility/ad_showcase/V08_h2_quartic/README.md), [water two-stretch model](../../reproducibility/ad_showcase/V08_water_two_stretch/README.md); explicit finite-domain and frozen-mode limits |

Use each workflow's separate `plot_*.py` script after generating its native and
reference JSON. The water workflows share mass normalization, IR/Raman
conversions and plot utilities. `CorrelatedEnergy` supplies relaxed MP2/CCSD
total-energy derivatives through the existing HF and correlation methods.

The original omegaB97X-D calculation includes long-range exchange and its
separate CHG geometry correction. Its raw finite-grid rotational residual is
retained, including the independent reference value. None of these finite
basis/grid comparisons establishes basis convergence or anharmonic accuracy.
