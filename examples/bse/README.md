# Molecular BSE examples

These examples separate the GradSCF calculation, independent checks and
plotting. Run ordinary Python scripts from the repository root with
`PYTHONPATH=src`; numerical verification for the AD showcase was performed
on `jy_gpu_1`. Each results directory records the physical approximation,
executed source, numerical gates and hardware.

| Example | Calculation and evidence |
| --- | --- |
| [Water electron-vibration baseline](water_ep_coupling.py) | RHF/3-21G, G0W0 and three singlet TDA-BSE states; screened RHF vertices with frozen GW correction and BSE kernel. Zero-temperature linear multiphonon, one-phonon Fan and Fan+DW are separate approximations. [Refreshed results](../../reproducibility/ad_showcase/X15_water_bse_baseline/README.md). |
| [Water temperature response](water_ep_temperature.py) | Fixed baseline electronic model and modes; Bose occupations at 0, 300, 600 and 1200 K, with a complete temperature JVP of the one-phonon optical spectrum. Run [the independent auxiliary-Hamiltonian check](compare_water_ep_temperature.py), then [plotting](plot_water_ep_temperature.py). [Results and raw weights](../../reproducibility/ad_showcase/X15_water_bse_temperature/README.md). |
| [Matched frozen and relaxed mode vertices](water_mode_vertices.py) | Water/6-31G* at one saved RHF minimum, with identical electronic origins, modes and dipoles. Compare projected RHF-Fock slopes with complete G0W0/TDA excitation-energy slopes. Run [finite differences](compare_water_mode_vertices.py), [diagonal vibronic spectra](water_diagonal_vibronic.py), then [plotting](plot_water_mode_vertices.py). [Results](../../reproducibility/ad_showcase/X15_water_matched_vertices/README.md). |
| [Full-BSE dynamic Raman response](../vibrations/water_bse_dynamic_raman.py) | All 70 full static-BSE states of water/6-31G*, with complete geometry derivatives of the frequency-dependent polarizability. [Independent resolvent check](../vibrations/compare_water_bse_resolvent.py), [plotting](../vibrations/plot_water_bse_dynamic_raman.py) and [results](../../reproducibility/ad_showcase/V06_water_bse_dynamic_raman/README.md). |
| [GW and BSE geometry response](../gw/water_geometry_response.py) | G0W0, evGW0 and evGW on explicitly continued QP branches, followed by TDA or full BSE. See the [GW index](../gw/README.md) for MolGW comparisons, branch selection and response limits. |

The matched diagonal-vertex example omits off-diagonal mode coupling in
both models; its vanishing antisymmetric-mode diagonal slope does not imply
that the mode has no vibronic effect. Its zero-temperature displaced harmonic
model is distinct from the baseline's coupled-state model and the finite-T
one-phonon extension. The latter changes only Bose populations and is not a
thermal multiphonon Franck–Condon calculation.

Dynamic Raman results use frequency-dependent Placzek activities and a static
BSE kernel. They are not a vibronic Kramers–Heisenberg–Dirac calculation.
All spectral widths are imposed resolution parameters, not calculated
lifetimes. Raw oscillator weights and finite-window losses are retained;
neither the three-state examples nor their trace sums establish a complete
electronic oscillator-strength sum rule.
