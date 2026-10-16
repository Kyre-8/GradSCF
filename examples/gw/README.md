# GW examples

Run ordinary Python scripts from the repository root with `PYTHONPATH=src`.
The examples set numerical parameters directly in code; independent external
comparisons and plotting are separate from native electronic calculations.
Float64, units, orbital windows and branch identity matter for the response
examples. A converged Dyson root need not be the largest-weight peak.

| Example | Scope and evidence |
| --- | --- |
| [Water native QP/BSE](water_primary_qp.py) | RHF/Cartesian 6-31G*, native eta=0 main-weight starting points, actual finite-eta G0W0 roots, all 70 TDA/full-BSE optical states. [Recorded calculation](../../reproducibility/ad_showcase/X02_water_gw_bse_response/README.md). |
| [Water forward convergence](water_forward_convergence.py) / [checks](compare_water_forward_convergence.py) / [figures](plot_water_forward_convergence.py) | One frozen 19-MO G0W0 reference: imaginary-frequency quadrature and separate G/W/optical windows. All solves converge, but smaller windows miss the declared accuracy targets. [Results and provenance](../../reproducibility/ad_showcase/GW_BSE_forward_gates/README.md); no basis-convergence claim. |
| [Fixed-QP BSE solver comparison](water_bse_solver_comparison.py) | Same six TDA/full states with dense/Davidson and direct/GMRES screening, true residuals, synchronized cold/warm times and fresh-process RSS. The [lazy-RPA acceptance](../../reproducibility/ad_showcase/GW_BSE_lazy_response/README.md) removes unnecessary pure-forward response solves and validates complete molecular JVPs. Historical timings remain separate; this is not an asymptotic scaling comparison. |
| [Water geometry response](water_geometry_response.py) | Moving native integrals, relaxed HF, QP/W and optical operators. Explicit local branch anchors; multistep derivative checks. `dOmega/dR` is not a variational excited-state total-energy force. |
| [Water dynamic BSE response](../vibrations/water_bse_dynamic_raman.py) | All 70 full-BSE states and three relaxed harmonic mode derivatives; independent resolvent and full spectral FD checks. [Results](../../reproducibility/ad_showcase/V06_water_bse_dynamic_raman/README.md). Frequency-dependent Placzek activity, not vibronic KHD. |
| [Water MolGW comparison](compare_water_molgw.py) / [figures](plot_water_molgw.py) | Saved-data independent energies/weights/optics and step convergence. Rejected coarse graphical-grid residue gates remain visible. |
| [Water QP removal spectrum](water_photoelectron_response.py) | Compose the verified GW JVP with actual `2Z` weights; spectral area, missing tails and geometry sensitivity. [Results](../../reproducibility/ad_showcase/X14_water_qp_removal/README.md). Satellites and photoionization matrix elements are omitted. |
| [Methane finite-eta resolvent](methane_resolvent_spectrum.py) / [figures](plot_methane_resolvent_spectrum.py) | Existing public CD resolvent representation, G0W0/evGW0/evGW, all 145 full-BSE states in the declared window. [Fresh MolGW comparisons](../../reproducibility/ad_showcase/MolGW_reference_build/methane_replay/README.md). |
| [QP numerical methods](qp_solvers.py) / [branch diagnostics](plot_qp_branches.py) | Shared root algorithms and the distinction between implicit local derivatives and discrete branch selection. [Recorded failures and response gates](../../reproducibility/ad_showcase/GW_BSE_response_gates/README.md). |
| [Outer implicit response](implicit_response.py) | Small fixed-frame method-specific checks. Do not interpret these as complete nuclear response without the moving-input workflow. |
| [Matsubara scGW](scgw_matsubara_h2.py) / [implicit scGW](scgw_implicit_response_h2.py) | Existing finite-temperature matrix self-consistency and its separate response contract. |
| [Periodic Si direct QP levels](pbc_silicon_qp.py) | Small 3D PBE/GTH-SZV, two k points, complete screening/internal sums with four requested output levels. Three converge; one retained failure is explicitly omitted from the QP energy plot. [Data, masks and figure](../../reproducibility/ad_showcase/PBC_silicon_gw/README.md). No continuous GW bands or bulk-convergence claim; use `PYTHONPATH=src:.`. |
| [Direct off-mesh GW](periodic_bands.py) / [independent check](compare_periodic_bands.py) | Complete OV+VO response, both contour sides and shifted internal bands. [H2](../../reproducibility/ad_showcase/periodic_gw_query/README.md) and [Si](../../reproducibility/ad_showcase/periodic_gw_query/silicon/README.md) short segments retain all requested QP certificates. Body-only, eager forward; no interpolation, degenerate-band self-energy or material-convergence claim. |
| [Electron-vibration spectrum](phonon_spectrum.py), [Cu(111)](cu111_photoemission.py), [adamantane](adamantane_photoemission.py) | Existing supplied/effective-vertex models and native molecular prototype. Each script declares frozen inputs, broadening and comparison limitations. |

The methane resolvent example does not repair or retroactively validate the
original facade run's recorded OpenBLAS failure. Its same-equation W gate and
independent reference results are explicit in the linked archive. The water
response adapter currently uses real closed-shell RHF; the original
omegaB97X-D SCF/TD examples do not establish a GW starting-point adapter.

The [AD showcase inventory](../AD_SHOWCASE_STATUS.md) distinguishes completed
components from pending higher-response, periodic and electron-vibration work.
