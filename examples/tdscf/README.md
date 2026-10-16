# Excited-state examples

Run these ordinary Python scripts from the repository root with `PYTHONPATH=src`.
Each result record states its basis, geometry, units, convergence gates, source
hashes, independent checks, and measured scope. Native calculations and PySCF
references are separate scripts; they do not share electronic integrals.

| Entry | Example and measured record |
| --- | --- |
| X01 | [Vertical absorption](formaldehyde_absorption.py): HF/PBE/PBE0/B3LYP/original omegaB97X-D, TDA/full TD, first six singlets; [record](../../reproducibility/ad_showcase/X01_formaldehyde_absorption/README.md). |
| X02 | [CIS geometry gradients](formaldehyde_cis_gradients.py): excitation and total excited energy derivatives with transition-density state matching; [record](../../reproducibility/ad_showcase/X02_formaldehyde_cis_gradients/README.md). |
| X03 | [Tracked CIS optimization](formaldehyde_cis_optimization.py): one local S1 minimum, independently checked stationarity and finite-difference gradient curvature; [record](../../reproducibility/ad_showcase/X03_formaldehyde_cis_optimization/README.md). |
| X04 | [Relaxed CIS density/dipole](formaldehyde_cis_density.py), [independent comparison](compare_formaldehyde_cis_density.py), and [plot](plot_formaldehyde_cis_density.py): total-energy response density versus unrelaxed wavefunction 1-RDM; [record](../../reproducibility/ad_showcase/X04_formaldehyde_cis_density/README.md). |
| X05 | [CIS NTOs and transition density](formaldehyde_cis_ntos.py): normalized attachment/detachment, full transition operators, NTO weights and electron–hole centroids; [record](../../reproducibility/ad_showcase/X05_formaldehyde_cis_ntos/README.md). |
| X07 | [Oscillator-strength geometry response](formaldehyde_oscillator_response.py): tracked bright/dark formaldehyde states, fixed harmonic mode directions and local/finite-amplitude intensity modulation; [record](../../reproducibility/ad_showcase/X07_formaldehyde_oscillator_response/README.md). |
| X08 | [Excited polarizability and Stark response](formaldehyde_cis_stark.py): total excited energy field derivatives, ground/excited tensor differences and independently checked field curves; [record](../../reproducibility/ad_showcase/X08_formaldehyde_cis_stark/README.md). |
| X10 | [Paired RHF/CIS minima, AD Hessians, emission and harmonic 0-0 bookkeeping](../../reproducibility/ad_showcase/X10_formaldehyde_cis_emission/README.md). |
| X11 | [Franck–Condon progression](formaldehyde_franck_condon.py), [plot](plot_formaldehyde_franck_condon.py): explicit common-ground-Eckart-slice double-harmonic model, raw versus model Duschinsky matrices, normalized unit-Condon probabilities and fixed-model 0–0 AD; [record and approximation](../../reproducibility/ad_showcase/X11_formaldehyde_franck_condon/README.md). |
| X12 | [Linear Herzberg–Teller activation](formaldehyde_herzberg_teller.py), [plot](plot_formaldehyde_herzberg_teller.py): frozen X11 surfaces with X07 full electronic transition-dipole derivatives; raw FC/HT/interference/total oscillator strengths, independent quadrature, closure and combined molecular gauge checks; [record and scope](../../reproducibility/ad_showcase/X12_formaldehyde_herzberg_teller/README.md). |

The shared helper in `gradscf_tools.excited` assembles physical inputs and state
tracking around the existing SCF, AO-to-MO and CI solvers. The examples do not
implement another CIS matrix or iterative solver. Isolated-state convergence,
root coverage, state tracking and response convergence are distinct checks.

X04 field rendering additionally uses scikit-image and its small dependencies;
the result README records the isolated plotting installation. Visualization
grids are independent of the exact AO-metric normalization checks. Later
catalog entries remain planned until their own gates and records exist.
