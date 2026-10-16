# Full configuration-interaction examples

Run the scripts from the repository root with `PYTHONPATH=src` and CPU
float64 enabled. Molecular cases use the native-integral environment; the
Hubbard example differentiates a model parameter rather than nuclear positions.

| Entry | Observable and boundary |
| --- | --- |
| [Ground state](ground_state.py) | H2 HF-to-FCI energies, density and spin diagnostics. |
| [Bond stretch](bond_stretch.py) | H2 energies and natural occupations at several geometries; no nuclear derivative is taken. |
| [Active space](active_space.py) | LiH with a folded doubly occupied core and fixed active orbitals; no orbital optimization. |
| [Parameter response](response.py) | Two-site Hubbard energy derivative with respect to hopping. |
| [H2 excited-state absorption](h2_excited_absorption.py) | Complete finite-space transition weights and moving-metric bond response; [independent reference](compare_h2_excited_absorption_pyscf.py), [plot](plot_h2_excited_absorption.py), [X09 record](../../reproducibility/ad_showcase/X09_h2_fci_esa/README.md). |
| [H3 nonadiabatic coupling](h3_nonadiabatic_coupling.py) | CI, orbital-frame and AO-cross-overlap response near a conical intersection; [independent reference](compare_h3_nac_pyscf.py), [plot](plot_h3_nac.py), [X13 record](../../reproducibility/ad_showcase/X13_h3_fci_nac/README.md). |

X13 uses neutral H3/STO-3G, all nine determinants in the fixed
`(N_alpha,N_beta)=(2,1)`, `M_s=+1/2` sector, and no HF calculation.
At exact degeneracy the two spatial states form a complete rank-two subspace;
their opposite-spin partners are not included, and individual-state NACs are
undefined. The raw lab-frame couplings retain their electronic rigid-motion
generators; ETF/ERF corrections are not silently imposed.

The small public [FCI overlap interface](../../src/gradscf/fci/README.md)
supports real coefficients in a shared `FCISpace`. A supplied orbital
cross-overlap matrix currently requires at most three occupied orbitals per
spin. The records describe exact validation gates and larger-space boundaries.
