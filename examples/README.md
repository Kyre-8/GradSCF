# GradSCF examples

Run from the repository root with `PYTHONPATH=src`, or use an editable GradSCF
installation. Examples are ordinary Python scripts; numerical settings are
written directly in the code. Individual files describe optional dependencies,
units, convergence criteria and measured outputs where available.

The [AD showcase plan](AD_SHOWCASE_PLAN.md) catalogs geometry derivatives,
IR/Raman spectra, excited-state properties and periodic response examples.
It distinguishes planned extensions from measured results. Selected figures
and data live in [`reproducibility/ad_showcase`](../reproducibility/ad_showcase/).
The [implementation inventory](AD_SHOWCASE_STATUS.md) records the verified
scope and remaining checks for each entry; the plan alone is not a feature list.

This directory contains the maintained runnable examples. Figures and numerical
outputs are organized in `reproducibility/ad_showcase/`; copied execution
scripts, job logs and development drafts are kept in the ignored
`artifacts/ad_showcase/` archive. A result directory does not contain a second
maintained copy of its example source.

Electronic-structure calculations and solvers are implemented under `src/`.
Examples define inputs and call those implementations. Do not import or execute
code from `reproducibility/` or the development archive.

## Directory guide

| Directory | Examples |
| --- | --- |
| [basis](basis/README.md) | Basis optimization, counterpoise and neural basis assembly. |
| [geometry](geometry/) | Nuclear gradients and independent derivative checks. |
| [response](response/) | Electric-field derivatives and static polarizabilities. |
| [vibrations](vibrations/) | Harmonic IR/Raman intensities and mode response. |
| [tdscf](tdscf/README.md) | Absorption spectra, state tracking, excited-state derivatives and relaxed CIS densities. |
| [bse](bse/README.md) | Optical spectra, temperature response and matched frozen/relaxed BSE electron-vibration vertices. |
| [cc](cc/) | Ground-state and EOM coupled cluster. |
| [ci](ci/) / [fci](fci/) | Truncated and full configuration interaction. |
| [dfpt](dfpt/README.md) | Static response, finite q and phonons. |
| [gw](gw/README.md) | Quasiparticles, complete molecular GW/BSE response, Matsubara scGW and electron-phonon effects. |
| [localization](localization/README.md) | Boys orbitals and optimization animations. |
| [mp](mp/README.md) | Perturbation series and convergence studies. |
| [ofdft](ofdft/) | Orbital-free DFT and external comparisons. |
| [pbc](pbc/) | Periodic basis, SCF, forces, temperature and symmetry. |
| [solvers](solvers/README.md) | Shared linear/eigenvalue solvers and degenerate response. |
| [training](training/README.md) | Neural XC, force supervision and response targets. |
| [xc](xc/README.md) | Complete traditional recipes, custom formulas and external-model adapters. |
| [thermally_excited_graphite](thermally_excited_graphite/README.md) | Graphite phonons, electronic-temperature scans, local anharmonic diagnostics and relative vibrational spectra. |

## Periodic foundations

| Example | Purpose |
| --- | --- |
| [Periodic H2](pbc/periodic_h2.py) | Gaussian/GTH periodic ground-state calculation. |
| [Basis library](pbc/periodic_basis.py) | Offline periodic basis and pseudopotential selection. |
| [Finite temperature](pbc/periodic_temperature.py) | Fermi occupations, free energy and static response. |
| [Forces](pbc/periodic_forces.py) | HF and OFDFT geometry derivatives. |
| [Spatial symmetry](pbc/periodic_symmetry.py) | Space-group and Cartesian AO/Bloch transformations. |
| [Irreducible integration](pbc/periodic_irreducible.py) | Full/irreducible SCF comparison and target contraction counts. |
| [Periodic XC training](training/periodic_xc.py) | Shared Sample/Trainer in fixed-density, explicit and implicit modes. |
| [DFPT and silicon phonons](dfpt/README.md) | Molecular/periodic response, external finite-q energies and material checks. |

Automatic spatial symmetry discovery requires `spglib` (the optional `symmetry`
extra). The small-cell examples demonstrate APIs and derivative consistency;
they are not converged material benchmarks. See the
[periodic scope and validation](../src/gradscf/pbc/README.md).

## Other methods

- [CI](ci/), [FCI](fci/), [CC](cc/) and [MP](mp/README.md): correlated wavefunctions,
  perturbation series, spectra and differentiable observables.
- [GW](gw/), [BSE](bse/) and [OFDFT](ofdft/): quasiparticles, optical response,
  electron-phonon effects and orbital-free models.
- [Training](training/README.md) and [basis optimization](basis/README.md):
  external functional models, reference targets and basis parameters.
- [Neural basis assembly](basis/nnao_basis.py) and
  [neural SCF force supervision](training/train_neural_scf_forces.py): native
  integral and gradient demonstrations.
- [Ethylene Boys orbitals](localization/ethylene_boys.py), [BODIPY Boys orbitals](localization/bodipy_boys.py)
  and their `animate_*` scripts: orbital localization and visualization.
- [Shared solvers](solvers/shared_solvers.py) and [degenerate subspaces](solvers/degenerate_subspace.py):
  differentiable solver interfaces and invariant observables.

Generated checkpoints, arrays and plots are written to the paths stated in each
script, usually relative to the working directory. Keep those outputs separate
from source control. Development archives under `artifacts/` are not required
to import or run these examples.
