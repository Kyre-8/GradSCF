# GradSCF examples

Run from the repository root with `PYTHONPATH=src`, or use an editable GradSCF
installation. Examples are ordinary Python scripts; numerical settings are
written directly in the code. Individual files describe optional dependencies,
units, convergence criteria and measured outputs where available.

## Periodic foundations

| Example | Purpose |
| --- | --- |
| [Periodic H2](periodic_h2.py) | Gaussian/GTH periodic ground-state calculation. |
| [Basis library](periodic_basis.py) | Offline periodic basis and pseudopotential selection. |
| [Finite temperature](periodic_temperature.py) | Fermi occupations, free energy and static response. |
| [Forces](periodic_forces.py) | HF and OFDFT geometry derivatives. |
| [Spatial symmetry](periodic_symmetry.py) | Space-group and Cartesian AO/Bloch transformations. |
| [Irreducible integration](periodic_irreducible.py) | Full/irreducible SCF comparison and target contraction counts. |
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
- [Ethylene Boys orbitals](ethylene_boys.py), [BODIPY Boys orbitals](bodipy_boys.py)
  and their `animate_*` scripts: orbital localization and visualization.
- [Shared solvers](shared_solvers.py) and [degenerate subspaces](degenerate_subspace.py):
  differentiable solver interfaces and invariant observables.

Generated checkpoints, arrays and plots are written to the paths stated in each
script, usually relative to the working directory. Keep those outputs separate
from source control. Development archives under `artifacts/` are not required
to import or run these examples.
