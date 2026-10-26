# Orbital localization

Run from the repository root with `PYTHONPATH=src` and a built native integral
library. GradSCF computes the reference HF orbitals; AD then acts on rotations
within the fixed occupied space. These scripts do not differentiate through HF
or calculate nuclear-coordinate response.

| Script | Purpose / prerequisite |
| --- | --- |
| [ethylene_boys.py](ethylene_boys.py) | Small ethylene calculation with gradient/HVP checks. |
| [archive_ethylene_boys.py](archive_ethylene_boys.py) | Reuse ethylene and its independent PySCF multistart comparison; archive A06 derivative and invariance checks. |
| [plot_ethylene_boys.py](plot_ethylene_boys.py) | Plot the saved A06 trajectory and checks. |
| [animate_ethylene_boys.py](animate_ethylene_boys.py) | Runs the ethylene calculation and renders each accepted optimization step. |
| [bodipy_boys.py](bodipy_boys.py) | Parent BODIPY calculation; saves its localization trajectory. |
| [animate_bodipy_boys.py](animate_bodipy_boys.py) | Renders the saved BODIPY trajectory in orbital pages. |
| [combine_bodipy_boys.py](combine_bodipy_boys.py) | Combines those page movies into a synchronized overview. |
| [animate_bodipy_boys_rotation.py](animate_bodipy_boys_rotation.py) | Uses the trajectory and overview to animate the orbital rotation matrix. |

Output locations remain `artifacts/bodipy_boys/` and
`artifacts/ethylene_boys_animation/` under the repository root. The animation
dependencies and BODIPY run order are documented in the
[BODIPY guide](bodipy_boys.md). Independent PySCF comparisons live in
[`tests/comparisons`](../../tests/comparisons/).

[A06 measured results](../../reproducibility/ad_showcase/A06_boys_localization/README.md)
record the current `jy_gpu_1` CPU float64 run, source hashes and fixed-subspace
boundary. It does not differentiate the reference SCF or nuclear geometry.
