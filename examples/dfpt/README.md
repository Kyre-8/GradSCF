# DFPT examples

Run from the repository root with `PYTHONPATH=src`. Scripts enable JAX float64
and expose their physical inputs directly. See the
[DFPT API](../../src/gradscf/dfpt/README.md) and
[finite-q conventions](../../src/gradscf/dfpt/FINITE_Q.md) for derivative scope.

| Script | Calculation | Prior output needed |
| --- | --- | --- |
| [01_molecular_response.py](01_molecular_response.py) | Molecular static response and Taylor coefficients. | None |
| [02_periodic_response.py](02_periodic_response.py) | Periodic static response. | None |
| [03_ofdft_response.py](03_ofdft_response.py) | Grid OFDFT screened response and Taylor energy coefficients. | None |
| [04_finite_q_response.py](04_finite_q_response.py) | Primitive-cell finite-q electronic response. | None |
| [03_external_finite_q.py](03_external_finite_q.py) | External XC energy, AD-derived finite-q kernel and screened response. | None |
| [05_atomic_phonons.py](05_atomic_phonons.py) | Atomic perturbations and phonon response. | None |
| [06_silicon_phonons.py](06_silicon_phonons.py) | Si q mesh, Fourier interpolation, dispersion and DOS. | None |
| [07_silicon_pyscf_reference.py](07_silicon_pyscf_reference.py) | Independent PySCF energy-difference curvature checks. | `silicon_phonons/qmesh.npz` from 06 |
| [08_silicon_convergence.py](08_silicon_convergence.py) | Separate FFT and basis sensitivity checks. | `silicon_phonons/qmesh.npz` from 06 |
| [09_plot_silicon_checks.py](09_plot_silicon_checks.py) | Plot sensitivity and independent curvature comparisons. | Results from 06, 07 and 08 |

The external-functional example supplies a real XC energy per primitive cell
for the physical q/-q perturbation. GradSCF differentiates that energy; the
example does not hand-code a response kernel. It uses a synthetic nonlinear
functional, not a fitted material XC model.

## Silicon workflow

Run 06, then 07 and 08, then 09, all from the same working directory. The scripts
write into `silicon_phonons/` and `silicon_convergence/`. Example 07 requires
PySCF; plotting requires Matplotlib. These material calculations are heavier
than the small-cell response demonstrations; 08 includes a DZVP case with a
measured process RSS near 31 GiB in the recorded run.

The inputs are fixed-geometry Si at a=5.43 Angstrom, PBE/GTH-PBE, initially
GTH-SZV, k/q=2x2x2 and FFT=31 cubed. This is a pilot and sensitivity study,
not a relaxed or fully basis/k/q-converged phonon prediction. Negative modes
remain visible, and raw ASR diagnostics are retained before any optional
Gamma projection. See the [IFC evidence](../../src/gradscf/pbc/IFC.md) and
[acceptance ledger](../../src/gradscf/pbc/ROADMAP.md).
