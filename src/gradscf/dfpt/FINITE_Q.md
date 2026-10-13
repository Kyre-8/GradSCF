# Static finite-q electronic response

The first finite-q implementation covers collinear periodic Gaussian
HF and semilocal KS references, including the existing global-hybrid kernel
algebra. The k mesh must be complete, uniformly weighted and closed under the
requested q. Both insulating integer occupations and finite-temperature Fermi
occupations are supported. The electronic-potential shortcut holds geometry
fixed; `QSource` also accepts overlap and direct-density variations for atomic
perturbations. See [atomic response and reference phonons](ATOMIC_RESPONSE.md).

This implements the finite-basis spectral form of the static Sternheimer
response described in Sec. II.C.1 of [Baroni et al., Rev. Mod. Phys. 73, 515
(2001)](https://doi.org/10.1103/RevModPhys.73.515). The code retains every band
in the AO basis and does not assemble the full response matrix.

## Momentum and density convention

Lattice vectors are rows; q is supplied in fractional reciprocal coordinates.
`momentum_transfer(kpoints, q)` returns a static positive-q mapping

\[
k_{p(k)}=k+q+G_k.
\]

The existing shared table uses a minus sign, so this adapter calls it with -q.
It also retains the integer reciprocal shifts G_k. For stored periodic Bloch
factors u_k, the lifted partner is

\[
\bar u_{k+q}(r)=e^{iG_k\cdot r}u_{p(k)}(r).
\]

The AO density block D_q[s,k] has row index k+q and column index k. Its physical
density is exp(iq.r) times the periodic envelope

\[
\rho_{s,q}(r)=\sum_k w_k\sum_{ab}
\bar u_{k+q,a}(r)D_{s,q;ab}(k)u_{k,b}^*(r).
\]

An individual q block is generally non-Hermitian. A real perturbation includes
the adjoint -q sector. At self-inverse q, such as q=(1/2,0,0) on a compatible
mesh, those sectors occupy the same blocks: supply the assembled physical
potential once, without subsequently doubling it. The linear API also accepts
complex amplitudes representing a single Fourier component.

AO derivative channels already contain the physical (grad+ik)u derivatives.
Multiplying partner channels by exp(iG_k.r) and applying the product rule gives
the physical density-gradient envelope. Adding another iq.rho would double
count the Bloch contribution.

## Independent response and screening

For a potential block V_q, first transform it to the ground-state MO frames:

\[
V^{MO}_{s,q}(k)=C_{s,p(k)}^\dagger V_{s,q}(k)C_{s,k}.
\]

The matrix-function derivative is

\[
[\chi_{0,q}V]^{MO}_{nm}(k)=
\frac{f_{n,p(k)}-f_{m,k}}{\epsilon_{n,p(k)}-\epsilon_{m,k}}
V^{MO}_{nm}(k).
\]

Transform back with C_partner and C_k dagger. At zero temperature, equal
occupations have zero response, including degeneracies wholly within occupied
or empty spaces; a positive global occupied/empty gap is required. Finite T
reuses the shared stable Fermi divided differences, including the coincident
energy limit f'(epsilon). Chemical potentials and stored occupations must
describe the same temperature. Infinite chemical potentials are valid only for
their completely empty/full spin channels; NaN chemical potentials are invalid.
For nonzero q, the first-order modulation-averaged particle count vanishes, so
there is no uniform chemical-potential response in this sector.

The self-consistent equation is

\[
\delta D_q=\chi_{0,q}(V_q+K_q\delta D_q),\qquad
(I-\chi_{0,q}K_q)\delta D_q=\chi_{0,q}V_q.
\]

Hartree uses 4*pi/|G+q|^2 on the periodic density envelope. LDA/GGA use JVPs
of the existing pointwise XC potential, with complex-linear extension from
real and imaginary field components. GGA projections retain both gradient
terms. Exact exchange reuses the ground-state FFT contraction, with lifted
partner AOs on its left side. When enabled, its Ewald term is
Madelung*S_partner*D_q*S_k, matching the finite-cell ground-state convention.

`dfpt.Problem` packs real and imaginary AO entries and calls the shared GMRES
solver with checked primal and transpose residuals. A bounded direct solve is
available through the existing `LinearSolverConfig`. No DFPT-local solver is
introduced. Reference metric, eigen-equation, density and sampling consistency
are checked before returning a valid response.

## API and AD boundary

```python
problem = dfpt.DFPT(mf, q=[1/3, 0., 0.])
result = problem.response(potential=vq)
```

`vq` is `(nk,nao,nao)` or `(2,nk,nao,nao)`, in Eh per perturbation unit.
`pbc.response.project_q_potential` converts a local grid envelope to these AO
blocks. `result.density` contains spin-resolved q blocks, and
`result.values['rho_spin']` contains their complex periodic density envelopes.
Energy and particle-number **first-order modulation averages** are zero; this
is not a claim of zero local charge response or zero second-order energy.

The array factory `dfpt.finite_q_problem(inputs, result, transfer, ...)` can be
constructed inside a differentiated function. Its ground-state reference must
also be built there if the loss requires relaxed SCF dependence. First response
at equal-energy Fermi pairs is tested; outer derivatives through degenerate MO
frames are not thereby guaranteed. q matching itself is host-static metadata.

Atomic displacement sources are now generated by the bounded commensurate
supercell adapter, including overlap/Pulay and direct density changes. Its
phonon driver also supports OFDFT through the supercell KKT response; this
spectral KS factory itself is not an OFDFT adapter. Scalable primitive-cell
phonons, arbitrary external neural functionals and dynamical response remain
outside this path. An arbitrary primitive-cell energy callback does not
determine its finite-q extension. Taylor series are disabled because higher
orders couple zero, q, 2q and further harmonics; iterating this one-sector linear
residual would not produce those physical corrections.

## Numerical checks

The supercell oracle uses H at (0.4,0.5,0.6) and (1.8,0.8,0.9) Bohr in a
6-Bohr cubic cell, GTH-SZV/GTH-Pade, a 3x1x1 k mesh, q=(1/3,0,0), and a
15-cubed FFT mesh. The independent three-cell calculation uses a 45x15x15
mesh. The AO perturbation is Fourier transformed into the supercell, both
perturbed SCFs are converged, and the density derivative is unfolded back.
The positive-q potential envelope is 0.03 Eh; the dimensionless centered
difference step is 1e-4. SCF tolerances are 1e-12 Eh in energy, 1e-11 in density,
and 1e-10 in the gradient criterion. Tests use CPU JAX float64.

| Reference | kBT / Eh | Maximum absolute AO response error |
|---|---:|---:|
| HF | 0 | 6.48e-10 |
| HF | 0.15 | 9.05e-10 |
| Handwritten PBE | 0 | 2.61e-10 |
| Handwritten PBE | 0.15 | 2.17e-8 |

The CPU PBE oracle above uses the independent JAX formula in
`tests/dfpt/test_pbe.py`; production-backend validation is recorded separately
below. Additional tests cover q=0 kernel agreement, shifted meshes, q/-q
conjugacy, self-inverse q, a filled spin channel in unrestricted HF, dense/GMRES
agreement, and response-loss gradients through a nondegenerate relaxed HF
reference. See `examples/dfpt/04_finite_q_response.py` for a runnable example.

## Production XC and CUDA validation (2026-10-13)

An isolated snapshot of production commit `1989eeb` was tested on `jy_gpu_1`
(host c20), using its `jax_scf` environment. The additional tests exercise the
actual XC backend without replacing the pointwise kernel. Environment:
Python 3.11.15, JAX/JAXlib 0.10.0, jax-xc 0.0.9, PySCF 2.13.0, CPU float64
and CUDA float64. GPU checks used one NVIDIA RTX 5090 (physical GPU 7, selected
by UUID). CPU comparisons used Intel Xeon Platinum 8457C with two OpenMP/BLAS
threads. The CUDA eigensolver and FFT-gradient probe passed before testing.

| Production PBE check | Measured result |
|---|---:|
| Zero-T finite-q vs supercell, maximum AO response error | 2.64122e-10 |
| kBT=0.15 Eh finite-q vs supercell, maximum AO response error | 2.17385e-8 |
| Relaxed-reference response-loss AD gradient | 3.463457417934745 |
| Same gradient, centered finite difference | 3.4634574188174834 |
| Gradient absolute difference | 8.83e-10 |

The response-loss test perturbs hcore by a scalar times a fixed Hermitian AO
matrix, reconverges at kBT=0.12 Eh, and differentiates the squared norm of the
screened finite-q density response. It checks the runtime XC kernel's dependence
on the relaxed density, including the higher XC derivatives entering backward.
This is a nondegenerate HF/PBE validation, not a guarantee for every functional
or degenerate MO-frame derivative.

Execution records are deliberately separated:

- The two production-PBE supercell tests completed successfully on CUDA:
  **2 passed**, 204.90 seconds.
- A subsequent CUDA batch individually passed the remaining **18 finite-q**
  checks and **4 existing periodic DFPT** checks. Its 900-second watchdog then
  expired while starting an additional ground-state DFT/PySCF comparison.
  The batch exit status was 124; it was not a clean completion of that entire
  selection. All 20 finite-q checks and all 4 periodic DFPT checks had reported
  PASSED across these two GPU invocations.
- The remaining ground-state comparisons were rerun as a separate CPU batch:
  **14 passed**, 74.43 seconds, covering SVWN/PBE/PBE0, restricted/unrestricted
  Gamma and k-point calculations, and spin-polarized PBE. Their energy tolerance
  against PySCF is 1e-6 Eh.

To reproduce the completed scope, use `PYTHONPATH=src JAX_ENABLE_X64=1`
and separate the response and ground-state selections:

```sh
JAX_PLATFORMS=cuda python -m pytest -vv -s tests/dfpt/test_finite_q.py tests/dfpt/test_periodic.py
JAX_PLATFORMS=cpu python -m pytest -vv -s tests/pbc/test_dft.py
```

The recorded GPU invocations were split as described above; the second also
selected the ground-state comparison file, which explains its timeout and the
separate CPU rerun. Timings include initial JIT compilation and
are not warmed-throughput benchmarks. New production-backend cases skip when
`jax_xc` is unavailable; the original handwritten PBE tests remain independent.
