# Static finite-q electronic response

The first finite-q implementation covers collinear periodic Gaussian
HF and semilocal KS references, including the existing global-hybrid kernel
algebra. The k mesh must be complete, uniformly weighted and closed under the
requested q. Both insulating integer occupations and finite-temperature Fermi
occupations are supported. All geometries and overlap matrices are fixed.

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

The current source is an electronic AO potential only. Atomic displacement
sources need additional overlap/Pulay and ionic-potential derivatives. Phonons,
finite-q OFDFT, arbitrary external neural functionals and dynamical response
are not provided here. An arbitrary primitive-cell energy callback does not
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

The PBE oracle uses the independent JAX formula in `tests/dfpt/test_pbe.py`;
it does not establish coverage of an unavailable jax-xc/libxc higher-derivative
backend. Additional tests cover q=0 kernel agreement, shifted meshes, q/-q
conjugacy, self-inverse q, a filled spin channel in unrestricted HF, dense/GMRES
agreement, and response-loss gradients through a nondegenerate relaxed HF
reference. See `examples/dfpt/04_finite_q_response.py` for a runnable example.
