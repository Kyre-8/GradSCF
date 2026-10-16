# A05 — Bounded exponent optimization with a fixed auxiliary basis

The new [ordinary Python example](../../../examples/basis/optimize_exponents.py)
optimizes two shared positive exponent scales for H2 at a fixed 1.4 bohr bond.
Its native optimization, normalization checks and independent PySCF energy/FD
comparison passed on `jy_gpu_1`. See [figure](bounded_exponents.png),
[native data](gradscf.json), [independent data](pyscf.json), and
[runtime/commands](../A03_A07_RUNTIME.md).

Each H has one three-primitive contracted s shell and one diffuse s shell,
for four Cartesian main AOs. The raw exponents/coefficients are stored in the
script and JSON. Geometry, raw coefficients, shell angular momenta,
primitive/contraction counts, AO order/dimension, and auxiliary basis remain
fixed. Two dimensionless parameters define `alpha=alpha0*exp(theta)`, with
`-0.7 <= theta_i <= 0.7`, sharing the contracted-shell and diffuse-shell scales
between both atoms. Exponents are in bohr⁻² and energy is in Hartree.

The auxiliary basis is fixed `def2-universal-jkfit`, Cartesian, with metric
threshold `1e-12`. Its unchanged metric factor is cached with a signature check.
Each objective evaluation rebuilds the normalized overlap, core Hamiltonian and
native three-center DF factors with the current exponents. For this four-AO
demonstration, `eri_DF[p,q,r,s]=sum_Q B[Q,p,q]*B[Q,r,s]` forms a small O(nao⁴)
tensor for the existing shared orbital HF solver. It is **DF-HF**, not exact
four-center HF and not a scalable full-tensor storage recommendation.

L-BFGS-B converged in 15 accepted updates:

| Quantity | Initial | Optimized |
| --- | ---: | ---: |
| DF-HF total energy / Eh | -1.1199118212206445 | -1.1200445639724483 |
| Contracted-shell scale | 1 | 1.0881421271643328 |
| Diffuse-shell scale | 1 | 1.7833229802393928 |
| DF minus exact four-center energy / Eh (independent PySCF) | -9.74358e-6 | -9.36895e-6 |

The energy decreases by `0.132743 mEh`; both final parameters are interior.
The projected optimizer gradient is `6.71e-11 Eh` or smaller. Every accepted or
trial evaluation must have stationary HF orbitals, finite energy/gradient,
unit shell self-overlap, and an overlap minimum eigenvalue above `1e-5`.
The largest orbital stationarity norm observed was `2.18e-15`. This establishes
a converged bounded local demonstration, not a global basis optimum or
transferable basis quality.

At both initial and final parameters, native and independently reconverged
PySCF central differences use `h=0.005,0.001,0.0002` in theta. At the smallest
step, the largest native and independent gradient errors are `2.58e-9 Eh`.
The independent DF energies agree within `4.45e-16 Eh`. Unit diagonal overlap
errors are at most `4.45e-16`; derivatives of that diagonal are at most
`2.23e-16`. These checks explicitly include the normalization chain. PySCF
uses the identical fixed auxiliary basis; exact four-center energies are
reported separately and are not substituted for the DF derivative oracle.

The native run took 34.21 s and the independent comparison 1.49 s, including
their respective setup. The existing native API supports the first exponent
derivatives used here. Exact four-center exponent AD, higher/mixed exponent
AD, auxiliary optimization and geometry response are outside this example.
No missing derivative is replaced with zero.

The complementary [existing contraction optimization](../../../examples/basis/optimize_bsse_contractions.py)
keeps exponents fixed and optimizes shared coefficients in a fixed scale gauge,
with differentiated physical AO normalization. Its recorded water-dimer BSSE
reduction is a separate historical result; lowering BSSE alone does not prove
better interaction energies. The [MP basis-response example](../../../examples/mp/implicit_gradient.py)
likewise differentiates a contraction parameter, not an exponent.
