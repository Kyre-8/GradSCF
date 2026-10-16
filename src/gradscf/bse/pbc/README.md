# Periodic static BSE body

This module provides a bounded, complex, closed-shell singlet TDA matrix for
finite models on a full equal-weight Gamma-centered k mesh.
It is eager and forward only. The direct q→0 head/wing integration needed for
production material binding energies is not implemented.

For cell-normalized orbitals and unit Euclidean transition amplitudes,

\[
H_{kvc,k'v'c'}=\Delta^{QP}_{kvc}\delta_{kk'}\delta_{vv'}\delta_{cc'}
 +\frac{2J_{kvc,k'v'c'}-W_{kvc,k'v'c'}}{N_k}.
\]

`build_tda_matrix` reuses the GW Fourier product builder and shared screened
linear solve. Its static polarizability includes both occupied→virtual and
virtual→occupied channels, with weights
`2*(f_m-f_n)/(Nk*(epsilon_m-epsilon_n))`. Doubling only the first channel
requires exact time-reversal pairing, which a coarse Bloch discretization
need not preserve. No final-matrix symmetrization is applied. See the
[ABINIT Adler–Wiser conventions](https://docs.abinit.org/variables/gw/#awtr).
This static assembly does not change the existing GW contour-deformation
screening implementation.

The auxiliary inverse is evaluated in transition space; the final exciton
matrix is dense and bounded by `max_transitions`. The memory estimate includes
both transition directions. Optical band windows leave the full supplied
occupied/virtual screening space intact. `solve_tda` reuses the shared complex
Davidson forward solver and checks its actual residuals.

`ks_dipoles` accepts current restricted LDA/GGA references and includes the
shared GTH nonlocal velocity term. It forms the KS length vertex
`d=i*v_KS/Delta_KS`. Fixed-orbital QP corrections change gaps while retaining
that vertex. `independent_particles` provides the matching QP-only control
without constructing an identity matrix. Both controls use the singlet/k
factor `sqrt(2/Nk)` once and report per-cell oscillator strengths, not a
macroscopic absorption coefficient. HF/hybrid/meta-GGA velocity completion
and BSE automatic differentiation are outside this interface.

Slab and wire inputs use the existing reduced-dimensional Coulomb boundary,
with `exxdiv=None` reference data. This is its finite-part potential gauge,
not an integration over the singular q cell. Before screening, the sampled
FFT orbital Gram matrix and the supplied AO metric must both be identity
to absolute tolerance `1e-9`. This verifies vanishing raw OV/VO heads in the
actual product grid. Only a negative exact-zero Coulomb head is removed
from square-root correlation factors; every negative non-head value is
rejected. The omitted bare head is restored once as
`-v0/(Nk*volume)*I` in the direct BSE term. A positive finite head stays in
the original factors. No absolute value, clipping or tensor symmetrization
is used. The source mesh must contain an exact-zero q point; a merely
near-zero point would not have had its bare head removed. Nonperiodic k/q
components are rejected before reciprocal wrapping.

`polarizability` evaluates both complex resonant and antiresonant length
response terms. It divides by the first lattice-vector length for a wire,
the first two vectors' area for a layer, and the volume for 3D, producing
units `bohr**(3-dimension)`. Vacuum padding therefore does not introduce an
arbitrary length/area normalization. This statement concerns normalization,
not convergence of the underlying finite-cell electronic calculation.
All spectral widths are imposed resolution parameters, not lifetimes.

The focused tests compare Gamma and two-k kernels with independent full
auxiliary inverses and HF-TDA, then check phase covariance, k permutation,
window consistency, q/−q reciprocity and optical normalization. The small
[periodic H2 example](../../../../examples/bse/periodic_hydrogen.py) separates
the GradSCF calculation from [plotting](../../../../examples/bse/plot_periodic_hydrogen.py).
The reduced-dimensional tests additionally compare with a signed-Coulomb
full auxiliary inverse and verify bare heads, raw momentum gates, sampled
metrics and complex line/sheet polarizabilities. The optional
`gradscf_tools.periodic_bse_reference` independently assembles Fourier pairs,
full Lehmann response and signed-v Dyson/BSE contractions in transition
space. It shares the original SCF inputs and Coulomb boundary and does not
serve as an independent SCF or cutoff-kernel validation.
