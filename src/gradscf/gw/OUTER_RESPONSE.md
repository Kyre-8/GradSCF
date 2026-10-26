# Molecular GW outer implicit response

`evgw_cd_restricted`, `evgw_cd_unrestricted`, and `qsgw_cd_restricted`
accept `differentiation=SCFDifferentiationConfig(mode="implicit")`.
The default remains the existing eager forward iteration. The opt-in path
compiles the numerical iteration, stops its trajectory, and attaches
`solvers.nonlinear.attach_root`; its JVP and transpose VJP use the shared
checked GMRES solver. `diff_mode` alone does not enable outer response.

For evGW, the differentiated variable is the full alpha/beta QP spectrum
(or the restricted spectrum), while the original mean-field spectrum,
coefficients, densities, and integrals are external parameters. The equations
are

```
R_p(E; theta) = E_p - e_mf,p(theta) - delta_v_p(theta)
               - Re Sigma_p(E_p; G(E, theta), W(E, theta)) = 0.
```

Unrequested orbitals instead obey `R_p = E_p - e_mf,p`; this preserves their
mean-field input response without singular zero rows. `update_w=False` gives
evGW0 for both references. W0 is fixed across the outer iteration but retains
its parameter response, including dependence on the initial spectrum. The
restricted path caches W0 once; the unrestricted path currently reevaluates
its arrays with fixed spectra. Both unrestricted spin channels currently
require at least one occupied and one virtual orbital; empty/full spin
channels are rejected before constructing a contour Fermi energy.

For restricted qsGW, the differentiated variable is the upper triangle of
a real symmetric effective Fock matrix in the fixed initial MO frame C0.
The residual is `F - F_qsGW[F; theta]`. Each map diagonalizes F and rebuilds
the occupied density, Hartree/exchange potential, G, W, and the mode-A static
correlation potential. The root therefore includes the orbital rotations
and induced density response. This avoids a redundant eigenvector-gauge
root. C0 must satisfy `C0.T @ S @ C0 = I`; it defines the metric and retained
subspace, and the code retains its dependence in transformed integrals and
returned AO coefficients. C0 need not be Euclidean-orthogonal. Pure rotations
of a complete initial frame leave physical energies and AO densities invariant.

The current qsGW response still differentiates the diagonalization inside
the static map and the returned orbital coefficients. It requires **all**
orbital eigenvalues to be isolated by more than
`1e-8 * (1 + max(abs(energy)))` Ha. Degenerate occupied or virtual manifolds
are explicitly rejected even when a density observable could admit a
spectral-block derivative. No arbitrary eigenvectors or regularized energy
denominators are differentiated at degeneracy. Initial energies only seed
the converged root. Real molecular inputs are required.

Forward nonconvergence raises, including compiled execution through a JAX
callback. Only unregularized converged-state response is accepted. Linear
solves check their actual primal/transpose residuals and return NaN response
when unresolved (including tested singular outer systems). As with other
local implicit methods, validity assumes a locally isolated solution and
unchanged occupation/residue branches. The convergence test certifies the
chosen finite contour grid, not basis/grid completeness. Periodic evGW outer
response and unrestricted qsGW are not added by these molecular paths.

Occupation labels stay fixed. Every updated spectrum must have a finite
positive gap between the lowest virtual and highest occupied level; ordering
and degeneracies inside a sector do not change that test. The evGW residual
also requires a finite positive local fixed-G/W slope on computed levels.
This slope is a validity check, not the full evGW response: the attached
outer Jacobian includes the changes of G and W. The numerical forward
closures and captured parameters are stopped before attaching that Jacobian,
so reverse mode does not traverse their dynamic iteration loops.

`tests/gw/test_gw_outer_response.py` compares CPU float64 compiled JVP/VJP
against central differences of independently reconverged eager solutions.
It covers initial-spectrum and individual DF-factor perturbations, separate
spin perturbations, W0 dependence, selected orbital windows, qsGW energy and
AO-density response, non-Euclidean initial frames and frame rotations,
nonconvergence, degeneracy rejection, and a singular outer response oracle.

The tests also compose unrestricted G0W0, evGW and evGW0 with spin-conserving
BSE and differentiate an excitation-energy/oscillator-strength objective.
The independent check reconverges the eager GW calculation at both finite-
difference points; it does not reuse the converged central screening matrix.

[The H2 example](../../../examples/gw/implicit_response.py) compares implicit
and reconverged finite-difference HOMO derivatives for all three restricted
outer methods. It uses native GradSCF HF once, then varies the AO factor scale
at fixed starting HF arrays; the derivative is not a nuclear or HF response.

End-to-end unrestricted G0W0/evGW/evGW0-to-BSE tests also differentiate a
lowest excitation plus oscillator strength through the QP spectrum, its actual
screening spectrum, both spin MO frames, and the electron-hole kernel. These
use genuine unequal occupations `(2, 1)` and a single AO-factor perturbation.
