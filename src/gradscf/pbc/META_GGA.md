# Periodic r2SCAN and kinetic-density response

The named `xc='r2scan'` expands into `mgga_x_r2scan + mgga_c_r2scan`.
The periodic Gamma and k-point facades use the existing SCF iterations and
implicit/explicit differentiation rules. No separate meta-GGA solver is used.
Other experimental meta-GGAs are not promoted by this change.

At each grid point the polarized XC variables are ordered as

    (rho_a, rho_b, grad_rho_a[3], grad_rho_b[3], tau_a, tau_b).

The convention is `tau_s = 1/2 sum_k w_k grad(phi_k) D_sk grad(phi_k)*`.
AO derivative channels already contain the Bloch derivative `(grad + i*k)u_k`.
The shared XC adapter receives complete spin-gradient vectors and orbital-derived
kinetic densities. AD differentiates the local energy density with respect to
all ten variables; the additional generalized Kohn-Sham matrix contribution is

    V_tau[k,p,q] = 1/2 integral v_tau grad(phi_kp)* . grad(phi_kq) dr.

For finite q, the transition kinetic density uses the k+q ket and k bra.
The field contraction and its dual potential projection both retain the 1/2
factor. Reciprocal folding multiplies all AO channels by the same phase;
adding another derivative of that phase would double count the Bloch derivative.
AD constructs the local 10-by-10 XC Hessian at every grid point once per
reference. Matrix applications reuse those coefficients, avoiding expansion of
the local XC derivative graph inside every Krylov/primal/transpose solver.
This requires O(100*ngrid) scalar storage; the AO response operator remains
matrix free. The Hessian retains its reference-density derivatives, so this
is not a frozen-kernel backward approximation. Existing
displacement and mixed geometry contractions include the tau AO variations,
so force constants use both basis motion and self-consistent density response.
At finite electronic temperature the geometry derivatives refer to free energy
at fixed temperature and spin electron counts.

The older unrestricted feature-bundle evaluator cannot retain the relative
spin-gradient angle in its synthetic-orbital bridge. Its MGGA path raises an
explicit error; use `eval_xc_energy_density_unrestricted_from_density_gradients`
with `tau_a` and `tau_b` instead. Missing kinetic density is never replaced with
zero in that MGGA interface. These named periodic calculations do not establish
support for arbitrary Laplacian-dependent functionals or current-density DFT.

Validation is in `tests/pbc/test_mgga.py` and the r2SCAN cases in
`tests/dfpt/test_displacement.py`, `test_finite_q.py`, and
`test_primitive_hessian.py`: LibXC local energies/potentials, energy-to-potential
consistency, q=0 JVP, independent supercell finite-q response/geometry, and
relaxed-reference response derivatives. Numerical material convergence and
anharmonic phonon self-energies remain separate from these implementation checks.

Functional reference: J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew and
J. Sun, *Accurate and Numerically Efficient r2SCAN Meta-Generalized Gradient
Approximation*, J. Phys. Chem. Lett. **11**, 8208-8215 (2020),
[doi:10.1021/acs.jpclett.0c02405](https://doi.org/10.1021/acs.jpclett.0c02405),
with its [published correction](https://doi.org/10.1021/acs.jpclett.0c03077).
The implementation uses the installed `jax_xc` functional, not a new fit.
