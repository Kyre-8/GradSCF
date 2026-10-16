# Projected phonons and finite-temperature spectral diagnostics

These interfaces reuse the periodic SCF, DFPT and shared solver infrastructure.
They provide projected harmonic modes, local frozen-mode energy expansions and
relative one-phonon spectra. They do not implement a full anharmonic phonon
self-energy or calibrated EELS scattering calculation.

## Projected force constants

`Phonons(mf, masses=...).project(directions, q=...)` accepts real Cartesian
directions of shape `(nvector,natom,3)`. Columns of B are the flattened
directions. For each column b, the existing atomic-response driver solves the
electronic response and returns Phi*b; it does not compute an entire Hessian
before projecting it. Gamma and nonzero q share the same action factory with
`Phonons.kernel`. Every call builds and validates a fresh reference snapshot.

With physical masses M in electron-mass units, the reduced problem is

    H = B† Phi B,  S = B† M B,
    H v = omega² S v.

The directions need not be orthonormal, but must be independent with a usable
positive mass metric. Response solves use unit directions and restore each
original amplitude afterwards. Hermiticity is checked in the mass-whitened
coordinates before symmetrization, so scaling B cannot hide a failed response.
The reported Hermiticity error is the raw projected force-constant norm.
If S=L L†, diagonalize L^-1 H L^-† with the existing dense
solver. The returned full mass-weighted mode is e=sqrt(M) B L^-† v; e†e=1.
Input masses are amu. Signed energies are hbar*omega in Hartree; negative
energies denote imaginary modes, consistent with the full phonon driver.

The result contains the projected force constants and mass metric, frequencies,
full Cartesian modes, linear-response residual, Hermiticity error and the full
dynamical residual norm

    || M^-1/2 Phi B L^-† v - e*omega² ||.

The last quantity detects leakage out of the requested subspace. A converged
electronic response and reduced eigensolve do not imply an invariant physical
mode subspace. Consumers must assess leakage on an appropriate scale. No ASR
projection is applied in `project`, and its eager mode analysis makes no new
eigenvector differentiation promise at degeneracy.

## Local anharmonic coefficients

`fit_frozen_mode(s, F, order=4)` fits a single displacement coordinate s in Bohr
to F(s)=sum(a_n*(s-s0)^n). It returns coefficients a_n, derivatives n!*a_n, fit
residuals, rank, condition number, origin and scale. Electronic states must be
independently converged at every input displacement; at finite electronic
temperature F is the appropriate electronic free energy.

The design matrix uses scaled displacements and energies relative to a sample
offset. Fits retain odd powers unless `even=True` is explicitly requested.
Orders 2 through 6 are supported. Compare multiple windows/orders: a small RMS
alone does not establish convergence of a fourth derivative. Eager validation
rejects invalid/rank-deficient/ill-conditioned data and nonfinite outputs.
JIT callers must also check the returned diagnostics as described in the API.

`directional_energy_taylor(energy_fn, R, d, order=4, args=(params,))` applies
JAX jet to E(R+s*d,params), with normalized coefficients and raw derivatives
reported separately. It keeps model parameters differentiable and propagates
unsupported-primitive errors. It does not silently switch to the fit routine.
The caller is responsible for electronic relaxation in energy_fn. General
high-order SCF geometry jets are not established by the analytic-function tests.

A single Gamma-coordinate expansion does not contain mixed-mode couplings at
other q points. Bulk temperature shifts and lifetimes require the appropriate
momentum-conserving vertices, occupations and Brillouin-zone self-energy
integrations. They remain separate development work.

## Relative one-phonon loss and gain

`one_phonon_spectrum(energies, strengths, energy_grid, temperature=T, fwhm=w)`
uses Hartree for energy/w and kelvin for T. For each positive mode energy E_nu,

    n_nu = 1 / (exp(E_nu / kBT) - 1),
    I_loss(E) = sum_nu S_nu (n_nu+1) G_w(E-E_nu),
    I_gain(E) = sum_nu S_nu n_nu G_w(E+E_nu).

G_w is a Gaussian of unit full-axis integral. S_nu is the integrated zero-T
loss strength, including any zero-point normalization. The Bose occupation is
shared with the existing GW phonon infrastructure. At T=0 the gain vanishes.
Integrated partner-line areas obey detailed balance; overlapping broadened
spectra do not in general obey that relation point by point. Zero or imaginary
modes are explicitly rejected rather than clipped. No lifetime contribution
is implied by the supplied Gaussian width.

`mode_scattering_weights` provides an optional coherent kinematic numerator

    W_nu = |sum_a f_a exp(-i Q.tau_a) (Q.e_a,nu) / sqrt(M_a)|².

Q is the full Cartesian scattering momentum G+q in inverse Bohr, tau is in
Bohr, and f_a is explicitly supplied by the caller. The modes use the native
lattice-translation displacement convention e_a(q)*exp(+i q.R). Atom phases
are included once. Divide W_nu by 2*E_nu for zero-point-normalized strengths.
No default charge/form-factor or Debye-Waller model is invented. Probe dynamics,
multiple scattering, momentum acceptance and absolute cross-section calibration
remain outside this utility. The total weight of an exactly degenerate cluster
is invariant under a unitary change of its mode basis.

See the readable [graphite example](../../../examples/thermally_excited_graphite/README.md).
