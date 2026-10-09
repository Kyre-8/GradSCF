# WGC99 kinetic energy and differentiable response

`KineticFunctional('wgc')` implements TF + full vW + the second-order Taylor
expansion of the density-dependent Wang–Govind–Carter kernel. Parameters are
fixed to alpha=(5+sqrt(5))/6, beta=(5-sqrt(5))/6, gamma=2.7, as used for the
ATLAS simple-metal comparison. This is distinct from the density-independent
Wang–Teter kernel. It currently requires a periodic FFT representation.

## Functional and conventions

Let theta=n-rho_s and eta=|G|/(2 k_F(rho_s)). The dimensionless reciprocal
kernel w satisfies

```
eta^2 w'' + (gamma-9) eta w' + 36 alpha beta w
    = 20 [1/L(eta) - 1 - 3 eta^2],
L(eta) = 1/2 + (1-eta^2)/(4 eta) log |(1+eta)/(1-eta)|.
```

Writing d1=dw/dlog(eta), d2=d²w/dlog(eta)², the four Fourier kernels are

```
K0  = w
K1  = -d1/(6 rho_s)
K2  = [d2+(6-gamma)d1]/(36 rho_s^2)
K12 = [d2+gamma d1]/(36 rho_s^2).
```

The real-space density-dependent kernel is approximated by
`K0 + K1(theta+theta') + K2(theta²+theta'²)/2 + K12 theta theta'`.
Its nonlocal energy is `C_TF integral n(r)^beta K(r,r') n(r')^alpha dr dr'`.
Each term is evaluated by periodic FFT convolution. `K0...K12` do not include
`C_TF`; the energy includes it once. G=0 kernels vanish.

The default rho_s is the current cell-average density. Alternatively pass
`kinetic_params={'reference_density': rho_s}` to hold a positive scalar
reference density fixed. This distinction matters for unconstrained density,
cell, and parameter derivatives. At fixed N and volume the constrained
minimizations use the same reference. Shape is checked during tracing;
finite-positive values are checked eagerly. Traced callers must preserve the
positive-domain contract themselves.

## Numerical kernel and AD boundary

The dimensionless ODE is integrated once on the host with SciPy DOP853 in
log(eta), rtol=2e-12, atol=2e-14. A 10,001-knot quintic Hermite table covers
eta=1e-6 to 200, with a knot at eta=1. Small/large-eta asymptotic expansions
cover its exterior; the high-eta initial condition includes eta^-2 and eta^-4
terms. Only these universal interpolation coefficients are cached. No physical
density, volume, lattice, or reciprocal vectors are frozen in that cache.

JAX evaluates the interpolant and its first derivative. The second derivative
is reconstructed from the defining ODE to preserve the uniform Lindhard
response. JAX differentiates the assembled energy for its potential and HVP;
`run_ofdft` retains the shared constrained implicit backward used by other
KEDFs. The fixed alpha/beta/gamma constants are not trainable parameters.

Fractional powers use a declared density floor of 1e-18 electron/Bohr³. The
kernel value at eta=1 is finite, but the exact Lindhard anomaly does not admit
all kernel derivatives. The existing response guard deliberately returns
nonfinite derivatives at that exact point instead of inventing a slope.
Implicit response also requires a converged, locally nonsingular constrained
stationary solution. Neither convergence nor a small gradient proves a global
minimum. WGC can be unreliable outside its intended density regime.

## Independent validation

- Analytic uniform energy and Lindhard second variation.
- Finite differences of energy, reference-density derivative and implicit
  optimized-density response.
- Independent NumPy implementation in `examples/ofdft/wgc_dftpy.py`, integrating
  the ODE directly in eta and evaluating an analytic potential. Its integration
  and differentiation do not call GradSCF or JAX. DFTpy dev has no native WGC;
  the adapter supplies its nonlocal term explicitly.
- A complete first density/potential frame from the pinned libKEDF fixture,
  with absolute tolerances 3e-6 Ha (energy) and 1e-5 Ha (potential). This legacy
  check has a coarser upstream ODE tolerance and an even skew-cell grid, so it
  is not a machine-precision oracle. See `tests/ofdft/data/README.md`.
- Independent self-consistent Al/Mg calculations and grid refinement are
  recorded in `examples/ofdft/WGC_REPRODUCTION.md`.

## References and attribution

- Wang, Govind and Carter, *Orbital-free kinetic-energy density functionals
  with a density-dependent kernel*, [Phys. Rev. B 60, 16350 (1999)](https://doi.org/10.1103/PhysRevB.60.16350).
- Mi et al., *ATLAS*, [arXiv:1507.07373](https://arxiv.org/abs/1507.07373).
  The paper supplies the parameter set; its exact Taylor order and complete
  figure input decks are not established here.
- Kernel conventions were checked against [libKEDF](https://github.com/EACcodes/libKEDF/tree/3dff53318ce7be52be5f45242ea8daf08a032866),
  including `WangGovindCarter`, `Taylor`, and its kernel ODE. Relevant copyright
  and BSD-3-Clause terms are retained in `LICENSE.libKEDF`.
- [PROFESS](https://github.com/EACcodes/PROFESS/tree/1e08c7a104ea2ffb69fb26fb5435f341ae86eb67)
  was inspected for the standard WGC parameter/expansion convention; no
  PROFESS runtime or Fortran source is included.
