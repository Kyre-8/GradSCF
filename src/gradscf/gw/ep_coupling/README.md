# Fixed-phonon electron self-energy

`gradscf.gw.ep_coupling` supplies harmonic propagators and Fan/Debye–Waller
terms for the existing matrix Matsubara scGW driver. It introduces no new
SCF, root or linear solver. The electronic Green function, electronic W,
density, chemical potential and total self-energy moment are self-consistent.
Phonon propagators, vertices and mean nuclear positions are fixed inputs.

```python
from gradscf.gw.ep_coupling import PhononModel
from gradscf.gw import scgw_matsubara_restricted
from gradscf.scf.autodiff import SCFDifferentiationConfig

phonons = PhononModel(energies=omega, couplings=g, quadratic=quadratic,
                     reference="external model or identified reference calculation")
result = scgw_matsubara_restricted(
    **electronic_inputs, phonons=phonons,
    differentiation=SCFDifferentiationConfig(mode="implicit"),
)
```

`electronic_inputs` has the same AO Hamiltonian/DF factors and initial MO
coefficients as ordinary scGW. In contrast, `g` and `quadratic` are already in
the **fixed initial orthonormal MO frame** defined by those coefficients.
They are not AO derivatives and do not follow instantaneous Fock orbitals.
If this frame changes, rotate the vertices along with the electronic data.

## Model and units

All energies use Hartree; beta is in Ha^-1. Define X_l=b_l+b_l^dagger and

```text
H_ep = sum_l g_l X_l + (1/2) sum_lm Lambda_lm X_l X_m
```

- `energies`: `(nmode,)`, finite and strictly positive.
- `couplings`: `(nmode,nmo,nmo)`, real symmetric for molecular scGW.
- `quadratic`: optional compact `(nmode,nmo,nmo)` diagonal-mode operators or
  full `(nmode,nmode,nmo,nmo)`. Both are Hermitian in orbital indices; full
  arrays are also symmetric in mode indices. Only diagonal mode entries
  enter fixed independent D0, so compact storage avoids unused mode-pair blocks.
- `reference`: static provenance string; no automatic screening or
  double-counting subtraction is inferred from it.

`quadratic=None` explicitly selects a linear-coupling model, not a complete
first-principles zero-point correction. Call `validate_model` before using
low-level kernels; the scGW driver calls it automatically, including under
JIT. Zero/negative modes are rejected, never replaced by an absolute value
or clipped. External mode construction must remove molecular rigid motions
or handle periodic acoustic limits appropriately.

## Equations and quadrature

The sign convention is D=-<T X X>. For independent harmonic modes,

```text
D_l(i nu) = -2 Omega_l / (nu^2 + Omega_l^2)
Sigma_Fan(tau) = -sum_l g_l G(tau) g_l D_l(tau)
Sigma_DW = (1/2) sum_l Lambda_ll (2 nB_l + 1)
M_Fan = sum_l (2 nB_l + 1) g_l g_l
```

`M_Fan` is the coefficient of 1/(i omega) in the dynamic self-energy. The
driver adds it to the GW moment used for reconstructing G(tau); DW instead
enters the static Fock matrix. Per-spin G is used, with no extra factor of
two in the electron Fan diagram.

To avoid truncating the soft-phonon reference contribution, `fan_self_energy`
subtracts the Green function of the current static Fock matrix in tau and
adds its analytic finite-temperature Fan poles on the fermion frequency grid.
The dressed correction still uses midpoint quadrature and requires an nw
convergence study. The reference subtraction does not replace the interacting
G in the diagram or freeze its response.

## Self-consistency and derivatives

Both eager iteration and the implicit residual call the same `_scgw_step`:
pure electronic GW + Fan supplies the dynamic self-energy and moment, and
Hartree–Fock + DW supplies the static matrix. The charge equation uses the
same interacting G. With implicit differentiation, the shared root/GMRES
and charge Schur complement include g, Omega and Lambda as live parameters.
Their iteration-independent status does not imply `stop_gradient`.

Existing scGW restrictions remain: real restricted molecular inputs,
resolvable static-reference spectra, a converged forward state, and a
resolved charge response. beta/nw/occupations and iteration controls are
static. There is no automatic geometry derivative or normal-mode generator.

## Outputs and energy boundary

`self_energy_iw`, `sigma_moment` and `fock_mo` contain the combined contributions.
`phonon_self_energy_iw`, `phonon_sigma_moment`, and `debye_waller` expose the
additional terms recomputed from the returned state. Their consistency with
the iterated quantities is bounded by the reported unmixed residual.

With a phonon model supplied, `total_energy` is **None**, even for zero model
couplings. `electronic_energy` contains only the electronic Hamiltonian
contribution evaluated on the returned G/density, excluding nuclear repulsion,
electron–phonon interaction and phonon energy. `correlation_energy` retains
its original electron–electron GW meaning. No closed coupled-system energy
or conserving phonon feedback is implied by a prescribed D0 bath.

The user must specify a reference consistent with their screened/bare vertex
choice. No force/tadpole equation, geometry relaxation, phonon Dyson update,
or subtraction of already included phonon polarization is performed here.

## Periodic Fan kernels

`PeriodicPhononModel(energies, couplings, k_plus_q, q_weights)` contains
energies `(nq,nmode)`, complex vertices `(nq,nk,nmode,norb,norb)`, integer
mapping `(nq,nk)` and normalized q weights `(nq,)`. As before, the stored
vertex maps **internal k+q to external k**. This is the Hermitian-conjugate
orientation of a conventional external-to-internal scattering matrix.
The model uses zero-point-normalized vertices and energies in Hartree.

`periodic_fan_self_energy(green_tau, fock, mu, model, grid)` now computes
frequency-domain Fan and its high-frequency moment. It uses a complex
Hermitian static reference at each k, adds the exact analytic reference poles,
and integrates only the dressed Green-function difference. The same pole
weights are used by the molecular and real-axis functions. Fock response
currently requires isolated eigenvalues at each k. G/vertex orbital frames,
grid resolution and the physical momentum mapping remain caller inputs.

`periodic_fan_self_energy_tau` remains the direct tau contraction. Both paths
accumulate one q slab at a time, with rematerialization for reverse mode,
avoiding a full `(ntau,nq,nk,norb,norb)` gathered Green tensor. They do not
construct Wannier vertices, apply acoustic cutoffs, supply periodic DW, or
provide a periodic scGW driver/phonon Dyson feedback. An executable model is
[periodic_phonon_fan.py](../../../../examples/gw/periodic_phonon_fan.py).

## Real-axis poles, linewidth and spectral matrix

- `fan_retarded(energies, model, frequencies, mu=..., beta=..., eta=...)`
  returns the full complex matrix from a specified canonical pole spectrum.
- `electron_linewidth` returns the molecular on-shell **full** width Gamma.
  `fan_linewidth` also permits rectangular `(mode,external,internal)` vertices
  for a single k/q block; sum its output with the supplied q weights.
- `spectral_function(fock, sigma, frequencies, ...)` evaluates the full-matrix
  retarded Dyson resolvent and `A=-(G-G†)/(2 pi i)`. Elementwise `-Im(G)/pi`
  is not the matrix spectral function for general complex orbital frames.

Electronic energies and Fock matrices are absolute; all query frequencies
are relative to mu. eta is positive and in Ha. Lorentzian linewidth uses the
same retarded poles. Gaussian linewidth uses
`exp[-(delta/eta)^2]/(sqrt(pi)*eta)` only as an on-shell delta approximation;
it is not used to patch the imaginary part of a complex retarded function.
No sign clipping or absolute-value causality repair is performed. The caller
supplies the desired causal sigma, including any electron-electron term and
consistent static potential.

These fixed-reference poles are **not** analytic continuation of matrix
scGW output. `SCGWResult.static_mo_energy` must not be interpreted as its
physical QP poles. See [phonon_spectrum.py](../../../../examples/gw/phonon_spectrum.py)
for an external two-level model and finite-window spectral checks.

## ElectronPhonon.jl reference

The inspected reference is [ElectronPhonon.jl](https://github.com/jaemolihm/ElectronPhonon.jl)
commit `a0ecee01b7413af5566134d94cad5b15431d6565`. Its separation of k/q traversal
from property kernels informed the q-slab accumulation here. Its
`src/selfenergy_electron.jl` computes positive on-shell halfwidths with
Gaussian broadening; this is not a complete self-consistent GW driver.
GradSCF uses full widths `Gamma=-2 Im Sigma^R`.

The comparison also accounts for upstream Rydberg units and its displacement
vertex normalization: its `g2` already contains the zero-point factor
`1/(2 Omega)` after the mode transformation. GradSCF's supplied `g` must
already include that normalization; it must not be applied a second time.

The original upstream numerical function, Fermi/Bose and Gaussian helpers
were executed using Julia 1.11.5 in a small dependency-free harness. Timing
and state containers are shimmed, and all modes are above the upstream
acoustic cutoff. No full-package, Wannier-interpolation or real-material
agreement is claimed. All 12 per-q and 6 accumulated widths match GradSCF to
`3.47e-18 Ha` absolute; the offline tolerance is `1e-13 Ha`.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/comparisons/compare_electronphonon_jl.py \
  --upstream /path/to/pinned/ElectronPhonon.jl --output /tmp/ep-julia-check \
  --julia /path/to/julia
```

The runner checks the pinned checkout and generates upstream extracts only
outside this repository. Tests use selected JSON reference arrays in
`tests/gw/data/electronphonon_jl`; provenance includes source, extract,
harness and runner hashes. Ordinary pytest requires neither Julia nor
network access. No upstream Julia production source is vendored.

## Verification and example

`tests/gw/test_ep_coupling.py` checks independent matrix pole sums, the dressed
Green correction and its grid convergence, thermal factors, moments, coordinate
covariance, input guards and complex periodic contractions. `test_scgw_ep.py`
checks zero-coupling recovery, analytic DW-only density, joint matrix/number
residuals, frame covariance and JIT JVP/VJP against reconverged finite differences.

See [fixed_phonon_scgw.py](../../../../examples/gw/fixed_phonon_scgw.py) for
native H2 HF followed by scGW with illustrative external mode parameters.
These parameters are not ab initio H2 electron–vibration couplings.

## References

- F. Giustino, *Rev. Mod. Phys.* **89**, 015003 (2017),
  [10.1103/RevModPhys.89.015003](https://doi.org/10.1103/RevModPhys.89.015003):
  Fan–Migdal, Debye–Waller and finite-temperature conventions.
- G. Stefanucci, R. van Leeuwen and E. Perfetto, *Phys. Rev. X* **13**, 031026
  (2023), [10.1103/PhysRevX.13.031026](https://doi.org/10.1103/PhysRevX.13.031026):
  Hedin–Baym equations and consistent coupled electron/phonon approximations.
- [Phonon Self-Energy Corrections: To Screen, or Not to Screen](https://arxiv.org/abs/2212.11806),
  *Phys. Rev. X* **13**, 041009 (2023): screened vertices and reference
  double-counting corrections needed before a phonon-feedback extension.

This fixed-D0 implementation is not a full solution of those coupled equations.


## Validation record (2026-10-04)

Apple M4 Pro / macOS arm64, Python 3.12.2, JAX 0.8.1, CPU float64.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/gw/test_ep_coupling.py tests/gw/test_scgw_ep.py \
  tests/gw/test_gw_scgw.py tests/gw/test_gw_scgw_response.py \
  tests/gw/test_gw_matsubara.py
```

**61 passed**, 17 JAX real/complex cotangent projection warnings, 201.31 s.
This includes the previous pure-electronic scGW response regressions.
The new kernel suite has 18 cases and the joint scGW extension has 7 cases.
No GPU, complete repository suite, or independent ab initio EP executable
was run. The 201.31 s is test wall time, not a performance benchmark.

The H2 example uses STO-3G at 0.74 Angstrom, native RHF tolerance 1e-12 Ha,
an external mode at 0.02 Ha, beta=8 Ha^-1, nw=24, mixing=0.3, scGW residual
tolerance 1e-9 Ha and particle tolerance 1e-13 electrons. It converges with
residual 8.63198e-10 Ha and particle-number error -2.93099e-14. The derivative
of D_MO[0,1] with respect to a linear-vertex scale is -0.0008926854028821704
from implicit AD and -0.0008926854037027109 from reconverged central
differences (step 1e-3). This is not a nuclear derivative or a room-temperature
H2 prediction. Output is also archived in comments in the example.

Earlier beta=20/nw=24 attempts did not satisfy the requested 1e-9 Ha threshold,
even after tightening particle tolerance and increasing iteration count.
A matched pure-electronic control also failed (1.50e-8 Ha residual after 200
steps), while beta=8 coupled calculations passed. Thus low-temperature/grid
convergence is not established by this example; these failures are not counted
as passes and do not identify a unique cause of the numerical floor.


## ElectronPhonon.jl extension validation (2026-10-04)

Final affected regression, CPU float64 on Apple M4 Pro with Python 3.12.2
and JAX 0.8.1:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/gw/test_ep_coupling.py tests/gw/test_ep_periodic.py \
  tests/gw/test_ep_spectral.py tests/gw/test_electronphonon_jl_fixture.py \
  tests/gw/test_scgw_ep.py tests/gw/test_gw_scgw.py \
  tests/gw/test_gw_scgw_response.py tests/gw/test_gw_matsubara.py
```

**78 passed**, 18 real/complex cotangent projection warnings, 212.19 s.
This includes the earlier 61 cases; counts must not be added. A separate
mixed-precision case exercises complex64 data with float64 q weights and
confirms complex128 accumulation. Periodic tests include independent full
U(2) transformations at each k, not only phase changes. No GPU or full
repository suite was run. Fock response at band degeneracy is not validated.

`phonon_spectrum.py` ran successfully: on-shell Gamma is approximately
[0.00204324, 0.00123711] Ha, finite-window diagonal spectral weights are
[0.99904094, 0.99903961] on [-2,2] Ha, and the minimum spectral-matrix
eigenvalue is positive. These are fixed-reference model spectra, not scGW
analytic continuation or experimental predictions.

`periodic_phonon_fan.py` also ran: its four-k/four-q one-band model gives a
k-independent self-energy of approximately -0.00154679i Ha at the lowest
positive Matsubara frequency 0.03141593 Ha, and moment 9.12392505e-5 Ha^2.
Actual example outputs are retained as comments at the ends of the scripts.

The executed Julia reference uses version 1.11.5 and the pinned commit above.
Per-q and total GradSCF full widths agree to 3.469446951953614e-18 Ha. The
upstream zero-temperature point is compared using beta=1e8 Ha^-1; for this
fixture's nonzero distances from mu, occupations reach the exact limiting
values numerically. Timing recorded in provenance includes startup/compilation
and is not a cross-software speed benchmark.

## Native isolated-molecule example

[Adamantane photoemission](../../../../examples/gw/adamantane_photoemission.py)
computes RHF/STO-3G modes and screened vertices with native integrals and the
shared implicit SCF response, then combines a frozen G0W0 correction with
Fan/DW. Its 72 modes and all 66 QP roots are computed; the HOMO is treated as
a complete three-dimensional subspace. Quadratic vertices use checked finite
displacements. [Results and experimental comparison](../../../../reproducibility/gw_bse/adamantane_native/README.md)
include raw spectra, numerical-resolution checks and method limits. The
calculated main peak differs from experiment by about 0.92 eV; this is an
implementation demonstration, not a quantitative experimental reproduction.
