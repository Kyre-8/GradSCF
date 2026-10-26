# Native water: one-phonon and multiphonon absorption

Run `examples/bse/water_ep_coupling.py` directly, without a CLI. Native GradSCF
integrals, stationary RHF/3-21G geometry/response, G0W0 and three-root singlet
TDA-BSE supply all electronic and vibrational inputs. No PySCF, experimental
curve or fitted coupling is used. CPU arm64, Python 3.12, JAX 0.8.1, float64;
the invocation below fixes the backend and BLAS thread count. The elapsed
time in JSON includes compilation and computation through the AD probe, before
figure rendering; it is not a warmup-excluded benchmark.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/private/tmp/gradscf-mpl-cache \
  python examples/bse/water_ep_coupling.py
```

## Electronic and vibrational inputs

| Quantity | Result |
| --- | --- |
| Optimized O-H bond | 1.82672351 bohr |
| RHF energy | -75.585959743008 Ha |
| Largest Cartesian gradient | 3.74e-8 Ha/bohr |
| Analytic Hessian asymmetry | 3.45e-11 Ha/bohr² |
| Bend, symmetric stretch, antisymmetric stretch | 1799.2880, 3812.3762, 3945.8321 cm⁻¹ |
| Three BSE excitations | 8.92633283, 11.13901191, 11.54453330 eV |
| Oscillator strengths | 0.00582640, approximately 0, 0.09305359 |
| Largest QP residual | 5.47e-11 Ha |

RHF linear vertices include the implicit stationary density response and
maximum-overlap orbital transport. Quadratic vertices use reconverged symmetric
displacements with maximum Cartesian motion 0.002 bohr. GW corrections and the
electronic BSE kernel are frozen during these displacements. Thus this is a
specified effective electronic Hamiltonian, not the full geometry derivative
of the correlated GW/BSE problem. The BSE electronic subspace contains only
three states; electronic-state and basis convergence are not established.

The ground initial state is the vibrational vacuum (0 K). The Fan/DW evaluation
uses beta=1e8 Ha⁻¹ to produce the same zero-occupation limit. No thermally
occupied initial levels, rotation, environment, Herzberg–Teller dipoles or
dissociation channel is included.

## What the four panels compare

- Upper left: fixed-nuclei BSE, one-phonon Fan, Fan+static DW, and the linear
  multiphonon Hamiltonian. Fan and the linear Hamiltonian use identical linear
  vertices with Lambda=None. Fan+DW is an additional quadratic correction,
  and is not identical to a fully quadratic multiphonon Hamiltonian.
- Upper right: diagonal-only one-phonon Fan versus the independent-mode
  Franck-Condon reference, with the same diagonal couplings. This isolates the
  single-phonon approximation from electronic-state mixing.
- Lower left: the final successive vibrational cutoffs. Legend cutoffs refer
  to the coupled two-state/three-mode sector and the separate bright-state/
  two-mode sector. The latter's antisymmetric mode stays in its vacuum.
- Lower right: the same final multiphonon lines at eta=10 and 3 meV. Their
  positions and strengths stay fixed while numerical line smoothing changes.

The C2v selection-rule residual is checked before block reduction (8.98e-13
Ha here). The antisymmetric mode has negligible diagonal couplings but mixes
the first bright state and the dark second state with a roughly 0.505 eV
off-diagonal vertex. It is retained in that sector. Unit tests compare full
and block optical resolvents, so the dark state is not omitted by an
oscillator-strength threshold.

The finite Hamiltonian is

    H_vib = H_X + sum_nu omega_nu b_nu† b_nu + sum_nu G_nu (b_nu+b_nu†).

It is assembled by `bse.ep_coupling.vibronic_hamiltonian`; diagonalization is
performed by the shared Hermitian solver. The ground-surface ZPE is subtracted.
Each sector contains all occupations with total quanta at most its cutoff.
No extra DW is added to this linear reference. Dipole-norm conservation and
positive transition energies are checked. Stop conditions require relative
spectral L1 changes below 0.5% in the 4–18 eV window for both eta values, and
below 0.1% separately for the coupled sector. Grid spacing is 2 meV. The actual
cutoffs, residuals and changes are recorded in `results.json`; these thresholds
certify this model and resolution, not exact molecular spectroscopy.

The final run uses cutoffs **28 / 40**, with sector dimensions **8990 / 861**.
The last successive-spectrum changes are **0.00869%** at eta=10 meV and
**0.05542%** at eta=3 meV. The independently converged coupled sector changes
by **0.02493% / 0.08604%** at the two resolutions. The largest eigen-residual
is **7.67e-15 Ha**. The executed computation phase took **115.10 s**, including
compilation; no GPU execution or performance comparison is claimed.

For the independent-mode Condon reference,

    S[nu,s] = (G[nu,s,s]/omega[nu])²,
    E_00[s] = E_X[s] - sum_nu G[nu,s,s]²/omega[nu],
    weight[n,s] = product_nu exp(-S[nu,s]) S[nu,s]^n_nu/n_nu!.

The diagonal bright-state Huang-Rhys sums are about 1.2182 and 4.5197. The
corresponding distributions of transition energies have standard deviations
0.5170 and 0.7275 eV. These are vibronic-envelope measures, not Lorentzian
lifetimes or experimental FWHM. The finite Franck-Condon summation retains raw
weights (no tail renormalization); retained weights are reported in JSON.
At the stated eta, an individual bare Lorentzian has FWHM=2 eta, i.e. 20 or
6 meV. Multiphonon redistribution therefore has a different origin from
choosing a broader plotting kernel.

## Differentiation and validation

The probe rescales all linear vertices with bare H_X, mode energies and
Condon dipoles held fixed, using a total-quanta cutoff of **4**. It evaluates
the resonant and antiresonant dipole resolvent with shared `solve_complex`.
The derivative is valid for this specified finite model; it is not a
converged nuclear-coordinate derivative of the entire workflow. AD and
recomputed central finite differences are saved in JSON and checked in the
script. Individual vibronic eigenvector derivatives are not used.

At photon energy 8.9932578150 eV, the finite-cutoff absorption is
1.1853226125 Mb. The coupling-scale AD derivative is 152.6769102802 Mb and
the recomputed central finite difference (step=1e-5) is 152.6767882203 Mb,
with relative difference 8.00e-7.

The matrix assembly and shared-response tests passed 37 tests, with two JAX
ComplexWarnings; the dedicated vibronic oracles include analytic Poisson
energies/weights, mode-pair quadratics at a truncation boundary, complex frame
covariance, full/block response equivalence and AD/finite-difference checks.
See `src/gradscf/bse/ep_coupling/VALIDATION.md` for commands and results.

## Physical interpretation and limits

The plotted broad envelope comes from many discrete vibronic transitions.
This bound harmonic model does not produce an irreversible lifetime simply
by adding modes. Real water's first ultraviolet band includes dissociation;
predicting it requires an appropriate excited-state potential and nuclear
dynamics/continuum treatment. These curves are an algorithmic comparison of
the specified frozen-kernel model, not an experimental water spectrum.

References:

- A. Baiardi, J. Bloino and V. Barone, *A general time dependent approach to
  vibronic spectroscopy including Franck-Condon, Herzberg-Teller and Dushinsky
  effects*, JCTC 9, 4097–4115 (2013),
  [doi:10.1021/ct400450k](https://doi.org/10.1021/ct400450k).
- *Three-dimensional wavepacket calculation for the photodissociation of water
  in the A state* (1992),
  [primary wavepacket study](https://www.sciencedirect.com/science/article/pii/0009261492859273).

## Artifacts

- `water_bse_phonons.png` / `.pdf`: the four-panel comparison.
- `absorption.csv`: photon energy in eV, cross sections in Mb; columns identify
  each approximation, cutoff and smaller-eta reference.
- `inputs.npz`: native coordinates/Hessian/directions, electronic and mode
  energies and vertices in Ha, transition dipoles in bohr.
- `vibronic_lines.npz`: final transition energies in Ha and dipoles in bohr,
  ordered by electronic sector rather than globally by energy.
- `results.json`: unrounded results, cutoffs, convergence metrics, AD/FD and
  backend/runtime metadata.
