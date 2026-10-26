# Molecular GW controls

```python
from gradscf import gto, scf, gw, bse

mf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')).run()
mygw = gw.GW(mf, method='evgw0', nw=100,
            max_cycle=80, conv_tol=1e-9).run()
print(mygw.mo_energy)             # Ha
print(mygw.result.qp_weight)      # local frequency derivative, dimensionless
response = bse.BSE(mygw, nroots=1, tda=False).run()
```

The restricted facade supports `g0w0`, `evgw` and `evgw0`. The latter two
reuse the existing eigenvalue fixed-point loop. `evgw0` fixes the original
mean-field spectrum in W while updating G; `evgw` updates both. Orbitals and
mean-field exchange subtraction stay fixed. W0 is built once per evGW0 call
and reused across its outer iterations. `max_cycle`,
`conv_tol` and `damp` configure only the outer eigenvalue loop. Functional
molecular drivers accept `differentiation=SCFDifferentiationConfig()` for
[outer implicit JVP/VJP](OUTER_RESPONSE.md); an inner implicit QP derivative
is not its substitute. `UGW` offers the same three forward method choices
for HF references; its W0 arrays are currently recomputed with fixed spectra.

## Three independent windows

- `run(orbs=...)`: which QP levels to correct; others keep their MF values.
- `g_orbitals=...`: intermediate orbitals in the correlation Green's function.
- `screening_occupied=...` and `screening_virtual=...`: transitions in W.

Indices are zero-based original MO indices, not renumbered active-space
indices. `None` means all allowed indices; an empty tuple removes the
corresponding correlation sum. These windows do not truncate mean-field J/K
or change electron count. All bare MO factors are still formed and stored.
Changing the optical window later in BSE does not alter W.

For example, excluding the first doubly occupied core orbital from G and W
uses `g_orbitals=range(1,nmo)` and `screening_occupied=range(1,nocc)`.
Explicit windows avoid element-dependent implicit frozen-core rules.

The restricted low-level `g0w0_cd_restricted` takes the same windows plus an
independent `screening_energy` spectrum. Its default is the existing G pole
spectrum, preserving older calls. `evgw_cd_restricted(update_w=False)` exposes
the fixed-W variant without another iteration implementation. These additions
currently target the real closed-shell molecular route; unrestricted/periodic
window controls and new self-consistency methods are not implied.

## Quasiparticle diagnostics and provenance

`GWResult` records QP energies, self-energy, residuals, computed/converged masks,
W energies, original-MO correlation windows and the producing method.
`qp_weight = 1/(1-d Re Sigma/d omega)` is evaluated at returned energies with
G/W fixed. Unrequested levels and singular slopes have NaN weights. A finite
weight does not certify root convergence or guarantee a physical main peak;
negative weights are not clipped. Nonlinear diagonal drivers test six seeds
with the shared scalar solver, deduplicate converged positive-slope candidates,
and choose the candidate closest to the on-shell energy. An explicit
continuation target takes precedence. `qp_candidate_count` counts only the
distinct accepted roots found by those seeds; it is not a complete frequency
scan. Tied candidates are uncertified. `qp_branch_valid`, `qp_on_shell` and
`qp_linearized` expose this choice; no largest-weight or physical-main-peak
identification is implied.

The native BSE adapter inherits the actual W spectrum and windows. G0W0/evGW0
therefore give W0; converged evGW supplies its updated W. Missing legacy
metadata retains the prior explicit-reference rules. Cached GW/BSE properties
are invalidated by source/settings changes; failed GW reruns clear their result.

The QP implicit VJP and fixed-input G0W0/BSE derivatives retain their existing
contracts. See [BSE properties](../bse/README.md), [scGW](SCGW.md), and
[executed MolGW comparisons](../bse/VALIDATION.md). COHSEX, spin-flip BSE,
periodic BSE and distributed execution remain
outside this increment.

## Periodic self-consistent finite models

`gw.pbc.build_periodic_model(mf)` constructs one fixed orthonormal source-k
Hamiltonian for `g0w0_model`, `evgw_cd_kpoints`, `qsgw_cd_kpoints` and
`scgw_matsubara_kpoints`. Complete complex OV/VO response, source weights and
Hartree/exchange conventions are shared. The new source drivers are eager
forward only and currently require `fc=False`; this is not an outer AD or
bulk-material convergence claim. A bare Ewald correction without a matched
screened Gamma-cell treatment must be assessed separately.

In the Matsubara driver, positive-frequency count and time integration are
independent controls. ``time_oversampling`` sets ``Nt=2*nw*factor``;
``time_quadrature='uniform'`` retains midpoint integration, while ``'gauss'``
uses positive Gauss-Legendre weights for the forward time integrals. Inverse
Matsubara sums keep their original 1/beta normalization. Matrix-pole Green
reference subtraction and the first two bosonic moments reduce finite-window
errors; they do not remove the need for frequency/time convergence checks.

Periodic scGW returns full frequency-dependent matrices and no QP energies.
`pbc.spectrum` provides an explicit causal Hermitian matrix realization with
PSD residues, fixed self-energy moment and held-out Matsubara frequencies,
followed by a separate continued Dyson solve. Its real-axis result remains
conditional on numerical-rank and frequency-grid stability. Static QP operator
interpolation and independently sampled Brillouin-zone DOS are labelled as
interpolations, rather than direct GW at every drawing point.
Static interpolation requires complete computed and converged QP masks; a
crossing between fixed occupied/empty diagonal-QP branches is rejected rather
than hidden by energy sorting. The GaAs comparison also checks the continued
electron count from all auxiliary-Hamiltonian poles with the original beta
and chemical potential. A finite-window DOS integral counts spectral states
and is recorded separately from the occupied electron count.

`continue_self_energy` uses the shared `solvers.poles.interpolate_causal`
Hermitian least-squares realization. It determines poles without a prescribed
grid or nonlinear optimizer and checks errors against the original matrices.
Different source-k pole sets use their exact union with zero residues at
unused poles. Numerical rank truncation and the unconstrained completion are
reported; the latter is fixed to zero and does not imply a unique spectrum.
The eager construction has no implicit differentiation rule.
An optional `frequency_cutoff` in Ha selects a range established by a separate
Matsubara convergence study. The input remains complete, excluded-frequency
errors are reported, and this option does not certify forward convergence.

For a nonmagnetic spinless source, pass `model=model` to
`continue_self_energy`. The existing k-point utility pairs k and -k, and
the source MO coefficients determine the antiunitary sewing matrices through
`C_k U_k = C_-k.conj()`. The original full-frequency state must satisfy
`Sigma_k(z) = U_k Sigma_-k(z).T U_k.conj().T`, together with the corresponding
Fock and first-moment relations. The check uses the same complex frequency;
complex-conjugating Sigma would also conjugate its frequency argument.
After this validation, paired residues on the common real pole set are
replaced by their convex time-reversal average. This preserves PSD and M1,
retains physical complex residues at non-TRIM k points, and recomputes all
train/holdout/moment certificates. It rejects a genuinely symmetry-breaking
source rather than imposing this assumption on it. The electron count must
still be checked separately. Without `model`, the generic unconstrained
matrix realization remains available. Neither choice proves real-axis
stability or identifies principal peaks.

The GaAs example records uncertified selected roots as NaN together with
their original candidates and masks. A retarded spectral map or DOS can
remain defined when a unique quasiparticle assignment fails. Each gap is
reported only when its own query sample has complete certified roots;
otherwise its JSON value and change are null.

The executed same-orbital BerkeleyGW comparison, including its finite-body
head adapter, frequency convergence, multiple roots and remaining full-curve
differences, is documented in [the periodic review](pbc/BERKELEYGW_REVIEW.md).

## Algorithm references

Window separation and fixed-W iteration were checked against MolGW 3.4 source
revision `b831818d7a845c36f295d036dc8ceef59daf9991`:
[m_selfenergy_evaluation.f90](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_selfenergy_evaluation.f90),
[m_selfenergy_tools.f90](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_selfenergy_tools.f90).
GradSCF retains its JAX contour-deformation implementation and shared solvers.
The executable comparison uses MolGW's independent analytic spectral route;
its graphical QP weights require a separate frequency-grid convergence check.


## Optional QP forward solvers

`GW(mf, qp_solver='newton')` and `GW(mf, qp_solver='hybrid')` use AD frequency
slopes for the inner G0W0 root calculation. `UGW` accepts the same choice.
The default remains secant; the measured small CPU cases do not show a speedup.
Forward method selection does not change the implicit backward. See
[algorithms, safeguards and measured costs](QP_SOLVERS.md).


## Contour-boundary treatment

Molecular diagonal G0W0/evGW/evGW0 self-energies now subtract Wc(0) from the
imaginary-axis quadrature and residue, then add the analytic static term.
This is essential when evGW evaluates omega exactly at a G pole: a strict
residue mask alone drops its half-residue contribution. Explicit per-spin
occupation signs also cover empty/full molecular spin channels. The formula
and independent single-pole/parameter-response checks are described in the
[expanded MolGW validation](../bse/VALIDATION.md).

The imaginary Green denominator uses eta->0; `eta` still broadens real-frequency
W. At finite eta this differs from the previous Green-broadened quadrature,
and is not a proof of exact boundary continuity or arbitrary higher AD.
Periodic Gamma drivers use the same subtraction with a matching zero-frequency
body cache and, when requested, a matching q0 head/wing cache. Molecular matrix
qsGW uses the corresponding static subtraction, with an independent analytic
two-orbital pole test; this does not establish full qsGW parity with MolGW.

The paired periodic k-mesh drivers use the same zero-frequency body and
head/wing caches as Gamma. General complex OV/VO covariances need the full
two-sided imaginary-axis response; the legacy even-W drivers reject an
unpaired finite model explicitly. `pbc.g0w0_bands` retains both Lehmann
branches and uses the complete requested matrix window at every query point.
Its 3D `fc=True` path replaces the correlation Gamma node with nonzero
Gauss-cell samples and shifted source/internal orbitals. The quadrature order
must be converged; low-dimensional or gapless screened cell integration is
not supplied by this path.

Exact HF references are supported by the full query at their source k points
with `fc=False`. Their reference potential is reconstructed as `F-h-J`,
including its actual SCF exchange boundary. Off-mesh HF and HF Gamma-cell
queries require further band/exchange support. Pure LDA/GGA queries retain
their arbitrary-k and gapped 3D cell paths. Numerical evidence and the
remaining differentiation boundaries are recorded in
[the 2026-10-18 repair validation](../../../reproducibility/ad_showcase/gw_repairs_20261018/README.md).


For one contracted intermediate state, let delta = omega - epsilon_m,
s = -1 (occupied) or +1 (virtual), and W0 = Wc(0). The regularized expression is

```
Sigma_m = -1/pi * integral_0^infinity
              delta/(delta**2 + nu**2) * [Wc(i*nu) - W0] dnu
          + s/2 * W0
          + (sign(delta)+s)/2 * [Wc(abs(delta)) - W0].
```

At zero real-axis broadening, Wc(i*nu)-W0 is O(nu^2). The subtracted
imaginary-axis integral and its first frequency derivative are regular at
delta=0. The residue coefficient is s/2 on that boundary. At finite eta,
the existing real-frequency response convention does not give exactly the
same Wc(0) as the imaginary-axis evaluation, so this identity is exact in
the eta->0 limit. Quadrature convergence remains necessary, including the
subtracted large-frequency tail.

## Molecular evGW resolvent acceleration

The restricted molecular evGW/evGW0 loop uses a direct-RPA transition-space resolvent.
For occupied-virtual factors V and positive gaps Delta, define

```
T = V sqrt(Delta)
M = diag(Delta**2) + 4 T.T T
G = T.T B_mn
Wc(z)[mn] = -4 G_mn.T (M - z**2 I)^-1 G_mn
```

The MO-pair contractions above are the diagonal of the pair-space quadratic
form. On the imaginary axis use z=i*nu and real gaps. To retain the existing
real-axis response, 1/(omega-Delta+2i*eta)+1/(-omega-Delta), construct a second
model with Delta replaced by Delta-i*eta and evaluate it at z=omega+i*eta.
Using the real matrix at z=omega+i*eta alone would change the finite-eta
response, particularly near a screening pole. The reference pole helpers use
that simpler convention only for checks; production preserves the CD response.

The shared `solvers.linear.factor_shifted` / `solve_shifted` implementation
reuses the real symmetric imaginary-axis factorization and a complex symmetric
real-axis factorization for many shifts. A defective or poorly conditioned
complex eigenbasis falls back to a dense solve. Spectral factors are stopped
primal data, while `custom_linear_solve` differentiates

```
(M-shift*I) dx = db - (dM-dshift*I) x.
```

Thus physical matrix, coupling, frequency and eta derivatives are retained;
no division by an eigenvalue gap is needed at degeneracy. Singular shifted
systems remain outside this derivative contract; no denominator clipping is
introduced. This combines the speed of a spectral solve with a basis-invariant
matrix-equation derivative, rather than treating spectral speed and correct
AD as mutually exclusive choices.

Residue evaluation requests only the external/intermediate MO pairs it needs,
including nonconsecutive G windows. evGW0 reuses both the factorization and the
imaginary-axis/static W values within one invocation. There is no global cache.
G0W0 still defaults to CD. Molecular evGW/evGW0 and restricted qsGW offer
opt-in [outer implicit response](OUTER_RESPONSE.md); eager forward remains default.

A serial C3H6/STO-3G/Weigend-RI comparison with 64 quadrature points tested
reused factorization against a per-frequency dense reference. Both sides of
the benchmark now retain the same finite-eta CD response. First-call results,
repeat samples, settings and timing boundaries are recorded in
[the benchmark report](../../../reproducibility/gw_bse/cycloalkane_scaling/resolvent_reuse.md).


## Fixed-phonon matrix scGW

`scgw_matsubara_restricted(..., phonons=PhononModel(...))` adds an external
harmonic Fan/Debye-Waller model inside each electronic iteration. The model
uses the fixed initial orthonormal MO frame; its frequencies and vertices
remain differentiable external parameters. The shared implicit root includes
the dynamic self-energy, total tail moment, static DW potential and particle
number. See [ep_coupling](ep_coupling/README.md) for conventions and boundaries.

With a phonon model, `total_energy` is None; `electronic_energy` is explicitly
only the electronic contribution, excluding nuclear repulsion and EP/phonon
energies. Fixed phonons do not imply a self-consistent phonon Dyson equation.
Complex periodic Fan kernels now include the Matsubara reference correction
and high-frequency moment; they are numerical maps, not a periodic scGW driver.
Direct fixed-reference retarded Fan, on-shell linewidths and matrix spectral
functions are also available; these do not analytically continue scGW output. An executable native H2 demonstration is in
[fixed_phonon_scgw.py](../../../examples/gw/fixed_phonon_scgw.py).
