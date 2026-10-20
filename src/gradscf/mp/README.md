# Molecular Moller--Plesset methods

`gradscf.mp` implements real restricted/unrestricted MP, including
specialized MP2/MP3 kernels and an order-driven residual/Taylor engine.
The eager interfaces reuse a converged GradSCF HF calculation:

```python
from gradscf import gto, scf, mp

mol = gto.M(atom="O 0 0 0; H 0 .75 .58; H 0 -.75 .58", basis="6-31g*")
mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10).run()
pt = mp.MP2(mf, frozen=1).run()
print(pt.e_corr, pt.e_tot)
pt3 = mp.MP3(mf, frozen=1).run()
print(pt3.e2, pt3.e3, pt3.e_tot)
```

`MP2` dispatches to `RMP2` or `UMP2`; restricted `MP3` uses `RMP3`, while UHF
MP3 uses the generic engine. `MP(mf, order=p)` defaults to that engine for every
finite p>=2. See [SERIES.md](SERIES.md) for order-dependent rank limits,
intermediate normalization and the complete-state/energy contract.
Explicit `scf.reference.RestrictedReference` and `UnrestrictedReference` MO
Hamiltonians are also accepted. The default `orbital_basis="canonical"` requires
the physical HF Fock matrix to be diagonal. `orbital_basis="noncanonical"`
accepts stationary HF orbitals with full occupied/virtual Fock blocks. ROHF,
complex orbitals, fractional occupations and relaxed/unrelaxed densities are
not part of this version. A level shift is never added to MP denominators.

## Array API and differentiation

```python
config = mp.MPConfig(order=3, with_t2=False)
result = mp.run_mp(h1_mo, eri_mo, nocc=nocc,
                   nuclear_repulsion=enuc, frozen=1, config=config)
```

Restricted `h1` and `eri` are real MO tensors in chemists' notation. For UHF,
`h1=(ha,hb)`, `eri=(gaa,gab,gbb)`, and `nocc=(nalpha,nbeta)`. Occupied orbitals
precede virtual orbitals in each spin channel. `nocc`, `frozen` and configuration
are static controls; numerical Hamiltonian arrays remain differentiable.

The specialized MP2 algebra and MP3 interaction action use JAX AD, including JIT, JVP/VJP
and higher derivatives on valid inputs. No iterative MP amplitude fixed point
is introduced for canonical orbitals. A complete parameter derivative still
needs the HF state and integral response; the eager facade is not a traced
SCF-to-MP function. `examples/mp/implicit_gradient.py` demonstrates the full
chain using the existing implicit orbital HF solver and reconverged finite
differences. It differentiates a dimensionless contraction coefficient in H2,
not nuclear coordinates.

For molecular geometry response, `gradscf_tools.correlated_response.CorrelatedEnergy`
connects the native integral and implicit HF responses to MP. Its `method="mp"`
accepts an arbitrary-order `MPConfig`; `method="mp2"` remains limited to order 2.
Both default to the noncanonical frame. This path differentiates the complete
occupied/virtual Fock operator rather than individual canonical eigenvectors.
Within-block MO degeneracy is allowed when the physical excitation resolvent
and HF response remain nonsingular. See [SERIES.md](SERIES.md).

## Results and validity

- `corrections` contains E2 through the requested order; `e2` and `e3` are
  shortcuts. `correlation_energy` is the sum of corrections, and
  `total_energy = reference_energy + correlation_energy`.
- `t2` means **first-order** doubles amplitudes at every energy order. Restricted
  amplitudes have shape `(nocc,nocc,nvir,nvir)`; unrestricted amplitudes are
  `(taa,tab,tbb)`, with antisymmetric same-spin blocks. `with_t2=False` omits
  these output tensors.
- `same_spin_energy` and `opposite_spin_energy` (facade `e_corr_ss/e_corr_os`)
  decompose **E2 only**, including when higher orders are requested.
- `canonical_error` is the maximum absolute off-diagonal physical Fock entry
  before freezing. The canonical input threshold is `1e-8 Ha`. Noncanonical
  inputs instead check occupied--virtual stationarity and the frozen boundary;
  a nonzero `canonical_error` can therefore be valid.
- `min_abs_denominator` diagnoses allowed doubles denominators in specialized
  kernels and all retained excitation gaps in the generic engine. Its default
  threshold is `1e-10 Ha`. Empty excitation spaces give zero energy and an
  infinite minimum denominator.
  Noncanonical MP2 currently checks the complete Cartesian four-axis spectrum,
  including Pauli-forbidden same-spin entries; this can be conservative for
  non-Aufbau references. The noncanonical series checks retained excitation sums.
- `valid=False` returns NaN total/correlation energy; eager `kernel()` raises.
  Reference values, diagnostics and still-valid lower-order components may
  remain available. Differentiation is supported only on valid inputs.
  Small denominators are diagnosed, never silently clipped or shifted.

`frozen` follows the shared post-HF convention: a core count or an orbital
index list, optionally separate alpha/beta lists for UMP2. Frozen occupied
electrons still contribute to the Fock and HF reference energy; frozen virtual
orbitals are excluded only from the correlated active space.
Noncanonical frozen spaces must remain invariant under the Fock operator and
must not split a degenerate occupied/virtual group. Pure MO-array derivatives
must preserve that partition. Moving frozen-core projector response for general
nuclear displacements is not implemented by the geometry adapter.

## Integral storage

The canonical MP2 facade transforms only occupied--virtual blocks from dense/s4 AO data.
For DF it retains `B[Q,i,a]` per spin and contracts one occupied-index slice
at a time. With `with_t2=False`, neither full MO ERIs nor the full MP2 doubles
tensor are constructed by the DF evaluation. The loop is checkpointed for AD.
This is an algorithmic storage property, not a measured GPU peak-memory claim.
The dense/s4 path still stores the selected `ovov` block. Direct-only SCF
without saved two-electron data is explicitly rejected.
Noncanonical MP2 solves a four-axis tensor equation and keeps its internal
amplitude tensor even when `with_t2=False`; its storage is O(nocc^2 nvir^2).

The specialized RMP3 kernel is an **in-core reference implementation**. It reuses the
existing full-MO post-HF adapter and restricted CC integral blocks; DF input
is expanded to full MO ERIs on this path. `with_t2=False` suppresses the output
but MP3 still needs first-order doubles internally. Streaming MP3 is a future
extension, not implied by MP2 DF support.

The generic engine also transforms full MO integrals and stores only retained
determinants. CI and MP share compact particle/hole string contractions for
Hamiltonian actions, without a determinant-pair connection table. The engine
reuses shared linear solves instead of diagonalization. Its storage is excitation-rank limited but
still combinatorial; declared capacity bounds can reject a high-order request.
Linear Hamiltonian nodes are shared across Taylor degrees and energy extraction;
[paired CPU measurements](GRAPH_REUSE.md) document the benefit and its limits.
For pth-order energies it computes complete state coefficients through
floor(p/2), retaining rank through twice that value; see SERIES.md.

`algorithm="tensor"` is an experimental alternative action representation for
that same Taylor engine. It contracts complete spin-resolved excitation blocks,
avoiding Slater--Condon connections and full spin-orbital ERIs. It still uses
full spatial MO integrals. Canonical tensor actions use dense antisymmetric
temporary blocks; the noncanonical tensor path uses the compact direct action
and preserves the same packed output chart. Performance
depends on the occupied/virtual dimensions and requested order. See
[the coordinate convention, capacity bounds and measurements](TENSOR_BLOCKS.md).

## Validation

The [convergence examples](../../../examples/mp/CONVERGENCE.md) reproduce
He2(2+) deceptive UMP convergence with native RMP/UMP2--30 and independent
FCI/RS checks. Ne/F- use a complete external high-order benchmark with native
HF/MP2/MP3 comparisons; their high-order curves are not GradSCF calculations.

`tests/mp` checks RHF water/6-31G* MP2 and UHF H3/STO-3G MP2 against independent
PySCF energies/amplitudes, with Cartesian orbitals and CPU float64. Tests also
cover frozen orbitals, selected dense/packed/DF transforms, pure JAX gradients
and HVPs, stale references, physical denominator validity and empty spaces.
MP3 on H2/3-21G and LiH/STO-3G is checked against independent determinant-space
Rayleigh--Schrodinger coefficients using PySCF FCI Hamiltonian actions, rather
than the CC expressions used by the implementation. Complete implicit-HF
parameter derivatives are checked against independently reconverged finite
differences.

The generic engine compares R/U energy and wavefunction coefficients with an
independent full-space perturbation recurrence. It also checks JIT, integral
gradients/HVPs and the complete implicit-HF parameter derivative through MP4.
See [VALIDATION.md](VALIDATION.md) for measured results and scope.
The [molecular comparison](../../../examples/mp/compare_molecules_pyscf.py)
also checks nine RHF/UHF cases with Cartesian 6-31G* through MP6, including
frozen cores, actual quadruple excitations and a fully alpha-polarized quartet.

Run from the repository root after building the native CPU integral library:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q tests/mp
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/molecular.py
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/implicit_gradient.py
```

See [REFERENCES.md](REFERENCES.md) for equations and attribution.
