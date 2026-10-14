# Periodic foundations

The native periodic path uses Cartesian Gaussian orbitals, GTH pseudopotentials
and FFT Coulomb integration on neutral 3D cells. Orbital basis data are offline;
see [basis provenance and coverage](../integrals/periodic/NOTICE.md).

## Sampling and weights

```python
kpts = cell.make_kpts((4, 4, 4), shift=(.5, .5, .5))
mf = scf.KRHF(cell, kpts=kpts).run()
```

`shift` is in grid steps; zero preserves the original Gamma-centered convention.
Absolute k points are in inverse Bohr, and lattice vectors are rows. SCF currently
evaluates the **full uniform Cartesian-product mesh**, including shifted meshes.
`kweights` can be supplied explicitly as strictly positive normalized weights.
The weights enter density, Hartree/exchange contractions, XC, energies, particle
counts, entropy, and semilocal band queries. They are independent of FFT grid
quadrature weights. Arbitrary weights do not automatically represent an
irreducible Brillouin zone: the missing AO/Bloch transformations must also be
implemented before evaluating only symmetry representatives.

For HF/global hybrids, `exxdiv='ewald'` retains the supercell Madelung correction
and requires equal weights. Use `exxdiv=None` for a general weighted quadrature;
this omits the singularity correction and is **not** a replacement finite-size
correction. Unsupported nonuniform Ewald calculations are rejected by the facade
and marked nonfinite in the array contraction path. Changing sampling requires
a new SCF calculation.

`pbc.kpoints.symmetry_reduce` builds orbit indices and multiplicities under supplied
reciprocal-space rotations and optional time reversal. It is metadata for scalar
sums, not an irreducible-SCF backend or an automatic space-group finder. Users
must supply symmetries valid for their Hamiltonian. Symmetry matching is discrete
host preprocessing, not part of the AD graph.

`KRKS(..., time_reversal=True)` (also `KRHF/KUHF/KUKS`) uses one member of
each k/-k pair for SCF-iteration diagonalization and expands the density by
complex conjugation. Input overlap/core matrices and pair weights must obey
time reversal; external XC potentials are checked in the iteration as well.
The converged state is certified against the original full-mesh fixed point.
Implicit backward retains the **full** residual and therefore includes
time-reversal-breaking perturbations. Explicit derivatives of a reduced
iteration require perturbations that preserve the declared symmetry.
Integrals, FFT J/K contractions, storage and final reported MO diagonalization
still use the full mesh. This is not general space-group/AO reduction or a
claim of an end-to-end speedup. A finite FFT grid that violates the matrix
symmetry checks must be refined, or the full-mesh path used.

`pbc.momentum.momentum_transfer_table(kpts, qpts=...)` is shared by periodic
response/correlation and GW. Coordinates are fractional reciprocal coordinates;
the table satisfies `k[j] = k[i] - q` modulo reciprocal lattice vectors. Shifted
k meshes generally require a separate unshifted q mesh. The former internal
`gw.pbc.momentum` module has moved here; exports from `gw.pbc` remain available.

## Finite-temperature SCF

```python
mf = scf.KRHF(cell, kpts=kpts, temperature=0.01).run()
print(mf.e_tot, mf.e_free, mf.entropy, mf.chemical_potential)
```

`temperature` is **kBT in Hartree**, not Kelvin. `None` retains fixed integer
occupations. Fermi-Dirac statistics are currently the only finite-temperature
scheme. Alpha and beta populations are fixed separately; unrestricted runs have
two chemical potentials and do not optimize their magnetization through a single
common chemical potential. A nonmagnetic odd-electron cell can have half-integer
spin populations, e.g. Al has `(1.5, 1.5)`. Such counts require the thermal path.
For traced calculations, `pbc.scf.run_kpoint_scf` is the array-level entry point;
temperature and inputs may be JAX tracers, while the FFT mesh remains static.

For each spin channel:

\[
f_{snk}=\frac{1}{1+\exp[(\epsilon_{snk}-\mu_s)/\tau]},\qquad
\sum_k w_k\sum_n f_{snk}=N_s,\quad \tau=k_BT.
\]

The shared bracketed solver finds each chemical potential. A shared implicit
root rule differentiates the number equation, including energy, temperature,
and particle-count dependence. The number residual uses electron/hole tails
relative to integer occupations, avoiding cancellation in a low-temperature gap.
Very small temperatures for which the active charge susceptibility underflows
are marked invalid rather than assigned an arbitrary chemical-potential gradient.
Empty/full spin channels have exact occupations and report infinite chemical
potentials; derivatives of the chemical potential itself at those endpoints are
not physically defined.

The converged density is the generalized Hermitian Fermi matrix:

\[
D_s(k)=X_k f_\tau(X_k^\dagger F_s(k)X_k-\mu_s I)X_k^\dagger,
\qquad X_k^\dagger S_kX_k=I.
\]

Density JVP/VJP uses a shared matrix-function Frechet rule. Its Fermi divided
differences use an analytic `expm1` expression and the derivative limit at
coincident energies, avoiding individual eigenvector derivatives and subtractive
cancellation. This guarantees the implemented **first** density response at
degeneracy; it does not promise higher derivatives or unique MO-coefficient
derivatives within a degenerate subspace.

The existing SCF loop now checks the Fermi-density fixed point as well as energy
and density changes. A vanishing commutator alone would not establish correct
thermal occupations. Projector level shifts are rejected in thermal mode.
Implicit backward uses this same fixed-point equation and the shared linear
solver. Temperature and populations are explicit leaves of its parameter tree.

`total_energy`/`e_tot` is the internal electronic-plus-ion energy. The separately
reported free energy is

\[
\mathcal F=E-\tau\mathcal S,\qquad
\mathcal S=-\sum_{skn}w_k[f\log f+(1-f)\log(1-f)] .
\]

`entropy` is the dimensionless quantity S/kB per cell. SCF convergence uses the
free energy of the current mixed density, with occupations obtained from its
metric-transformed density matrix. Its logarithms use machine-epsilon clipping
at 0/1; low-temperature entropy has an absolute roundoff floor and is not a
tail-relative-precision observable. At a stationary state, the thermal gradient
satisfies `dF/dtemperature = -entropy`; geometry derivatives must likewise use
the appropriate thermodynamic potential.

## Response and consumer boundaries

Static finite-q electronic response is now available through
`dfpt.DFPT(mf, q=...).response(potential=...)` for HF and semilocal KS on a
closed, equally weighted k mesh. Both integer insulating and finite-temperature
Fermi references are supported. It retains the k+q folding phases and reuses
the FFT exchange and shared linear solvers. This is separate from the TD/GW
occupation restrictions below. See [finite-q conventions and checks](../dfpt/FINITE_Q.md).
Atomic-displacement/overlap sources and bounded commensurate HF/KS/OFDFT
reference phonons are now available; see [atomic response](../dfpt/ATOMIC_RESPONSE.md).
Primitive HF/KS displacement sources and `Phonons.kernel(method='primitive')`
now construct geometry curvature and electronic response without a supercell.
The bounded supercell path remains an independent reference and supports OFDFT.
Primitive response still requires an equal-weight closed k mesh and FFT
convergence; full AO contractions and k-pair exchange work remain.

`Phonons.mesh(qmesh, method='primitive')` samples unweighted force constants and
returns a `ForceConstants` object for shortest-image Fourier interpolation,
band paths and Gaussian DOS. The input q mesh must close on the reference k
mesh. Raw ASR diagnostics are retained; Gamma projection is opt-in and imaginary
frequencies stay visible. See [IFC conventions, units and validation](IFC.md)
for the phase gauge, numerical postprocessing boundary and polar limitations.

### External XC functionals

The same `dft.Functional(input_fn, energy_fn)` used for molecules accepts
periodic SCF inputs. `input_fn` receives the **current** Hermitian density
`(2,nk,nao,nao)`, real spin densities, density gradients, kinetic-energy density
and density Laplacian. It can also access AO values/covariant derivatives,
overlap, grid coordinates/weights, k points/weights, lattice and atomic
coordinates. `atom_charges` are full nuclear charges; `ionic_charges` are GTH
valence charges. Grid and k-point weights are different quantities.
`energy_fn(params, features)` returns a real scalar XC energy per cell in Eh;
the user supplies the whole XC model, including any desired baseline.

```python
mf = dft.KRKS(cell, kpts=kpts, functional=functional, params=params,
              temperature=.01, time_reversal=True).run()
response = dfpt.DFPT(mf).response(potential=v1)
mf.run(params=new_params)  # Recompute before constructing a new response.
```

Here `dft` is `gradscf.pbc.dft`; construct `functional` using
`gradscf.dft.Functional`. Potential/kernel evaluation uses the shared derivative
module. For complex density blocks, its convention is
`dE = Re sum_sk w_k Tr(V_sk dD_sk)`: conjugate the JAX gradient and divide by
k weights once. The kernel is the JVP of this physical potential, with no
locality approximation or four-index Hessian allocation.

For parameter training, use `pbc.scf.run_kpoint_scf(..., functional=..., params=...)`
and `dfpt.periodic_problem(inputs, result, ..., functional=..., params=...)`
inside the differentiated function. Model parameters are explicit leaves of
the SCF/response residual. Fixed-density, explicit and implicit energy training
are demonstrated in `examples/training/periodic_xc.py`; the molecular `Trainer`
facade is not yet a periodic sample adapter. Rebuild response snapshots after
changing parameters. Response-loss parameter gradients are tested for
nondegenerate thermal references; this does not extend the higher-derivative
guarantees at degeneracy.

`dfpt.DFPT(mf).response(potential=...)` supports static q=0 thermal response using
the Fermi-density residual, including changes of occupations and chemical
potentials. It returns density, internal energy, free energy, entropy, and
electron-number responses. The fixed-integer KS projector chart is still used
for zero-temperature references. Thermal Taylor series are explicitly unavailable.

Periodic TDHF/TDDFT and GW remain zero-temperature, integer-occupation,
equal-weight consumers. They reject thermal/nonuniform references rather than
silently replacing their occupations. Optical postprocessing, bands, DFPT, and
GW share reference freshness checks, including temperature, k points, weights,
and exchange correction.
External periodic models are currently supported by SCF and q=0 DFPT, but not
by the TD/GW, optical or band-query consumers: an arbitrary energy model does
not specify their missing off-grid/dynamical operators.

## Forces and stress

Periodic HF/KS and Cell-based OFDFT share `pbc.grad`:

```python
gradient = mf.nuc_grad_method()  # Also calc.nuc_grad_method() for periodic OFDFT.
de = gradient.kernel()          # dE/dR, or dF/dR at finite temperature.
forces = gradient.result.forces # -de, in Eh/Bohr.
stress = gradient.stress()      # Tensile-positive, in Eh/Bohr**3.
pressure = -stress.trace()/3
```

Each `kernel()` rebuilds integrals and reconverges from the default seed using
the existing explicit/implicit policy. An existing SCF reference must have
current settings. A density differing from the supplied reference by more than
`1e-6*(1+norm(reference))` is rejected, rather than silently returning another
solution's derivatives. Following an arbitrary metastable branch is not
implemented. Failed primal/adjoint calculations leave no result cache.
`stress()` reuses the latest successful `kernel()` result; rerun `kernel()` after
changing the source. The source SCF result is not replaced.

For row lattice A and Cartesian positions R, let G_R=dE/dR and G_A=dE/dA:

\[
\mathbf F=-G_R,\qquad
\sigma=\frac{\operatorname{sym}(G_R^{\mathsf T}R+G_A^{\mathsf T}A)}{\det A}.
\]

The symmetric strain convention is `R(e)=R@(I+e).T`, `A(e)=A@(I+e).T`.
Forces hold the cell fixed; stress holds fractional positions and fractional
k points fixed. Electron/spin counts, kBT, Gaussian exponents and contractions
are fixed. `cell.parameters_at(coordinates)` moves every atom-attached shell;
floating centers require an explicit geometry callback. Moving overlaps,
Gaussian centers, FFT weights, reciprocal vectors, GTH potentials, Ewald and
exchange corrections remain in AD. Thermal derivatives use free energy.
The derivative driver uses the full k mesh even for a time-reversal source.

`pbc.grad.energy_and_derivatives(energy_fn, params, coordinates, lattice)` is
the pure JAX entry for custom models and parameter/geometry differentiation.
Its callback returns a real scalar, rebuilds geometry-dependent inputs and owns
convergence checks. Existing derivative-order limits still apply; no higher
thermal derivatives at degeneracy are implied. OFDFT supports grid and Gamma
Gaussian amplitudes and retains its local-pseudopotential restriction.

FFT mesh and Ewald image topology remain fixed during AD: this differentiates
the discretized energy, not an adaptive cutoff. Check force/stress mesh
convergence separately. For the off-grid H2 regression, the largest residual
total-force component decreases from `1.18e-5` at mesh 31 cubed to
`2.82e-8 Eh/Bohr` at mesh 41 cubed; rigid-translation energy differences reproduce
the residual. No artificial zero-force projection is applied.
`examples/periodic_forces.py` demonstrates HF and OFDFT.

## Validation and remaining steps

Focused tests cover weighted energy/Fock consistency (HF and a handwritten PBE
oracle), particle conservation, chemical-potential derivatives, degenerate real
and complex Fermi response, thermal SCF JIT/explicit/implicit gradients, and
PySCF Fermi-smearing energy/free-energy/entropy/occupation comparisons. Al static
response is checked against independently perturbed SCF solutions. Optional
`jax_xc` comparisons remain separate from the handwritten PBE tests.

`examples/periodic_temperature.py` demonstrates FCC Al with a small shifted mesh,
free-energy AD and q=0 response. Its coarse sampling is an API demonstration,
not a material convergence benchmark.

The next foundation work remains physical space-group/AO transformations,
irreducible J/K evaluation and periodic Trainer sample adapters.
Native finite-q OFDFT phonons, Born charges/LO-TO corrections, general phonon
interpolation, metallic TD/GW/BSE, other smearing schemes, SOC and
macroscopic Berry-phase fields are not supplied by this increment.
