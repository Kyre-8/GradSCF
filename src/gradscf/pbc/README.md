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
irreducible SCF evaluation, further XC/model adapters, and the force/stress
interface. Finite-q phonons, metallic TD/GW/BSE, other smearing schemes, SOC and
macroscopic Berry-phase fields are not supplied by this increment.
