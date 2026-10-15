# Basis examples

- [bsse_counterpoise.py](bsse_counterpoise.py): RHF counterpoise correction.
- [optimize_bsse_contractions.py](optimize_bsse_contractions.py): differentiable
  contraction optimization for a fixed water dimer.
- [nnao_basis.py](nnao_basis.py): MACE-conditioned H-X basis assembly and native
  overlap integrals; requires the optional MACE stack and native library.

## RHF counterpoise example

bsse_counterpoise.py directly calls the public GradSCF API:

    mf = scf.RHF(mol)
    energy = mf.kernel()

The only helper constructs molecular inputs with zero-charge ghost centers.
The library prepares integrals and performs SCF. Five energies at each fixed
dimer geometry give raw interaction, CP interaction and the BSSE correction.
Reference output is included as comments at the end of the script.

Run from the GradSCF repository root:

    PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python examples/basis/bsse_counterpoise.py

Requires GradSCF dependencies and its built native CPU integral library.
The script selects CPU float64 and reads the bundled S22 water, ammonia
and formic-acid dimer geometries. Edit the systems and bases tuples to
choose cases.

The public SCF assembly currently uses Cartesian AOs, explicitly selected
with cart=True. The def2-SVP results therefore use six Cartesian d
functions per d shell. The appended comments were recomputed with this
convention; they differ from the earlier five-function spherical d results.

Ghost centers retain their element labels and basis functions while their
nuclear charges and electron contributions are zero. Their physical
fragment Hamiltonian and SCF state are recomputed. Pure HF ghost energy
calculations skip unused XC quadrature. This example validates HF energies;
ghost DFT quadrature and grid-dependent response features are not validated.
No density fitting or learned NNAO basis is used.

    raw  = E_AB(full) - E_A(own) - E_B(own)
    CP   = E_AB(full) - E_A(full) - E_B(full)
    BSSE = CP - raw

All values are in kcal/mol. Positive BSSE raises the interaction energy.
Fragment deformation is excluded. These are fixed-geometry HF interaction
energies, not correlated binding-energy reference values.

- [PySCF simple HF example](https://github.com/pyscf/pyscf/blob/master/examples/scf/00-simple_hf.py)
- [S22 database](https://doi.org/10.1039/B600027D)
- [Boys–Bernardi counterpoise](https://doi.org/10.1080/00268977000101561)

## Differentiable contraction optimization

optimize_bsse_contractions.py reduces counterpoise BSSE for the fixed S22 water dimer. It optimizes six shared O 2s/O 2p/H 1s coefficient variables, retaining fixed exponents, core coefficients and AO dimension. Physical AO normalization is differentiated.

    PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/basis/optimize_bsse_contractions.py

The example uses existing public RHF and implicit shared-orbital HF APIs; no new source API is required. Small fixed native primitive tensors are contracted in JAX, and jax.value_and_grad supplies the BSSE gradient to SLSQP. Each own-fragment energy may increase by at most 1e-4 Ha; coefficient changes are bounded in a fixed scale gauge. Runtime checks reject invalid gradients, unconverged states and unsatisfied final constraints.

Reference energies and optimization results are recorded as comments at the script end. This is a one-geometry demonstration of BSSE control; the CP interaction energy changes as well, so lower BSSE alone does not prove higher accuracy.
