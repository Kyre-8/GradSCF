# DABNA-2 response benchmark inputs

The monomer is the unchanged 96-atom gas-phase PBEh-3c geometry used by the
existing MR-TADF campaign. Its SHA256 is
`039b06decb1c4d8c2ec3ebd4f34d9c78c13be478009f232fda252a5b5eb46f51`.
The dimer is a rigid computational benchmark, with both internal monomer
geometries preserved. It is not an optimized aggregate or a crystal structure.

| Cartesian def2-SVP space | Monomer | Rigid dimer |
| --- | ---: | ---: |
| Atoms | 96 | 192 |
| Electrons | 388 | 776 |
| AOs | 1060 | 2120 |
| Occupied orbitals | 194 | 388 |
| Virtual orbitals | 866 | 1732 |
| Complete single-excitation space | 168004 | 672016 |

`prepare_dimer.py` performs a deterministic rigid placement search. The saved
placement has a 5.50-Angstrom normal displacement, a (1.00, -0.50)-Angstrom
in-plane slip, and a 255-degree rotation about the core-plane normal. Its
minimum interfragment nuclear distance is 2.561339 Angstrom.
`dimer_provenance.json` records the transformation, search, input/output
hashes and the remote geometry check. Regeneration is simply
`python prepare_dimer.py` on the authorized compute node.

## Complete molecular TDDFT example

`dft_response.py` runs a new PBE ground-state SCF followed by the lowest 20
singlet excitations of full TDDFT. It retains Cartesian def2-SVP, the complete
occupied/virtual orbital space, and the unchanged level-3 integration grid.
AO values are evaluated in blocks; changing a workspace block size does not
remove quadrature points. Native direct J/K avoids storing four-center ERIs,
and Davidson uses response actions without constructing the complete A/B
matrices. This example does not use density fitting or TDA.

Run on the authorized compute node from the repository root:

```sh
PYTHONPATH=src python examples/tdscf/dabna2/dft_response.py
PYTHONPATH=src python examples/tdscf/dabna2/plot_spectrum.py
```

The default output directory is `artifacts/dabna2/monomer_pbe_level3/`.
`status.json` records the current stage and convergence status; a running stage
is not a completed calculation. A successful run saves `spectrum.json`,
`spectrum.npz`, and the TDDFT amplitudes. The plot displays oscillator strengths
as sticks, without a fitted linewidth or vibrational structure.

The SCF density and orbitals are saved before the response solve. A density
checkpoint may supply `run_case(..., init_density=density)` as a compatible
initial guess; it still requires a new SCF solve. To restart only TDDFT from
the converged, saved AO/grid reference, use the existing cache reader:

```python
import h5py
from gradscf import tdscf
from gradscf.data.hdf5_cache import read_restricted_molecule

with h5py.File("artifacts/dabna2/monomer_pbe_level3/scf_reference.h5") as archive:
    reference = read_restricted_molecule(archive["reference"])

td = tdscf.TDDFT(reference, xc_functional="pbe", nstates=20,
    eigensolver="davidson", davidson_tol=1e-6, davidson_max_iter=512,
    davidson_max_subspace=160, max_memory=8000.,
    response_kernel_options=dict(two_electron_mode="direct", integral_cache="none",
        direct_tol=1e-12, direct_workers=16, grid_block_size=256))
result = td.kernel()
```

The explicit `xc_functional="pbe"` is required on reload: the cache preserves
the numerical reference but does not serialize the XC specification. The
response workspace budget is in decimal MB and is not a process RSS limit.
CPU affinity, thread limits and timeouts are set by the external job runner.

The complete monomer calculation and the dimer electronic-structure calculation
must be distinguished from earlier AO-only and individual-response-action
benchmarks. See measured memory, timing and current validation limits in
[the direct-response notes](../../../src/gradscf/tddft/DIRECT_RESPONSE.md).
