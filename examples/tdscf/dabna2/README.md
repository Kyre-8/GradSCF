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

The current electronic-structure validation is limited to one matched-orbital
monomer response action. A 20-root spectrum and the dimer electronic-structure
run have not completed. See the measured memory, timing and remaining limits
in [the direct-response notes](../../../src/gradscf/tddft/DIRECT_RESPONSE.md).
