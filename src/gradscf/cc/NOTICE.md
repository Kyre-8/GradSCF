# Adapted PySCF code

The following numerical contraction code is adapted from **PySCF 2.9.0**,
licensed under Apache License 2.0. The upstream copyright notices apply to these
adaptations. A copy of the license is distributed as [LICENSE.pyscf](LICENSE.pyscf).
Other original GradSCF files retain the repository's license.

| GradSCF file | Upstream source | Adaptation |
| --- | --- | --- |
| `_equations.py` | [`pyscf/cc/rccsd.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/rccsd.py), `update_amps` | JAX contractions and functional diagonal updates; return the physical residual instead of divided amplitude updates; real in-core blocks; no PySCF object/dependency |
| `_intermediates.py` | [`pyscf/cc/rintermediates.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/rintermediates.py), nine restricted intermediates | JAX array/contraction operations and direct full-MO blocks |
| `triples.py` | [`pyscf/cc/ccsd_t_slow.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/ccsd_t_slow.py), `kernel` and `r3` | Virtual triples streamed in a JAX loop; permutation-table contraction replaces the expanded 36 terms; no logger or PySCF runtime |
| `_spin_equations.py` | [`pyscf/cc/gccsd.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/gccsd.py), `update_amps` and energy contractions | Real JAX physical residual with full Fock diagonal; no denominator division, solver loop, logging or PySCF object |
| `_spin_intermediates.py` | [`pyscf/cc/gintermediates.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/gintermediates.py), `make_tau` and six `cc_` intermediates | JAX operations on in-core antisymmetrized spin-orbital blocks |
| `_spin_triples.py` | [`pyscf/cc/gccsd_t.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/gccsd_t.py), `kernel` | Real canonical collinear inputs, static JAX loop, spin/Pauli-allowed denominator diagnostics, convergence/symmetry checks, separated connected/singles terms |
| `_spin_triples.py`, `_general_correction` | [`pyscf/cc/gccsd_t_slow.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/gccsd_t_slow.py), `kernel` moment contractions | JAX W/V moments including F_vo*T2; common tensor resolvent replaces diagonal division to support noncanonical inputs and degenerate-factor response; explicit capacity guard |
| `_streamed_triples.py` | The same `gccsd_t_slow` moment equations; current PySCF `ccsd_t_rdm.py`/`ccsd_t_lambda.py` consulted for bounded triples workspaces | Independently derived panel JVP/VJP and second response from D X = W; full occupied/virtual matrix cotangents, recomputed moment pullbacks and neighbouring panels; restricted bridge expands only needed integral blocks |
| `_spin_density.py` | [`pyscf/cc/gccsd_rdm.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/gccsd_rdm.py), `_gamma1_intermediates`, `_gamma2_intermediates`, active portions of `_make_rdm1/2` | Pure real JAX contractions and functional block assembly; reference/frozen terms restored separately by GradSCF |
| `_qci_equations.py` | [`pyscf/cc/qcisd_slow.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/qcisd_slow.py), `update_amps` | Shared JAX intermediates with QCI singles selection; full Fock diagonals produce physical residuals directly; no iterative update/denominator division |
| `rccsd.py`, QCI energy branch | [`pyscf/cc/qcisd.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/qcisd.py), `energy` | Use tau=t2, retain 2*Fov*t1; reuse shared energy contractions |
| `triples.py`, QCI variant | [`pyscf/cc/qcisd_t_slow.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/qcisd_t_slow.py), `kernel` | QCI singles weight and F_vo*T2 in the shared streamed permutation kernel; QCI-specific residual/method validation |
| `eom/_charged.py` | [`pyscf/cc/eom_rccsd.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/eom_rccsd.py), `ipccsd_matvec`, `eaccsd_matvec`, charged `_IMDS` blocks | Unpartitioned real spatial JAX contractions; functional packed actions; no PySCF runtime or eigensolver; EA virtual loop becomes a full contraction |
| `eom/_intermediates.py` | [`pyscf/cc/rintermediates.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/rintermediates.py), twelve EOM W intermediates | JAX contractions and direct full-MO blocks; reuse existing ground intermediates; retain the Wvvvv*t1 term at zero t1 so its derivative is correct |

SHA-256 hashes of the installed upstream source files used in this adaptation:

```text
eom_rccsd.py     9dd17a373014ebfed6a341734acdbe1a10fb0713a4054aebdb7685033a98ef9a
rccsd.py         35391234363b6ae26893891e18c468e636458ac209bfdad9606c15112457a552
rintermediates.py c7f3311b15c3482315bc5c8aaf570334dd71170c4b9a901ad6968bd97c903666
ccsd_t_slow.py   4fd3638e0176639781ecb138575a226912c99301a1923b60d20351eec36ac960
gccsd.py         6f01125b4dbac4af88034655cf3ac4ca8535c0c76336a5ae1e19e67f6840c42f
gintermediates.py f8cbb93d5bc92eca817d3e29dd79795843fff9dcaef2f5ec0d8394555136083e
gccsd_t.py       e9c07fc2ccbb3f31364d66330e6d4c83607a63a5929844cb81f0940c0802b4f3
gccsd_t_slow.py  4b09954250221dab24ee8e62c5616bb83c01155c97f4c01ee292015d9b24425e
gccsd_rdm.py     abdda0be8065755cf28767a646ca72b70fdae3c080ee9dc4ca47e8ca3696548f
qcisd_slow.py    ad688e8e5e0829bac41b78e4e423dbafb64f8ec23779123aca788faf49cd3886
qcisd.py         ea47d219df5418a67af5ec59db0f44711cad44723f9bcc3b42c4fb77f84aee90
qcisd_t_slow.py  84506867d3bc10c1c74fe516507283f4833c3172fb8f3642a8e2be76ad3db379
```

PySCF is a numerical comparison dependency in tests, not a dependency of these
production kernels. See [REFERENCES.md](REFERENCES.md) for scientific method
attribution, which is distinct from the source-code attribution above.

The restricted IP/EA Dyson endpoint contractions in `eom/properties.py` are
adapted separately from PySCF **2.13.0**,
[`cc/momgfccsd.py`](https://github.com/pyscf/pyscf/blob/v2.13.0/pyscf/cc/momgfccsd.py),
`build_bra_hole`, `build_bra_part`, `build_ket_hole`, and `build_ket_part`.
The installed upstream file SHA-256 is
`eefcee6acdfbe5e88aaa38fc10903c2f78aa6ce3e6183b8b711f8857ad2344fb`.
Only the physical endpoint contractions are adapted to real JAX arrays and
the existing EOM packing. The particle-moment outer minus sign is absorbed in
the particle ket. No moment reconstruction or numerical solver is copied.

OpenMolcas was subsequently consulted for triples-method definitions and
iteration controls. Its inspected revision and source hashes are recorded in
[OPENMOLCAS.md](OPENMOLCAS.md). No OpenMolcas Fortran source was incorporated.
