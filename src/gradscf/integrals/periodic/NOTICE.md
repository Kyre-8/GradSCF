# GTH data provenance

`gth_data.json` contains basis and pseudopotential parameters exported from the
PySCF 2.9.0 distribution (`pyscf.pbc.gto.basis` and `pyscf.pbc.gto.pseudo`).
The snapshot contains 25 basis families and 378 valid element/family entries,
covering 71 distinct elements. It includes the original GTH SZV/DZV/DZVP/TZVP/
TZV2P/QZV2P/QZV3P families, augmented and correlation-consistent variants, and
MOLOPT/MOLOPT-SR families. Coverage differs by family; use
`gradscf.pbc.gto.basis_names(element)` and `basis_elements(name)` to query it.
PySCF-style family names and per-element CP2K header aliases are supported,
ignoring case, spaces, hyphens and underscores. No data are downloaded at runtime.

The original default gth-pade/gth-pbe/gth-blyp pseudopotentials are unchanged.
74 additional core/valence variants are available through explicit `-qN`
suffixes. Basis headers with a known valence count are checked against the
chosen potential; mismatches are rejected rather than silently selecting a
different Hamiltonian. Legacy basis headers without q labels have unknown
valence metadata. Raw basis/parameter dictionaries remain accepted; the caller
owns their physical pairing.

`provenance` in the JSON records upstream version, URL, and SHA256 hashes of
each input file. Reproduce the snapshot with the test-only PySCF 2.9.0 tool:

```sh
python tests/pbc/export_gth_data.py --check
```

Two upstream oxygen blocks, `gth-aug-tzvp/O` and `gth-aug-tzv2p/O`, have incomplete
contraction rows and are rejected by the upstream parser. They are explicitly
listed in `unavailable` and not exported as valid bases. No coefficients are
invented or repaired. Existing default basis and potential values are checked
for exact preservation during export.

Data availability does not imply complete-basis convergence, a suitable FFT
grid, a matching XC parametrization, or all-electron/relativistic capability.
In particular, this expansion does not add PAW/USPP or spin-orbit operators.

PySCF is copyright The PySCF Developers and distributed under Apache License
2.0. The full license is bundled at
`gradscf/integrals/_native/vendor/pyscf/LICENSE` in this distribution.
Upstream source: https://github.com/pyscf/pyscf/tree/v2.9.0/pyscf/pbc/gto

Numerical algorithms in this directory implement analytic Cartesian Gaussian
Fourier transforms, normalized GTH projectors, Ewald summation and FFT Coulomb
convolution. No PySCF Python API is imported at runtime.
