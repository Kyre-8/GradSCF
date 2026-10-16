# CI theory, numerical methods, and reference software

Bibliographic identifiers were checked against publisher records and the PySCF
authors' [software paper](https://arxiv.org/abs/2002.12531) on 2026-09-21.
Machine-readable entries are in [references.bib](references.bib).
Theory citations, software comparisons, and repository code reuse serve
different purposes and are identified separately below.

## Reference-to-implementation map

| Code or claim | Reference | Role and scope |
| --- | --- | --- |
| `hamiltonian.py`: one-/two-electron matrix elements between determinants | Slater1929; Condon1930 | Original theoretical foundations of the Slater–Condon rules; not a claim to reproduce either paper's program |
| `space.py`, `solve_ci`: rank-truncated CI and FCI limit | Sherrill1999 | Review of the CI hierarchy, determinant versus spin-adapted spaces, and computational formulations |
| `solve_cis`: singles states and spin-adapted CIS | Foresman1992 | Established CIS theory/application reference; not a claim that CIS was first introduced in 1992 |
| `corrections.py`, `tddft/cisd.py`: canonical singlet CIS(D) | HeadGordon1994 | Original CIS(D) method; the explicit working expression used by the independent test is also documented in the Q-Chem manual below |
| CI eigenvalue/eigenvector differentiation through `solvers` | Xie2020 | Related implicit eigensolver differentiation theory; method-level coverage is established by repository tests, not by the citation. General CI/CIS(D) retains its first-order scope; isolated singlet CIS has separate second-response validation. |
| PySCF CISD, HF-TDA and FCI test oracles; public API conventions | Sun2020 | Reference software attribution, separate from the theoretical definition of CI or CIS(D) |
| `make_uci_space`, UCISD/UCISDT/UCISDTQ | Slater1929; Condon1930; Sherrill1999 | The same determinant CI hierarchy in a fixed (N-alpha,N-beta) sector with separate orbital frames; no new perturbation model or spin adaptation |
| `solve_ucis` | Singles projection of the same Hamiltonian, subtracting the HF determinant energy | PySCF UHF/TDA comparison for a stationary UHF reference; this does not implement spin-flip CIS or a ROHF response theory |
| `CID`, `UCID`, explicit excitation ranks | Reference plus all double substitutions; same Slater–Condon Hamiltonian and variational principle | Independent rank-projected PySCF FCI matrix, including frozen spaces; no pair-only or CCD interpretation |
| `spin_square` | S^2 = Sz^2 + (S+ S- + S- S+)/2, contracted with normalized mixed-spin 2-RDM and cross-frame overlaps | PySCF `fci.spin_op` oracle for common/distinct frames, pure-spin states and frozen electrons; diagnostics only |

The production singlet CIS(D) adapter reuses the existing repository function
`gradscf.tddft.cisd.restricted_cisd_second_order_correction`. The project-level
[code-origins statement](../../../README.md#code-origins) applies to that existing
code. The new CI tests use PySCF to supply comparison results and FCI Hamiltonian
matrices; they do not call an ORCA or Q-Chem executable. Naming a software manual
as an equation reference does not establish numerical agreement with that code.

## Bibliography

1. **Slater1929** — J. C. Slater, “The Theory of Complex Spectra,”
   *Physical Review* **34**, 1293 (1929).
   [DOI: 10.1103/PhysRev.34.1293](https://doi.org/10.1103/PhysRev.34.1293).
2. **Condon1930** — E. U. Condon, “The Theory of Complex Spectra,”
   *Physical Review* **36**, 1121 (1930).
   [DOI: 10.1103/PhysRev.36.1121](https://doi.org/10.1103/PhysRev.36.1121).
3. **Foresman1992** — J. B. Foresman, M. Head-Gordon, J. A. Pople, and
   M. J. Frisch, “Toward a systematic molecular orbital theory for excited
   states,” *The Journal of Physical Chemistry* **96**, 135–149 (1992).
   [DOI: 10.1021/j100180a030](https://doi.org/10.1021/j100180a030).
4. **HeadGordon1994** — M. Head-Gordon, R. J. Rico, M. Oumi, and T. J. Lee,
   “A doubles correction to electronic excited states from configuration
   interaction in the space of single substitutions,” *Chemical Physics Letters*
   **219**, 21–29 (1994).
   [DOI: 10.1016/0009-2614(94)00070-0](https://doi.org/10.1016/0009-2614%2894%2900070-0).
5. **Sherrill1999** — C. D. Sherrill and H. F. Schaefer III, “The Configuration
   Interaction Method: Advances in Highly Correlated Approaches,”
   *Advances in Quantum Chemistry* **34**, 143–269 (1999).
   [DOI: 10.1016/S0065-3276(08)60532-8](https://doi.org/10.1016/S0065-3276%2808%2960532-8).
6. **Xie2020** — H. Xie, J.-G. Liu, and L. Wang, “Automatic differentiation of
   dominant eigensolver and its applications in quantum physics,”
   *Physical Review B* **101**, 245139 (2020).
   [DOI: 10.1103/PhysRevB.101.245139](https://doi.org/10.1103/PhysRevB.101.245139).
7. **Sun2020** — Q. Sun, X. Zhang, S. Banerjee, et al., “Recent developments in
   the PySCF program package,” *The Journal of Chemical Physics* **153**, 024109
   (2020). [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074).

## Equation-level companion and validation boundaries

The [Q-Chem 5.1 manual, section 7.6.1, equations 7.38–7.40](https://manual.q-chem.com/5.1/sect-excorr.html)
provides the working decomposition used in `tests/ci/test_cis_d.py`:

```text
delta omega = <CIS|V|U2 HF> + <CIS|V|T2 U1 HF> - E_MP2
```

The test assembles the doubles contribution and disconnected-triples term in a
determinant basis, then subtracts the MP2 ground-state contribution. It is an
independent implementation of this expression using PySCF FCI matrix elements,
not a direct comparison to a separate package's CIS(D) routine. The 1994 paper
supplies the method attribution; the manual supplies the inspected equation-level
companion. Publisher metadata/abstract checks are not a claim that the full
original articles were audited equation by equation.

The cited CIS(D) test validates the real RHF **singlet** path and specified frozen
spaces. It does not validate every use of the older correction function with
TDDFT amplitudes, empirical double-hybrid scaling, or unrestricted references.
The normalization conversion (unit spatial CIS norm to restricted-TDA norm
squared 1/2) is an implementation convention documented in the CI README and
checked against the determinant oracle.

`CISDTQ` denotes an explicit variational singles-through-quadruples space.
Unspecified perturbative names such as `CISD(T)` or `CISDT(Q)` must not be
attributed to these references or exported as implemented methods. A future
correction needs its own definition, source, denominators and validation.

The [PySCF CI API](https://pyscf.org/user/ci.html) and
[Psi4 DETCI manual](https://psicode.org/psi4manual/master/detci.html) remain useful
software documentation, but do not replace the method bibliography.
