# FCI references and implementation provenance

| Reference | Role |
| --- | --- |
| P. J. Knowles and N. C. Handy, *A new determinant-based full configuration interaction method*, Chem. Phys. Lett. **111**, 315–321 (1984). DOI [10.1016/0009-2614(84)85513-X](https://doi.org/10.1016/0009-2614(84)85513-X) | Determinant/string-based direct CI as a foundation for complete active spaces. This implementation uses explicit JAX spin-free excitation actions, not the original optimized program. |
| Q. Sun et al., *Recent developments in the PySCF program package*, J. Chem. Phys. **153**, 024109 (2020). DOI [10.1063/5.0006074](https://doi.org/10.1063/5.0006074) | Reference software and API conventions; separate from the theoretical definition of FCI. |
| [PySCF FCI API](https://pyscf.org/pyscf_api_docs/pyscf.fci.html) and installed PySCF 2.9.0 `fci/direct_spin1.py`, `cistring.py`, `spin_op.py` | Independent Hamiltonian actions, spin diagnostics, energy/RDM/transition-RDM numerical oracles and alpha/beta ordering conventions. |

No PySCF solver runs in the GradSCF production path. The JAX contractions and
resource guards are implemented here; the existing GradSCF fermionic phase
helper is shared by CI and FCI. Numerical forward and backward rules are reused
from `gradscf.solvers`. No additional method-local eigen/linear solver is added.

The spin-free identity and RDM definitions are written explicitly in README.md.
Named formulas, software API resemblance and executed numerical validation are
kept distinct. A forward PySCF comparison does not establish GPU performance,
complex-orbital support, or complete nuclear/higher derivatives.
