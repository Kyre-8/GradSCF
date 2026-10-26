# Restricted QCISD and QCISD(T)

`gradscf.cc.QCISD` implements real, closed-shell quadratic configuration
interaction with singles and doubles. It is distinct from variational CISD,
linearized CCSD and full CCSD. The method reference is Pople, Head-Gordon and
Raghavachari, *J. Chem. Phys.* **87**, 5968–5975 (1987),
[doi:10.1063/1.453520](https://doi.org/10.1063/1.453520).

The residual is adapted from PySCF 2.9.0 `qcisd_slow.update_amps`, reusing the
existing restricted intermediates. Its physical residual is evaluated directly
with full Fock diagonals; there is no private QCI iteration, DIIS or adjoint
solver. Source attribution, hashes and licensing are in [NOTICE.md](NOTICE.md).
Tests compare against PySCF's separate optimized `qcisd.QCISD` implementation.

## Public interface

```python
from gradscf import cc, gto, scf

mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = scf.RHF(mol).run()
myqci = cc.QCISD(mf).run()  # also mf.QCISD().run()
print(myqci.e_tot, myqci.e_corr, myqci.converged)
print(myqci.e_tot + myqci.qcisd_t())
```

`kernel()` returns `(e_corr, t1, t2)`; `.run()` returns the object. Frozen
occupied/virtual orbitals and restarts use the same conventions as restricted
CC. `cc.CC(mf, method="qcisd")` is equivalent. Functional code uses
`cc.run_cc(h1, eri, nocc=..., config=cc.CCConfig(method="qcisd"))`.
The new numerical method ID is appended; existing method IDs are unchanged.

## Equations and response

QCISD adds specific quadratic terms to truncated CI equations to obtain its
size-consistent approximation. It does not keep all nonlinear CCSD terms.
The implemented residual has polynomial degree at most two in the amplitudes;
tests verify that property and energy additivity of noninteracting H4 fragments.
The energy convention is

\[
E_{\rm corr}=2\sum_{ia}F_{ia}t_i^a+
\sum_{ijab}\big[2(ia|jb)-(ib|ja)\big]t_{ij}^{ab}.
\]

In particular, it has no CCSD `t1*t1` energy contribution. The `2*Fov*t1` term
matches optimized PySCF `qcisd.energy`; its slow QCISD helper instead zeroes t1
in the energy call, which agrees for a stationary RHF reference but is not the
off-stationary convention used here. Random noncanonical Fock tests also use
the optimized implementation as the oracle. Physical molecular applications in
this stage use a converged closed-shell HF reference.

The shared nonlinear solver converges `R_QCI(t)=0` and attaches its implicit
response. The functional derivative therefore includes the QCISD amplitude
Jacobian, not the CCSD Jacobian. `solve_lambda()` solves the model-specific
stationary adjoint. `make_rdm1()` returns the real symmetric, orbital-unrelaxed
one-electron **model response density** from the QCISD Lagrangian. Its electron
count and contractions with one-electron perturbations are tested against
reconverged finite differences. It is not a QCISD(T)-corrected density.
The existing CCSD 2-RDM contractions are not applied to QCISD;
`make_rdm2()` rejects this method explicitly.

## The triples correction

`myqci.qcisd_t()` and `myqci.triples()` select QCISD(T). The functional interface
requires the explicit `variant="qcisd(t)"` in `evaluate_triples` or
`triples_correction`. The canonical restricted permutation kernel is shared,
but the singles-dependent weight is **twice** the CCSD(T) weight for the same
amplitudes. Converged QCI and CCSD amplitudes also differ; doubling a CCSD(T)
correction is not QCISD(T).

The implementation follows the working contractions in
[PySCF qcisd_t_slow](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/qcisd_t_slow.py),
including its F_vo*T2 term. The active Fock matrix must satisfy the canonical
check. `TriplesResult` contains connected and applied singles components that
sum to the correction, with `variant_id=2`. A QCI correction requires matching
converged QCI amplitudes; a CCSD correction requires CCSD amplitudes. The
facade rejects cross-model requests and stale reference/frozen settings, while
invalid low-level results are marked invalid/NaN, including empty triples spaces.

For total derivatives, add the QCI triples correction inside the differentiated
function, preserving the returned QCISD state's amplitude response. The default
functional triples variant is still CCSD(T), so it must not be omitted for QCI.

## Scope and validation

This stage covers real restricted molecular QCISD and canonical QCISD(T),
full-MO in-core integrals, frozen spaces, JIT and first-order fixed-MO integral
response. Unrestricted/ROHF QCI, semicanonical QCI triples, QCISD(TQ), QCI 2-RDMs,
triples-corrected densities and complete nuclear gradients are not implemented.
The canonical correction retains the 20,000-virtual-triples default limit and
does not materialize a full six-index T3 tensor. This is not a DF/local QCI engine.

`tests/cc/test_qcisd.py` includes random-amplitude residual/energy oracles for
canonical and noncanonical blocks, H4/STO-3G full/frozen energies and amplitudes,
H2O/STO-3G energies, triples weighting checks, quadratic residual degree,
fragment additivity, energy/amplitude/density response, and failure/scope guards.
Energy tolerances are 2e-9 Hartree, triples 2e-10 Hartree, random residuals 2e-12,
and finite-difference derivatives 2e-7 with step 1e-4. These are implementation
validation tolerances, not physical error estimates.

The standalone [example](../../../examples/cc/restricted_qcisd.py) uses native
GradSCF RHF throughout. Executed results are recorded in [VALIDATION.md](VALIDATION.md).
