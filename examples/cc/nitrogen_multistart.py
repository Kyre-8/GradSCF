"""Native restricted SCF branch selection before CCSD and EOM, without a CLI."""

import jax

jax.config.update("jax_enable_x64", True)
from gradscf import gto, scf, cc

mf = scf.RHF(
    gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g"),
    conv_tol=1e-12,
    conv_tol_density=1e-10,
    conv_tol_grad=1e-9,
    max_cycle=150,
)
selection = mf.multistart(amplitudes=(0.05, 0.15, 0.4), seed=20260923)
for attempt in selection.attempts:
    print(
        "rotation:",
        attempt.amplitude,
        "energy:",
        attempt.energy,
        "converged:",
        attempt.converged,
    )
if not selection.converged:
    raise RuntimeError("No finite converged SCF candidate")
mycc = cc.CCSD(selection.selected, conv_tol=1e-12, residual_tol=1e-11).run()
eom = cc.EOMEE(mycc, nroots=3, solver="davidson", max_space=48).run()
print("selected candidate:", selection.selected_index)
print("CCSD total energy (Hartree):", float(mycc.e_tot))
print("EE energies (Hartree):", eom.e)
print("isolated-root response valid:", eom.result.response_valid)
# This selects the lowest of explicit converged candidates. It neither proves
# global SCF stability nor differentiates the discrete branch-selection step.
