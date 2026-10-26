"""Water/STO-3G: compare optional QP forward algorithms at fixed G0/W0.

The physical GW approximation and implicit backward are unchanged. See
src/gradscf/gw/QP_SOLVERS.md for measured costs and convergence limits.
"""
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf import gto, scf, gw

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='sto-3g')
mf = scf.RHF(mol, conv_tol=1e-12).run()
reference = None
for solver in ('secant', 'newton', 'hybrid'):
    calculation = gw.GW(mf, nw=100, qp_solver=solver).run()
    assert calculation.converged
    print(solver)
    print('QP energies / Ha:', calculation.mo_energy)
    print('Maximum residual / Ha:', float(np.max(np.abs(calculation.result.qp_residual))))
    if reference is None:
        reference = np.asarray(calculation.mo_energy)
    else:
        np.testing.assert_allclose(calculation.mo_energy, reference, rtol=0, atol=1e-6)
# Timing this complete example includes SCF, G/W preparation and compilation.
# The dedicated benchmark isolates the repeated QP solve with prepared G/W.

# Measured on CPU float64, JAX 0.8.1; all three printed the same energies:
# [-20.03651539, -1.18601408, -.61602597, -.41729818, -.33074393,
#    .60883361, .74357588] Ha
# Maximum residual / Ha:
# secant  2.385631362594731e-10
# newton  2.8247369721068338e-14
# hybrid  2.1073306988306229e-10
