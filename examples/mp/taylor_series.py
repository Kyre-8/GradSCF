"""Generate MP2 through MP6 from one residual and normalized Taylor series.

H6/STO-3G, native GradSCF RHF, real canonical MOs and CPU float64.
The generic engine uses excitation-limited determinant connections, not a full
FCI eigensolve. A high order can exhaust the finite excitation space of a small
molecule; this does not make a partial sum a converged or variational energy.
"""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, mp

mol = gto.M(atom="H 0 0 0; H 0 0 .8; H 0 0 1.7; H 0 0 2.6; H 0 0 3.5; H 0 0 4.4",
            basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200).run()
if not mf.converged:
    raise RuntimeError("Converge the canonical HF reference first")

pt = mp.MP(mf, order=6, with_coefficients=True).run()
print("RHF / Ha: %.12f" % mf.e_tot)
print("Retained determinants:", pt.space.size)
print("Maximum excitation rank:", max(pt.space.ranks))
print("Wavefunction coefficients through order:", pt.wavefunction_coefficients.shape[0]-1)
print("Linear response residual / Ha: %.3e" % pt.result.residual_norm)

total = float(mf.e_tot)
for order, correction in enumerate(np.asarray(pt.corrections), 2):
    total += correction
    print("MP%d  correction = % .12f  partial total = % .12f" % (order, correction, total))

# Independent existing low-order kernels remain available as reference paths.
np.testing.assert_allclose(pt.e2, mp.MP2(mf).run().e2, atol=1e-10)
np.testing.assert_allclose(pt.e3, mp.MP3(mf).run().e3, atol=1e-10)

# Measured output (CPU, JAX 0.8.1, float64):
# RHF / Ha: -3.172676142364
# Retained determinants: 400
# Maximum excitation rank: 6
# Wavefunction coefficients through order: 3
# Linear response residual / Ha: 4.813e-17
# MP2  correction = -0.051659165678  partial total = -3.224335308042
# MP3  correction = -0.016557683628  partial total = -3.240892991670
# MP4  correction = -0.006267145387  partial total = -3.247160137057
# MP5  correction = -0.002416510625  partial total = -3.249576647682
# MP6  correction = -0.000954427910  partial total = -3.250531075592
