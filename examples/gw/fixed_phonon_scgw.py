"""H2 scGW with an externally specified one-mode vibrational model.

The phonon energy and vertices below are illustrative model inputs, not an
ab initio H2 normal mode or derivative. They live in the initial HF MO frame.
beta=8 Ha^-1 is a finite-temperature numerical example, not room temperature.
Run with PYTHONPATH=src JAX_PLATFORMS=cpu; there is no CLI or PySCF calculation.
"""

import jax
import jax.numpy as jnp
from gradscf import gto, scf, gw
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
from gradscf.gw.ep_coupling import PhononModel
from gradscf.scf.autodiff import SCFDifferentiationConfig

jax.config.update("jax_enable_x64", True)

mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-12).run()
reference = mf.scf_result
factors = eri_pair_matrix_to_df_factors(mf._scf_inputs.eri_pair_matrix, nao=2, tol=1e-12)
vertices = jnp.array([[[.004, .003], [.003, -.006]]])
quadratic = jnp.array([[[[.0001, 0.], [0., -.0001]]]])
backward = SCFDifferentiationConfig(tolerance=1e-9, max_iter=80)


def calculate(scale, differentiation=None):
    phonons = PhononModel(jnp.array([.02]), scale * vertices, quadratic,
                         reference="external illustrative model")
    return gw.scgw_matsubara_restricted(
        mo_energy=reference.mo_energy, mo_coeff=reference.mo_coeff, nocc=1,
        df_factors=factors, hcore_matrix=reference.hcore_matrix,
        nuclear_repulsion=reference.nuclear_repulsion, phonons=phonons,
        beta=8., nw=24, max_iter=200, mixing=.3, tol=1e-9, particle_tol=1e-13,
        differentiation=differentiation,
    )


result = calculate(1.)
value, derivative = jax.jit(jax.value_and_grad(
    lambda scale: calculate(scale, backward).density_mo[0, 1]))(1.)
step = 1e-3
finite_difference = (calculate(1 + step).density_mo[0, 1]
                     - calculate(1 - step).density_mo[0, 1]) / (2 * step)

print("Converged:", bool(result.converged))
print("Unmixed residual / Ha:", float(result.fixed_point_residual))
print("Particle-number error:", float(result.particle_number_error))
print("Chemical potential / Ha:", float(result.chemical_potential))
print("Density element D[0,1]:", float(value))
print("d D[0,1] / d scale (AD, FD):", float(derivative), float(finite_difference))
print("Coupled total energy:", result.total_energy)  # Deliberately undefined for the fixed bath.
assert abs(derivative - finite_difference) < 2e-6

# CPU float64 output (2026-10-04, JAX 0.8.1):
# Converged: True
# Unmixed residual / Ha: 8.631976990381833e-10
# Particle-number error: -2.930988785010413e-14
# Chemical potential / Ha: 0.045253529652736386
# Density element D[0,1]: -0.00044626729152923714
# d D[0,1] / d scale (AD, FD): -0.0008926854028821704 -0.0008926854037027109
# Coupled total energy: None
