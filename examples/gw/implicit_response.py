"""H2: implicit response of converged evGW, evGW0 and qsGW.

The derivative is with respect to a scale of the AO Coulomb factors, with
the starting HF arrays held fixed. It is neither a nuclear derivative nor
a derivative through HF. Each finite-difference point reconverges GW.
Run with JAX_PLATFORMS=cpu for the float64 CPU reference calculation.
"""

import jax
import jax.numpy as jnp
from gradscf import gto, scf, gw
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
from gradscf.scf.autodiff import SCFDifferentiationConfig

jax.config.update("jax_enable_x64", True)

mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-12).run()
reference = mf.scf_result
factors = eri_pair_matrix_to_df_factors(mf._scf_inputs.eri_pair_matrix, nao=2, tol=1e-12)
response = SCFDifferentiationConfig(tolerance=1e-10, max_iter=100)


def homo(scale, method, differentiation):
    common = dict(mo_energy=reference.mo_energy, mo_coeff=reference.mo_coeff,
                  nocc=1, df_factors=scale * factors,
                  hcore_matrix=reference.hcore_matrix,
                  nw=48, max_iter=100, tol=1e-10, damping=.3,
                  differentiation=differentiation)
    if method == "qsgw":
        result = gw.qsgw_cd_restricted(**common, tol_density=1e-10)
    else:
        result = gw.evgw_cd_restricted(**common, update_w=method == "evgw",
            fock_matrix=reference.fock_matrix, density_matrix=reference.density_matrix)
    return result.mo_energy[0]


step = 1e-4
for method in ("evgw", "evgw0", "qsgw"):
    energy, derivative = jax.jit(jax.value_and_grad(lambda x: homo(x, method, response)))(1.)
    finite_difference = (homo(1 + step, method, None) - homo(1 - step, method, None)) / (2 * step)
    print(f"{method:6s}: HOMO = {float(energy):.10f} Ha; "
          f"AD = {float(derivative):.10f}; FD = {float(finite_difference):.10f} Ha/scale")
    assert jnp.abs(derivative - finite_difference) < 2e-6

# CPU float64 output (2026-10-04):
# evgw  : HOMO = -0.5967014247 Ha; AD = 1.2977251783; FD = 1.2977251714 Ha/scale
# evgw0 : HOMO = -0.5968406353 Ha; AD = 1.2935202972; FD = 1.2935202908 Ha/scale
# qsgw  : HOMO = -0.5967014247 Ha; AD = 1.2977251783; FD = 1.2977251714 Ha/scale
# This two-orbital example has coincident evGW/qsGW HOMO values; their methods
# are different. Tests also cover off-diagonal perturbations and orbital rotations.
