"""X14: H2 EOM-IP/EA energies and Dyson weights through the complete HF/CC chain.

STO-3G has two spatial orbitals here. CCSD and both charged EOM spaces are
complete for this particular two-electron example, allowing an exact FCI check.
This statement does not extend to EA-CCSD in a larger H2 basis. The weights are
biorthogonal residues, not squared right-EOM amplitudes or GW residues.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import cc, gto, scf
from gradscf.ao2mo import transform_integrals
from gradscf.solvers import LinearSolverConfig
from gradscf_tools.molecular_response import MolecularEnergy


output = Path("reproducibility/ad_showcase/X14_h2_eom_dyson")
output.mkdir(parents=True, exist_ok=True)
mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-13,
             conv_tol_density=1e-12).run()
reference = MolecularEnergy(mf, gradient_tolerance=1e-11)
coordinates = reference.coordinates
direction = jnp.array([[0., 0., -.5], [0., 0., .5]])  # dR_HH/dt = 1
cc_config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
response_config = LinearSolverConfig(method="direct", rtol=1e-11, max_dense=3)


def calculate(coords):
    state, inputs = reference.solve_with_inputs(coords)
    # The stationary occupied/virtual frame moves with the AO metric. CCSD
    # and EOM use this noncanonical frame; there is no frozen-MO derivative.
    h1, eri = transform_integrals(inputs["hcore"], state.mo_coeff, eri=inputs["eri"])
    ground = cc.run_cc(h1, eri, nocc=1, config=cc_config)
    left = cc.solve_lambda(h1, eri, ground, nocc=1, config=cc_config)
    energies, weights, diagnostics = [], [], {}
    valid = state.stationary & ground.converged & left.converged
    diagnostics["cc_residual"] = ground.residual_norm
    diagnostics["lambda_residual"] = left.residual_norm
    for sector in ("ip", "ea"):
        cfg = cc.EOMConfig(sector=sector, nroots=2, conv_tol=1e-11,
                           vector_response=True, response_config=response_config)
        result = cc.run_eom(h1, eri, ground, nocc=1, config=cfg, cc_config=cc_config)
        dyson = cc.dyson_orbitals(ground, left, result)
        valid &= jnp.all(dyson.valid & dyson.response_valid)
        energies.append(result.energies)
        weights.append(dyson.weights)
        diagnostics[sector+"_right_residual"] = jnp.max(result.residual_norms)
        diagnostics[sector+"_left_residual"] = jnp.max(result.left_residual_norms)
        diagnostics[sector+"_gap"] = jnp.min(result.spectral_gaps)
        diagnostics[sector+"_condition"] = jnp.max(result.condition_numbers)
    values = jnp.concatenate(energies+weights)  # IP, EA, one-spin IP/EA weights
    return jnp.where(valid, values, jnp.nan), diagnostics


forward = jax.jit(lambda coords: calculate(coords)[0])
response = jax.jit(lambda coords: jax.jvp(forward, (coords,), (direction,)))
start = perf_counter()
values = np.asarray(forward(coordinates))
cold_forward = perf_counter()-start
start = perf_counter()
_, tangent = response(coordinates)
tangent = np.asarray(tangent)
cold_response = perf_counter()-start
start = perf_counter()
np.asarray(response(coordinates)[1])
warm_response = perf_counter()-start
if not np.all(np.isfinite(values)) or not np.all(np.isfinite(tangent)):
    raise RuntimeError("The HF, CC, Lambda, EOM or response solve failed.")

checks = []
for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    plus = np.asarray(forward(coordinates+step*direction))
    minus = np.asarray(forward(coordinates-step*direction))
    numerical = (plus-minus)/(2*step)
    checks.append(dict(step_bohr=step, plus_values=plus.tolist(), minus_values=minus.tolist(),
                       derivative=numerical.tolist(),
                       max_error=float(np.max(np.abs(numerical-tangent)))))
    print(f"h={step:.1e} bohr: complete-chain error {checks[-1]['max_error']:.3e}", flush=True)
np.testing.assert_allclose(tangent, checks[-1]["derivative"], atol=2e-7, rtol=2e-6)
np.testing.assert_allclose(values[4:6].sum(), 1., atol=2e-10)
np.testing.assert_allclose(values[6:8].sum(), 1., atol=2e-10)
np.testing.assert_allclose([tangent[4:6].sum(), tangent[6:8].sum()], 0., atol=2e-9)

cotangent = jnp.array([.3, -.2, .2, .1, .7, -.4, .6, -.3])
reverse = jax.jit(jax.grad(lambda t: cotangent @ forward(coordinates+t*direction)))(0.)
np.testing.assert_allclose(reverse, cotangent @ tangent, atol=2e-9)
_, diagnostics = calculate(coordinates)
record = dict(basis="sto-3g", cart=True, bond_bohr=1.4,
    coordinates_bohr=np.asarray(coordinates).tolist(), direction=np.asarray(direction).tolist(),
    energy_convention="IP=E(N-1)-E(N); EA=E(N+1)-E(N)", weight_convention="one spin",
    labels=["IP1", "IP2", "EA1", "EA2", "S_IP1", "S_IP2", "S_EA1", "S_EA2"],
    values=values.tolist(), derivative_per_bohr=tangent.tolist(), finite_differences=checks,
    reverse_check_error=float(abs(reverse-cotangent @ tangent)),
    diagnostics={key: float(value) for key, value in diagnostics.items()},
    cold_forward_seconds=cold_forward, cold_response_seconds=cold_response,
    warm_response_seconds=warm_response)
(output/"gradscf.json").write_text(json.dumps(record, indent=2)+"\n")
for label, value, derivative in zip(record["labels"], values, tangent):
    print(f"{label:6s} {value: .12f}   d/dR_HH={derivative: .12f}", flush=True)

# Measured on jy_gpu_1, CPU float64, 2026-10-16; energies in Hartree:
# IP1    0.598764596067   d/dR_HH=-0.195676302117
# IP2    1.375959358529   d/dR_HH=-0.847922908991
# EA1    0.690829386828   d/dR_HH=-0.522798046336
# EA2    1.490925411647   d/dR_HH=-1.171474761025
# Both sectors have one-spin weights [0.987295204908, 0.012704795092]
# and weight derivatives [-0.024160241500, +0.024160241500] / bohr.
# Native reconverged FD errors at h=1e-3,3e-4,1e-4,3e-5,1e-5 bohr:
# [2.703e-7, 2.432e-8, 2.679e-9, 1.164e-10, 1.830e-10].
# Independent FCI: max value difference 9.992e-16; final FD error 1.608e-10.
# Cold forward / cold response / warm response: 18.407 / 20.561 / 0.002236 s.
