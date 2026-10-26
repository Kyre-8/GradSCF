"""R03: selected nonlinear field response of hydrogen fluoride.

Fixed R(HF)=1.7328 bohr, RHF / Cartesian 6-31+G*. The molecular z axis
points from H to F. For E(F)=E0-mu F-alpha F²/2-beta F³/6-gamma F⁴/24,
all four response coefficients are negative energy derivatives.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.molecular_response import MolecularEnergy


mol = gto.M(atom="H 0 0 0; F 0 0 1.7328", basis="6-31+g*", unit="Bohr", cart=True)
mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11,
             integral_backend="native").run()
energy = MolecularEnergy(mf)
coordinates = energy.coordinates
axial_energy = lambda field: energy(coordinates, jnp.array([0., 0., field]))
mu = lambda field: -jax.grad(axial_energy)(field)
alpha = jax.grad(mu)
beta = jax.grad(alpha)
gamma = jax.grad(beta)
response = jax.jit(lambda field: jnp.stack((mu(field), alpha(field), beta(field), gamma(field))))
start = perf_counter()
values = np.asarray(response(0.))
cold = perf_counter()-start
if not np.isfinite(values).all():
    raise RuntimeError("The nonlinear field-response chain did not converge.")
print("mu_z, alpha_zz, beta_zzz, gamma_zzzz / au:", values, flush=True)

# Check one mixed permutation in addition to the selected axial derivatives.
field_alpha = lambda field: -jax.hessian(energy, argnums=1)(coordinates, field)
field_beta = jax.jit(jax.jacfwd(field_alpha))(jnp.zeros(3))
if not np.isfinite(field_beta).all():
    raise RuntimeError("A mixed third field derivative is nonfinite.")
np.testing.assert_allclose(field_beta, field_beta.transpose(2, 0, 1), atol=1e-7, rtol=1e-8)
checks = []
lower_response = jax.jit(lambda field: jnp.stack((alpha(field), beta(field))))
for step in (3e-3, 1e-3, 3e-4):
    numerical = np.asarray((lower_response(step)-lower_response(-step))/(2*step))
    if not np.isfinite(numerical).all():
        raise RuntimeError(f"Nonconverged nonlinear response at field step {step}.")
    errors = np.abs(numerical-values[2:])
    checks.append(dict(field_step_au=step, beta_fd=float(numerical[0]), gamma_fd=float(numerical[1]),
                       beta_error=float(errors[0]), gamma_error=float(errors[1])))
    print(f"step={step:.1e}: beta/gamma FD errors {errors}", flush=True)
np.testing.assert_allclose(numerical, values[2:], atol=1e-5, rtol=5e-5)
fields = np.linspace(-.005, .005, 21)
dipole_function = jax.jit(mu)
dipoles = np.array([float(dipole_function(field)) for field in fields])
if not np.isfinite(dipoles).all():
    raise RuntimeError("A field-scan SCF response did not converge.")

output = Path("reproducibility/ad_showcase/R03_hf_hyperpolarizability")
output.mkdir(parents=True, exist_ok=True)
data = dict(method="RHF", basis="6-31+g*", cart=True, symbols=["H", "F"],
            coordinates_bohr=np.asarray(coordinates).tolist(),
            mu_z=float(values[0]), alpha_zz=float(values[1]), beta_zzz=float(values[2]),
            gamma_zzzz=float(values[3]), beta_tensor=np.asarray(field_beta).tolist(),
            finite_field_checks=checks, fields_au=fields.tolist(), dipoles_au=dipoles.tolist(),
            cold_four_response_seconds=cold, response_units="atomic units",
            electronic_response="implicit", origin_bohr=[0., 0., 0.])
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")

# jy_gpu_1, CPU float64, 2026-10-15:
# mu_z, alpha_zz, beta_zzz, gamma_zzzz / au:
# [-0.82079529 4.55004223 10.89022317 169.95012676]
# steps 3e-3 / 1e-3 / 3e-4 au:
# beta FD errors  [1.031e-3, 1.145e-4, 1.031e-5]
# gamma FD errors [2.573e-2, 2.859e-3, 2.573e-4]
