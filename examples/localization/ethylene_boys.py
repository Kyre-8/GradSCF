"""Foster-Boys localization of the occupied orbitals of ethylene.

GradSCF supplies RHF orbitals and AO integrals. After RHF, the occupied space
is fixed: AD differentiates only orbital rotations, not SCF or coordinates.
Position integrals are in Bohr and Boys scores in Bohr**2.
The resulting occupied MO coefficients are stored in c_boys and the rotation
in u_opt. rotation_history records the selected optimization trajectory.
This script has no CLI and does not write files.

Reference: Foster and Boys, Rev. Mod. Phys. 32, 300 (1960).
https://doi.org/10.1103/RevModPhys.32.300
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from scipy.optimize import minimize

from gradscf import dft, gto, integrals

jax.config.update("jax_enable_x64", True)

# Run RHF once. All eight occupied orbitals, including the C 1s cores, are
# localized together. This is a fixed demonstration geometry, in Angstrom.
mol = gto.M(
    atom="""
    C  -0.6695   0.0000   0.0000
    C   0.6695   0.0000   0.0000
    H  -1.2321  -0.9289   0.0000
    H  -1.2321   0.9289   0.0000
    H   1.2321  -0.9289   0.0000
    H   1.2321   0.9289   0.0000
    """,
    basis="cc-pvdz",
    unit="Angstrom",
    cart=True,
)
mf = dft.RKS(mol, xc="hf", integral_backend="native", conv_tol=1e-12).run()
assert mf.converged
c_occ = jnp.asarray(mf.mo_coeff[:, np.asarray(mf.mo_occ) > 0])
nocc = c_occ.shape[1]

topology, parameters = integrals.prepare_basis(
    mol.atom, mol.basis, unit=mol.unit, cart=mol.cart,
)
plan = integrals.make_plan(topology, backend="native")
overlap = plan.evaluate("overlap", parameters)
dipole_ao = plan.evaluate("dipole", parameters, origin=jnp.zeros(3))
dipole_mo = jnp.einsum("mp,xmn,nq->xpq", c_occ, dipole_ao, c_occ)

# U = exp(K), with K[p,q] = theta and K[q,p] = -theta for p < q.
p, q = np.triu_indices(nocc, 1)
nparam = len(p)


def rotation(theta):
    kappa = jnp.zeros((nocc, nocc), dtype=theta.dtype)
    kappa = kappa.at[p, q].set(theta).at[q, p].set(-theta)
    return expm(kappa)


def boys_loss(theta, dipoles):
    u = rotation(theta)
    rotated = jnp.einsum("pi,xpq,qj->xij", u, dipoles, u)
    centers = jnp.diagonal(rotated, axis1=1, axis2=2)
    return -jnp.sum(centers**2)


value_and_grad = jax.jit(jax.value_and_grad(boys_loss))
gradient = jax.jit(jax.grad(boys_loss))


@jax.jit
def hessp(theta, vector, dipoles):
    return jax.jvp(lambda t: gradient(t, dipoles), (theta,), (vector,))[1]


def objective(theta, dipoles):
    value, grad = value_and_grad(jnp.asarray(theta), dipoles)
    return float(value), np.asarray(grad)


# Canonical ethylene orbitals can be a high-symmetry stationary point.
# Compare multiple starts; a small gradient alone does not establish a minimum.
rng = np.random.default_rng(0)
starts = [np.zeros(nparam)]
starts += [1e-3 * rng.normal(size=nparam) for _ in range(7)]
results = []
histories = []
for theta0 in starts:
    history = [theta0.copy()]
    result = minimize(
        objective, theta0, args=(dipole_mo,), jac=True, method="L-BFGS-B",
        callback=lambda theta: history.append(np.array(theta, copy=True)),
        options={"gtol": 1e-9, "ftol": 1e-15, "maxiter": 500, "maxls": 40},
    )
    results.append(result)
    histories.append(history)
    print(f"Boys score = {-result.fun:.12f}, iterations = {result.nit}")

best_index = min(range(len(results)), key=lambda i: results[i].fun)
best = results[best_index]
u_base = rotation(jnp.asarray(best.x))
rotation_history = [np.asarray(rotation(jnp.asarray(t))) for t in histories[best_index]]
iteration_stages = ["Initial"] + ["L-BFGS-B"] * (len(rotation_history) - 1)

# Refine in a fresh local rotation chart. Newton-CG uses the AD HVP directly.
dipoles_base = jnp.einsum("pi,xpq,qj->xij", u_base, dipole_mo, u_base)
polish_history = []
polished = minimize(
    objective, np.zeros(nparam), args=(dipoles_base,), jac=True,
    hessp=hessp, method="Newton-CG",
    callback=lambda theta: polish_history.append(np.array(theta, copy=True)),
    options={"xtol": 1e-14, "maxiter": 100},
)
u_opt = u_base @ rotation(jnp.asarray(polished.x))
rotation_history += [np.asarray(u_base @ rotation(jnp.asarray(t))) for t in polish_history]
iteration_stages += ["Newton-CG"] * len(polish_history)
c_boys = c_occ @ u_opt
dipoles_local = jnp.einsum("pi,xpq,qj->xij", u_opt, dipole_mo, u_opt)
local_gradient = gradient(jnp.zeros(nparam), dipoles_local)

# Occupied rotations preserve S-orthogonality and the RHF density.
orth_error = jnp.linalg.norm(c_boys.T @ overlap @ c_boys - jnp.eye(nocc))
density_error = jnp.linalg.norm(2 * (c_boys @ c_boys.T - c_occ @ c_occ.T))
score_initial = -boys_loss(jnp.zeros(nparam), dipole_mo)
score_final = jnp.sum(jnp.diagonal(dipoles_local, axis1=1, axis2=2)**2)
print(f"RHF energy / Ha       = {float(mf.e_tot):.12f}")
print(f"Initial Boys score    = {float(score_initial):.12f}")
print(f"Final Boys score      = {float(score_final):.12f}")
print(f"Local gradient norm   = {float(jnp.linalg.norm(local_gradient)):.3e}")
print(f"Orthogonality error   = {float(orth_error):.3e}")
print(f"Density error         = {float(density_error):.3e}")
assert orth_error < 1e-10
assert density_error < 1e-10
assert jnp.linalg.norm(local_gradient) < 1e-8
assert score_final > score_initial + 1e-6

# Check global-coordinate AD against finite differences at a nonzero theta.
theta = jnp.asarray(0.1 * rng.normal(size=nparam))
vector = jnp.asarray(rng.normal(size=nparam))
vector /= jnp.linalg.norm(vector)
step = 1e-5
grad = gradient(theta, dipole_mo)
fd = (boys_loss(theta + step * vector, dipole_mo)
      - boys_loss(theta - step * vector, dipole_mo)) / (2 * step)
hvp = hessp(theta, vector, dipole_mo)
fd_hvp = (gradient(theta + step * vector, dipole_mo)
          - gradient(theta - step * vector, dipole_mo)) / (2 * step)
np.testing.assert_allclose(grad @ vector, fd, atol=2e-8, rtol=0)
np.testing.assert_allclose(hvp, fd_hvp, atol=2e-7, rtol=0)

# The analytic formula is for a LOCAL increment at the current orbitals,
# not for the nonzero global exponential coordinates used immediately above.
u = rotation(theta)
current_dipoles = jnp.einsum("pi,xpq,qj->xij", u, dipole_mo, u)
centers = jnp.diagonal(current_dipoles, axis1=1, axis2=2)
analytic = 4 * jnp.sum(
    (centers[:, p] - centers[:, q]) * current_dipoles[:, p, q], axis=0,
)
local_ad = gradient(jnp.zeros(nparam), current_dipoles)
np.testing.assert_allclose(local_ad, analytic, atol=1e-10, rtol=0)
print("Analytic gradient, finite-difference gradient and HVP checks passed.")
