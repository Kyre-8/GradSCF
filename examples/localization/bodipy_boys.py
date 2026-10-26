"""Foster-Boys localization of the occupied orbitals of BODIPY.

GradSCF supplies RHF orbitals and AO integrals. After RHF, the occupied space
is fixed: AD differentiates only orbital rotations, not SCF or coordinates.
Position integrals are in Bohr and Boys scores in Bohr**2.
The resulting occupied MO coefficients are stored in c_boys and the rotation
in u_opt. rotation_history records the selected optimization trajectory.
This script has no CLI. It saves the trajectory for a separate animation,
so rendering does not require another SCF or localization calculation.

Reference: Foster and Boys, Rev. Mod. Phys. 32, 300 (1960).
https://doi.org/10.1103/RevModPhys.32.300
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from jax.scipy.linalg import expm
from scipy.optimize import minimize

from gradscf import dft, gto, integrals

jax.config.update("jax_enable_x64", True)

# Run RHF once. All 49 occupied orbitals, including the core orbitals, are
# localized together. This is a fixed UFF demonstration geometry, in Angstrom.
# Parent identity: PubChem CID 25058173, C9H7BF2N2, charge 0, singlet.
# Geometry: RDKit 2025.03.3, ETKDGv3 seed 11, converged UFF; not HF optimized.
# InChIKey: GUHHEAYOTAJBPT-UHFFFAOYSA-N.
# https://pubchem.ncbi.nlm.nih.gov/compound/25058173
mol = gto.M(
    atom="""
    F -0.05230623  2.61911226 -1.19096121
    B -0.03696946  1.78290644 -0.00003323
    F -0.05225361  2.61916201  1.19086139
    N -1.26805823  0.85176145  0.00000877
    C -2.57027905  1.18928449  0.00000440
    C -3.34541754  0.04220480  0.00000207
    C -2.47528689 -1.03565151  0.00000840
    C -1.20590652 -0.49625074  0.00000944
    C  0.06933733 -1.23670365 -0.00000371
    C  1.19947477 -0.52494758 -0.00003899
    C  2.56791415 -1.01407129  0.00000269
    C  3.33737731  0.07188435  0.00003825
    C  2.46591716  1.25626535  0.00000995
    N  1.22492751  0.89622434 -0.00004127
    H -2.94192947  2.20609059 -0.00000510
    H -4.42612052 -0.00372969 -0.00002147
    H -2.73980397 -2.08479703 -0.00000613
    H  0.08851797 -2.31923827  0.00004164
    H  2.90663509 -2.04174164  0.00002950
    H  4.41906517  0.07256562  0.00010994
    H  2.79820747  2.28681811  0.00005746
    """,
    basis="cc-pvdz",
    unit="Angstrom",
    cart=True,
)
print("BODIPY: GradSCF shell-direct RHF/cc-pVDZ (245 Cartesian AOs)", flush=True)
mf = dft.RKS(mol, xc="hf", integral_backend="native", conv_tol=1e-11, max_cycle=150)
mf = mf.direct_scf().run()
assert mf.converged
hf_energy = float(mf.e_tot)
print(f"RHF converged: {hf_energy:.12f} Ha", flush=True)
c_occ = jnp.asarray(mf.mo_coeff[:, np.asarray(mf.mo_occ) > 0])
nocc = c_occ.shape[1]
assert mol.nelectron == 98 and nocc == 49

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


# Canonical BODIPY orbitals can be a high-symmetry stationary point.
# Compare multiple starts; a small gradient alone does not establish a minimum.
rng = np.random.default_rng(0)
starts = [np.zeros(nparam)]
starts += [1e-3 * rng.normal(size=nparam) for _ in range(3)]
results = []
histories = []
for theta0 in starts:
    history = [theta0.copy()]
    result = minimize(
        objective, theta0, args=(dipole_mo,), jac=True, method="L-BFGS-B",
        callback=lambda theta: history.append(np.array(theta, copy=True)),
        options={"gtol": 1e-9, "ftol": 1e-15, "maxiter": 100, "maxls": 40},
    )
    results.append(result)
    histories.append(history)
    print(f"Boys score = {-result.fun:.12f}, iterations = {result.nit}", flush=True)

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
    options={"xtol": 1e-14, "maxiter": 200},
)
u_opt = u_base @ rotation(jnp.asarray(polished.x))
rotation_history += [np.asarray(u_base @ rotation(jnp.asarray(t))) for t in polish_history]
iteration_stages += ["Newton-CG"] * len(polish_history)

# At this larger score, energy changes can round to zero before the orbital
# gradient is resolved. Refine stationarity using 1/2 |g|^2 and its gradient
# H @ g, evaluated by the same AD HVP. No SCF or dense Hessian is needed.
refinement_base = u_opt
refinement_dipoles = jnp.einsum("pi,xpq,qj->xij", u_opt, dipole_mo, u_opt)


def stationarity_objective(theta, dipoles):
    g = gradient(jnp.asarray(theta), dipoles)
    return float(.5 * (g @ g)), np.asarray(hessp(jnp.asarray(theta), g, dipoles))


refinement_history = []
if jnp.linalg.norm(gradient(jnp.zeros(nparam), refinement_dipoles)) > 1e-7:
    def record_refinement(theta):
        refinement_history.append(np.array(theta, copy=True))
        u = rotation(jnp.asarray(theta))
        d = jnp.einsum("pi,xpq,qj->xij", u, refinement_dipoles, u)
        if jnp.linalg.norm(gradient(jnp.zeros(nparam), d)) < 1e-7:
            raise StopIteration

    refined = minimize(
        stationarity_objective, np.zeros(nparam), args=(refinement_dipoles,),
        jac=True, method="L-BFGS-B", callback=record_refinement,
        options={"gtol": 1e-12, "ftol": 0., "maxiter": 500, "maxls": 40},
    )
    u_opt = refinement_base @ rotation(jnp.asarray(refined.x))
    rotation_history += [np.asarray(refinement_base @ rotation(jnp.asarray(t)))
                         for t in refinement_history]
    iteration_stages += ["Stationarity refinement"] * len(refinement_history)

c_boys = c_occ @ u_opt
dipoles_local = jnp.einsum("pi,xpq,qj->xij", u_opt, dipole_mo, u_opt)
local_gradient = gradient(jnp.zeros(nparam), dipoles_local)

# Occupied rotations preserve S-orthogonality and the RHF density.
orth_error = jnp.linalg.norm(c_boys.T @ overlap @ c_boys - jnp.eye(nocc))
density_error = jnp.linalg.norm(2 * (c_boys @ c_boys.T - c_occ @ c_occ.T))
score_initial = -boys_loss(jnp.zeros(nparam), dipole_mo)
score_final = jnp.sum(jnp.diagonal(dipoles_local, axis1=1, axis2=2)**2)
print(f"RHF energy / Ha       = {hf_energy:.12f}")
print(f"Initial Boys score    = {float(score_initial):.12f}")
print(f"Final Boys score      = {float(score_final):.12f}")
print(f"Local gradient norm   = {float(jnp.linalg.norm(local_gradient)):.3e}")
print(f"Orthogonality error   = {float(orth_error):.3e}")
print(f"Density error         = {float(density_error):.3e}")
assert orth_error < 1e-10
assert density_error < 1e-10
assert jnp.linalg.norm(local_gradient) < 1e-7
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

# Numerical output only; PySCF comparison and rendering are separate scripts.
output = Path(__file__).resolve().parents[2] / "artifacts" / "bodipy_boys"
output.mkdir(parents=True, exist_ok=True)
np.savez(output / "trajectory.npz", c_occ=np.asarray(c_occ), c_boys=np.asarray(c_boys),
         rotations=np.asarray(rotation_history), stages=np.asarray(iteration_stages),
         dipole_mo=np.asarray(dipole_mo), dipole_ao=np.asarray(dipole_ao),
         overlap=np.asarray(overlap), coords_bohr=np.asarray(parameters.nuclear_coords),
         symbols=np.asarray(mol.to_spec().symbols), atom=np.asarray(mol.atom),
         basis=np.asarray(mol.basis), cart=mol.cart, hf_energy=hf_energy)
print(f"Saved {output / 'trajectory.npz'}", flush=True)
