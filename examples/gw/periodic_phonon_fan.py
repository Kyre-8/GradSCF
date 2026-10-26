"""One-band periodic model: complex Matsubara Fan kernel with explicit k/q.

This illustrates the periodic self-energy map, not a periodic scGW solver.
The electronic and phonon models are prescribed (no Wannier interpolation).
Run with PYTHONPATH=src JAX_PLATFORMS=cpu. Energies are in Hartree.
"""

import jax
import jax.numpy as jnp
from gradscf.gw.matsubara import matsubara_grid
from gradscf.gw.ep_coupling import PeriodicPhononModel, periodic_fan_self_energy

jax.config.update("jax_enable_x64", True)

nk = nq = 4
kpoints = jnp.arange(nk) / nk
energies = -.06 * jnp.cos(2 * jnp.pi * kpoints)
grid = matsubara_grid(nw=24, beta=100.)
mu = 0.
green_tau = -jnp.exp(-grid.tau[:, None] * (energies - mu)
                     - jnp.logaddexp(0., -grid.beta * (energies - mu)))
phonons = PeriodicPhononModel(
    energies=jnp.full((nq, 1), .012),
    couplings=jnp.full((nq, nk, 1, 1, 1), .007, dtype=jnp.complex128),
    k_plus_q=(jnp.arange(nq)[:, None] + jnp.arange(nk)[None, :]) % nk,
    q_weights=jnp.full(nq, 1 / nq),
    reference="external one-band local-coupling model",
)
result = jax.jit(periodic_fan_self_energy)(
    green_tau[:, :, None, None], energies[:, None, None], mu, phonons, grid)

print("Lowest positive Matsubara frequency / Ha:", float(grid.fermion[grid.nw]))
print("Sigma at this frequency / Ha:", result["sigma_iw"][grid.nw, :, 0, 0])
print("Fan high-frequency moment / Ha^2:", result["sigma_moment"][:, 0, 0].real)
# Uniform q integration and local vertices make the self-energy k independent.
assert jnp.max(jnp.abs(result["sigma_iw"] - result["sigma_iw"][:, :1])) < 1e-12

# CPU complex128 output (2026-10-04, JAX 0.8.1):
# Lowest positive Matsubara frequency / Ha: 0.031415926535897934
# Sigma at this frequency / Ha: approximately -0.00154679j at all four k points.
# Fan high-frequency moment / Ha^2: [9.12392505e-05] at each k point.
