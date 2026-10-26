"""Two-level external phonon model: retarded Fan, linewidth, and spectrum.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/gw/phonon_spectrum.py
All energies are Ha, beta is Ha^-1, and frequencies are relative to mu.
These illustrative fixed-reference poles are evaluated directly on the
real axis; this is not an analytic continuation of the scGW example.
No ab initio phonons, PySCF calculation, CLI, or plotting package is used.
"""

import jax
import jax.numpy as jnp

from gradscf.gw.ep_coupling import (
    PhononModel, electron_linewidth, fan_retarded, spectral_function,
)

jax.config.update("jax_enable_x64", True)

energies = jnp.array([-.12, .16])
mu, beta, eta = .01, 150., .003
phonons = PhononModel(
    energies=jnp.array([.025, .04]),
    couplings=jnp.array([[[.014, .008j], [-.008j, -.01]],
                         [[.005, .004], [.004, .008]]]),
    reference="illustrative external two-level model",
)
frequencies = jnp.linspace(-2., 2., 16001)
sigma = fan_retarded(energies, phonons, frequencies, mu=mu, beta=beta, eta=eta)
spectrum = spectral_function(jnp.diag(energies), sigma, frequencies, mu=mu, eta=eta)
diagonal = jnp.diagonal(spectrum, axis1=-2, axis2=-1).real
integral = jnp.sum((diagonal[1:] + diagonal[:-1]) * jnp.diff(frequencies)[:, None] / 2, axis=0)
peaks = frequencies[jnp.argmax(diagonal, axis=0)]
linewidth = electron_linewidth(energies, phonons, mu=mu, beta=beta, eta=eta)

print("Backend:", jax.default_backend(), "dtype:", spectrum.dtype)
print("On-shell full linewidth Gamma / Ha:", linewidth)
print("Dominant finite-grid spectral maxima relative to mu / Ha:", peaks)
print("Finite-window integrated diagonal spectral weight:", integral)
print("Minimum spectral-matrix eigenvalue / Ha^-1:", jnp.linalg.eigvalsh(spectrum).min())
assert jnp.max(jnp.abs(integral - 1)) < .0011
assert jnp.linalg.eigvalsh(spectrum).min() >= -1e-12

# CPU complex128 output (2026-10-04, JAX 0.8.1):
# On-shell full linewidth Gamma / Ha: [0.00204324 0.00123711]
# Dominant finite-grid spectral maxima relative to mu / Ha: [-0.12375  0.1455]
# Finite-window integrated diagonal spectral weight: [0.99904094 0.99903961]
# Minimum spectral-matrix eigenvalue / Ha^-1: 0.00020661853537677738
