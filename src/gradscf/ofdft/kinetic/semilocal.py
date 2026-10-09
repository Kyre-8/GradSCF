"""Spin-unpolarized atomic-unit TF/vW kinetic energies."""
import jax.numpy as jnp

TF_CONSTANT = .3 * (3*jnp.pi**2)**(2/3)


def thomas_fermi(rho, weights):
    """Integral of C_TF n^(5/3); expects nonnegative density."""
    return TF_CONSTANT * jnp.sum(weights * rho**(5/3))
