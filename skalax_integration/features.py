"""Build the Skala input feature dictionary from a GradSCF molecule.

GradSCF and Skala share the libxc / PySCF convention for the three local
features that Skala consumes:

* ``rho``  : electron density on the quadrature grid, with ``sum(w * rho) = N_e``.
* ``grad`` : density gradient ``grad(rho)`` in Bohr^-1.
* ``kin``  : kinetic-energy density ``tau = 0.5 * sum_i n_i |grad phi_i|^2``.

GradSCF's restricted (closed-shell) molecule stores one total density matrix
``P = P_alpha + P_beta``.  Skala expects explicit spin channels
``[alpha, beta]``; for a closed shell these split symmetrically,
``alpha = beta = total / 2``, which is the mapping implemented here.
Coordinates are expected to already be in Bohr, matching GradSCF's internal
grid/atom coordinates.
"""

from __future__ import annotations

import jax.numpy as jnp


def build_skala_features(molecule, density_total):
    """Return the Skala ``mol`` dict for a total density matrix.

    Parameters
    ----------
    molecule : RestrictedMolecule
        GradSCF molecule carrying ``ao`` (``(ngrid, nao)``), ``ao_deriv1``
        (``(4, ngrid, nao)`` with ``[value, dx, dy, dz]``), ``grid``
        (``weights``/``coords``) and ``atom_coords``.
    density_total : jax.Array
        Spin-summed density matrix of shape ``(nao, nao)``.

    Returns
    -------
    dict[str, jax.Array]
        ``density`` ``(2, ngrid)``, ``grad`` ``(2, 3, ngrid)``,
        ``kin`` ``(2, ngrid)``, ``grid_coords`` ``(ngrid, 3)``,
        ``grid_weights`` ``(ngrid,)``, ``coarse_0_atomic_coords``
        ``(natom, 3)``.
    """
    ao = jnp.asarray(molecule.ao)
    ao_deriv1 = jnp.asarray(molecule.ao_deriv1)
    density = jnp.asarray(density_total)

    rho = jnp.einsum("pq,gp,gq->g", density, ao, ao)
    grad = 2.0 * jnp.einsum("xrp,pq,rq->xr", ao_deriv1[1:4], density, ao)
    tau = 0.5 * jnp.einsum(
        "xrp,pq,xrq->r", ao_deriv1[1:4], density, ao_deriv1[1:4]
    )

    # Closed-shell spin splitting: alpha = beta = total / 2.
    rho_a = rho_b = 0.5 * rho
    grad_a = grad_b = 0.5 * grad
    tau_a = tau_b = 0.5 * tau

    return {
        "density": jnp.stack([rho_a, rho_b]),
        "grad": jnp.stack([grad_a, grad_b]),
        "kin": jnp.stack([tau_a, tau_b]),
        "grid_coords": jnp.asarray(molecule.grid.coords),
        "grid_weights": jnp.asarray(molecule.grid.weights),
        "coarse_0_atomic_coords": jnp.asarray(molecule.atom_coords),
    }
