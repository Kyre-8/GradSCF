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

The grid quantities are obtained from the density matrix with two contractions
over the AO matrix, ``rho = sum_pq P_pq A_p A_q`` and likewise for the gradient
and the kinetic density.  Reverse-mode differentiation of those contractions
w.r.t. ``P`` is what the differentiable SCF needs in order to build
``dE/dP`` -- and a naive implementation makes XLA materialise a
``(ngrid, nao, nao)`` intermediate, i.e. ~9.5 GB for def2-svp with a level-2
grid.  ``chunk_size`` therefore evaluates them in grid blocks inside a
checkpointed scan, which keeps only ``(chunk, nao, nao)`` live at a time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _grid_features_chunked(ao, ao_deriv1, density, chunk_size):
    """``(rho, grad, tau)`` for the total density, evaluated in grid blocks."""
    ngrid, nao = ao.shape
    n_blocks = -(-ngrid // chunk_size)
    padded = n_blocks * chunk_size
    if padded != ngrid:
        pad = padded - ngrid
        ao = jnp.concatenate(
            [ao, jnp.zeros((pad, nao), dtype=ao.dtype)], axis=0
        )
        ao_deriv1 = jnp.concatenate(
            [ao_deriv1, jnp.zeros((4, pad, nao), dtype=ao_deriv1.dtype)], axis=1
        )
    d1 = ao_deriv1[1:4]
    dtype = density.dtype

    def body(carry, k):
        rho_out, grad_out, tau_out = carry
        off = k * chunk_size
        a = jax.lax.dynamic_slice_in_dim(ao, off, chunk_size, axis=0)
        ad = jax.lax.dynamic_slice_in_dim(d1, off, chunk_size, axis=1)
        rho = jnp.einsum("pq,gp,gq->g", density, a, a)
        grad = 2.0 * jnp.einsum("xgp,pq,gq->xg", ad, density, a)
        tau = 0.5 * jnp.einsum("xgp,pq,xgq->g", ad, density, ad)
        rho_out = jax.lax.dynamic_update_slice_in_dim(rho_out, rho, off, axis=0)
        grad_out = jax.lax.dynamic_update_slice_in_dim(grad_out, grad, off, axis=1)
        tau_out = jax.lax.dynamic_update_slice_in_dim(tau_out, tau, off, axis=0)
        return (rho_out, grad_out, tau_out), None

    init = (
        jnp.zeros((padded,), dtype=dtype),
        jnp.zeros((3, padded), dtype=dtype),
        jnp.zeros((padded,), dtype=dtype),
    )
    # ``jax.checkpoint`` on the body keeps XLA from materialising the full
    # stacked ``(n_blocks, chunk, nao, nao)`` residual of the reverse pass.
    (rho, grad, tau), _ = jax.lax.scan(
        jax.checkpoint(body), init, jnp.arange(n_blocks)
    )
    return rho[:ngrid], grad[:, :ngrid], tau[:ngrid]


def build_skala_features(molecule, density_total, chunk_size: int = 0):
    """Return the Skala ``mol`` dict for a total density matrix.

    Parameters
    ----------
    molecule : RestrictedMolecule
        GradSCF molecule carrying ``ao`` (``(ngrid, nao)``), ``ao_deriv1``
        (``(4, ngrid, nao)`` with ``[value, dx, dy, dz]``), ``grid``
        (``weights``/``coords``) and ``atom_coords``.
    density_total : jax.Array
        Spin-summed density matrix of shape ``(nao, nao)``.
    chunk_size : int
        Grid block size used when building ``rho``/``grad``/``tau``.  ``0``
        (default) evaluates the whole grid at once; a positive value caps the
        reverse-mode allocation at ``chunk_size * nao**2``.

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

    if chunk_size and int(ao.shape[0]) > int(chunk_size):
        rho, grad, tau = _grid_features_chunked(
            ao, ao_deriv1, density, int(chunk_size)
        )
    else:
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
