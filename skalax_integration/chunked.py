"""Memory-bounded (chunked) evaluation of the Skala functional.

Skala evaluates everything on the full DFT quadrature grid at once, and its
non-local branch allocates edge tensors of *static* size
``num_fine * num_coarse`` (padding out the cutoff-masked edges).  For dense
grids / large basis sets that exceeds GPU memory.

This module evaluates exactly the same quantities in fine-grid chunks.  It is
mathematically identical because every cross-chunk coupling in Skala is a sum:

* the XC energy is ``sum_i w_i e_i``            -> accumulate per chunk;
* the non-local fine->coarse aggregation is
  ``h_coarse[a] = sum_{fine points i of a} contribution_i``
  -> accumulate into a global ``(num_coarse, hidden)`` buffer in pass 1, then
     use that *final* buffer in pass 2.  (Doing pass 1 per chunk and combining
     afterwards would be wrong: ``tp_up`` is non-linear.)

Padding: the last block is padded to a static size using far-away coordinates
(so no cutoff edge is created) and zero integration weights (so no energy
contribution).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

from skalax.functional.base import enhancement_density_inner_product
from skalax.functional.functions import (
    exp_radial_func,
    normalization_envelope,
    polynomial_envelope,
    prepare_features,
    vect_cdist,
)
from skalax.utils.scatter import scatter_sum


_FAR = 1.0e6  # padded-point coordinate: far outside any radius_cutoff


def _pad_molecule(mol: dict, n_pad: int) -> dict:
    if n_pad <= 0:
        return mol

    def pad_last(x, n, axis):
        shape = list(x.shape)
        shape[axis] = n
        return jnp.concatenate([x, jnp.zeros(shape, dtype=x.dtype)], axis=axis)

    out = dict(mol)
    out["density"] = pad_last(mol["density"], n_pad, axis=1)
    out["grad"] = pad_last(mol["grad"], n_pad, axis=2)
    out["kin"] = pad_last(mol["kin"], n_pad, axis=1)
    out["grid_weights"] = pad_last(mol["grid_weights"], n_pad, axis=0)
    out["grid_coords"] = jnp.concatenate(
        [mol["grid_coords"], jnp.full((n_pad, 3), _FAR, dtype=mol["grid_coords"].dtype)],
        axis=0,
    )
    return out


def chunked_get_exc(model, mol: dict, chunk_size: int = 8192):
    """Total XC energy (scalar, Hartree) evaluated in fine-grid chunks."""
    n_points = int(mol["grid_coords"].shape[0])
    n_blocks = max(1, -(-n_points // chunk_size))
    mol = _pad_molecule(mol, n_blocks * chunk_size - n_points)

    coords = mol["grid_coords"]
    weights = mol["grid_weights"]
    density = mol["density"]
    grad = mol["grad"]
    kin = mol["kin"]
    coarse = mol["coarse_0_atomic_coords"]

    nl = model.non_local_model
    non_local = nl is not None
    n_coarse = int(coarse.shape[0])
    # ``h_coarse`` lives in ``hidden_irreps`` (16x0e+16x1e+16x2e+16x3e -> 256),
    # not in the scalar width ``hidden_nf`` (= 16).
    hidden = int(nl.hidden_irreps.dim) if non_local else 1
    # ``jnp.arange`` needs a concrete ``stop``; wrapping the block count in a
    # jnp array makes JAX treat it as a tracer under ``jit``.
    block_ids = jnp.arange(n_blocks)

    def block_inputs(k):
        off = k * chunk_size
        sl = jax.lax.dynamic_slice_in_dim(coords, off, chunk_size, axis=0)
        di = jax.lax.dynamic_slice_in_dim(density, off, chunk_size, axis=1)
        gi = jax.lax.dynamic_slice_in_dim(grad, off, chunk_size, axis=2)
        ki = jax.lax.dynamic_slice_in_dim(kin, off, chunk_size, axis=1)
        wi = jax.lax.dynamic_slice_in_dim(weights, off, chunk_size, axis=0)
        sub = {
            "density": di, "grad": gi, "kin": ki, "grid_coords": sl,
            "grid_weights": wi, "coarse_0_atomic_coords": coarse,
        }
        features_ab, features_ba = prepare_features(sub)
        spin = jnp.concatenate([features_ab, features_ba], axis=0).astype(model.dtype)
        spin = model._input_model(spin)
        a, b = jnp.split(spin, 2, axis=0)
        return (a + b) / 2, di, wi, sl

    def edge_terms(h_mid, coord_block, weight_block):
        """Cutoff-masked edge tensors for this block (static size chunk*n_coarse)."""
        directions, distances = vect_cdist(coord_block, coarse)
        if nl.radius_cutoff != float("inf"):
            up_weight = normalization_envelope(distances, nl.radius_cutoff)
        else:
            up_weight = jnp.ones_like(distances)
        n_fine, n_c = distances.shape
        n_edges = n_fine * n_c
        mask = distances <= nl.radius_cutoff
        idx = jnp.argwhere(mask, size=n_edges, fill_value=0)
        fine_idx = idx[:, 0]
        coarse_idx = idx[:, 1]
        edge_mask = (jnp.arange(n_edges) < mask.sum()).astype(h_mid.dtype)
        edge_dist = distances[fine_idx, coarse_idx]
        edge_dir = directions[fine_idx, coarse_idx]
        up_w = up_weight[fine_idx, coarse_idx] * edge_mask
        dist_ft = exp_radial_func(edge_dist, nl.hidden_nf) * edge_mask[:, None]
        if nl.radius_cutoff != float("inf"):
            dist_ft = dist_ft * polynomial_envelope(
                edge_dist, nl.radius_cutoff, 8
            )[:, None]
        dir_ft = e3nn.spherical_harmonics(
            nl.sph_irreps, edge_dir, normalize=False, normalization="norm"
        ).array * edge_mask[:, None]
        return fine_idx, coarse_idx, dist_ft, dir_ft, up_w, weight_block[fine_idx]

    # ---- pass 1: accumulate the global coarse buffer -------------------
    def pass1(h_coarse, k):
        h_mid, _, w_block, coord_block = block_inputs(k)
        if not non_local:
            return h_coarse, None
        h_pre = jax.nn.silu(
            h_mid @ nl.pre_down_linear.weight.T + nl.pre_down_linear.bias
        )
        fine_idx, coarse_idx, dist_ft, dir_ft, _, edge_w = edge_terms(
            h_mid, coord_block, w_block
        )
        down = nl.tp_down(h_pre[fine_idx], dir_ft)
        down = nl._mul_repeat(dist_ft, down, nl.hidden_irreps)
        down = down.astype(jnp.float64) * edge_w[:, None].astype(jnp.float64)
        contribution = scatter_sum(
            down, coarse_idx, dim=0, dim_size=n_coarse
        ).astype(h_coarse.dtype)
        return h_coarse + contribution, None

    if non_local:
        h_coarse, _ = jax.lax.scan(
            jax.checkpoint(pass1),
            jnp.zeros((n_coarse, hidden), dtype=model.dtype),
            block_ids,
        )
    else:
        h_coarse = jnp.zeros((1, 1), dtype=model.dtype)

    # ---- pass 2: broadcast back and accumulate the energy --------------
    def pass2(acc, k):
        h_mid, d_block, w_block, coord_block = block_inputs(k)
        if non_local:
            fine_idx, coarse_idx, dist_ft, dir_ft, up_w, _ = edge_terms(
                h_mid, coord_block, w_block
            )
            up = nl.tp_up(h_coarse[coarse_idx], dir_ft)
            denom = scatter_sum(up_w, fine_idx, dim=0, dim_size=chunk_size)[fine_idx]
            up = nl._mul_repeat(dist_ft * (up_w / (denom + 0.1))[:, None], up, nl.out_irreps)
            h_nl = scatter_sum(up, fine_idx, dim=0, dim_size=chunk_size)
            h_nl = jax.nn.silu(
                h_nl @ nl.post_up_linear.weight.T + nl.post_up_linear.bias
            )
            density_sum = d_block.sum(0).reshape(-1, 1)
            h_nl = h_nl * jnp.exp(-density_sum).astype(model.dtype)
            features = jnp.concatenate([h_mid, h_nl], axis=-1)
        else:
            features = h_mid
        enhancement = model._output_model(features)
        e_density = enhancement_density_inner_product(
            enhancement_factor=enhancement, density=d_block
        ).astype(jnp.float64)
        return acc + (e_density * w_block.astype(jnp.float64)).sum(), None

    total, _ = jax.lax.scan(
        jax.checkpoint(pass2), jnp.asarray(0.0, jnp.float64), block_ids
    )
    return total
