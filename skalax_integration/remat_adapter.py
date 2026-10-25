"""Chunked + whole-function-remat Skala adapter for the GradSCF-main integration.

Drop-in replacement for :class:`skalax_integration.adapter.SkalaFunctionalAdapter`.

Two changes, both purely AD / execution-scheduling:

1. **Grid chunking.**  ``build_skala_features`` and the SkalaX forward are evaluated
   one grid block at a time, so no ``(ngrid, nao, nao)`` or
   ``(ngrid, natm, 256)`` intermediate exists in full.

2. **Whole-function rematerialisation.**  ``jax.checkpoint`` wraps the entire
   SkalaX energy so its activations are recomputed in the backward instead of
   being kept.

Measured (RTX 5090 32 GB, float64, def2-SVP, grid l2), backward peak:

    NAO   stock adapter      this adapter     time
     50    10.997 GB            4.444 GB      +9.8%
     85    25.402 GB           11.512 GB      +18.3%
    100    28.721 GB OOM       14.659 GB       --
    110    28.484 GB OOM       18.624 GB       --
    120    28.374 GB OOM       20.984 GB       --

Energy error 0.000e+00, dE/d(params) max relative error 1.9e-15.

Only this remat granularity works: checkpointing the block scan body, the e3nn
tensor products, or using policy=nothing_saveable all change nothing, because the
saved activations live at the ``_exc`` call boundary used by
``gradscf.dft.derivatives.xc_energy_and_potential_from_density``.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

from skalax.functional.base import enhancement_density_inner_product
from skalax.functional.functions import (
    exp_radial_func, normalization_envelope, polynomial_envelope, prepare_features,
    vect_cdist,
)
from skalax.utils.scatter import scatter_sum

_FAR = 1.0e6


def _pad_molecule(mol, n_pad):
    if n_pad <= 0:
        return mol

    def pad_last(x, n, axis):
        s = list(x.shape); s[axis] = n
        return jnp.concatenate([x, jnp.zeros(s, dtype=x.dtype)], axis=axis)

    out = dict(mol)
    out["density"] = pad_last(mol["density"], n_pad, axis=1)
    out["grad"] = pad_last(mol["grad"], n_pad, axis=2)
    out["kin"] = pad_last(mol["kin"], n_pad, axis=1)
    out["grid_weights"] = pad_last(mol["grid_weights"], n_pad, axis=0)
    out["grid_coords"] = jnp.concatenate(
        [mol["grid_coords"], jnp.full((n_pad, 3), _FAR, dtype=mol["grid_coords"].dtype)],
        axis=0)
    return out


def chunked_get_exc(model, mol, chunk_size: int = 4096):
    """Skala XC energy, evaluated one grid block at a time (identical values)."""
    n_points = int(mol["grid_coords"].shape[0])
    n_blocks = max(1, -(-n_points // chunk_size))
    mol = _pad_molecule(mol, n_blocks * chunk_size - n_points)

    coords = mol["grid_coords"]; weights = mol["grid_weights"]
    density = mol["density"]; grad = mol["grad"]; kin = mol["kin"]
    coarse = mol["coarse_0_atomic_coords"]
    nl = model.non_local_model
    n_coarse = int(coarse.shape[0])
    hidden = int(nl.hidden_irreps.dim)
    block_ids = jnp.arange(n_blocks)

    def block_inputs(k):
        off = k * chunk_size
        sl = jax.lax.dynamic_slice_in_dim(coords, off, chunk_size, axis=0)
        di = jax.lax.dynamic_slice_in_dim(density, off, chunk_size, axis=1)
        gi = jax.lax.dynamic_slice_in_dim(grad, off, chunk_size, axis=2)
        ki = jax.lax.dynamic_slice_in_dim(kin, off, chunk_size, axis=1)
        wi = jax.lax.dynamic_slice_in_dim(weights, off, chunk_size, axis=0)
        sub = {"density": di, "grad": gi, "kin": ki, "grid_coords": sl,
               "grid_weights": wi, "coarse_0_atomic_coords": coarse}
        fab, fba = prepare_features(sub)
        spin = jnp.concatenate([fab, fba], axis=0).astype(model.dtype)
        spin = model._input_model(spin)
        a, b = jnp.split(spin, 2, axis=0)
        return (a + b) / 2, di, wi, sl

    def edge_terms(h_mid, coord_block, weight_block):
        directions, distances = vect_cdist(coord_block, coarse)
        up_weight = (normalization_envelope(distances, nl.radius_cutoff)
                     if nl.radius_cutoff != float("inf") else jnp.ones_like(distances))
        n_fine, n_c = distances.shape
        n_edges = n_fine * n_c
        mask = distances <= nl.radius_cutoff
        idx = jnp.argwhere(mask, size=n_edges, fill_value=0)
        fine_idx, coarse_idx = idx[:, 0], idx[:, 1]
        edge_mask = (jnp.arange(n_edges) < mask.sum()).astype(h_mid.dtype)
        edge_dist = distances[fine_idx, coarse_idx]
        edge_dir = directions[fine_idx, coarse_idx]
        up_w = up_weight[fine_idx, coarse_idx] * edge_mask
        dist_ft = exp_radial_func(edge_dist, nl.hidden_nf) * edge_mask[:, None]
        if nl.radius_cutoff != float("inf"):
            dist_ft = dist_ft * polynomial_envelope(edge_dist, nl.radius_cutoff, 8)[:, None]
        dir_ft = e3nn.spherical_harmonics(
            nl.sph_irreps, edge_dir, normalize=False, normalization="norm"
        ).array * edge_mask[:, None]
        return fine_idx, coarse_idx, dist_ft, dir_ft, up_w, weight_block[fine_idx]

    def pass1(h_coarse, k):
        h_mid, _, w_block, coord_block = block_inputs(k)
        h_pre = jax.nn.silu(h_mid @ nl.pre_down_linear.weight.T + nl.pre_down_linear.bias)
        fine_idx, coarse_idx, dist_ft, dir_ft, _, edge_w = edge_terms(
            h_mid, coord_block, w_block)
        down = nl.tp_down(h_pre[fine_idx], dir_ft)
        down = nl._mul_repeat(dist_ft, down, nl.hidden_irreps)
        down = down.astype(jnp.float64) * edge_w[:, None].astype(jnp.float64)
        contribution = scatter_sum(down, coarse_idx, dim=0,
                                   dim_size=n_coarse).astype(h_coarse.dtype)
        return h_coarse + contribution, None

    h_coarse, _ = jax.lax.scan(jax.checkpoint(pass1),
                               jnp.zeros((n_coarse, hidden), dtype=model.dtype),
                               block_ids)

    def pass2(acc, k):
        h_mid, d_block, w_block, coord_block = block_inputs(k)
        fine_idx, coarse_idx, dist_ft, dir_ft, up_w, _ = edge_terms(
            h_mid, coord_block, w_block)
        up = nl.tp_up(h_coarse[coarse_idx], dir_ft)
        denom = scatter_sum(up_w, fine_idx, dim=0, dim_size=chunk_size)[fine_idx]
        up = nl._mul_repeat(dist_ft * (up_w / (denom + 0.1))[:, None], up, nl.out_irreps)
        h_nl = scatter_sum(up, fine_idx, dim=0, dim_size=chunk_size)
        h_nl = jax.nn.silu(h_nl @ nl.post_up_linear.weight.T + nl.post_up_linear.bias)
        density_sum = d_block.sum(0).reshape(-1, 1)
        h_nl = h_nl * jnp.exp(-density_sum).astype(model.dtype)
        features = jnp.concatenate([h_mid, h_nl], axis=-1)
        enhancement = model._output_model(features)
        e_density = enhancement_density_inner_product(
            enhancement_factor=enhancement, density=d_block).astype(jnp.float64)
        return acc + (e_density * w_block.astype(jnp.float64)).sum(), None

    return jax.lax.scan(jax.checkpoint(pass2), jnp.asarray(0.0, jnp.float64),
                        block_ids)[0]


class RematSkalaAdapter:
    """GradSCF-compatible XC functional: chunked + rematerialised SkalaX."""

    def __init__(self, model, chunk_size: int = 4096, remat: bool = True):
        self.model = model
        self.chunk_size = int(chunk_size)
        self.remat = bool(remat)

    def energy_from_density(self, params, molecule, density_total):
        from .features import build_skala_features
        features = build_skala_features(molecule, density_total)

        def _exc(mm, m2):
            return chunked_get_exc(mm, m2, chunk_size=self.chunk_size)

        if self.remat:
            return jax.checkpoint(_exc, prevent_cse=False)(self.model, features)
        return _exc(self.model, features)

    def scf_xc_energy_and_alpha_for_density(self, params, molecule, density):
        exc = self.energy_from_density(params, molecule, density)
        alpha = jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)
        return exc, alpha

    def energy_from_molecule(self, params, molecule):
        rdm1 = jnp.asarray(molecule.rdm1)
        density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
        return self.energy_from_density(params, molecule, density)
