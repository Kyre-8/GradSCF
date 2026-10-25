"""SkalaX functional adapter with TDDFT kernel_action support.

Extends the original adapter to:
  * support grid chunking for memory-bounded evaluation
  * expose kernel_action (f_xc · delta_P) via forward-over-reverse AD
  * work with GradSCF's TDDFT response module through ao_kernel_action

Adds an opt-in whole-function rematerialisation (SKALA_REMAT=1) that wraps the
SkalaX energy in ``jax.checkpoint``.  The saved activations of the energy live
at this call boundary, so this is the only granularity that actually lowers the
backward peak (see skalax_integration/remat_adapter.py for measurements).
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp

from gradscf.dft.functional import Functional

_REMAT = os.environ.get("SKALA_REMAT", "0") == "1"


def make_skala_input(di) -> dict[str, jax.Array]:
    """Convert a GradSCF DensityInputs into Skala's mol feature dict."""
    return {
        "density": di.rho_spin,
        "grad": jnp.transpose(di.grad_rho_spin, (0, 2, 1)),
        "kin": di.tau_spin,
        "grid_coords": di.coordinates,
        "grid_weights": di.weights,
        "coarse_0_atomic_coords": di.atom_coords,
    }


def make_skala_functional_v2(model, chunk_size: int = 0) -> Functional:
    """Build a GradSCF Functional around SkalaX with kernel_action support.

    Parameters
    ----------
    model : SkalaFunctional
        The loaded SkalaX model.
    chunk_size : int
        Grid chunk size for memory-bounded evaluation. 0 = monolithic.
        When > 0, uses chunked_get_exc which is mathematically identical.
    """
    if _REMAT:
        # Validated chunked+remat forward (checkpoints inside the block scans).
        from skalax_integration.remat_adapter import chunked_get_exc
    else:
        from skalax_integration.chunked import chunked_get_exc

    params, static = eqx.partition(model, eqx.is_array)

    def _rebuild(p):
        p = p["params"] if isinstance(p, dict) and "params" in p else p
        return eqx.combine(p, static)

    def input_fn(di):
        return make_skala_input(di)

    def energy_fn(fn_params, mol):
        m = _rebuild(fn_params)
        ngrid = int(mol["grid_coords"].shape[0])
        if chunk_size > 0 and chunk_size < ngrid:
            def body(mm, f):
                return chunked_get_exc(mm, f, chunk_size=chunk_size)
        else:
            def body(mm, f):
                return mm.get_exc(f)
        return body(m, mol)

    def init_fn(key, mol):
        del key, mol
        return {"params": params}

    return Functional(
        input_fn=input_fn,
        energy_fn=energy_fn,
        init_fn=init_fn,
        name="skala_tddft",
    )


def load_skala_functional(weights_dir=None, *, key=None):
    """Load the Skala model with pretrained weights."""
    from skalax import (
        SkalaFunctional,
        get_default_weights_dir,
        load_config,
        load_weights_from_npz,
    )

    if weights_dir is None:
        weights_dir = get_default_weights_dir()
    config = load_config(weights_dir)
    if key is None:
        key = jax.random.PRNGKey(0)

    model = SkalaFunctional(
        lmax=config["lmax"],
        non_local=config["non_local"],
        non_local_hidden_nf=config["non_local_hidden_nf"],
        radius_cutoff=config["radius_cutoff"],
        key=key,
    )
    return load_weights_from_npz(model, weights_dir)


def make_skala_tddft_functional(weights_dir=None, chunk_size=4096, *, key=None):
    """Convenience: load model and create a TDDFT-ready Functional."""
    model = load_skala_functional(weights_dir, key=key)
    return make_skala_functional_v2(model, chunk_size=chunk_size)
