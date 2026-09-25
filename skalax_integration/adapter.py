"""Wrap skalax's ``SkalaFunctional`` as a GradSCF differentiable-SCF XC functional.

GradSCF's :class:`~gradscf.scf.differentiable.DifferentiableSCF` consumes an XC
functional through a small duck-typed interface rather than a class hierarchy.
The method :meth:`SkalaFunctionalAdapter.scf_xc_energy_and_alpha_for_density`
returns the scalar XC energy together with the exact-exchange fraction
``alpha``.  The SCF driver differentiates that energy with respect to the
density matrix to build the XC potential, so the adapter only needs to be
differentiable in JAX.  Skala is a pure (semi-local + non-local neural) density
functional, so ``alpha`` is always zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .features import build_skala_features


class SkalaFunctionalAdapter:
    """GradSCF-compatible functional backed by a loaded Skala model.

    Parameters
    ----------
    model : skalax.functional.model.SkalaFunctional
        Equinox module with pretrained weights already loaded.
    """

    def __init__(self, model):
        self.model = model

    def energy_from_density(self, params, molecule, density_total):
        """Scalar Skala XC energy for a total density matrix (Hartree)."""
        features = build_skala_features(molecule, density_total)
        return self.model.get_exc(features)

    def scf_xc_energy_and_alpha_for_density(self, params, molecule, density):
        """Return ``(exc, alpha)``; the SCF driver ADs ``exc`` to get Vxc."""
        exc = self.energy_from_density(params, molecule, density)
        alpha = jnp.asarray(0.0, dtype=jnp.asarray(density).dtype)
        return exc, alpha

    def energy_from_molecule(self, params, molecule):
        """Scalar Skala XC energy from a molecule's ``rdm1`` (Hartree)."""
        rdm1 = jnp.asarray(molecule.rdm1)
        density = rdm1.sum(axis=0) if rdm1.ndim == 3 else rdm1
        return self.energy_from_density(params, molecule, density)


def load_skala_functional(weights_dir=None, *, key=None):
    """Build a :class:`skalax.SkalaFunctional` and load the bundled weights."""
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
