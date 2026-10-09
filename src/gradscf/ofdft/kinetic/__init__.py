"""KEDF energy callbacks with explicit, trainable PyTree parameters."""
from dataclasses import dataclass, field
from typing import Callable
import jax.numpy as jnp
from .semilocal import thomas_fermi, TF_CONSTANT
from .nonlocal_kernel import wang_teter


@dataclass(frozen=True)
class KineticFunctional:
    """TF, vW, TF+vW, WT, or second-order WGC99. Parameters 'tf'/'vw' override weights.

    WT uses the standard TF+full-vW response kernel; configurable weights are
    intentionally limited to the semilocal family.
    """
    name: str = 'tfvw'

    def __post_init__(self):
        object.__setattr__(self, 'name', self.name.lower())
        if self.name not in ('tf', 'vw', 'tfvw', 'wt', 'wgc'):
            raise ValueError('KEDF must be tf, vw, tfvw, wt or wgc; or supply an energy callback.')

    def __call__(self, params, features):
        # n^(5/3)=|phi|^(10/3): differentiate the latter to avoid 0*inf
        # at density nodes in an amplitude Hessian.
        tf_energy = (thomas_fermi(features.rho, features.weights) if features.phi is None
                     else TF_CONSTANT*jnp.sum(features.weights*jnp.abs(features.phi)**(10/3)))
        if self.name == 'wgc':
            from .wgc import wang_govind_carter
            if set(params)-{'reference_density'}:
                raise ValueError('WGC only accepts an optional reference_density parameter.')
            return tf_energy+features.vw_energy+wang_govind_carter(features,params.get('reference_density'))
        if self.name == 'wt':
            if params:
                raise ValueError('Standard WT does not accept adjustable TF/vW weights.')
            return tf_energy+features.vw_energy+wang_teter(features)
        tf = params.get('tf', 0. if self.name == 'vw' else 1.)
        vw = params.get('vw', 0. if self.name == 'tf' else 1.)
        return tf*tf_energy+vw*features.vw_energy


@dataclass(frozen=True)
class NeuralKineticFunctional:
    """Add energy_fn(network_params, features) to an explicit baseline.

    The callback returns a scalar ENERGY in Hartree, not an energy per electron
    or a potential. Params are {'network': PyTree, 'baseline': optional PyTree}.
    Flax/Haiku/pure-JAX apply functions can be wrapped without binding weights in
    a closure. AD of this energy supplies consistent potentials and HVPs.
    Smooth networks are needed for density response/training through the minimum.
    This adapter imposes no positivity or exact constraints on a custom model.
    """
    energy_fn: Callable
    baseline: KineticFunctional = field(default_factory=KineticFunctional)

    def __call__(self, params, features):
        return self.baseline(params.get('baseline', {}), features)+self.energy_fn(params['network'], features)


__all__ = ['KineticFunctional', 'NeuralKineticFunctional', 'thomas_fermi', 'wang_teter']
