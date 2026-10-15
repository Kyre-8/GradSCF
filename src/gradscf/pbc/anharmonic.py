"""Local projected energy expansions, not bulk anharmonic phonon self-energies.

A frozen-mode scan must independently reconverge the electronic state at each
displacement, using the same cell, electron count, electronic temperature and
energy convention. At fixed electronic temperature use the appropriate free
energy. Neither a one-direction fit nor a Gamma-point Taylor expansion supplies
Brillouin-zone mode couplings, thermal frequency shifts or phonon linewidths.
"""
import math
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.jet import jet


class FrozenModeFit(NamedTuple):
    coefficients: object
    derivatives: object
    residuals: object
    rms_residual: object
    rank: object
    condition_number: object
    origin_bohr: object
    scale_bohr: object


class DirectionalTaylor(NamedTuple):
    coefficients: object
    derivatives: object


def _order(order):
    if isinstance(order, bool) or not isinstance(order, Integral) or not 2 <= order <= 6:
        raise ValueError('order must be an integer from 2 through 6')


def _concrete(value):
    return not isinstance(value, jax.core.Tracer)


def _finite(value, name):
    if _concrete(value) and not np.all(np.isfinite(np.asarray(value))):
        raise ValueError(f'{name} must be finite')


def fit_frozen_mode(displacements_bohr, energies_hartree, *, order=4,
                    origin_bohr=0., even=False, rcond=None,
                    max_condition=1e10, max_residual_hartree=None):
    """Fit E(s) = sum(a[n] * (s-origin_bohr)**n) with JAX least squares.

    Coefficients include the Taylor 1/n! normalization, so ``derivatives[n]``
    is n! times ``coefficients[n]``; both include n=0. Units are Hartree/Bohr**n.
    The scalar coordinate is a displacement in Bohr, not a mass-weighted normal
    coordinate. The user supplies its direction/normalization externally.

    Coordinates are scaled by max(abs(s-origin)) before fitting, and an energy
    offset is removed to reduce cancellation. No reflection symmetry is assumed:
    ``even=True`` explicitly removes odd powers about the supplied origin.
    ``residuals`` are fitted minus supplied energies (Hartree); ``rms_residual``
    is their RMS, not an uncertainty estimate. A small residual cannot establish
    SCF convergence or adequacy of the sampling window/polynomial order.

    Eager calls reject nonfinite inputs/outputs, rank-deficient or poorly
    conditioned data and optionally excessive RMS residual. Inside JAX transformations, data-dependent
    Python checks are unavailable; callers must validate their grid eagerly and
    check the returned rank, condition and residual diagnostics. Order, evenness,
    rcond and thresholds are static settings. Energy parameters remain
    differentiable, including under jit. Float64 is recommended for energy scans.
    """
    _order(order)
    if not isinstance(even, (bool, np.bool_)):
        raise ValueError('even must be boolean')
    if not np.isfinite(max_condition) or max_condition <= 0:
        raise ValueError('max_condition must be finite and positive')
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
        raise ValueError('rcond must be finite and nonnegative')
    if max_residual_hartree is not None and (
            not np.isfinite(max_residual_hartree) or max_residual_hartree < 0):
        raise ValueError('max_residual_hartree must be finite and nonnegative')
    s, energy, origin = map(jnp.asarray, (displacements_bohr, energies_hartree, origin_bohr))
    if s.ndim != 1 or energy.shape != s.shape or origin.shape != ():
        raise ValueError('displacement and energy shapes must match 1D arrays; origin must be scalar')
    if any(jnp.iscomplexobj(x) for x in (s, energy, origin)):
        raise ValueError('displacements, energies and origin must be real')
    dtype = jnp.result_type(s, energy, origin, 1.)
    s, energy, origin = (x.astype(dtype) for x in (s, energy, origin))
    for value, name in ((s, 'displacements'), (energy, 'energies'), (origin, 'origin')):
        _finite(value, name)
    powers = tuple(range(0, order+1, 2 if even else 1))
    if s.size < len(powers):
        raise ValueError('insufficient samples for requested polynomial order')
    scale = jnp.max(jnp.abs(s-origin))
    _finite(scale, 'coordinate scale')
    if _concrete(scale) and float(scale) <= 0:
        raise ValueError('displacements must span a nonzero range about origin')
    reduced = (s-origin)/scale
    design = jnp.stack([reduced**n for n in powers], axis=1)
    centered_energy = energy-energy[0]
    _finite(centered_energy, 'offset energies')
    fitted, _, rank, singular_values = jnp.linalg.lstsq(design, centered_energy, rcond=rcond)
    condition = singular_values[0]/singular_values[-1]
    if _concrete(rank) and int(rank) != len(powers):
        raise ValueError('polynomial design matrix is rank deficient')
    if _concrete(condition) and (not np.isfinite(float(condition)) or float(condition) > max_condition):
        raise ValueError('scaled polynomial design matrix exceeds condition limit')
    residuals = design @ fitted - centered_energy
    rms = jnp.sqrt(jnp.mean(residuals**2))
    _finite(residuals, 'fit residuals')
    _finite(rms, 'fit RMS residual')
    if max_residual_hartree is not None and _concrete(rms) and float(rms) > max_residual_hartree:
        raise ValueError('fit RMS residual exceeds max_residual_hartree')
    coefficients = jnp.zeros(order+1, dtype=dtype).at[jnp.array(powers)].set(
        fitted/scale**jnp.array(powers))
    coefficients = coefficients.at[0].add(energy[0])
    derivatives = coefficients*jnp.asarray([math.factorial(n) for n in range(order+1)], dtype=dtype)
    _finite(coefficients, 'fit coefficients')
    _finite(derivatives, 'fit derivatives')
    return FrozenModeFit(coefficients, derivatives, residuals, rms, rank, condition, origin, scale)


def directional_energy_taylor(energy_fn, coordinates, direction, *, order=4, args=()):
    """Taylor-expand scalar energy_fn(coordinates+s*direction, *args) at s=0.

    Coordinates are in Bohr and direction is dimensionless; it is not normalized
    internally. Coefficients and n! times coefficients include n=0 and have units
    Hartree/Bohr**n if energy_fn returns Hartree. Parameters in ``args`` are kept
    live for differentiation. This applies JAX jet directly to the energy
    function; it neither solves an SCF root nor substitutes a root Taylor series.
    The supplied energy function is responsible for electronic relaxation.

    All primitives traversed by the function must support jet. Unsupported
    primitives propagate their explicit JAX error: there is no finite-difference,
    zero-derivative or alternate-backend fallback.
    """
    _order(order)
    x, v = jnp.asarray(coordinates), jnp.asarray(direction)
    if x.shape != v.shape:
        raise ValueError('coordinates and direction must have identical shape')
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(v):
        raise ValueError('coordinates and direction must be real')
    dtype = jnp.result_type(x, v, 1.)
    x, v = x.astype(dtype), v.astype(dtype)
    _finite(x, 'coordinates')
    _finite(v, 'direction')

    def scalar_energy(y):
        value = jnp.asarray(energy_fn(y, *args))
        if value.shape != () or jnp.iscomplexobj(value):
            raise ValueError('energy_fn must return a real scalar')
        return value

    value, terms = jet(scalar_energy, (x,), ((v,)+(jnp.zeros_like(x),)*(order-1),),
                       factorial_scaled=False)
    coefficients = jnp.stack([value, *terms])
    _finite(coefficients, 'energy Taylor coefficients')
    derivatives = coefficients*jnp.asarray([math.factorial(n) for n in range(order+1)], dtype=coefficients.dtype)
    _finite(derivatives, 'energy Taylor derivatives')
    return DirectionalTaylor(coefficients, derivatives)
