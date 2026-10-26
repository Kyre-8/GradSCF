"""OFDFT response using the existing energy in a fixed reference metric chart."""
from dataclasses import replace
import jax
import jax.numpy as jnp
from ..ofdft import OFDFTConfig, KineticFunctional
from ..ofdft.problem import _to_sphere, _to_coefficients, _validate
from ..ofdft.energy import energy_components, amplitude
from .core import Problem


def ofdft_problem(data, result, *, kinetic=None, kinetic_params=None, xc_energy_fn=None,
                  xc_params=None, config=None, tolerance=1e-7):
    """Reuse a converged OFDFT state; Gaussian, periodic and periodic-Gaussian.

    A fixed reference whitening map avoids repeated Cholesky in Taylor AD.
    The physical metric at the perturbed inputs remains in the KKT constraint,
    so overlap/volume/electron-number response is not frozen.
    Reference validity is checked in the original raw KKT units. The returned
    Problem and Response residuals divide stationarity by the reference N;
    the state stores physical chemical potential, and the constraint is
    (electron_count-N)/N_ref. Linear tolerances act on these normalized equations.
    """
    _validate(data)
    cfg = config or OFDFTConfig()
    kinetic = KineticFunctional() if kinetic is None else kinetic
    args = (data, {} if kinetic_params is None else kinetic_params,
            {} if xc_params is None else xc_params)
    number = jnp.asarray(data.nelectron)
    x = _to_sphere(result.coefficients, data)
    if data.representation == 'periodic':
        transform = jnp.sqrt(number/data.weights)
        coefficients = lambda y: transform*y
    else:
        transform = _to_coefficients(jnp.eye(x.size), data)
        coefficients = lambda y: transform@y

    def energy(y, a):
        d, p, xp = a
        return energy_components(coefficients(y), d, kinetic=kinetic, kinetic_params=p,
                                  xc=cfg.xc, xc_energy_fn=xc_energy_fn, xc_params=xp).total

    def count(y, a):
        q, d = coefficients(y), a[0]
        return jnp.sum(d.weights*q*q) if data.representation=='periodic' else q@d.overlap@q

    gradient, metric_gradient = jax.grad(energy), jax.grad(count)
    initial = jnp.concatenate((x, jnp.atleast_1d(result.chemical_potential)))

    def raw_residual(state, a):
        y, mu = state[:-1], state[-1]
        return jnp.concatenate((gradient(y, a)-mu*metric_gradient(y, a),
                                jnp.atleast_1d((count(y, a)-a[0].nelectron)/number)))

    reference = raw_residual(initial, args)
    valid = result.converged & jnp.all(jnp.isfinite(reference)) & (jnp.max(jnp.abs(reference), initial=0.) <= tolerance)

    def residual(state, a):
        return raw_residual(state, a).at[:-1].divide(number)

    def quantities(state, a):
        y = state[:-1]
        return {'density': amplitude(coefficients(y), a[0])**2,
                'energy': energy(y, a), 'electron_number': count(y, a),
                'chemical_potential': state[-1]}

    def potential_tangent(value):
        value = jnp.asarray(value)
        if value.shape != data.external_potential.shape:
            raise ValueError('Potential perturbation must match OFDFT external_potential shape')
        if jnp.iscomplexobj(value):
            raise ValueError('OFDFT potential perturbation must be real')
        value = value.astype(data.external_potential.dtype)
        if data.representation == 'gaussian':
            value *= jnp.where(jnp.max(jnp.abs(value-value.T)) <= 1e-10, 1., jnp.nan)
        zero = jax.tree.map(jnp.zeros_like, args)
        return (replace(zero[0], external_potential=value), zero[1], zero[2])

    def potential_path(tangent, t):
        return (replace(data, external_potential=data.external_potential+t*tangent[0].external_potential), args[1], args[2])
    return Problem(residual, initial, args, quantities, valid, tolerance,
                    potential_tangent, potential_path)
