"""Small one/two-coordinate polynomial vibrational models, in atomic units.

This module assembles matrices and certifies supplied quartic polynomials. It
contains no electronic, optimization or eigenvalue solver. Coordinates are in
bohr with explicit electron-unit masses, or mass-weighted with unit masses.
Polynomial coefficients multiply monomials directly (factorials included).
"""
from fractions import Fraction
from math import comb, inf, nextafter, prod

import jax
import jax.numpy as jnp
import numpy as np


def oscillator_powers(frequency, mass, size, *, max_power=4):
    """Return P Q^k P after forming powers in a padded oscillator space.

    Padding before projection is essential even for diagonal Q^4 matrix
    elements at the top retained state. Powers of an already truncated Q
    would omit paths that leave and return to the retained space.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Oscillator matrices require float64 enabled.")
    if (np.ndim(frequency) or np.ndim(mass) or np.iscomplexobj(frequency) or np.iscomplexobj(mass)
            or not np.isfinite(frequency) or not np.isfinite(mass) or frequency <= 0 or mass <= 0):
        raise ValueError("Frequency and mass must be finite and positive.")
    if type(size) is not int or size < 1 or type(max_power) is not int or not 0 <= max_power <= 4:
        raise ValueError("Size must be positive and maximum power an integer from zero to four.")
    padded = size+max_power
    ladder = jnp.diag(jnp.sqrt(jnp.arange(1, padded, dtype=jnp.float64)), 1)
    coordinate = (ladder+ladder.T)/jnp.sqrt(2*mass*frequency)
    powers = [jnp.eye(padded)]
    for _ in range(max_power):
        powers.append(powers[-1]@coordinate)
    return tuple(matrix[:size, :size] for matrix in powers)


def _inputs(frequencies, masses, sizes, coefficients):
    frequencies, masses = np.asarray(frequencies), np.asarray(masses)
    sizes = tuple(sizes)
    if len(sizes) not in (1, 2) or frequencies.shape != (len(sizes),) or masses.shape != frequencies.shape:
        raise ValueError("Supply one or two oscillator coordinates with matching frequencies and masses.")
    if any(type(size) is not int or size < 1 for size in sizes) or prod(sizes) > 1024:
        raise ValueError("Positive static sizes must give at most 1024 basis states.")
    for powers, value in coefficients.items():
        if len(powers) != len(sizes) or any(type(p) is not int or p < 0 for p in powers) or sum(powers) > 4:
            raise ValueError("Monomial powers must match the coordinates and have total degree <=4.")
        if np.ndim(value) or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError("Polynomial coefficients must be finite real scalars.")
    tables = [oscillator_powers(w, m, n) for w, m, n in zip(frequencies, masses, sizes)]
    return frequencies, masses, sizes, tables


def _matrix(tables, sizes, coefficients):
    matrix = jnp.zeros((prod(sizes), prod(sizes)), dtype=jnp.float64)
    for powers, value in coefficients.items():
        operator = tables[0][powers[0]]
        if len(sizes) == 2:
            operator = jnp.kron(operator, tables[1][powers[1]])
        matrix = matrix+value*operator
    return .5*(matrix+matrix.T)


def polynomial_matrix(frequencies, masses, sizes, coefficients):
    """Multiplicative polynomial operator in the product oscillator basis."""
    *_, sizes, tables = _inputs(frequencies, masses, sizes, coefficients)
    return _matrix(tables, sizes, coefficients)


def polynomial_hamiltonian(frequencies, masses, sizes, coefficients):
    """T+V using exact projected kinetic energy and a degree<=4 potential.

    The caller owns any eigenproblem, confinement check and physical validity
    window. The two-coordinate kinetic energy is diagonal in these coordinates;
    no Watson, rotational, Coriolis or frozen-mode backreaction terms are added.
    """
    frequencies, masses, sizes, tables = _inputs(frequencies, masses, sizes, coefficients)
    states = np.array(list(np.ndindex(*sizes)))
    harmonic = jnp.diag(jnp.asarray(((states+.5)*frequencies).sum(axis=1)))
    correction = dict(coefficients)
    for axis in range(len(sizes)):
        powers = tuple(2 if index == axis else 0 for index in range(len(sizes)))
        correction[powers] = correction.get(powers, 0.)-.5*masses[axis]*frequencies[axis]**2
    return harmonic+_matrix(tables, sizes, correction)


def _bernstein(coefficients, left, right):
    """Exact power-to-Bernstein conversion on a rational interval."""
    degree = len(coefficients)-1
    powers = [sum(coefficients[k]*comb(k, j)*left**(k-j)*(right-left)**j
                  for k in range(j, degree+1)) for j in range(degree+1)]
    return [sum(powers[j]*Fraction(comb(i, j), comb(degree, j)) for j in range(i+1))
            for i in range(degree+1)]


def _angular_lower_bound(coefficients, max_depth):
    """Certify an even-degree binary form on the complete unit circle.

    Supplied float coefficients are treated as exact binary rationals. Two
    cones (1,t)/(1+t²)^.5 and (t,1)/(1+t²)^.5 with |t|<=1 cover the circle
    up to the irrelevant overall sign. No angular sampling substitutes for
    the Bernstein lower bounds. Failure to certify is indeterminate.
    """
    degree = len(coefficients)-1
    lower, depth_used = None, 0
    for cone, values in enumerate((coefficients, coefficients[::-1])):
        stack = [(Fraction(-1), Fraction(1), 0)]
        while stack:
            left, right, depth = stack.pop()
            depth_used = max(depth_used, depth)
            for point in (left, (left+right)/2, right):
                value = sum(c*point**i for i, c in enumerate(values))
                if value <= 0:
                    return "not_positive", None, dict(depth=depth_used, witness_cone=cone,
                                                      witness_ratio=str(point))
            bound = min(_bernstein(values, left, right))
            if bound > 0:
                normalized = bound/(1+max(left**2, right**2))**(degree//2)
                lower = normalized if lower is None else min(lower, normalized)
            elif depth == max_depth:
                return "indeterminate", None, dict(depth=depth_used)
            else:
                middle = (left+right)/2
                stack.extend(((left, middle, depth+1), (middle, right, depth+1)))
    return "positive", lower, dict(depth=depth_used)


def _convolve(first, second):
    output = [Fraction(0)]*(len(first)+len(second)-1)
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            output[i+j] += left*right
    return output


def quartic_certificate(coefficients, *, max_depth=12):
    """Confinement and sufficient unique-stationary-point certificate.

    V=g.q+K[q,q]/2+C[q,q,q]/6+D[q,q,q,q]/24. Positive D on all
    directions makes V coercive. If F=8KD-3C² is also everywhere positive,
    radial curvature is bounded below by m=F_min/(8D_max)>0. Every stationary
    point then lies in ||q||<=||g||_1/m. A positive Hessian lower bound inside
    that ball proves uniqueness, including a small retained linear term.
    Failure of this sufficient test is indeterminate, not a claim of multiple
    minima. Exact rational arithmetic certifies the supplied float polynomial,
    not uncertainty in its electronic force constants or a Taylor remainder.
    """
    if not coefficients or type(max_depth) is not int or not 0 <= max_depth <= 16:
        raise ValueError("Supply a polynomial and a static subdivision depth from zero to sixteen.")
    dimension = len(next(iter(coefficients)))
    if dimension not in (1, 2):
        raise ValueError("The quartic certificate supports one or two coordinates.")
    exact = {}
    for powers, value in coefficients.items():
        if len(powers) != dimension or any(type(p) is not int or p < 0 for p in powers) or sum(powers) > 4:
            raise ValueError("Invalid polynomial powers.")
        if np.ndim(value) or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError("Polynomial coefficients must be finite real scalars.")
        exact[powers] = Fraction.from_float(float(value))

    def form(degree, factor):
        return [factor*exact.get((degree-i, i) if dimension == 2 else (degree,), Fraction(0))
                for i in range(degree+1 if dimension == 2 else 1)]

    quadratic, cubic, quartic = form(2, 2), form(3, 6), form(4, 24)
    radial = [8*left-3*right for left, right in zip(_convolve(quadratic, quartic),
                                                   _convolve(cubic, cubic))]
    bound = (lambda values: _angular_lower_bound(values, max_depth)) if dimension == 2 else (
        lambda values: ("positive", values[0], {}) if values[0] > 0 else ("not_positive", None, {}))
    status, quartic_min, details = bound(quartic)
    result = dict(quartic_status=status, stationary_status="indeterminate",
                  method="exact rational Bernstein bounds" if dimension == 2 else "exact scalar inequalities",
                  quartic_certificate=details)
    if status != "positive":
        return result
    result["quartic_derivative_lower_bound"] = nextafter(float(quartic_min), -inf)
    radial_status, radial_min, details = bound(radial)
    result.update(radial_status=radial_status, radial_certificate=details)
    if radial_status != "positive":
        return result
    minimum = radial_min/(8*sum(abs(value) for value in quartic))
    linear_bound = sum(abs(value) for powers, value in exact.items() if sum(powers) == 1)
    radius = linear_bound/minimum
    cubic_bound = 6*sum(abs(value) for powers, value in exact.items() if sum(powers) == 3)
    quartic_bound = 12*sum(abs(value) for powers, value in exact.items() if sum(powers) == 4)
    hessian_lower = minimum-cubic_bound*radius-quartic_bound*radius**2
    result.update(radial_curvature_lower_bound=nextafter(float(minimum), -inf),
                  stationary_radius=nextafter(float(radius), inf),
                  hessian_lower_bound_in_radius=nextafter(float(hessian_lower), -inf))
    if hessian_lower > 0:
        result["stationary_status"] = "unique"
    return result


__all__ = ["oscillator_powers", "polynomial_matrix", "polynomial_hamiltonian", "quartic_certificate"]
