"""Host-only eta=0 pole seeds, using the existing shared Hermitian solver.

This numerical branch selection is not a differentiable observable. It does
not replace the finite-eta CD equation or certify its global maximum-Z root.
"""
from numbers import Integral, Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian


class PoleCandidates(NamedTuple):
    """Four leading groups per requested orbital, in descending weight order.

    Energies are group centroids in Hartree; weights are complete central
    projector weights, not individual vectors inside a degenerate group.
    Padding has energy NaN, weight zero and size zero.
    """
    energies: np.ndarray
    weights: np.ndarray
    sizes: np.ndarray


class PoleSeedResult(NamedTuple):
    """Arrays follow ``orbs``; invalid seeds are NaN and must not be used.

    ``weight_gap`` is dimensionless; ``pole_gap`` and ``residual_norms`` are
    in Hartree. Residuals are the maximum over the complete shared-solver
    spectrum for each orbital. Diagnostics remain available for invalid seeds.
    """
    orbs: tuple[int, ...]
    seeds: np.ndarray
    weights: np.ndarray
    candidates: PoleCandidates
    weight_gap: np.ndarray
    pole_gap: np.ndarray
    residual_norms: np.ndarray
    valid: np.ndarray


def _real_array(name, value):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64, copy=False)


def pole_seeds(base_energy, mo_energy, nocc, poles, couplings, *, orbs=None,
               max_dense=2048, gap_tol=1e-10, weight_tol=1e-8):
    """Select isolated largest-weight groups of a real restricted eta=0 model.

    ``base_energy`` already includes the static correction, epsilon_p+delta_v.
    ``mo_energy`` contains the G pole energies, with the first ``nocc`` occupied.
    Positive ``poles`` and symmetric real ``couplings`` of shape (npole,nmo,nmo)
    use the convention returned by :func:`gradscf.gw.poles.rpa_pole_expansion`.
    No extra exchange or spin factor is added here.

    For orbital p, the arrowhead diagonal is (base_p, epsilon_m+s_m*Omega_nu),
    with s=-1 occupied/+1 virtual, and vertices C_nu,mp/sqrt(2*Omega_nu).
    The squared central projection is its eta=0 spectral weight. Adjacent
    eigenvalues within ``gap_tol`` plus their measured residuals form a group;
    its complete projector weight is summed before ranking. The selected
    group must be a singleton, separated from every bath pole and from the
    next group in weight by ``weight_tol``. Otherwise ``valid`` is false.

    Eager host orchestration only: JIT/AD tracers are rejected and JAX float64
    must be enabled. The dense dimension 1+npole*nmo is checked before assembly.
    The shared dense primal uses a 1e-10 Eh residual gate; only its stopped
    raw diagnostics are used. All outputs follow requested ``orbs`` (all by
    default). Use accepted seeds in the existing nonlinear QP driver; finite
    eta, geometry continuation, open shells and periodic systems need their
    own validation. Individual canonical MO branches are assumed defined.
    """
    arguments = (base_energy, mo_energy, nocc, poles, couplings, orbs,
                 max_dense, gap_tol, weight_tol)
    if any(isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(arguments)):
        raise TypeError("pole_seeds is host-only; construct numerical seeds outside JIT and AD")
    if not jax.config.x64_enabled:
        raise ValueError("pole_seeds requires JAX float64 enabled")
    base = _real_array("base_energy", base_energy)
    energy = _real_array("mo_energy", mo_energy)
    omega = _real_array("poles", poles)
    coupling = _real_array("couplings", couplings)
    if base.ndim != 1 or base.size < 2:
        raise ValueError("base_energy must have shape (nmo,) with nmo >= 2")
    if energy.shape != base.shape:
        raise ValueError("mo_energy must have the same (nmo,) shape as base_energy")
    nmo = base.size
    if not isinstance(nocc, Integral) or isinstance(nocc, bool) or not 0 < nocc < nmo:
        raise ValueError("nocc must be an integer with nonempty occupied and virtual spaces")
    if omega.ndim != 1 or not omega.size or np.any(omega <= 0):
        raise ValueError("poles must be a nonempty vector of positive frequencies")
    if coupling.shape != (omega.size, nmo, nmo):
        raise ValueError("couplings must have shape (npole,nmo,nmo)")
    symmetry_tol = 64*np.finfo(np.float64).eps*max(1., float(np.max(abs(coupling))))
    if np.max(abs(coupling-coupling.swapaxes(1, 2))) > symmetry_tol:
        raise ValueError("couplings must be symmetric in their real MO-pair indices")
    if not isinstance(max_dense, Integral) or isinstance(max_dense, bool) or max_dense < 1:
        raise ValueError("max_dense must be a positive integer")
    for name, value in (("gap_tol", gap_tol), ("weight_tol", weight_tol)):
        if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    try:
        selected = tuple(range(nmo)) if orbs is None else tuple(orbs)
    except TypeError as error:
        raise ValueError("orbs must contain distinct in-range integer indices") from error
    if (not selected or any(not isinstance(p, Integral) or isinstance(p, bool)
                            or not 0 <= p < nmo for p in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("orbs must contain distinct in-range integer indices")
    selected = tuple(map(int, selected))
    dimension = 1+omega.size*nmo
    if dimension > max_dense:
        raise ValueError(f"Pole arrowhead dimension {dimension} exceeds max_dense={max_dense}")

    count = len(selected)
    seeds, weights = np.full(count, np.nan), np.full(count, np.nan)
    weight_gap, pole_gap, residuals = (np.full(count, np.nan) for _ in range(3))
    candidates = PoleCandidates(np.full((count, 4), np.nan), np.zeros((count, 4)),
                                np.zeros((count, 4), dtype=int))
    valid = np.zeros(count, dtype=bool)
    sign = np.where(np.arange(nmo) < nocc, -1., 1.)
    bath = (energy[None, :]+omega[:, None]*sign[None, :]).ravel()
    config = EigenSolverConfig(method="dense", nroots=dimension, max_dense=int(max_dense), atol=1e-10)
    response = EigenResponseConfig(target="subspace")
    for row, p in enumerate(selected):
        matrix = np.diag(np.r_[base[p], bath])
        matrix[0, 1:] = matrix[1:, 0] = (coupling[:, :, p]/np.sqrt(2*omega)[:, None]).ravel()
        solved = solve_hermitian(jax.lax.stop_gradient(jnp.asarray(matrix)), config=config, response=response)
        values, vectors, errors = map(np.asarray,
            (solved.raw_values, solved.raw_vectors, solved.raw_residual_norms))
        residuals[row] = np.max(errors)
        if not all(np.isfinite(array).all() for array in (values, vectors, errors)):
            continue
        order = np.argsort(values, kind="stable")
        values, errors, projection = values[order], errors[order], vectors[0, order]**2
        cuts = np.flatnonzero(np.diff(values) > gap_tol+errors[:-1]+errors[1:])+1
        groups = np.split(np.arange(dimension), cuts)
        ranked = sorted(((float(np.mean(values[group])), float(projection[group].sum()), len(group))
                         for group in groups), key=lambda item: -item[1])
        for column, (value, weight, size) in enumerate(ranked[:4]):
            candidates.energies[row, column] = value
            candidates.weights[row, column] = weight
            candidates.sizes[row, column] = size
        value, weights[row], size = ranked[0]
        weight_gap[row] = weights[row]-(ranked[1][1] if len(ranked) > 1 else 0.)
        pole_gap[row] = np.min(abs(value-bath))
        valid[row] = (np.all(np.asarray(solved.converged)) and np.all(np.asarray(solved.raw_present))
                      and size == 1 and weight_gap[row] > weight_tol
                      and pole_gap[row] > gap_tol+residuals[row])
        if valid[row]:
            seeds[row] = value
    return PoleSeedResult(selected, seeds, weights, candidates, weight_gap, pole_gap, residuals, valid)


__all__ = ["PoleCandidates", "PoleSeedResult", "pole_seeds"]
