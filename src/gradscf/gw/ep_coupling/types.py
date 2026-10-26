"""Fixed external phonon model and input validation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ...scf._pytree import pytree_dataclass


@pytree_dataclass(static_fields=("reference",))
@dataclass(frozen=True)
class PhononModel:
    """Molecular modes in the fixed initial orthonormal MO frame.

    energies: (nmode,), strictly positive phonon energies in Ha.
    couplings: (nmode, norb, norb), Hermitian linear vertices in Ha.
    quadratic: optional (nmode, norb, norb) diagonal-mode vertices or
        (nmode, nmode, norb, norb) in Ha. Full mode-pair tensors must be
        symmetric in mode indices; both layouts are Hermitian in orbital
        indices. The diagonal gives fixed-D0 DW; it is not inferred from g.
    reference: provenance only; no implicit double-counting correction.

    Call validate_model before using numerical kernels. Construction does
    not validate values, allowing this type to also represent AD tangents.
    """

    energies: jax.Array
    couplings: jax.Array
    quadratic: jax.Array | None = None
    reference: str = "external"


def _require(condition, message):
    """Reject concrete or traced invalid inputs without modifying physics."""
    def check(value):
        if not bool(np.asarray(value)):
            raise ValueError(message)
    if isinstance(condition, jax.core.Tracer):
        jax.debug.callback(check, condition, ordered=True)
    else:
        check(condition)


def validate_model(model, *, norb=None, real=True):
    """Check shape, finite positive modes, and vertex symmetries.

    real=True is required by the molecular scGW driver. Value validation
    also executes under JIT (an invalid traced value raises an XLA callback
    error containing the original ValueError). No frequency is clamped.
    """
    if not isinstance(model, PhononModel):
        raise TypeError("model must be a PhononModel")
    omega, g = jnp.asarray(model.energies), jnp.asarray(model.couplings)
    if omega.ndim != 1 or not omega.shape[0]:
        raise ValueError("phonon energies must have nonempty shape (nmode,)")
    if g.ndim != 3 or g.shape[0] != omega.shape[0] or g.shape[1] != g.shape[2] or not g.shape[1]:
        raise ValueError("couplings must have shape (nmode, norb, norb)")
    if norb is not None and g.shape[1] != norb:
        raise ValueError("couplings shape does not match norb")
    if jnp.iscomplexobj(omega):
        raise ValueError("phonon energies must be real and positive")
    if real and jnp.iscomplexobj(g):
        raise ValueError("molecular scGW requires real phonon couplings")
    if not isinstance(model.reference, str):
        raise TypeError("reference must be a static provenance string")
    _require(jnp.all(jnp.isfinite(omega) & (omega > 0)), "phonon energies must be finite and strictly positive")
    _require(jnp.all(jnp.isfinite(g)), "phonon couplings must be finite")
    _require(jnp.allclose(g, g.swapaxes(-1, -2).conj(), rtol=1e-10, atol=1e-12),
             "phonon couplings must be Hermitian")
    if model.quadratic is not None:
        quadratic = jnp.asarray(model.quadratic)
        diagonal_shape = (omega.shape[0],g.shape[1],g.shape[1])
        if quadratic.shape not in (diagonal_shape,(omega.shape[0],)+diagonal_shape):
            raise ValueError("quadratic must have shape (nmode,norb,norb) or (nmode,nmode,norb,norb)")
        if real and jnp.iscomplexobj(quadratic):
            raise ValueError("molecular scGW requires real quadratic vertices")
        _require(jnp.all(jnp.isfinite(quadratic)), "quadratic vertices must be finite")
        _require(jnp.allclose(quadratic, quadratic.swapaxes(-1, -2).conj(), rtol=1e-10, atol=1e-12),
                 "quadratic vertices must be Hermitian in orbital indices")
        if quadratic.ndim == 4:
            _require(jnp.allclose(quadratic, quadratic.swapaxes(0, 1), rtol=1e-10, atol=1e-12),
                     "quadratic vertices must be symmetric in mode indices")


@pytree_dataclass(static_fields=("reference",))
@dataclass(frozen=True)
class PeriodicPhononModel:
    """Fixed independent q modes; complex vertices map internal k+q to k.

    energies (nq,nmode), couplings (nq,nk,nmode,norb,norb), k_plus_q (nq,nk)
    and normalized q_weights (nq,). All energies and zero-point-normalized
    vertices are in Ha. The reference label does not alter the supplied data.
    This is a Fan model, without periodic DW or a periodic scGW driver.
    """
    energies: jax.Array
    couplings: jax.Array
    k_plus_q: jax.Array
    q_weights: jax.Array
    reference: str = "external"


def _validate_q_data(couplings, k_plus_q, q_weights, *, nk, norb):
    g, mapping, weights = map(jnp.asarray, (couplings, k_plus_q, q_weights))
    if g.ndim != 5 or g.shape[1] != nk or g.shape[-2:] != (norb, norb):
        raise ValueError("couplings must have shape (nq,nk,nmode,norb,norb)")
    nq, _, nmode, _, _ = g.shape
    if not nq or not nmode or not nk or not norb:
        raise ValueError("periodic mode and orbital dimensions must be nonempty")
    if mapping.shape != (nq, nk) or weights.shape != (nq,):
        raise ValueError("mapping or q_weights shape is inconsistent")
    if not jnp.issubdtype(mapping.dtype, jnp.integer):
        raise ValueError("k_plus_q mapping must contain integer indices")
    if jnp.iscomplexobj(weights):
        raise ValueError("q_weights must be real nonnegative normalized weights")
    _require(jnp.all((mapping >= 0) & (mapping < nk)), "k_plus_q mapping index out of bounds")
    _require(jnp.all(jnp.isfinite(weights) & (weights >= 0))
             & jnp.isclose(jnp.sum(weights), 1., rtol=1e-10, atol=1e-12),
             "q_weights must be finite nonnegative normalized weights")
    _require(jnp.all(jnp.isfinite(g)), "periodic couplings must be finite")
    return g, mapping, weights


def validate_periodic_model(model, *, nk, norb):
    if not isinstance(model, PeriodicPhononModel):
        raise TypeError("model must be a PeriodicPhononModel")
    g, _, _ = _validate_q_data(model.couplings, model.k_plus_q, model.q_weights,
                              nk=nk, norb=norb)
    omega = jnp.asarray(model.energies)
    if omega.shape != (g.shape[0], g.shape[2]) or jnp.iscomplexobj(omega):
        raise ValueError("periodic phonon energies must have real shape (nq,nmode)")
    _require(jnp.all(jnp.isfinite(omega) & (omega > 0)), "phonon energies must be finite and positive")
    if not isinstance(model.reference, str):
        raise TypeError("reference must be a static provenance string")
