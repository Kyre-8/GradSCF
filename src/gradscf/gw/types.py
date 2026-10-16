"""Result container for GW calculations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..scf._pytree import pytree_dataclass


@pytree_dataclass(static_fields=("nw", "g_orbitals", "screening_occupied", "screening_virtual", "method"))
@dataclass(frozen=True)
class GWResult:
    """Quasiparticle energies from a (G0W0/evGW/qsGW/scGW) calculation.

    Attributes
    ----------
    mo_energy:
        Quasiparticle energies, shape ``(nmo,)`` (restricted) or
        ``(2, nmo)`` (unrestricted), or ``(nk, nmo)`` (periodic restricted).
        Orbitals not included in the
        self-energy evaluation are prefilled with the mean-field values
        (unlike PySCF, which fills them with zeros).
    mo_coeff:
        Orbital coefficients (identical to the mean-field input for G0W0;
        updated for qsGW/scGW).
    converged:
        True when every requested Dyson residual meets the tolerance.
        Secant solves additionally require |dx| < tol. A scalar boolean
        array under JAX transformations, rather than static pytree metadata.
        For qsGW, this instead certifies the static Fock residual together
        with the undamped spectral and density updates.
    converged_mask:
        Per-orbital convergence flags (False where the secant iteration did
        not meet the step and residual tolerances; the returned energy is
        then the best-residual iterate). Orbitals not requested are marked True.
    sigma_qp:
        Diagonal correlation self-energy at the converged QP energies
        (complex), same shape as ``mo_energy``; ``None`` when not stored.
    nw:
        Number of imaginary-axis quadrature points used.
    qp_residual:
        Dyson residual at ``mo_energy`` for the supplied G/W pole spectrum
        (Ha). Unrequested orbitals are filled with zero. ``None`` for
        methods which do not store this diagnostic.
    qp_computed_mask:
        True where a QP calculation was actually requested and performed;
        inspect converged_mask as well to require successful convergence.
        G0W0 evaluate-only calls mark all entries False. evGW fills this mask
        when its outer equations converge. None means coverage is not recorded
        by the producing driver. MF-filled levels must not be inferred from
        converged_mask or zero qp_residual alone.
    qp_weight:
        Local Z=1/(1-d Re Sigma/d omega) at the returned energy, with G/W
        spectra and residue branch fixed. Restricted CD paths provide it;
        other methods may leave it None. Unrequested levels and singular
        slopes are NaN. Inspect QP coverage/convergence as well; Z is not a
        convergence certificate or a differentiable root-selection rule.
    g_orbitals, screening_occupied, screening_virtual:
        Recorded original-MO correlation windows. For restricted periodic CD,
        these band-index windows apply at every screening/internal k point;
        qp_computed_mask separately records selected output k/band pairs.
        None denotes unrecorded metadata; explicit empty tuples mean no sums.
    method:
        CD method/backend label, or None for drivers without recorded metadata.
    screening_energy:
        Actual independent-particle pole spectrum used to build W, in the
        returned orbital frame. Restricted/unrestricted CD drivers record it;
        at a converged evGW fixed point it equals mo_energy. None means the
        producing driver supplies no such static-screening provenance. This
        field does not imply differentiability of an outer self-consistency loop.
    """

    mo_energy: jnp.ndarray
    mo_coeff: jnp.ndarray
    converged: bool | jnp.ndarray
    sigma_qp: jnp.ndarray | None = None
    converged_mask: jnp.ndarray | None = None
    nw: int | None = None
    qp_residual: jnp.ndarray | None = None
    # True only for levels whose QP energies were actually computed. Legacy
    # convergence masks mark unrequested MF-filled levels True and are distinct.
    # None means the producing driver has not supplied coverage metadata.
    qp_computed_mask: jnp.ndarray | None = None
    screening_energy: jnp.ndarray | None = None

    # Local frequency derivative at the returned energy, with G and W fixed.
    # This is diagnostic, not a convergence certificate; unrequested levels are NaN.
    qp_weight: jnp.ndarray | None = None
    g_orbitals: tuple[int, ...] | None = None
    screening_occupied: tuple[int, ...] | None = None
    screening_virtual: tuple[int, ...] | None = None
    method: str | None = None


__all__ = ["GWResult"]
