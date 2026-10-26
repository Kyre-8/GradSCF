"""Eager checked G0W0/evGW/evGW0 and explicit fixed-MO reference snapshots."""

from dataclasses import dataclass
import jax.numpy as jnp
from ..scf.reference import _array_signature


@dataclass(frozen=True)
class BSEReference:
    """Explicit real MO inputs. Omitted masks mean user-supplied QP energies.

    The caller supplies one orthonormal frame, common to the spectra, factors
    and optional dipoles. Screening energies are always explicit.
    """

    qp_energy: object
    screening_energy: object
    mo_factors: object
    nocc: int | tuple[int, int]
    dipole_mo: object = None
    qp_computed_mask: object = None
    qp_converged_mask: object = None
    screening_occupied: tuple[int, ...] | None = None
    screening_virtual: tuple[int, ...] | None = None

    def __post_init__(self):
        for name in ("screening_occupied", "screening_virtual"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_gw_result(cls, result, *, mo_factors, nocc, dipole_mo=None):
        """Build a fixed-frame snapshot from a recorded real CD result.

        G0W0, converged evGW and evGW0 record their actual screening spectrum.
        Caller-supplied factors/dipoles must be in result.mo_coeff's frame.
        Unknown screening or QP coverage is rejected, never inferred. This
        adapter preserves the producing driver's differentiation contract.
        Use nocc=(nalpha,nbeta) and spin-resolved factors for unrestricted data.
        """
        from ..gw.types import GWResult

        if not isinstance(result, GWResult):
            raise TypeError("Expected a GWResult")
        if result.screening_energy is None:
            raise ValueError("GW result has no recorded screening spectrum")
        if result.qp_computed_mask is None or result.converged_mask is None:
            raise ValueError("GW result has no QP coverage/convergence metadata")
        qp, screening = jnp.asarray(result.mo_energy), jnp.asarray(
            result.screening_energy
        )
        spin = isinstance(nocc, tuple)
        if (qp.ndim != (2 if spin else 1) or (spin and qp.shape[0] != 2)
                or screening.shape != qp.shape or jnp.iscomplexobj(qp)):
            raise ValueError("BSE requires a real GW result consistent with nocc")
        return cls(
            qp,
            screening,
            mo_factors,
            nocc,
            dipole_mo,
            result.qp_computed_mask,
            result.converged_mask,
            result.screening_occupied,
            result.screening_virtual,
        )


def source_signature(source):
    if isinstance(source, BSEReference):
        arrays = (
            source.qp_energy,
            source.screening_energy,
            source.mo_factors,
            source.dipole_mo,
            source.qp_computed_mask,
            source.qp_converged_mask,
        )
        return source.nocc, source.screening_occupied, source.screening_virtual, tuple(
            None if x is None else _array_signature(x) for x in arrays
        )
    from ..gw.rgw import GW
    from ..gw.ugw import UGW

    if not isinstance(source, (GW, UGW)):
        raise NotImplementedError(
            "BSE accepts a real GW/UGW facade or explicit BSEReference"
        )
    return source.state_signature()


def reference_from_source(source, *, max_aux=1024, max_factor_elements=20_000_000):
    if isinstance(source, BSEReference):
        return source
    from ..gw.rgw import GW
    from ..gw.ugw import UGW

    if not isinstance(source, (GW, UGW)):
        raise NotImplementedError(
            "BSE accepts a real GW/UGW facade or explicit BSEReference"
        )
    data = source.get_bse_inputs(
        max_aux=max_aux, max_factor_elements=max_factor_elements
    )
    dipole = None
    if data["dipole_ao"] is not None:
        c = jnp.asarray(data["mo_coeff"])
        dipole = jnp.einsum(
            "xmn,...mi,...nj->...xij", data["dipole_ao"], c, c, precision="highest"
        )
    return BSEReference(
        data["qp_energy"],
        data["screening_energy"],
        data["mo_factors"],
        data["nocc"],
        dipole,
        data["qp_computed_mask"],
        data["qp_converged_mask"],
        data["screening_occupied"],
        data["screening_virtual"],
    )
