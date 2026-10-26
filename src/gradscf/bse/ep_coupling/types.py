"""Fixed vibrational vertices in a neutral-excitation basis."""

from dataclasses import dataclass

from ...scf._pytree import pytree_dataclass
from ...gw.ep_coupling.types import (
    PhononModel as _PhononModel,
    validate_model as _validate,
)


@pytree_dataclass(static_fields=("reference",))
@dataclass(frozen=True)
class PhononModel(_PhononModel):
    """Neutral-excitation vertices, relative to the vibrational ground surface.

    Reuses the GW phonon data layout: energies (nmode,), couplings
    (nmode,nstate,nstate), and optional compact diagonal or full mode-pair
    quadratic vertices, all in Ha. Q=b+b† and H_X(Q)=H_X+G Q+Lambda QQ/2.
    Quadratic vertices are operator derivatives, not Hessians of individual
    eigenvalues which would already include linear-vertex state mixing.
    The phonon bath is fixed, with negligible excitation population.
    """


def validate_model(model, *, nstates=None):
    if not isinstance(model, PhononModel):
        raise TypeError(
            "Expected a BSE ep_coupling.PhononModel in the excitation basis"
        )
    _validate(model, norb=nstates, real=False)
