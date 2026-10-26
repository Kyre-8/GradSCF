"""Eager molecular BSE facade; functional numerical path is run_bse."""

from dataclasses import fields
import numpy as np
from .types import BSEConfig
from .space import make_bse_space, SpinBSESpace
from .reference import reference_from_source, source_signature
from .response import run_bse
from .properties import transition_dipoles, oscillator_strengths, polarizability, absorption_cross_section


class BSE:
    def __init__(
        self,
        source,
        *,
        occupied=None,
        virtual=None,
        screening_occupied=None,
        screening_virtual=None,
        **kwargs,
    ):
        from ..gw.ugw import UGW
        if isinstance(source, UGW) or isinstance(getattr(source, "nocc", None), tuple):
            kwargs.setdefault("singlet", None)
        cfg = BSEConfig(**kwargs)
        self.source = source
        self.occupied = None if occupied is None else tuple(occupied)
        self.virtual = None if virtual is None else tuple(virtual)
        self.screening_occupied = (
            None if screening_occupied is None else tuple(screening_occupied)
        )
        self.screening_virtual = (
            None if screening_virtual is None else tuple(screening_virtual)
        )
        for field in fields(cfg):
            setattr(self, field.name, getattr(cfg, field.name))
        self.result = self.reference = self.space = self.e = self.xy = None
        self.converged = None
        self._state_key = None

    def _config(self):
        return BSEConfig(
            **{field.name: getattr(self, field.name) for field in fields(BSEConfig)}
        )

    def _key(self):
        return (
            source_signature(self.source),
            self._config(),
            self.occupied,
            self.virtual,
            self.screening_occupied,
            self.screening_virtual,
        )

    def kernel(self):
        self._state_key = None
        self.converged = None
        cfg = self._config()
        ref = reference_from_source(
            self.source,
            max_aux=cfg.max_aux,
            max_factor_elements=cfg.max_factor_elements,
        )
        self.reference = ref
        nmo = np.shape(ref.qp_energy)[-1]
        self.space = space = make_bse_space(
            nmo, ref.nocc, occupied=self.occupied, virtual=self.virtual
        )
        screen = make_bse_space(
            nmo,
            ref.nocc,
            occupied=ref.screening_occupied if self.screening_occupied is None else self.screening_occupied,
            virtual=ref.screening_virtual if self.screening_virtual is None else self.screening_virtual,
        )
        channels = space.channels if isinstance(space, SpinBSESpace) else (space,)
        for mask in (ref.qp_computed_mask, ref.qp_converged_mask):
            if mask is not None:
                valid = np.shape(mask) == np.shape(ref.qp_energy) and np.asarray(mask).dtype == np.bool_
                if valid:
                    masks = mask if isinstance(space, SpinBSESpace) else (mask,)
                    valid = all(np.all(np.asarray(m)[list(c.occupied + c.virtual)])
                                for c, m in zip(channels, masks))
                if not valid:
                    raise ValueError(
                        "Every selected BSE level requires an actually computed, converged QP energy"
                    )
        self.result = run_bse(
            ref.qp_energy,
            ref.screening_energy,
            ref.mo_factors,
            space,
            screening_space=screen,
            qp_computed_mask=ref.qp_computed_mask,
            qp_converged_mask=ref.qp_converged_mask,
            config=cfg,
        )
        if not bool(self.result.screening_valid):
            raise ValueError(
                "BSE screening requires finite symmetric factors and positive screening gaps"
            )
        self.e = self.result.excitation_energies
        self.xy = (self.result.x_amplitudes, self.result.y_amplitudes)
        self.converged = np.asarray(self.result.converged)
        self._state_key = self._key()
        return self.e, self.xy

    def run(self):
        self.kernel()
        return self

    def _ready(self):
        if self.result is None or self._state_key is None:
            raise RuntimeError("Run BSE before requesting optical properties")
        if self._key() != self._state_key:
            raise RuntimeError("BSE source or settings changed; run kernel() again")
        if not np.all(self.result.converged & self.result.stable):
            raise RuntimeError(
                "Converge stable BSE roots before requesting optical properties"
            )
        if self.reference.dipole_mo is None:
            raise ValueError("BSE optical properties require dipole_mo inputs")

    def transition_dipole(self):
        self._ready()
        return transition_dipoles(self.result, self.reference.dipole_mo, self.space)

    def oscillator_strength(self):
        self._ready()
        return oscillator_strengths(self.result, self.reference.dipole_mo, self.space)


    def polarizability(self, omega=0., *, eta=0.):
        """Retarded tensor (a0^3), with frequencies and broadening in Ha."""
        self._ready()
        return polarizability(self.result, self.reference.dipole_mo, self.space, omega, eta=eta)

    def absorption_cross_section(self, omega, *, eta=.01, polarization=None, unit='au'):
        """Finite-root absorption, averaged or polarized, in a0^2 or Mb."""
        self._ready()
        return absorption_cross_section(self.result, self.reference.dipole_mo, self.space,
            omega, eta=eta, polarization=polarization, unit=unit)
