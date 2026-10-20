"""Short eager MP interfaces; numerical differentiation belongs to run_mp."""
from dataclasses import fields
import numpy as np

from ..scf.reference import (reference_from_source, unrestricted_reference_from_source,
                             is_unrestricted_source, RestrictedReference, UnrestrictedReference)
from .types import MPConfig
from .mp2 import run_mp, evaluate_mp2
from .integrals import from_scf
from ..scf.facade import RKS, UKS


class MP:
    """Generic order-driven MP; defaults to the residual/Taylor series engine."""
    unrestricted = None
    method_order = None

    def __init__(self, mf, *, order=None, frozen=None, **kwargs):
        spin = is_unrestricted_source(mf)
        if self.unrestricted is not None and spin != self.unrestricted:
            raise ValueError("Reference spin does not match restricted/unrestricted MP class")
        order = (self.method_order or 2) if order is None else order
        if self.method_order is not None and order != self.method_order:
            raise ValueError("Use MP(mf, order=...) to select a different perturbation order")
        self.mf, self.frozen = mf, frozen
        self._unrestricted = spin
        kwargs.setdefault("algorithm", "series" if self.method_order is None else "auto")
        cfg = MPConfig(order=order, **kwargs)
        for field in fields(cfg):
            setattr(self, field.name, getattr(cfg, field.name))
        self.result = None
        self.e_corr = self.e_tot = self.e2 = self.e3 = self.t2 = None
        self.e_corr_ss = self.e_corr_os = None
        self.converged = False
        self.corrections = self.wavefunction_coefficients = self.space = None

    def kernel(self):
        self.converged = False
        self.result = None
        self.e_corr = self.e_tot = self.e2 = self.e3 = self.t2 = None
        self.e_corr_ss = self.e_corr_os = None
        self.corrections = self.wavefunction_coefficients = self.space = None
        cfg = MPConfig(**{f.name: getattr(self, f.name) for f in fields(MPConfig)})
        series = cfg._use_series(self._unrestricted)
        explicit = isinstance(self.mf, (RestrictedReference, UnrestrictedReference))
        if series or cfg.order == 3 or explicit:
            if not explicit:
                if not isinstance(self.mf, (RKS, UKS)):
                    raise NotImplementedError("Canonical MP requires RHF/UHF or explicit MO references")
                self.mf._check_reference_source()
            if series:
                from .series import make_series_space
                nmo = np.shape(self.mf.h1[0] if self._unrestricted else self.mf.h1)[0] if explicit else self.mf.mo_coeff.shape[-1]
                if explicit:
                    no = self.mf.nocc
                else:
                    occ = np.asarray(self.mf.mo_occ)
                    no = (tuple(int(np.count_nonzero(x)) for x in occ) if self._unrestricted
                          else int(np.count_nonzero(occ)))
                self.space = make_series_space(nmo, no, frozen=self.frozen, config=cfg)
            ref = (unrestricted_reference_from_source if self._unrestricted else reference_from_source)(self.mf)
            result = run_mp(ref.h1, ref.eri, nocc=ref.nocc,
                frozen=self.frozen, nuclear_repulsion=ref.nuclear_repulsion, config=cfg)
        else:
            result = evaluate_mp2(from_scf(self.mf, frozen=self.frozen,
                                           frozen_gap_tol=cfg.denominator_tol), cfg)
        self.result = result
        self.e_corr, self.e_tot, self.e2, self.e3 = (result.correlation_energy,
            result.total_energy, result.e2, result.e3)
        self.t2 = result.t2
        self.e_corr_ss, self.e_corr_os = result.same_spin_energy, result.opposite_spin_energy
        self.corrections = result.corrections
        self.wavefunction_coefficients = result.wavefunction_coefficients
        self.converged = bool(result.valid)
        if not self.converged:
            raise RuntimeError("MP requires stationary HF data in the selected orbital basis and resolved physical denominators")
        return self.e_corr, self.t2

    def run(self):
        self.kernel()
        return self


class RMP2(MP):
    unrestricted = False
    method_order = 2


class UMP2(RMP2):
    unrestricted = True


class RMP3(RMP2):
    method_order = 3


def MP2(mf, **kwargs):
    return (UMP2 if is_unrestricted_source(mf) else RMP2)(mf, **kwargs)


def MP3(mf, **kwargs):
    if is_unrestricted_source(mf):
        return MP(mf, order=3, **kwargs)
    return RMP3(mf, **kwargs)
