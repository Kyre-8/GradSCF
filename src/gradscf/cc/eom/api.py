"""Checked eager EOM-CCSD facades; numerical work belongs to shared solvers."""

from dataclasses import fields
import numpy as np
from ...scf.reference import _array_signature
from .types import EOMConfig, EOMPrecisionDiagnostics
from .amplitudes import EOMAmplitudeSpace
from .operators import run_eom, run_eom_from_integrals


class _EOM:
    def __init__(self, ground, *, sector, **kwargs):
        cfg = EOMConfig(sector=sector, **kwargs)
        self.ground = ground
        self._sector = sector
        for field in fields(cfg):
            setattr(self, field.name, getattr(cfg, field.name))
        self.result = self.e = self.right_vectors = self.left_vectors = self.space = (
            None
        )
        self.converged = None
        self._state_key = None

    def _config(self):
        if self.sector != self._sector:
            raise ValueError("Use the matching EOM sector class")
        return EOMConfig(**{f.name: getattr(self, f.name) for f in fields(EOMConfig)})

    def _source(self):
        from ..api import CC

        if not isinstance(self.ground, CC):
            raise TypeError(
                "EOM requires a GradSCF CCSD object; use run_eom for explicit arrays"
            )
        ref = self.ground._ready()
        if self.ground.method != "ccsd":
            raise ValueError("EOM-CCSD requires CCSD amplitudes")
        if not isinstance(ref.nocc, int):
            raise NotImplementedError(
                "Initial EOM-CCSD supports restricted references only"
            )
        for public, stored in (
            (self.ground.t1, self.ground.result.t1),
            (self.ground.t2, self.ground.result.t2),
        ):
            if _array_signature(public) != _array_signature(stored):
                raise RuntimeError("CC amplitudes changed; run CC kernel again")
        return ref

    def _key(self):
        self._source()
        return (
            self.ground._physical_state_key(),
            id(self.ground.result),
            self.ground._config(),
            _array_signature(self.ground.result.t1),
            _array_signature(self.ground.result.t2),
            self._config(),
        )

    def check_source(self):
        if self._state_key is None:
            raise RuntimeError("Run EOM kernel first")
        if self._key() != self._state_key:
            raise RuntimeError("EOM source or settings changed; run kernel again")

    def kernel(self):
        self._state_key = None
        self.result = self.e = self.right_vectors = self.left_vectors = self.space = (
            None
        )
        self.converged = None
        cfg = self._config()
        ref = self._source()
        if self.ground._integrals is not None:
            self.result = run_eom_from_integrals(self.ground._integrals,
                self.ground.result, config=cfg, cc_config=self.ground._config())
        else:
            self.result = run_eom(ref.h1, ref.eri, self.ground.result, nocc=ref.nocc,
                frozen=self.ground.frozen, config=cfg, cc_config=self.ground._config())
        self.e = self.result.energies
        self.right_vectors, self.left_vectors = (
            self.result.right_vectors,
            self.result.left_vectors,
        )
        self.converged = np.asarray(self.result.converged)
        self.space = EOMAmplitudeSpace(*self.ground.t1.shape, cfg.sector)
        self._state_key = self._key()
        if cfg.nroots == 1:
            return self.e[0], self.right_vectors[:, 0]
        return self.e, self.right_vectors.T

    def run(self):
        self.kernel()
        return self

    def diagnostics(
        self, *, scf_gradient_tol=None, cc_residual_tol=None, eom_residual_tol=None
    ):
        """Read fresh stored layer results; never rerun or silently tighten solvers."""
        from ...scf.facade import RKS

        self.check_source()
        cc_tol = (
            self.ground.residual_tol
            if cc_residual_tol is None
            else float(cc_residual_tol)
        )
        eom_tol = self.conv_tol if eom_residual_tol is None else float(eom_residual_tol)
        targets = (cc_tol, eom_tol) + (
            () if scf_gradient_tol is None else (float(scf_gradient_tol),)
        )
        if any(not np.isfinite(t) or t <= 0 for t in targets):
            raise ValueError("Residual targets must be finite and positive")
        source = self.ground.mf
        scf = (
            source.diagnostics(gradient_tol=scf_gradient_tol)
            if isinstance(source, RKS)
            else None
        )
        return EOMPrecisionDiagnostics(
            scf, self.ground.result, self.result, cc_tol, eom_tol
        )

    def amplitudes(self, root=0):
        self.check_source()
        if not isinstance(root, int) or not 0 <= root < self.nroots:
            raise ValueError("Invalid EOM root")
        return self.space.unpack(self.right_vectors[:, root])

    def dyson(self, *, max_elements=20_000_000):
        """One-spin IP/EA Dyson amplitudes and biorthogonal spectral weights."""
        from .properties import dyson_orbitals

        self.check_source()
        if self.sector not in {"ip", "ea"}:
            raise ValueError("Dyson orbitals require the IP or EA sector")
        self.ground.solve_lambda()
        return dyson_orbitals(self.ground.result, self.ground.lambda_result,
                              self.result, max_elements=max_elements)


class EOMEE(_EOM):
    """Restricted singlet EOM-EE-CCSD; not an all-spin EE dispatcher."""

    def __init__(self, ground, **kwargs):
        super().__init__(ground, sector="ee", **kwargs)


class EOMIP(_EOM):
    """Restricted doublet EOM-IP-CCSD; energies E(N-1)-E(N)."""

    def __init__(self, ground, **kwargs):
        super().__init__(ground, sector="ip", **kwargs)


class EOMEA(_EOM):
    """Restricted doublet EOM-EA-CCSD; energies E(N+1)-E(N), not their negative."""

    def __init__(self, ground, **kwargs):
        super().__init__(ground, sector="ea", **kwargs)
