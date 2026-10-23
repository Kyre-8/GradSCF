"""Short eager MP interfaces; numerical differentiation belongs to run_mp."""
from dataclasses import fields, replace
import jax
import jax.numpy as jnp
import numpy as np

from .._memory import resolve_memory_mb, tree_storage_bytes
from ..ao2mo.scf import prepare_scf_source
from ..ao2mo import frozen_indices, unrestricted_frozen_indices
from ..scf.reference import (reference_from_source, unrestricted_reference_from_source,
                             is_unrestricted_source, RestrictedReference, UnrestrictedReference)
from .types import MPConfig
from .mp2 import run_mp, evaluate_mp2
from .integrals import from_orbitals
from ._memory import preparation_bytes
from ..scf.facade import RKS, UKS


class MP:
    """Generic order-driven MP; defaults to the residual/Taylor series engine.

    ``max_memory`` plans known integral/state/work arrays in decimal MB. None inherits
    the SCF source's budget, or the shared default for an explicit MO reference.
    Compiler, allocator and arbitrary outer AD residency remain separate.
    """
    unrestricted = None
    method_order = None

    def __init__(self, mf, *, order=None, frozen=None, max_memory=None, **kwargs):
        spin = is_unrestricted_source(mf)
        if self.unrestricted is not None and spin != self.unrestricted:
            raise ValueError("Reference spin does not match restricted/unrestricted MP class")
        order = (self.method_order or 2) if order is None else order
        if self.method_order is not None and order != self.method_order:
            raise ValueError("Use MP(mf, order=...) to select a different perturbation order")
        self.mf, self.frozen = mf, frozen
        self.max_memory = resolve_memory_mb(max_memory, mf)
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
        if not explicit and not isinstance(self.mf, (RKS, UKS)):
            raise NotImplementedError("MP requires RHF/UHF or explicit MO references")
        if not explicit:
            self.mf._check_reference_source()
        nmo = (np.shape(self.mf.h1[0] if self._unrestricted else self.mf.h1)[0]
               if explicit else self.mf.mo_coeff.shape[-1])
        occ = None if explicit else np.asarray(self.mf.mo_occ)
        no = (self.mf.nocc if explicit else tuple(int(np.count_nonzero(x)) for x in occ)
              if self._unrestricted else int(np.count_nonzero(occ)))
        if series:
            from .series import make_series_space
            self.space = make_series_space(nmo, no, frozen=self.frozen, config=cfg, packed_only=True)
        if not explicit and (not series or cfg.orbital_basis == 'noncanonical'):
            # Source validation reads solved data once; no full-MO reference.
            estimate = replace(cfg, order=2, with_t2=True) if cfg.order == 3 and not series else cfg
            future = preparation_bytes(nmo, no, self.frozen, estimate, jnp.float64) if not series else 0
            if series:
                from ._series_memory import history_bytes
                future = history_bytes(self.space, cfg, jnp.float64)
            frame = prepare_scf_source(self.mf, max_memory=self.max_memory,
                reserved_bytes=tree_storage_bytes(self.mf, self.space)+future)
            owned = (self.mf, self.space, frame.coefficients, frame.focks,
                     frame.reference_energy, frame.ao_kwargs)
            caller = tree_storage_bytes(self.mf, self.space,
                exclude=(frame.coefficients, frame.focks, frame.reference_energy, frame.ao_kwargs))
            common = dict(frozen=self.frozen, max_memory=self.max_memory)
            if series:
                from .series import run_from_orbitals
                result = run_from_orbitals(frame.coefficients, frame.focks, frame.nocc,
                    frame.reference_energy, config=cfg, reserved_bytes=caller, **common, **frame.ao_kwargs)
            elif cfg.order == 3:
                from .mp3 import run_from_orbitals
                result = run_from_orbitals(frame.coefficients, frame.focks, frame.nocc,
                    frame.reference_energy, config=cfg, reserved_bytes=caller, **common, **frame.ao_kwargs)
            else:
                dtype = (jnp.float64 if 'direct_basis' in frame.ao_kwargs else
                    jnp.result_type(*jax.tree.leaves((frame.coefficients, frame.focks, frame.ao_kwargs))))
                future = preparation_bytes(nmo, no, self.frozen, cfg, dtype)
                data = from_orbitals(frame.coefficients, frame.focks, frame.nocc, frame.reference_energy,
                    frozen_gap_tol=cfg.denominator_tol, reserved_bytes=caller+future,
                    **common, **frame.ao_kwargs)
                result = evaluate_mp2(data, cfg, max_memory=self.max_memory,
                    reserved_bytes=tree_storage_bytes(owned, exclude=(data,)))
        else:
            # Explicit MO references and canonical series keep their old chart.
            caller = tree_storage_bytes(self.mf, self.space)
            ref = (unrestricted_reference_from_source if self._unrestricted else reference_from_source)(
                self.mf, max_memory=self.max_memory, reserved_bytes=caller)
            caller = tree_storage_bytes(self.mf, self.space,
                exclude=(ref.h1, ref.eri, ref.nuclear_repulsion))
            result = run_mp(ref.h1, ref.eri, nocc=ref.nocc,
                frozen=self.frozen, nuclear_repulsion=ref.nuclear_repulsion, config=cfg,
                max_memory=self.max_memory, reserved_bytes=caller)
            if series and cfg.algorithm == 'tensor' and cfg.orbital_basis == 'canonical':
                self._expand_small_space(ref, cfg, caller)
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

    def _expand_small_space(self, reference, config, reserved):
        """Retain legacy small-space expansion only after a byte preflight."""
        from ._series_memory import plan_series
        from ._tensor import _with_dense_maps
        hs = reference.h1 if self._unrestricted else (reference.h1,)
        gs = reference.eri if self._unrestricted else (reference.eri,)
        dtype = jnp.result_type(*hs, *gs)
        focks = tuple(jax.ShapeDtypeStruct(h.shape, dtype) for h in hs)
        frozen = (unrestricted_frozen_indices(self.space.nmo, reference.nocc, self.frozen)
                  if self._unrestricted else frozen_indices(self.space.nmo, reference.nocc, self.frozen))
        memory = plan_series(hs, gs, focks, self.space, reference.nocc, frozen,
                             config, self.max_memory, reserved_bytes=reserved)
        if not memory.packed:
            self.space = _with_dense_maps(self.space)

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
