"""PySCF-style ground-state CC facades, without solver copies."""

from dataclasses import fields
import jax
import jax.numpy as jnp
from .._memory import array_device, resolve_memory_mb, tree_storage_bytes
from ..scf.reference import (reference_from_source, UnrestrictedReference,
                             unrestricted_reference_from_source, is_unrestricted_source)
from gradscf.ao2mo import frozen_indices, unrestricted_frozen_indices
from ..scf.reference import reference_state_signature
from .types import CCConfig
from .ground import _prepare_ground, run_cc, run_cc_from_integrals
from .integrals import from_orbitals
from .uccsd import run_ucc, run_ucc_from_integrals
from .unrestricted_integrals import from_orbitals as ucc_from_orbitals
from .lambda_equations import solve_lambda, solve_lambda_from_integrals
from .triples import evaluate_triples, evaluate_triples_from_integrals
from .properties import (_density_from_lambda, _rdm2_from_lambda,
                         _density_from_integrals, _density_ucc_from_integrals,
                         _rdm2_from_states, _density_preflight,
                         _contract_rdm2_from_states)


class CC:
    """Ground-state CC with an inherited AO->MO budget in decimal MB.

    ``max_memory=None`` inherits the SCF budget or uses the shared default for
    explicit MO references. Restricted SCF sources retain selected integral
    blocks and orbital/AO metadata for posthoc consumers, as do UHF SCF sources.
    Explicit MO and ROHF references retain their original representation.
    Known source, integral and solver arrays share that budget. Runtime,
    compilation and arbitrary external AD tapes remain outside this array plan.
    """
    def __init__(self, mf, *, method="ccsd", frozen=None, max_memory=None, **kwargs):
        cfg = CCConfig(method=method, **kwargs)
        self.mf = mf
        self.max_memory = resolve_memory_mb(max_memory, mf)
        self.frozen = frozen
        for f in fields(cfg):
            setattr(self, f.name, getattr(cfg, f.name))
        self.result = self.reference = None
        self.e_corr = self.e_tot = self.t1 = self.t2 = None
        self.l1 = self.l2 = self.lambda_result = None
        self.converged = self.converged_lambda = False
        self._state_key = None
        self._integrals = self._nmo = None

    def _physical_state_key(self):
        ref = self.reference
        frozen = (unrestricted_frozen_indices(self._nmo, ref.nocc, self.frozen)
                  if isinstance(ref.nocc, (tuple, list))
                  else frozen_indices(self._nmo, ref.nocc, self.frozen))
        return (
            self.method,
            frozen,
            reference_state_signature(self.mf),
        )

    def _config(self):
        return CCConfig(**{f.name: getattr(self, f.name) for f in fields(CCConfig)})

    def _held_bytes(self, *exclude, extra=()):
        return tree_storage_bytes(getattr(self.mf, '__dict__', {}), vars(self), extra,
                                  exclude=exclude)

    def _restricted_integrals(self, ref, cfg, t1, t2, reservation):
        def prepare(mode):
            return from_orbitals(
                ref.coefficients, ref.focks[0], ref.nocc, ref.reference_energy,
                frozen=self.frozen, max_memory=self.max_memory, model=cfg.method,
                vvvv=mode, reserved_bytes=reservation, **ref.ao_kwargs)

        if 'direct_basis' not in ref.ao_kwargs or cfg.method == 'ccs':
            return prepare('auto')
        for mode in ('dense', 'direct'):
            try:
                # Separate preparation and ground peaks: do not charge their
                # mutually exclusive workspaces as simultaneous residency.
                with jax.default_device(array_device(ref.coefficients)):
                    candidate = jax.eval_shape(lambda: prepare(mode))
                    held = self._held_bytes(candidate, t1, t2)
                    jax.eval_shape(lambda ints: _prepare_ground(
                        ints, cfg, t1, t2, self.max_memory, held, preview=True)[1], candidate)
            except ValueError as error:
                if mode != 'dense' or 'budget' not in str(error):
                    raise
            else:
                # Only the selected representation is ever materialized.
                return prepare(mode)

    def kernel(self, t1=None, t2=None):
        # A failed rebuild cannot leave a previous snapshot available to
        # Lambda/properties/EOM, including when source validation raises.
        self.result = self.reference = self._integrals = self._nmo = None
        self.e_corr = self.e_tot = self.t1 = self.t2 = None
        self.l1 = self.l2 = self.lambda_result = None
        self.converged = self.converged_lambda = False
        self._state_key = None
        cfg = self._config()
        unrestricted = is_unrestricted_source(self.mf)
        from ..scf.facade import RKS, UKS
        if isinstance(self.mf, (RKS, UKS)):
            from ..ao2mo.scf import prepare_scf_source
            self.reference = ref = prepare_scf_source(self.mf, max_memory=self.max_memory)
            paid = jax.tree.leaves((ref.coefficients, ref.focks, ref.reference_energy, ref.ao_kwargs))
            reservation = self._held_bytes(*paid, extra=(t1, t2))
            if unrestricted:
                self._nmo = ref.coefficients[0].shape[1]
                self._integrals, occupied, virtual = ucc_from_orbitals(
                    ref.coefficients, ref.focks, ref.nocc, ref.reference_energy,
                    frozen=self.frozen, max_memory=self.max_memory, vvvv='blocks',
                    reserved_bytes=reservation,
                    **ref.ao_kwargs)
                self.result = run_ucc_from_integrals(self._integrals, occupied, virtual,
                    config=cfg, t1=t1, t2=t2, max_memory=self.max_memory,
                    reserved_bytes=self._held_bytes(self._integrals, t1, t2))
            else:
                self._nmo = ref.coefficients.shape[1]
                self._integrals = self._restricted_integrals(ref, cfg, t1, t2, reservation)
                self.result = run_cc_from_integrals(self._integrals, config=cfg, t1=t1, t2=t2,
                    max_memory=self.max_memory,
                    reserved_bytes=self._held_bytes(self._integrals, t1, t2))
        else:
            self.reference = ref = (unrestricted_reference_from_source(
                                        self.mf, max_memory=self.max_memory)
                                    if unrestricted else reference_from_source(
                                        self.mf, max_memory=self.max_memory))
            self._nmo = ref.h1[0].shape[0] if unrestricted else ref.h1.shape[0]
            self.result = (run_ucc if unrestricted else run_cc)(
                ref.h1, ref.eri, nocc=ref.nocc,
                nuclear_repulsion=ref.nuclear_repulsion, frozen=self.frozen,
                config=cfg, t1=t1, t2=t2, max_memory=self.max_memory,
                reserved_bytes=self._held_bytes(ref.h1, ref.eri, ref.nuclear_repulsion, t1, t2))
        self.e_corr, self.e_tot = (
            self.result.correlation_energy,
            self.result.total_energy,
        )
        self.t1, self.t2 = self.result.t1, self.result.t2
        self.converged = bool(self.result.converged)
        self._state_key = self._physical_state_key()
        return self.e_corr, self.t1, self.t2

    def run(self):
        self.kernel()
        return self

    def _ready(self):
        if self.result is None or not self.converged:
            raise RuntimeError("Converge the CC ground state first")
        if self._physical_state_key() != self._state_key:
            raise RuntimeError(
                "CC reference, method or frozen space changed; run kernel() again"
            )
        return self.reference

    def EOMEE(self, **kwargs):
        """Create a restricted singlet EOM-EE-CCSD calculation."""
        from .eom import EOMEE

        return EOMEE(self, **kwargs)

    def EOMIP(self, **kwargs):
        from .eom import EOMIP

        return EOMIP(self, **kwargs)

    def EOMEA(self, **kwargs):
        from .eom import EOMEA

        return EOMEA(self, **kwargs)

    def eomee_ccsd_singlet(self, **kwargs):
        return self.EOMEE(**kwargs).kernel()

    def ipccsd(self, **kwargs):
        return self.EOMIP(**kwargs).kernel()

    def eaccsd(self, **kwargs):
        return self.EOMEA(**kwargs).kernel()

    def solve_lambda(self, *, _extra=()):
        ref = self._ready()
        self.lambda_result = (solve_lambda_from_integrals(
            self._integrals, self.result, config=self._config(), max_memory=self.max_memory,
            reserved_bytes=self._held_bytes(self._integrals, self.result, extra=_extra))
            if self._integrals is not None else solve_lambda(
                ref.h1, ref.eri, self.result, nocc=ref.nocc,
                frozen=self.frozen, config=self._config(), max_memory=self.max_memory,
                reserved_bytes=self._held_bytes(ref.h1, ref.eri, self.result, extra=_extra)))
        self.l1, self.l2 = self.lambda_result.l1, self.lambda_result.l2
        self.converged_lambda = bool(self.lambda_result.converged)
        return self.l1, self.l2

    def triples(self, *, variant=None, orbital_basis="canonical",
                max_triples_elements=2_000_000, algorithm="auto",
                max_workspace_elements=2_000_000):
        """Return CCSD(T) or QCISD(T) for this model, with state diagnostics.

        An explicit variant can select restricted CCSD+T(CCSD). Cross-model
        amplitude reuse is rejected; functional callers select the variant explicitly.
        """
        ref = self._ready()
        if variant is None:
            variant = "qcisd(t)" if self.method == "qcisd" else "ccsd(t)"
        required = "qcisd" if variant == "qcisd(t)" else "ccsd"
        if self.method != required:
            raise ValueError("Noniterative triples require matching CCSD or QCISD amplitudes")
        options = dict(
            denominator_tol=self.denominator_tol,
            residual_tol=self.residual_tol,
            variant=variant,
            orbital_basis=orbital_basis,
            max_triples_elements=max_triples_elements,
            algorithm=algorithm,
            max_workspace_elements=max_workspace_elements,
            max_memory=self.max_memory,
            reserved_bytes=self._held_bytes(self._integrals, self.result)
                if self._integrals is not None else self._held_bytes(ref.h1, ref.eri, self.result),
        )
        result = (evaluate_triples_from_integrals(self._integrals, self.result, **options)
            if self._integrals is not None else evaluate_triples(
                ref.h1, ref.eri, self.result, nocc=ref.nocc, frozen=self.frozen, **options))
        if not bool(result.valid):
            raise ValueError(
                "Triples require matching converged CCSD/QCISD amplitudes, canonical orbitals "
                "(unless orbital_basis='semicanonical'), and a resolved denominator solve"
            )
        return result

    def ccsd_t(self, *, orbital_basis="canonical", max_triples_elements=2_000_000,
               algorithm="auto", max_workspace_elements=2_000_000):
        return self.triples(variant="ccsd(t)", orbital_basis=orbital_basis,
                            max_triples_elements=max_triples_elements, algorithm=algorithm,
                            max_workspace_elements=max_workspace_elements).energy

    def qcisd_t(self):
        """Canonical restricted QCISD(T), requiring converged QCISD amplitudes."""
        return self.triples(variant="qcisd(t)").energy

    def make_rdm1(self):
        ref = self._ready()
        self.solve_lambda()
        if not self.converged_lambda:
            raise RuntimeError("Converge Lambda before evaluating the CC density")
        return self._rdm1_with_left(ref)

    def _rdm1_with_left(self, ref, *, extra=()):
        if self._integrals is not None:
            if isinstance(ref.nocc, (tuple, list)):
                return _density_ucc_from_integrals(self._integrals, self.result,
                    self.lambda_result, nmo=self._nmo, nocc=ref.nocc,
                    frozen=self.frozen, config=self._config(), max_memory=self.max_memory,
                    reserved_bytes=self._held_bytes(self._integrals, self.result, self.lambda_result, extra=extra))
            density = _density_from_integrals(self._integrals, self.result,
                self.lambda_result, nmo=self._nmo, nocc=ref.nocc,
                frozen=self.frozen, config=self._config(), max_memory=self.max_memory,
                reserved_bytes=self._held_bytes(self._integrals, self.result, self.lambda_result, extra=extra))
            return density.astype(jnp.result_type(ref.coefficients, ref.hcore))
        return _density_from_lambda(
            ref.h1,
            ref.eri,
            self.result,
            self.lambda_result,
            nocc=ref.nocc,
            frozen=self.frozen,
            config=self._config(),
            max_memory=self.max_memory,
            reserved_bytes=self._held_bytes(ref.h1, ref.eri, self.result, self.lambda_result, extra=extra),
        )

    def _density_metadata(self, ref):
        if self._integrals is None:
            h = ref.h1[0] if isinstance(ref.nocc,(tuple,list)) else ref.h1
            return h.shape[0],h.dtype
        coefficients = ref.coefficients if isinstance(ref.nocc,(tuple,list)) else (ref.coefficients,)
        return self._nmo,jnp.result_type(*coefficients,ref.hcore)

    def make_rdm2(self, *, indices=None, spin=None):
        """Full spatial density or four requested MO axes and a spin block."""
        ref = self._ready()
        nmo,dtype = self._density_metadata(ref)
        _density_preflight(self.result,nmo=nmo,nocc=ref.nocc,dtype=dtype,
            indices=indices,spin=spin,max_memory=self.max_memory,
            reserved_bytes=self._held_bytes())
        self.solve_lambda()
        if not self.converged_lambda:
            raise RuntimeError("Converge Lambda before evaluating the CC density")
        if self._integrals is not None:
            return _rdm2_from_states(self.result, self.lambda_result, nmo=nmo,
                nocc=ref.nocc, frozen=self.frozen,dtype=dtype,config=self._config(),
                indices=indices,spin=spin,max_memory=self.max_memory,
                reserved_bytes=self._held_bytes(self.result,self.lambda_result))
        return _rdm2_from_lambda(ref.h1, ref.eri, self.result, self.lambda_result,
            nocc=ref.nocc,frozen=self.frozen,config=self._config(),indices=indices,spin=spin,
            max_memory=self.max_memory,
            reserved_bytes=self._held_bytes(ref.h1,ref.eri,self.result,self.lambda_result))

    def expectation_value(self, h1=None, eri=None):
        """Contract real MO observables without exporting a full two-body density.

        Unrestricted inputs are (h_alpha,h_beta) and (eri_aa,eri_ab,eri_bb).
        Both observables and the original stationary T/Lambda remain live AD
        dependencies. The reference orbitals are held fixed by this property.
        """
        ref = self._ready()
        nmo,dtype = self._density_metadata(ref)
        unrestricted = isinstance(ref.nocc,(tuple,list))
        if h1 is None and eri is None:
            raise ValueError('Supply a one- or two-body observable')
        one = () if h1 is None else tuple(map(jnp.asarray,h1)) if unrestricted else (jnp.asarray(h1),)
        two = () if eri is None else tuple(map(jnp.asarray,eri)) if unrestricted else (jnp.asarray(eri),)
        if one and (len(one) != (2 if unrestricted else 1) or any(x.shape != (nmo,nmo) for x in one)):
            raise ValueError('One-body observable must have full spatial MO shape (nmo,nmo)')
        if two and (len(two) != (3 if unrestricted else 1) or any(x.shape != (nmo,)*4 for x in two)):
            raise ValueError('Two-body observable must have full spatial MO shape (nmo,)*4')
        # NumPy observables may own a distinct backend copy after conversion.
        extra = h1,eri,one,two
        self.solve_lambda(_extra=extra)
        if not self.converged_lambda:
            raise RuntimeError('Converge Lambda before evaluating the CC density')
        value = jnp.asarray(0.,jnp.result_type(dtype,*one,*two))
        if one:
            dm1 = self._rdm1_with_left(ref,extra=extra)
            densities = dm1 if unrestricted else (dm1,)
            value = sum((jnp.einsum('pq,qp->',op,dm) for op,dm in zip(one,densities)),value)
            del dm1,densities
        if two:
            value = value+_contract_rdm2_from_states(two if unrestricted else two[0],self.result,self.lambda_result,
                nmo=nmo,nocc=ref.nocc,dtype=dtype,frozen=self.frozen,config=self._config(),
                max_memory=self.max_memory,reserved_bytes=self._held_bytes(
                    self.result,self.lambda_result,*two,extra=(extra,one,two,value)))
        return value


class CCSD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="ccsd", **kwargs)


class QCISD(CC):
    """Real restricted quadratic CI singles and doubles; distinct from CCSD."""
    def __init__(self, mf, **kwargs):
        if is_unrestricted_source(mf):
            raise NotImplementedError("QCISD currently requires a restricted closed-shell reference")
        super().__init__(mf, method="qcisd", **kwargs)


class RCCSD(CCSD):
    def __init__(self, mf, **kwargs):
        if is_unrestricted_source(mf):
            raise ValueError("RCCSD requires a restricted closed-shell reference")
        super().__init__(mf, **kwargs)


class UCCSD(CCSD):
    def __init__(self, mf, **kwargs):
        if not is_unrestricted_source(mf):
            raise ValueError("UCCSD requires a UHF/ROHF or UnrestrictedReference")
        super().__init__(mf, **kwargs)


class UCCD(CC):
    def __init__(self, mf, **kwargs):
        if not is_unrestricted_source(mf):
            raise ValueError("UCCD requires a UHF/ROHF or UnrestrictedReference")
        super().__init__(mf, method="ccd", **kwargs)


class CCD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="ccd", **kwargs)


class CCS(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="ccs", **kwargs)


class CC2(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="cc2", **kwargs)


class LCCD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="lccd", **kwargs)


class LCCSD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="lccsd", **kwargs)
