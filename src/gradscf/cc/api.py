"""PySCF-style ground-state CC facades, without solver copies."""

from dataclasses import fields
import jax.numpy as jnp
from .._memory import resolve_memory_mb
from ..scf.reference import (reference_from_source, UnrestrictedReference,
                             unrestricted_reference_from_source, is_unrestricted_source)
from gradscf.ao2mo import frozen_indices, unrestricted_frozen_indices
from ..scf.reference import reference_state_signature
from .types import CCConfig
from .ground import run_cc, run_cc_from_integrals
from .integrals import from_orbitals
from .uccsd import run_ucc, run_ucc_from_integrals
from .unrestricted_integrals import from_orbitals as ucc_from_orbitals
from .lambda_equations import solve_lambda, solve_lambda_from_integrals
from .triples import evaluate_triples, evaluate_triples_from_integrals
from .properties import (_density_from_lambda, _rdm2_from_lambda,
                         _density_from_integrals, _density_ucc_from_integrals,
                         _rdm2_from_states)


class CC:
    """Ground-state CC with an inherited AO->MO budget in decimal MB.

    ``max_memory=None`` inherits the SCF budget or uses the shared default for
    explicit MO references. Restricted SCF sources retain selected integral
    blocks and orbital/AO metadata for posthoc consumers, as do UHF SCF sources.
    Explicit MO and ROHF references retain their original representation.
    The transformation budget does not cap the downstream amplitude solver.
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
            if unrestricted:
                self._nmo = ref.coefficients[0].shape[1]
                self._integrals, occupied, virtual = ucc_from_orbitals(
                    ref.coefficients, ref.focks, ref.nocc, ref.reference_energy,
                    frozen=self.frozen, max_memory=self.max_memory, vvvv='blocks',
                    **ref.ao_kwargs)
                self.result = run_ucc_from_integrals(self._integrals, occupied, virtual,
                                                      config=cfg, t1=t1, t2=t2)
            else:
                self._nmo = ref.coefficients.shape[1]
                self._integrals = from_orbitals(
                    ref.coefficients, ref.focks[0], ref.nocc, ref.reference_energy,
                    frozen=self.frozen, max_memory=self.max_memory, model=cfg.method,
                    **ref.ao_kwargs)
                self.result = run_cc_from_integrals(self._integrals, config=cfg, t1=t1, t2=t2)
        else:
            self.reference = ref = (unrestricted_reference_from_source(
                                        self.mf, max_memory=self.max_memory)
                                    if unrestricted else reference_from_source(
                                        self.mf, max_memory=self.max_memory))
            self._nmo = ref.h1[0].shape[0] if unrestricted else ref.h1.shape[0]
            self.result = (run_ucc if unrestricted else run_cc)(
                ref.h1, ref.eri, nocc=ref.nocc,
                nuclear_repulsion=ref.nuclear_repulsion, frozen=self.frozen,
                config=cfg, t1=t1, t2=t2)
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

    def solve_lambda(self):
        ref = self._ready()
        self.lambda_result = (solve_lambda_from_integrals(
            self._integrals, self.result, config=self._config())
            if self._integrals is not None else solve_lambda(
                ref.h1, ref.eri, self.result, nocc=ref.nocc,
                frozen=self.frozen, config=self._config()))
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
        if self._integrals is not None:
            if isinstance(ref.nocc, (tuple, list)):
                return _density_ucc_from_integrals(self._integrals, self.result,
                    self.lambda_result, nmo=self._nmo, nocc=ref.nocc,
                    frozen=self.frozen, config=self._config())
            density = _density_from_integrals(self._integrals, self.result,
                self.lambda_result, nmo=self._nmo, nocc=ref.nocc,
                frozen=self.frozen, config=self._config())
            return density.astype(jnp.result_type(ref.coefficients, ref.hcore))
        return _density_from_lambda(
            ref.h1,
            ref.eri,
            self.result,
            self.lambda_result,
            nocc=ref.nocc,
            frozen=self.frozen,
            config=self._config(),
        )

    def make_rdm2(self):
        ref = self._ready()
        self.solve_lambda()
        if not self.converged_lambda:
            raise RuntimeError("Converge Lambda before evaluating the CC density")
        if self._integrals is not None:
            coefficients = (ref.coefficients if isinstance(ref.nocc, (tuple, list))
                            else (ref.coefficients,))
            return _rdm2_from_states(self.result, self.lambda_result, nmo=self._nmo,
                nocc=ref.nocc, frozen=self.frozen,
                dtype=jnp.result_type(*coefficients, ref.hcore), config=self._config())
        return _rdm2_from_lambda(ref.h1, ref.eri, self.result, self.lambda_result,
                                 nocc=ref.nocc, frozen=self.frozen, config=self._config())


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
