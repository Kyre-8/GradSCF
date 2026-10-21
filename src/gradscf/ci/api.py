"""PySCF-style facades backed by GradSCF's JAX CI kernels."""
from dataclasses import replace

import numpy as np

from .._memory import resolve_memory_mb
from .integrals import reference_from_source
from .space import make_ci_space, make_uci_space, _selected_ranks
from .spin import spin_square
from .solver import solve_ci, solve_cis, solve_cis_response, solve_ucis
from gradscf.integrals.molecular.ao2mo import spin_orbital_integrals
from .solver import restricted_fock
from .corrections import cis_d_correction
from .types import CIConfig
from .properties import make_rdm1, make_rdm2, make_rdm12
from ..scf.reference import unrestricted_reference_from_source, is_unrestricted_source
from ..scf.reference import reference_state_signature, UnrestrictedReference
from gradscf.integrals.molecular.ao2mo import frozen_indices, unrestricted_frozen_indices


class CI:
    """Single-reference CI through max_excitation, conserving the reference M_s.

    kernel() returns (e_corr, ci). One root gives a scalar and a vector;
    multiple roots give an energy array and coefficient columns. result always
    retains a root axis. This facade performs eager input validation; use the
    functional solve_ci interface under jit/grad.

    ``max_memory`` budgets AO->MO transforms in decimal MB. None inherits the
    SCF budget or uses the shared default for an explicit MO reference.
    """
    def __init__(self, mf, *, max_excitation=2, frozen=None, nroots=1,
                 solver="davidson", conv_tol=1e-9, max_cycle=100, max_space=None,
                 gradient_mode="eigenvalue_only", adjoint_tol=1e-10,
                 adjoint_max_cycle=100, max_determinants=1_000_000, excitation_ranks=None,
                 max_memory=None):
        self.mf = mf
        self.max_memory = resolve_memory_mb(max_memory, mf)
        self.max_excitation = max_excitation
        self.excitation_ranks = _selected_ranks(max_excitation, excitation_ranks)
        self.frozen = frozen
        self.nroots = nroots
        self.solver = solver
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.max_space = max_space
        self.gradient_mode = gradient_mode
        self.adjoint_tol = adjoint_tol
        self.adjoint_max_cycle = adjoint_max_cycle
        self.max_determinants = max_determinants
        self.result = self.reference = self.space = None
        self.e_tot = self.e_corr = self.ci = self.converged = None
        self._state_key = None

    def _physical_state_key(self):
        ref = self.reference
        frozen = (unrestricted_frozen_indices(ref.h1[0].shape[0], ref.nocc, self.frozen)
                  if isinstance(ref, UnrestrictedReference)
                  else frozen_indices(ref.h1.shape[0], ref.nocc, self.frozen))
        return (self.max_excitation, _selected_ranks(self.max_excitation, self.excitation_ranks),
                frozen, reference_state_signature(self.mf))

    def _config(self):
        return CIConfig(nroots=self.nroots, solver=self.solver, conv_tol=self.conv_tol,
                        max_cycle=self.max_cycle, max_space=self.max_space,
                        gradient_mode=self.gradient_mode, adjoint_tol=self.adjoint_tol,
                        adjoint_max_cycle=self.adjoint_max_cycle)

    def kernel(self):
        if is_unrestricted_source(self.mf):
            return UCI.kernel(self)
        self.reference = ref = reference_from_source(self.mf, max_memory=self.max_memory)
        self.space = make_ci_space(ref.h1.shape[0], ref.nocc, max_excitation=self.max_excitation,
                                   frozen=self.frozen, max_determinants=self.max_determinants,
                                   excitation_ranks=self.excitation_ranks)
        self.result = solve_ci(ref.h1, ref.eri, self.space,
                               nuclear_repulsion=ref.nuclear_repulsion, config=self._config())
        self._state_key = self._physical_state_key()
        self.e_tot = self.result.total_energies
        self.e_corr = self.result.correlation_energies
        self.ci = self.result.coefficients
        self.converged = np.asarray(self.result.converged)
        if self.nroots == 1:
            self.e_tot, self.e_corr, self.ci = self.e_tot[0], self.e_corr[0], self.ci[:, 0]
            self.converged = bool(self.converged[0])
        return self.e_corr, self.ci

    def run(self):
        self.kernel()
        return self

    def _density_vector(self, root):
        if self.result is None or self.space is None or not hasattr(self.result, "coefficients"):
            raise RuntimeError("Run determinant CI before evaluating its density")
        if self._physical_state_key() != self._state_key:
            raise RuntimeError("CI reference, excitation rank or frozen space changed; run kernel() again")
        if not isinstance(root, (int, np.integer)) or not 0 <= root < len(self.result.converged):
            raise ValueError("root must select an available CI root")
        if not bool(self.result.converged[root]):
            raise RuntimeError("Converge the selected CI root before evaluating its density")
        return self.result.coefficients[:, root]

    def make_rdm1(self, *, root=0):
        return make_rdm1(self._density_vector(root), self.space)

    def make_rdm2(self, *, root=0):
        return make_rdm2(self._density_vector(root), self.space)

    def make_rdm12(self, *, root=0):
        return make_rdm12(self._density_vector(root), self.space)

    def spin_square(self, *, root=0, overlap_ab=None):
        """Total-spin diagnostic; SCF-backed UHF supplies its actual MO overlap."""
        vector = self._density_vector(root)
        if isinstance(self.reference, UnrestrictedReference) and overlap_ab is None:
            if not hasattr(self.mf, "mo_coeff"):
                raise ValueError("An explicit UnrestrictedReference requires overlap_ab")
            coeff = np.asarray(self.mf.mo_coeff)
            if coeff.ndim == 2:  # ROHF: a common orthonormal spatial frame.
                overlap_ab = np.eye(coeff.shape[1])
            else:
                overlap_ab = coeff[0].T @ np.asarray(self.mf.reference.overlap_matrix) @ coeff[1]
        return spin_square(vector, self.space, overlap_ab=overlap_ab)


class CID(CI):
    """Reference plus all double substitutions; no single substitutions."""
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=2, excitation_ranks=(0, 2), **kwargs)


class CISD(CI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=2, **kwargs)


class CISDT(CI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=3, **kwargs)


class CISDTQ(CI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=4, **kwargs)


class UCI(CI):
    """Variational CI in a fixed (Nalpha,Nbeta) sector, not spin adapted."""
    def kernel(self):
        self.reference = ref = unrestricted_reference_from_source(self.mf, max_memory=self.max_memory)
        self.space = make_uci_space(ref.h1[0].shape[0], ref.nocc,
                                    max_excitation=self.max_excitation, frozen=self.frozen,
                                    max_determinants=self.max_determinants,
                                    excitation_ranks=self.excitation_ranks)
        self.result = solve_ci(ref.h1, ref.eri, self.space,
                               nuclear_repulsion=ref.nuclear_repulsion, config=self._config())
        self._state_key = self._physical_state_key()
        self.e_tot, self.e_corr, self.ci = (self.result.total_energies,
                                          self.result.correlation_energies,
                                          self.result.coefficients)
        self.converged = np.asarray(self.result.converged)
        if self.nroots == 1:
            self.e_tot, self.e_corr, self.ci = self.e_tot[0], self.e_corr[0], self.ci[:, 0]
            self.converged = bool(self.converged[0])
        return self.e_corr, self.ci


class UCID(UCI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=2, excitation_ranks=(0, 2), **kwargs)


class UCISD(UCI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=2, **kwargs)


class UCISDT(UCI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=3, **kwargs)


class UCISDTQ(UCI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=4, **kwargs)


class UCIS(UCI):
    """Spin-conserving UHF/TDA singles; no singlet/triplet label."""
    def __init__(self, mf, **kwargs):
        if kwargs.get("excitation_ranks") is not None:
            raise ValueError("UCIS has a fixed singles space; excitation_ranks is not supported")
        super().__init__(mf, max_excitation=1, **kwargs)
        self.e = self.amplitudes = None

    def kernel(self):
        if self.excitation_ranks is not None:
            raise ValueError("UCIS has a fixed singles space; excitation_ranks is not supported")
        self.reference = ref = unrestricted_reference_from_source(self.mf, max_memory=self.max_memory)
        h, g = spin_orbital_integrals(ref.h1, ref.eri)
        n = h.shape[0]//2
        occ = np.asarray(list(range(ref.nocc[0]))+list(range(n, n+ref.nocc[1])), dtype=int)
        f = h+g[:, :, occ, occ].sum(-1)-g[:, occ, occ, :].sum(1)
        for spin, no in enumerate(ref.nocc):
            if np.max(np.abs(np.asarray(f)[spin*n:spin*n+no, spin*n+no:(spin+1)*n]), initial=0.) > 1e-7:
                raise ValueError("UCIS requires a stationary UHF reference; ROHF orbitals need not satisfy this condition")
        self.result = solve_ucis(ref.h1, ref.eri, nocc=ref.nocc, frozen=self.frozen,
                                 config=self._config(), max_determinants=self.max_determinants)
        self.e, self.amplitudes = self.result.excitation_energies, self.result.amplitudes
        self.converged = np.asarray(self.result.converged)
        return self.e, self.amplitudes


class CIS(CI):
    """Spin-adapted HF singles; e is an excitation-energy array in Hartree.

    Non-DF SCF sources use the shared AO response action instead of a full MO
    ERI transformation. ``max_memory`` (decimal MB) budgets this AO path or
    each MO transform and defaults to the source SCF budget;
    explicit CIReference and DF inputs retain the MO-integral implementation.
    """
    _use_ao_response = True

    def __init__(self, mf, *, singlet=None, max_memory=None, **kwargs):
        if kwargs.get("excitation_ranks") is not None:
            raise ValueError("CIS has a fixed singles space; excitation_ranks is not supported")
        if is_unrestricted_source(mf) and singlet is not None:
            raise ValueError("Unrestricted CIS has no singlet/triplet selection; use UCIS")
        super().__init__(mf, max_excitation=1, max_memory=max_memory, **kwargs)
        self.singlet = None if is_unrestricted_source(mf) else (True if singlet is None else singlet)
        self.e = self.amplitudes = None

    def kernel(self):
        if self.excitation_ranks is not None:
            raise ValueError("CIS has a fixed singles space; excitation_ranks is not supported")
        if is_unrestricted_source(self.mf):
            return UCIS.kernel(self)
        from ..scf.facade import RKS
        from ..scf.reference import as_reference

        if (self._use_ao_response and isinstance(self.mf, RKS)
                and self.mf.jk_backend != "df"):
            if str(self.mf.xc).strip().lower() != "hf":
                raise NotImplementedError("CIS requires a closed-shell HF reference")
            if (not self.mf.converged or self.mf.scf_result is None
                    or self.mf._scf_inputs is None):
                raise RuntimeError("Run and converge the HF SCF before CIS")
            # The AO adapter retains the source's convergence/freshness checks.
            # Use its actual Fock matrix, including noncanonical block response.
            self.reference = ref = replace(
                as_reference(self.mf), fock_matrix=self.mf.scf_result.fock_matrix)
            coeff = np.asarray(self.mf.mo_coeff)
            nocc = self.mf._scf_inputs.nelectron // 2
            if coeff.ndim != 2 or np.iscomplexobj(coeff) or self.mf._scf_inputs.nelectron % 2:
                raise NotImplementedError("Only real closed-shell HF references are supported")
            expected = np.zeros(coeff.shape[1])
            expected[:nocc] = 2
            if not np.allclose(np.asarray(self.mf.mo_occ), expected, atol=1e-10, rtol=0):
                raise ValueError("CIS requires integer occupations, occupied orbitals first")
            fock = coeff.T @ np.asarray(ref.fock_matrix) @ coeff
            if np.max(np.abs(fock[:nocc, nocc:]), initial=0.) > 1e-7:
                raise ValueError("CIS requires a stationary HF reference (occupied-virtual Fock block must vanish)")
            self.result = solve_cis_response(
                ref, singlet=self.singlet, frozen=self.frozen,
                config=self._config(), max_memory=self.max_memory)
            self.e, self.amplitudes = self.result.excitation_energies, self.result.amplitudes
            self.converged = np.asarray(self.result.converged)
            return self.e, self.amplitudes
        self.reference = ref = reference_from_source(self.mf, max_memory=self.max_memory)
        fock = np.asarray(restricted_fock(ref.h1, ref.eri, ref.nocc))
        if np.max(np.abs(fock[:ref.nocc, ref.nocc:]), initial=0.) > 1e-7:
            raise ValueError("CIS requires a stationary HF reference (occupied-virtual Fock block must vanish)")
        self.result = solve_cis(ref.h1, ref.eri, nocc=ref.nocc, singlet=self.singlet,
                                frozen=self.frozen, config=self._config())
        self.e, self.amplitudes = self.result.excitation_energies, self.result.amplitudes
        self.converged = np.asarray(self.result.converged)
        return self.e, self.amplitudes


class CIS_D(CIS):
    """Canonical RHF singlet CIS(D), with complete first-order CIS-vector response."""
    # The doubles correction requires the canonical MO ERIs after the CIS solve.
    _use_ao_response = False

    def __init__(self, mf, *, singlet=True, denominator_tol=1e-10, **kwargs):
        if is_unrestricted_source(mf):
            raise NotImplementedError("CIS(D) currently requires a closed-shell RHF reference")
        if not singlet:
            raise NotImplementedError("The CIS(D) facade currently supports singlet roots only")
        if kwargs.get("gradient_mode", "implicit_eigenvector") != "implicit_eigenvector":
            raise ValueError("CIS(D) requires gradient_mode='implicit_eigenvector'")
        kwargs["gradient_mode"] = "implicit_eigenvector"
        super().__init__(mf, singlet=True, **kwargs)
        self.denominator_tol = denominator_tol
        self.e_cis = self.correction = self.cis_result = None

    def kernel(self):
        super().kernel()
        self.cis_result = self.result
        ref = self.reference
        if ref.mo_energy is None:
            raise ValueError("CIS(D) requires canonical HF mo_energy")
        fock = np.asarray(restricted_fock(ref.h1, ref.eri, ref.nocc))
        eps = np.asarray(ref.mo_energy)
        if eps.shape != (ref.h1.shape[0],) or not np.allclose(
                fock, np.diag(eps), atol=1e-7, rtol=0):
            raise ValueError("CIS(D) requires canonical HF orbitals and consistent mo_energy")
        self.result = cis_d_correction(ref.eri, ref.mo_energy, self.cis_result,
                                       nocc=ref.nocc, frozen=self.frozen,
                                       denominator_tol=self.denominator_tol)
        self.e_cis, self.correction = self.result.cis_energies, self.result.corrections
        self.e = self.result.excitation_energies
        self.converged = np.asarray(self.result.valid)
        return self.e, self.amplitudes
