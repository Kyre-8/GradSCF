"""Relaxed molecular MP/CI/FCI/CC energies using shared integrals and solvers."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf import cc, ci, fci, mp
from gradscf._memory import resolve_memory_mb, tree_storage_bytes, workspace_bytes
from gradscf.ao2mo import transform_integrals, transform_unrestricted_integrals
from gradscf.ao2mo.reference import transform_fock
from gradscf.ao2mo.frame import project_frame
from gradscf.cc.integrals import from_orbitals as cc_from_orbitals
from gradscf.cc.unrestricted_integrals import from_orbitals as ucc_from_orbitals
from gradscf.mp.integrals import from_orbitals
from gradscf.mp.mp2 import evaluate_mp2
from gradscf.mp.series import run_from_orbitals as series_from_orbitals
from gradscf.ci.integrals import from_orbitals as ci_from_orbitals
from gradscf.ci.solver import solve_ci_from_integrals
from gradscf.fci.reference import _active_from_orbitals, orbital_selection
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian
from .molecular_response import MolecularEnergy

_CI_RANKS = {"cid": 2, "cisd": 2, "cisdt": 3, "cisdtq": 4}


class CorrelatedEnergy:
    """E(R,F,V) for real HF-based MP, CI/FCI or CCSD/(T), in atomic units.

    The HF state, AO integrals, MO transformation and correlation solution are
    all rebuilt inside the differentiated map. This adapter adds no solver.
    MP and CCSD use the transported occupied/virtual frame directly by default.
    MP's noncanonical Fock resolvent permits internal MO degeneracy. An explicit
    canonical MP configuration instead requires isolated canonical orbitals.
    CCSD(T) adds the bounded-panel general-reference correction in the same
    transported frame. Its JVP/VJP supports second-order response; third-order
    requests raise explicitly. Ordinary MP/CCSD response behavior is unchanged.
    MP2 and CC transform only their required occupied/virtual blocks.
    Explicit ``jk_backend='direct'`` also removes full moving AO ERIs through
    the native shell/direct operators. MP2's noncanonical resolvent still owns
    O(o**2*v**2) state; CC retains its bare virtual block. Higher noncanonical
    MP transforms selected normal-order kernels for the same Taylor residual;
    its finite wavefunction histories still grow with the excitation space.
    CID/CISD/CISDT/CISDTQ use the same selected-kernel CI producer and support
    fixed ``frozen`` selections. FCI requires a common spatial frame and uses
    explicit ``core``/``active`` selections; its active-space four-index output
    is retained. Neither method performs a full-MO preparation before those
    selected producers. All frame, integral and SCF response remains live.
    A selected frozen/core/active model projects each initial HF occupied and
    virtual anchor block into the current stationary subspace, then uses a
    Cholesky-orthonormal frame. This explicitly fixes a differentiable local
    gauge; it is not the arbitrary frame produced by optimizer recentering
    away from the anchor geometry. Anchor rank loss remains nonfinite.
    Rotations across selected partitions define a different correlation model.
    ``max_memory`` inherits the SCF source budget and plans MO transformation
    arrays; complete solver/AD process residency requires separate accounting.
    Frozen/core selections are limited to CI/FCI; no density fitting or DFT
    reference is used here.

    Correlated dipoles and polarizabilities must be derivatives of this total
    energy. The reference's stationary dipole is only the HF observable.
    """

    def __init__(self, mf, method, *, config=None, gradient_tolerance=1e-10,
                 jk_backend="full", max_memory=None, frozen=None, core=0, active=None):
        self.method = method.lower()
        if self.method not in {"mp", "mp2", "ccsd", "ccsd(t)", "fci", *_CI_RANKS} or str(mf.xc).lower() != "hf":
            raise ValueError("Use MP, CI, FCI or CCSD/(T) with a real HF reference.")
        if self.method in _CI_RANKS or self.method == "fci":
            mf._check_reference_source()
        self.max_memory = resolve_memory_mb(max_memory, mf)
        self.reference = MolecularEnergy(mf, gradient_tolerance=gradient_tolerance,
                                         jk_backend=jk_backend)
        self.coordinates = self.reference.coordinates
        occupations = np.asarray(mf.mo_occ)
        self.nocc = (tuple(int(np.count_nonzero(row)) for row in occupations)
                     if self.reference.unrestricted else int(np.count_nonzero(occupations)))
        self.nmo = mf.mo_coeff.shape[-1]
        if self.method in _CI_RANKS or self.method == "fci":
            self._init_configuration(config, frozen, core, active)
            return
        if frozen is not None or core != 0 or active is not None:
            raise ValueError("Frozen/core/active selections are currently supported by CI/FCI only.")
        if config is None:
            config = (mp.MPConfig(order=2, with_t2=False, orbital_basis="noncanonical")
                      if self.method in {"mp", "mp2"}
                      else cc.CCConfig(conv_tol=1e-11, residual_tol=1e-10))
        if self.method in {"mp", "mp2"}:
            if not isinstance(config, mp.MPConfig):
                raise ValueError("The MP adapter requires an MPConfig.")
            if self.method == "mp2" and config.order != 2:
                raise ValueError("The MP2 adapter requires an order-2 MPConfig; use method='mp' for higher orders.")
        elif not isinstance(config, cc.CCConfig) or config.method != "ccsd":
            raise ValueError("The CCSD adapter requires a CCSD configuration.")
        self.config = config
        self._selected_mp2 = (self.method in {"mp", "mp2"} and config.order == 2
                              and not config._use_series(self.reference.unrestricted))
        self._selected_mp_series = (self.method == "mp" and config.order > 2
            and config.orbital_basis == "noncanonical"
            and config._use_series(self.reference.unrestricted))
        self._selected_cc = self.method in {"ccsd", "ccsd(t)"}
        if jk_backend == "direct" and not (self._selected_mp2 or self._selected_cc
                                          or self._selected_mp_series):
            raise NotImplementedError(
                "Direct correlated inputs require MP2, noncanonical MP series or CCSD/(T).")

    def __call__(self, coordinates, field=None, *, potential=None):
        if self.method in _CI_RANKS or self.method == "fci":
            return self._configuration_energy(coordinates, field, potential)
        state, inputs = self.reference.solve_with_inputs(coordinates, field, potential=potential)
        coefficients, valid = state.mo_coeff, state.stationary
        if self.method in {"mp", "mp2"} and self.config.orbital_basis == "canonical":
            def canonicalize(coeff, fock):
                fock_mo = coeff.T @ fock @ coeff
                orbitals = solve_hermitian(.5*(fock_mo+fock_mo.T),
                    config=EigenSolverConfig(method="dense", nroots=self.nmo, atol=1e-10),
                    response=EigenResponseConfig(target="eigenpairs"))
                return coeff @ orbitals.vectors, jnp.all(orbitals.response_valid)
            if self.reference.unrestricted:
                alpha, alpha_valid = canonicalize(coefficients[0], state.fock_matrix[0])
                beta, beta_valid = canonicalize(coefficients[1], state.fock_matrix[1])
                coefficients, valid = (alpha, beta), valid & alpha_valid & beta_valid
            else:
                coefficients, canonical_valid = canonicalize(coefficients, state.fock_matrix.mean(axis=0))
                valid &= canonical_valid
        if self._selected_mp2 or self._selected_cc or self._selected_mp_series:
            cs = tuple(coefficients) if self.reference.unrestricted else (coefficients,)
            fs = tuple(state.fock_matrix) if self.reference.unrestricted else (state.fock_matrix.mean(axis=0),)
            focks = tuple(transform_fock(f, c) for c, f in zip(cs, fs))
            options = dict(eri=inputs["eri"], direct_basis=inputs.get("direct_basis"),
                           max_memory=self.max_memory)
            if self._selected_mp2:
                data = from_orbitals(coefficients, focks, self.nocc, state.total_energy,
                    frozen_gap_tol=self.config.denominator_tol, **options)
                result = evaluate_mp2(data, self.config)
            elif not self._selected_cc:
                direct = inputs.get("direct_basis")
                ao = {"direct_basis": direct} if direct is not None else {
                    "eri" if inputs["eri"].ndim == 4 else "eri_pair_matrix": inputs["eri"]}
                source = (state, inputs, self.reference.parameters, self.reference.coefficients,
                    self.reference.coordinates, self.reference.occupations, self.reference.plan.topology)
                reserved = tree_storage_bytes(*source, fs,
                    exclude=(cs, focks, state.total_energy, *jax.tree.leaves(ao)))
                result = series_from_orbitals(coefficients, focks, self.nocc, state.total_energy,
                    config=self.config, max_memory=self.max_memory, reserved_bytes=reserved, **ao)
            elif self.reference.unrestricted:
                data, occupied, virtual = ucc_from_orbitals(coefficients, focks,
                    self.nocc, state.total_energy, vvvv='blocks', **options)
                result = cc.run_ucc_from_integrals(data, occupied, virtual, config=self.config)
            else:
                data = cc_from_orbitals(coefficients, focks[0], self.nocc,
                                       state.total_energy, **options)
                result = cc.run_cc_from_integrals(data, config=self.config)
        else:
            transform = transform_unrestricted_integrals if self.reference.unrestricted else transform_integrals
            h1, eri = transform(inputs["hcore"], coefficients, eri=inputs["eri"],
                                max_memory=self.max_memory)
            solve = (mp.run_mp if self.method in {"mp", "mp2"} else
                     cc.run_ucc if self.reference.unrestricted else cc.run_cc)
            result = solve(h1, eri, nocc=self.nocc, nuclear_repulsion=inputs["nuclear_repulsion"],
                           config=self.config)
        valid &= result.valid if self.method in {"mp", "mp2"} else result.converged
        total = result.total_energy
        if self.method == "ccsd(t)":
            options = dict(orbital_basis="semicanonical", algorithm="streamed",
                denominator_tol=self.config.denominator_tol, residual_tol=self.config.residual_tol)
            correction = (cc.evaluate_triples_from_integrals(data, result, **options)
                if self._selected_cc else cc.evaluate_triples(
                    h1, eri, result, nocc=self.nocc, **options))
            total += correction.energy
            valid &= correction.valid
        return jnp.where(valid, total, jnp.nan)

    def _init_configuration(self, config, frozen, core, active):
        """Fix topology only; every numerical orbital/SCF input stays live."""
        if self.method == "fci":
            if self.reference.unrestricted:
                raise NotImplementedError("FCI requires a common spatial frame; use UCI for UHF orbitals.")
            if frozen is not None:
                raise ValueError("FCI uses core/active selections; CI uses frozen.")
            self.core, self.active = orbital_selection(self.nmo, core, active)
            nelec = self.nocc-len(self.core)
            self.space = fci.make_fci_space(len(self.active), (nelec, nelec))
            self._selected_frame = bool(self.core or self.active != tuple(range(self.nmo)))
            self.config = EigenSolverConfig(atol=1e-11, maxiter=150) if config is None else config
            if not isinstance(self.config, EigenSolverConfig):
                raise ValueError("The FCI adapter requires an EigenSolverConfig.")
        else:
            if core != 0 or active is not None:
                raise ValueError("CI uses frozen selections; core/active are FCI options.")
            build = ci.make_uci_space if self.reference.unrestricted else ci.make_ci_space
            self.space = build(self.nmo, self.nocc, max_excitation=_CI_RANKS[self.method],
                excitation_ranks=(0, 2) if self.method == "cid" else None, frozen=frozen)
            self._selected_frame = (any(self.space.frozen) if self.reference.unrestricted
                                    else bool(self.space.frozen))
            self.config = ci.CIConfig(conv_tol=1e-11, max_cycle=150) if config is None else config
            if not isinstance(self.config, ci.CIConfig):
                raise ValueError("The CI adapter requires a CIConfig.")

    def _configuration_energy(self, coordinates, field, potential):
        state, inputs = self.reference.solve_with_inputs(coordinates, field, potential=potential)
        coefficients = state.mo_coeff
        source = (state, inputs, self.reference.parameters, self.reference.coefficients,
                  self.reference.coordinates, self.reference.occupations, self.reference.plan.topology)
        if self._selected_frame:
            shape = coefficients.shape[-2:]
            item = jnp.result_type(coefficients, inputs['overlap'], self.reference.coefficients).itemsize
            workspace_bytes(self.max_memory, resident_bytes=tree_storage_bytes(*source)
                +(4*shape[0]*shape[1]+6*shape[1]**2)*item,
                device=getattr(coefficients, 'device', None))
            if self.reference.unrestricted:
                coefficients = jnp.stack([project_frame(c, a, inputs['overlap'], no)
                    for c, a, no in zip(coefficients, self.reference.coefficients, self.nocc)])
            else:
                coefficients = project_frame(coefficients, self.reference.coefficients,
                                               inputs['overlap'], self.nocc)
        direct = inputs.get("direct_basis")
        options = {"direct_basis": direct} if direct is not None else {
            "eri" if inputs["eri"].ndim == 4 else "eri_pair_matrix": inputs["eri"]}
        if self.method == "fci":
            reserved = tree_storage_bytes(*source,
                exclude=(inputs["hcore"], coefficients, inputs["nuclear_repulsion"],
                         *jax.tree.leaves(options)))
            h, g, constant = _active_from_orbitals(inputs["hcore"], coefficients,
                core=self.core, active=self.active, ecore=inputs["nuclear_repulsion"],
                max_memory=self.max_memory, reserved_bytes=reserved, **options)
            result = fci.solve_fci(h, g, self.space, ecore=constant, config=self.config,
                max_memory=self.max_memory,
                reserved_bytes=tree_storage_bytes(*source, exclude=(h, g, constant)))
        else:
            cs = tuple(coefficients) if self.reference.unrestricted else (coefficients,)
            fs = tuple(state.fock_matrix) if self.reference.unrestricted else (state.fock_matrix.mean(axis=0),)
            focks = tuple(transform_fock(f, c) for c, f in zip(cs, fs))
            reserved = tree_storage_bytes(*source, fs,
                exclude=(cs, focks, state.total_energy, *jax.tree.leaves(options)))
            data = ci_from_orbitals(coefficients, focks if self.reference.unrestricted else focks[0],
                state.total_energy, self.space, max_memory=self.max_memory,
                reserved_bytes=reserved, **options)
            result = solve_ci_from_integrals(data, self.space, config=self.config,
                max_memory=self.max_memory,
                reserved_bytes=tree_storage_bytes(*source, fs, focks, exclude=(data,)))
        return jnp.where(state.stationary & result.converged[0], result.total_energies[0], jnp.nan)

    def dipole(self, coordinates, field=None, *, potential=None):
        """Return -dE_total/dF, including reference and amplitude response."""
        field = jnp.zeros(3) if field is None else jnp.asarray(field)
        return -jax.grad(lambda applied: self(coordinates, applied, potential=potential))(field)
