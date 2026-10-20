"""CI Hamiltonian orchestration using the common eigensolver and AD rules."""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from ..solvers import LinearOperator, EigenSolverConfig, EigenResponseConfig, LinearSolverConfig, solve_hermitian
from .hamiltonian import build_direct_hamiltonian, validate_integrals
from .space import frozen_indices
from .types import CIConfig, CIResult, CISResult, UCISResult
from .space import make_uci_space
from ..fci.cistring import excite


def _eigenpairs(apply, diagonal, config, response=None, reference_index=None):
    dim = diagonal.size
    if config.nroots > dim:
        raise ValueError("nroots exceeds operator dimension")
    operator = LinearOperator(
        (dim, dim),
        diagonal.dtype,
        lambda v: apply(v[:, None])[:, 0],
        diagonal=diagonal,
        matmat=apply,
    )
    capacity = min(dim, (max(40, 4*(config.nroots+1)) if config.max_space is None
                         else config.max_space))
    shared = EigenSolverConfig(
        method=config.solver,
        nroots=config.nroots,
        atol=config.conv_tol,
        maxiter=config.max_cycle,
        max_subspace=capacity,
    )
    initial = None
    if config.solver == "davidson":
        count = min(config.nroots+1, dim)
        if reference_index is None:
            indices = jnp.argsort(diagonal)[:count]
        else:
            order = jnp.argsort(jnp.where(jnp.arange(dim) == reference_index, jnp.inf, diagonal))
            indices = jnp.concatenate((jnp.asarray([reference_index]), order[:count-1]))
        units = jax.nn.one_hot(indices, dim, dtype=diagonal.dtype).T
        width = min(capacity, 2*count)
        extra = jax.random.normal(jax.random.PRNGKey(0), (dim, max(0, width-count)),
                                  dtype=diagonal.dtype) / jnp.sqrt(float(dim))
        # Keep reference/low-diagonal vectors exact. Separate full-support
        # columns allow searches outside their invariant spin/symmetry sectors.
        initial = jnp.concatenate((units, extra), axis=1)
    if response is None:
        response = EigenResponseConfig(
            target=(
                "eigenpairs"
                if config.gradient_mode == "implicit_eigenvector"
                else "eigenvalues"
            ),
            linear_config=LinearSolverConfig(
                rtol=config.adjoint_tol, maxiter=config.adjoint_max_cycle
            ),
        )
    else:
        if not isinstance(response, EigenResponseConfig):
            raise TypeError("response must be an EigenResponseConfig")
        if (config.gradient_mode != "eigenvalue_only" or config.adjoint_tol != 1e-10
                or config.adjoint_max_cycle != 100):
            raise ValueError("Use response controls or legacy CI config AD controls, not both")
        if response.target == "subspace":
            raise ValueError("CI individual-root results do not support a subspace response")
    result = solve_hermitian(operator, config=shared, response=response, initial_vectors=initial)
    return result.values, result.vectors, result.residual_norms, result.converged


def solve_ci(h1, eri, space, *, nuclear_repulsion=0.0, config=None):
    """Variational CI in a static fixed-M_s space, with first-order energy AD.

    Frozen-core energy is retained because frozen electrons remain in the
    determinants. Derivatives assume converged, isolated roots and fixed topology.
    """
    config = CIConfig() if config is None else config
    op = build_direct_hamiltonian(h1, eri, space)
    energies, vectors, residuals, converged = _eigenpairs(op.apply, op.diagonal, config, reference_index=0)
    reference = op.diagonal[0]
    return CIResult(energies + nuclear_repulsion, energies - reference,
                    reference + nuclear_repulsion, vectors, residuals, converged)


def restricted_fock(h1, eri, nocc):
    return h1 + 2 * jnp.einsum("pqii->pq", eri[:, :, :nocc, :nocc]) - jnp.einsum(
        "piiq->pq", eri[:, :nocc, :nocc, :])


def solve_ucis(h1, eri, *, nocc, frozen=None, config=None, max_determinants=1_000_000):
    """Spin-conserving UHF singles, relative to the reference determinant.

    Requires a stationary UHF reference for equivalence to UHF/TDA. No singlet/
    triplet projection or spin-flip excitations. Explicit MO inputs are checked
    by the eager UCIS facade; this functional interface is JIT-compatible.
    """
    config = CIConfig() if config is None else config
    n = h1[0].shape[0]
    full = make_uci_space(n, nocc, max_excitation=1, frozen=frozen,
                          max_determinants=max_determinants)
    if full.size == 1:
        raise ValueError("UCIS has no active single excitations")
    full_op = build_direct_hamiltonian(h1, eri, full)
    space = replace(full, determinants=full.determinants[1:], ranks=full.ranks[1:])
    op = build_direct_hamiltonian(h1, eri, space)
    energies, vectors, norms, converged = _eigenpairs(op.apply, op.diagonal, config)
    amplitudes = []
    lookup = {d: k for k, d in enumerate(space.determinants)}
    for spin, (no, fr) in enumerate(zip(nocc, full.frozen)):
        occ = [p for p in range(no) if p not in fr]
        vir = [p for p in range(no, n) if p not in fr]
        links = [excite(full.determinants[0], (spin*n+i,), (spin*n+a,))
                 for i in occ for a in vir]
        indices = np.asarray([lookup[d] for d, _ in links], dtype=np.int32)
        signs = jnp.asarray([s for _, s in links], dtype=vectors.dtype)
        amplitudes.append((vectors[indices]*signs[:, None]).T.reshape(config.nroots, len(occ), len(vir)))
    return UCISResult(energies-full_op.diagonal[0], tuple(amplitudes), norms, converged,
                       config.gradient_mode == "implicit_eigenvector")


def solve_cis(h1, eri, *, nocc, singlet=True, frozen=None, config=None, response=None):
    """Spin-adapted HF singles with unit-normalized spatial amplitudes.

    A(ia,jb) = delta(ij) F(ab) - delta(ab) F(ij)
               + 2 (ia|jb) [singlet only] - (ij|ab).
    The occupied-virtual Fock block must vanish (stationary HF reference).
    Optional ``response`` uses the shared EigenResponseConfig for individual
    energies/eigenpairs, including bounded direct linear response. It cannot
    be combined with nondefault legacy config AD controls. Subspace invariants
    are not represented by this result type and are explicitly rejected.
    CIS reference: Foresman et al. (1992), doi:10.1021/j100180a030.
    See ci/REFERENCES.md for the separate numerical-response references.
    """
    config = CIConfig() if config is None else config
    h1, eri = validate_integrals(h1, eri)
    nmo = h1.shape[0]
    if not isinstance(nocc, int) or nocc < 1 or nocc >= nmo:
        raise ValueError("CIS requires occupied and virtual orbitals")
    frozen = frozen_indices(nmo, nocc, frozen)
    occ = jnp.asarray([i for i in range(nocc) if i not in frozen], dtype=jnp.int32)
    vir = jnp.asarray([i for i in range(nocc, nmo) if i not in frozen], dtype=jnp.int32)
    no, nv = occ.size, vir.size
    if no * nv == 0:
        raise ValueError("CIS has no active single excitations")
    fock = restricted_fock(h1, eri, nocc)
    f_oo, f_vv = fock[jnp.ix_(occ, occ)], fock[jnp.ix_(vir, vir)]
    exchange = eri[jnp.ix_(occ, occ, vir, vir)]
    coulomb = eri[jnp.ix_(occ, vir, occ, vir)]

    def apply(vectors):
        x = vectors.reshape(no, nv, -1)
        y = jnp.einsum("ab,ibk->iak", f_vv, x) - jnp.einsum("ij,jak->iak", f_oo, x)
        y = y - jnp.einsum("ijab,jbk->iak", exchange, x)
        if singlet:
            y = y + 2 * jnp.einsum("iajb,jbk->iak", coulomb, x)
        return y.reshape(no * nv, -1)

    diagonal = (jnp.diag(f_vv)[None, :] - jnp.diag(f_oo)[:, None]
                - jnp.einsum("iiaa->ia", exchange))
    if singlet:
        diagonal = diagonal + 2 * jnp.einsum("iaia->ia", coulomb)
    energies, vectors, residuals, converged = _eigenpairs(apply, diagonal.ravel(), config, response)
    return CISResult(energies, vectors.T.reshape(config.nroots, no, nv), residuals, converged,
                     (response.target == "eigenpairs" if response is not None
                      else config.gradient_mode == "implicit_eigenvector"), singlet)


def solve_cis_response(molecule, *, singlet=True, frozen=None, config=None,
                       response=None, max_memory=2000.0):
    """AO-action counterpart of ``solve_cis`` for a restricted HF reference.

    The reference carries AO J/K data (or a direct provider) and the full AO
    Fock matrix. Only the active orbital columns are selected; the reference
    density and its occupied/virtual Fock blocks remain unchanged. As in
    ``solve_cis``, the caller supplies a stationary real HF reference and the
    spatial amplitudes have unit norm. ``max_memory`` is in decimal MB.
    """
    from ..tddft._semilocal_response import SemilocalResponseFunctional
    from ..tddft._utils import _restricted_channel
    from ..tddft.response import build_restricted_tda_operator
    from ..tddft.response_options import ResponseKernelOptions, plan_response_memory

    config = CIConfig() if config is None else config
    coeff, occupations, orbital_energies = _restricted_channel(molecule)
    nmo = coeff.shape[1]
    nocc = molecule.nocc
    if not isinstance(nocc, int) or nocc < 1 or nocc >= nmo:
        raise ValueError("CIS requires occupied and virtual orbitals")
    frozen = frozen_indices(nmo, nocc, frozen)
    occupied = [i for i in range(nocc) if i not in frozen]
    virtual = [i for i in range(nocc, nmo) if i not in frozen]
    no, nv = len(occupied), len(virtual)
    if no * nv == 0:
        raise ValueError("CIS has no active single excitations")
    active = jnp.asarray(occupied + virtual, dtype=jnp.int32)
    active_reference = replace(molecule, mo_coeff=coeff[:, active],
                               mo_occ=occupations[active],
                               mo_energy=orbital_energies[active], nocc=no)
    plan = plan_response_memory(coeff.shape[0], no*nv, config.nroots,
                                coeff.dtype, max_memory,
                                max_subspace=(no*nv if config.solver == "dense"
                                              else config.max_space),
                                direct_basis=getattr(active_reference, "direct_basis", None))
    config = replace(config, max_space=plan.max_subspace)
    vind, diagonal, _ = build_restricted_tda_operator(
        active_reference, SemilocalResponseFunctional("hf"), singlet=singlet,
        response_kernel_options=ResponseKernelOptions(
            two_electron_mode="direct", max_memory=max_memory))
    # The TD action consumes row vectors, while the common CI solver uses
    # columns. Do not inherit restricted TDA's sqrt(1/2) amplitude convention.
    energies, vectors, residuals, converged = _eigenpairs(
        lambda columns: vind(columns.T).T, diagonal, config, response)
    return CISResult(energies, vectors.T.reshape(config.nroots, no, nv),
                     residuals, converged,
                     (response.target == "eigenpairs" if response is not None
                      else config.gradient_mode == "implicit_eigenvector"), singlet)
