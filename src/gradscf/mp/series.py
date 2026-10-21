"""MP of arbitrary finite order from a projected residual and Taylor lifting.

The state uses intermediate normalization. The kth wavefunction has excitation
rank at most 2k for a two-body Hamiltonian. A Rayleigh quotient of the state
through k gives exact energy coefficients through 2k+1 (Wigner's rule).
No FCI product space, dense Hamiltonian or eigensolve is constructed here.
Retained state coefficients still grow combinatorially; this is a bounded
in-core generic engine, not the streaming DF-MP2 algorithm.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.jet import jet

from ..ci.space import make_ci_space, make_uci_space
from ..ci import hamiltonian
from ..fci.cistring import excite
from ..solvers import LinearOperator, LinearSolverConfig, solve_linear
from .types import MPResult
from .integrals import _reference_data, canonical_error, reference_error, _active_orbitals


def make_series_space(nmo, nocc, *, frozen=None, config):
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    if config.algorithm == "tensor":
        from ._tensor import make_tensor_space
        return make_tensor_space(nmo, nocc, max_excitation=2*k, frozen=frozen,
            max_determinants=config.max_determinants, max_tensor_elements=config.max_tensor_elements,
            packed_only=config.orbital_basis == "noncanonical")
    build = make_uci_space if isinstance(nocc, tuple) else make_ci_space
    return build(nmo, nocc, max_excitation=2*k, frozen=frozen,
                 max_determinants=config.max_determinants)


def _jet(fun, primals, coefficients):
    return jet(fun, primals, coefficients, factorial_scaled=False)


def _amplitudes(first, space, unrestricted):
    n = space.nmo
    ns = space.nocc if unrestricted else (space.nocc, space.nocc)
    fr = space.frozen if unrestricted else (space.frozen, space.frozen)
    selections = [_active_orbitals(n, no, f) for no, f in zip(ns, fr)]
    lookup = {d: i for i, d in enumerate(space.determinants)}
    def block(a, b):
        oa, va = selections[a]
        ob, vb = selections[b]
        indices, phases = [], []
        for i in oa:
            for j in ob:
                for x in va:
                    for y in vb:
                        det, sign = excite(space.determinants[0], (a*n+int(i), b*n+int(j)),
                                           (a*n+int(x), b*n+int(y)))
                        indices.append(lookup[det] if sign else 0)
                        phases.append(sign)
        shape = (len(oa), len(ob), len(va), len(vb))
        return (first[jnp.asarray(indices, dtype=jnp.int32)] * jnp.asarray(phases)).reshape(shape)
    return (block(0, 0), block(0, 1), block(1, 1)) if unrestricted else block(0, 1)


def _spectral_focks(focks, nocc, frozen):
    """Numerical excitation spectrum, without differentiating its orbital gauge."""
    ns = nocc if isinstance(nocc, tuple) else (nocc,)
    fr = frozen if isinstance(nocc, tuple) else (frozen,)
    result = []
    for fock, no, frozen_spin in zip(focks, ns, fr):
        diagonal = jnp.diag(jax.lax.stop_gradient(fock))
        for indices in _active_orbitals(fock.shape[0], no, frozen_spin):
            index = jnp.asarray(indices, dtype=jnp.int32)
            block = jax.lax.stop_gradient(fock[jnp.ix_(index, index)])
            diagonal = diagonal.at[index].set(jnp.linalg.eigvalsh(block))
        result.append(jnp.diag(diagonal))
    return tuple(result)


def _action_ranks(config, maximum, degree):
    """Exact ranks needed by the two-body recurrence and terminal energy bra."""
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    input_rank = min(maximum, 2*degree)
    output_rank = min(maximum, 2*degree+2)
    if degree == k:
        # At energy order p, this W Psi_k node pairs only with a bra of
        # degree at most p-k-1, hence excitation rank at most 2(p-k-1).
        output_rank = min(output_rank, max(0, 2*(config.order-k-1)))
    return input_rank, output_rank


def _direct_actions(h1, eri, focks, reference_energy, nocc, frozen, space, config):
    """One physical direct operator, optionally transported to the tensor chart."""
    tensor = config.algorithm == "tensor"
    unrestricted = isinstance(nocc, tuple)
    ns = nocc if unrestricted else (nocc,)
    if tensor:
        build = make_uci_space if unrestricted else make_ci_space
        determinants = build(space.nmo, nocc, max_excitation=max(space.ranks), frozen=frozen,
                             max_determinants=config.max_determinants)
        permutation, phases = space.determinant_map(determinants.determinants)
        indices, signs = jnp.asarray(permutation), jnp.asarray(phases)
        inverse = jnp.asarray(np.argsort(permutation))
        to_physical = lambda c: (c*signs)[inverse]
        from_physical = lambda c: c[indices]*signs
    else:
        determinants = space
        to_physical = from_physical = lambda c: c
    # H0 contains the complete occupied/virtual blocks. Its cross block is
    # excluded even for approximately stationary input, and checked below.
    blocks = []
    for f, no in zip(focks, ns):
        occupied = jnp.arange(f.shape[0]) < no
        blocks.append(jnp.where(occupied[:, None] == occupied[None, :], f, 0.))
    f0 = sum(jnp.trace(f[:no, :no]) for f, no in zip(focks, ns))*(1 if unrestricted else 2)
    one_body = hamiltonian.build_one_body(tuple(blocks) if unrestricted else blocks[0],
                                         determinants, block_diagonal=True)
    # Diagonals do not transform with the signs of the coefficient chart.
    gaps = (one_body.diagonal-f0)[indices] if tensor else one_body.diagonal-f0
    if config.orbital_basis == "noncanonical":
        fock_action = lambda c: from_physical(one_body.apply(to_physical(c)))-f0*c
        spectra = _spectral_focks(focks, nocc, determinants.frozen)
        spectral = hamiltonian.build_one_body(spectra if unrestricted else spectra[0],
                                             determinants, block_diagonal=True)
        spectral_constant = sum(jnp.trace(f[:no, :no]) for f, no in zip(spectra, ns))*(1 if unrestricted else 2)
        spectral_gaps = spectral.diagonal-spectral_constant
        minimum = jnp.min(jnp.abs(spectral_gaps[1:]), initial=jnp.inf)
    else:
        fock_action = lambda c: gaps*c
        minimum = jnp.min(jnp.abs(gaps[1:]), initial=jnp.inf)
    ranks = jnp.asarray(space.ranks)
    maximum = max(space.ranks)
    actions = {}

    def fluctuation(c, degree):
        input_rank, output_rank = _action_ranks(config, maximum, degree)
        key = (input_rank, output_rank)
        if key not in actions:
            build = (hamiltonian.build_fluctuation if config.orbital_basis == "noncanonical"
                     else hamiltonian.build_direct_hamiltonian)
            actions[key] = build(h1, eri, determinants,
                max_input_rank=input_rank, max_output_rank=output_rank)
        state = jnp.where(ranks <= input_rank, c, 0.)
        value = from_physical(actions[key].apply(to_physical(state)))
        if config.orbital_basis == "canonical":
            value = value-reference_energy*state
            if degree:
                value = value-fock_action(state)
        return jnp.where(ranks <= output_rank, value, 0.)

    return gaps, fock_action, fluctuation, minimum, determinants


def run_series(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0., config):
    nuclear_repulsion = jnp.asarray(nuclear_repulsion)
    if nuclear_repulsion.shape != () or jnp.iscomplexobj(nuclear_repulsion):
        raise ValueError("nuclear_repulsion must be a real scalar")
    hs, gs, focks, reference_energy = _reference_data(h1, eri, nocc)
    unrestricted = isinstance(nocc, tuple)
    space = make_series_space(hs[0].shape[0], nocc, frozen=frozen, config=config)
    n = space.nmo
    if config.algorithm == "tensor" and config.orbital_basis == "canonical":
        from ._tensor import build_tensor_action
        gaps = space.gaps(focks)
        action = build_tensor_action(gs if unrestricted else gs[0], focks, space)
        maximum = max(sum(block.excitation) for block in space.blocks)
        def fluctuation(c, degree):
            input_rank, output_rank = _action_ranks(config, maximum, degree)
            return action.apply(c, max_input_rank=input_rank, max_output_rank=output_rank)
        alpha_rank, beta_rank = space.spin_ranks()
        amplitudes = space.amplitudes
        fock_action = lambda c: gaps*c
        minimum = jnp.min(jnp.abs(gaps[1:]), initial=jnp.inf)
    else:
        gaps, fock_action, fluctuation, minimum, determinant_space = _direct_actions(
            h1, eri, focks, reference_energy, nocc, frozen, space, config)
        if config.algorithm == "tensor":
            alpha_rank, beta_rank = space.spin_ranks()
            amplitudes = space.amplitudes
        else:
            reference_bits = determinant_space.determinants[0]
            alpha_rank = np.array([((d ^ reference_bits) & ((1 << n)-1)).bit_count()//2
                                   for d in determinant_space.determinants])
            beta_rank = np.array([((d ^ reference_bits) >> n).bit_count()//2
                                  for d in determinant_space.determinants])
            amplitudes = lambda first: _amplitudes(first, determinant_space, unrestricted)
    h0 = fock_action  # A lambda-independent E_HF shift cannot enter E2..En.
    phi = jnp.zeros(space.size, reference_energy.dtype).at[0].set(1.)
    # W is linear in the state: share W Psi_j across all subsequent degrees.
    ws = [fluctuation(phi, 0)]
    def nonlinear(y, lam, w):
        return jnp.concatenate((-y[-1]*y[:-1]+lam*w[1:],
                                (lam*w[0]-y[-1])[None]))
    zero = jnp.zeros_like(phi)
    lam0 = jnp.zeros((), zero.dtype)
    def jacobian(y):
        # At lambda=0, c=0 and E_corr=0, the residual Jacobian is exactly
        # diag(Q F_N Q, -1). Its live Fock action retains the full input AD.
        state = jnp.concatenate((jnp.zeros(1, y.dtype), y[:-1]))
        return jnp.concatenate((fock_action(state)[1:], -y[-1:]))
    jacdiag = jnp.concatenate((gaps[1:], -jnp.ones(1, zero.dtype)))
    safe = jnp.where(jnp.abs(jacdiag) > config.denominator_tol, jacdiag, 1.)
    operator = LinearOperator((space.size, space.size), zero.dtype, jacobian, diagonal=jacdiag)
    linear = LinearSolverConfig(method="direct" if space.size <= 128 else "gmres",
        max_dense=128, rtol=1e-11, atol=1e-13, maxiter=100, restart=min(40, space.size))
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    ys, waves = [], [phi]
    solved, norm = jnp.asarray(True), lam0
    for degree in range(1, k+1):
        inputs = (tuple(ys+[zero]), tuple([jnp.ones_like(lam0)]+[lam0]*(degree-1)),
                  tuple(ws[1:]+[jnp.zeros_like(phi)]))
        # The unknown kth coefficient is zero in this source evaluation, so
        # its linear F_N term vanishes. Keep the direct operator outside jet.
        _, terms = _jet(nonlinear, (zero, lam0, ws[0]), inputs)
        response = solve_linear(operator, -terms[-1], config=linear,
            preconditioner=lambda v: v/safe, transpose_preconditioner=lambda v: v/safe)
        ys.append(response.solution)
        waves.append(jnp.concatenate((jnp.zeros(1, zero.dtype), response.solution[:-1])))
        ws.append(fluctuation(waves[-1], degree))
        solved = solved & response.converged
        norm = jnp.maximum(norm, response.residual_norm)
    def rayleigh(c, hc):
        return jnp.vdot(c, hc)/jnp.vdot(c, c)
    # Wigner 2k+1: omitted higher state coefficients cannot affect these energies.
    coefficients = tuple(waves[1:]+[jnp.zeros_like(phi)]*max(0, config.order-k))[:config.order]
    hc0 = jnp.zeros_like(phi)  # The normal-ordered block Fock operator annihilates Phi0.
    hcs = [h0(waves[j])+ws[j-1] for j in range(1, k+1)]+[ws[-1]]
    hcs = tuple(hcs+[jnp.zeros_like(phi)]*max(0, config.order-len(hcs)))[:config.order]
    _, es = _jet(rayleigh, (phi, hc0), (coefficients, hcs))
    corrections = jnp.stack(es[1:])
    first = waves[1]
    # Project first-order channels, then apply the same generic energy functional.
    def channel(mask):
        c1 = jnp.where(jnp.asarray(mask), first, 0.)
        _, terms = _jet(rayleigh, (phi, hc0),
            ((c1, jnp.zeros_like(phi)), (h0(c1)+ws[0], fluctuation(c1, 1))))
        return terms[1]
    ss = channel(((alpha_rank == 2) & (beta_rank == 0)) | ((alpha_rank == 0) & (beta_rank == 2)))
    os = channel((alpha_rank == 1) & (beta_rank == 1))
    error = canonical_error(focks)
    stationary_error = reference_error(focks, nocc, frozen=frozen,
        orbital_basis=config.orbital_basis, frozen_gap_tol=config.denominator_tol)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in (*hs, *gs, *focks)]))
    valid = (finite & jnp.isfinite(nuclear_repulsion) & (stationary_error < config.canonical_tol)
             & (minimum > config.denominator_tol) & solved & jnp.all(jnp.isfinite(corrections)))
    # Multiplicative guard rejects invalid JVP and VJP as well as primal values.
    mask = lambda x: x*jnp.where(valid, jnp.ones_like(x), jnp.full_like(x, jnp.nan))
    corr = jnp.sum(corrections)
    e3 = corrections[1] if config.order >= 3 else jnp.zeros_like(corr)
    t2 = jax.tree.map(mask, amplitudes(first)) if config.with_t2 else None
    return MPResult(mask(reference_energy+nuclear_repulsion+corr), mask(corr),
        reference_energy+nuclear_repulsion, mask(corrections[0]), mask(e3), mask(ss), mask(os), t2,
        minimum, error, valid, mask(corrections),
        mask(jnp.stack(waves)) if config.with_coefficients else None, norm)
