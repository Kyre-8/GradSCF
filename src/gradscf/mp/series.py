"""MP of arbitrary finite order from a projected residual and Taylor lifting.

The state uses intermediate normalization. The kth wavefunction has excitation
rank at most 2k for a two-body Hamiltonian. A Rayleigh quotient of the state
through k gives exact energy coefficients through 2k+1 (Wigner's rule).
No FCI product space, dense Hamiltonian or eigensolve is constructed here.
Retained state coefficients still grow combinatorially; this is a bounded
in-core generic engine, not the streaming DF-MP2 algorithm.
"""
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.jet import jet

from ..ci.space import make_ci_space, make_uci_space
from ..ci import hamiltonian
from ..fci.cistring import excite
from ..ao2mo import (validate_integrals, validate_unrestricted_integrals,
                     frozen_indices, unrestricted_frozen_indices)
from ..solvers import LinearOperator, LinearSolverConfig, solve_linear
from .._memory import (DEFAULT_MAX_MEMORY_MB, tree_storage_bytes,
                      validate_memory_mb, validate_reserved_bytes, workspace_bytes)
from .types import MPResult
from .integrals import _reference_data, canonical_error, reference_error, _active_orbitals


def make_series_space(nmo, nocc, *, frozen=None, config, packed_only=False):
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    if config.algorithm == "tensor":
        from ._tensor import _with_dense_maps, make_tensor_space
        options = dict(max_excitation=2*k, frozen=frozen,
            max_determinants=config.max_determinants, max_tensor_elements=config.max_tensor_elements,
            packed_only=True)
        space = make_tensor_space(nmo, nocc, **options)
        if (not packed_only and config.orbital_basis == "canonical"
                and space.dense_elements <= config.max_tensor_elements):
            # Preserve the faster dense contractions when their full expansion
            # fits; otherwise retain the identical packed excitation space.
            space = _with_dense_maps(space)
        return space
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


def _minimum_gap(focks, nocc, determinants):
    ns = nocc if isinstance(nocc, tuple) else (nocc,)
    spectra = _spectral_focks(focks, nocc, determinants.frozen)
    spectral = hamiltonian.build_one_body(spectra if isinstance(nocc, tuple) else spectra[0],
                                         determinants, block_diagonal=True)
    constant = sum(jnp.trace(f[:no, :no]) for f, no in zip(spectra, ns)) * (1 if isinstance(nocc, tuple) else 2)
    return jnp.min(jnp.abs((spectral.diagonal - constant)[1:]), initial=jnp.inf)


def _coefficient_chart(space, config):
    if config.algorithm == "tensor":
        from ._tensor_packed import _physical_space
        determinants, indices, signs, inverse = _physical_space(space)
        return (determinants, indices, lambda c: (c * signs)[inverse],
                lambda c: c[indices] * signs)
    return space, None, lambda c: c, lambda c: c


def _state_outputs(space, determinants, config, unrestricted):
    if config.algorithm == "tensor":
        alpha, beta = space.spin_ranks()
        return space.amplitudes, alpha, beta
    reference_bits = determinants.determinants[0]
    n = space.nmo
    alpha = np.array([((d ^ reference_bits) & ((1 << n) - 1)).bit_count() // 2
                      for d in determinants.determinants])
    beta = np.array([((d ^ reference_bits) >> n).bit_count() // 2
                     for d in determinants.determinants])
    return lambda first: _amplitudes(first, determinants, unrestricted), alpha, beta


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


def _direct_actions(h1, eri, focks, reference_energy, nocc, frozen, space, config,
                    max_memory, memory):
    """One physical direct operator, optionally transported to the tensor chart."""
    tensor = config.algorithm == "tensor"
    unrestricted = isinstance(nocc, tuple)
    ns = nocc if unrestricted else (nocc,)
    determinants, indices, to_physical, from_physical = _coefficient_chart(space, config)
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
        minimum = _minimum_gap(focks, nocc, determinants)
    else:
        fock_action = lambda c: gaps*c
        minimum = jnp.min(jnp.abs(gaps[1:]), initial=jnp.inf)
    ranks = jnp.asarray(space.ranks)
    maximum = max(space.ranks)
    # Construct the finite set outside any rematerialized action. Storing a
    # nested checkpoint tracer in a lazy Python cache would leak its AD scope.
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    keys = {_action_ranks(config, maximum, degree) for degree in range(k + 1)}
    keys.add(_action_ranks(config, maximum, 1))
    build = (hamiltonian.build_fluctuation if config.orbital_basis == "noncanonical"
             else hamiltonian.build_direct_hamiltonian)
    actions = {key: build(h1, eri, determinants, max_input_rank=key[0],
        max_output_rank=key[1], max_memory=max_memory,
        reserved_bytes=memory.action_reserved_bytes) for key in sorted(keys)}

    def fluctuation(c, degree):
        input_rank, output_rank = _action_ranks(config, maximum, degree)
        key = (input_rank, output_rank)
        state = jnp.where(ranks <= input_rank, c, 0.)
        value = from_physical(actions[key].apply(to_physical(state)))
        if config.orbital_basis == "canonical":
            value = value-reference_energy*state
            if degree:
                value = value-fock_action(state)
        return jnp.where(ranks <= output_rank, value, 0.)

    return gaps, fock_action, fluctuation, minimum, determinants


def _compiled_config(config):
    return replace(config, denominator_tol=1e-10, canonical_tol=1e-8,
                   with_t2=bool(config.with_t2), with_coefficients=bool(config.with_coefficients))


def run_series(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0., config,
               max_memory=DEFAULT_MAX_MEMORY_MB):
    """Validate static metadata and reuse a graph with live physical inputs."""
    max_memory = validate_memory_mb(max_memory)
    nuclear_repulsion = jnp.asarray(nuclear_repulsion)
    if nuclear_repulsion.shape != () or jnp.iscomplexobj(nuclear_repulsion):
        raise ValueError("nuclear_repulsion must be a real scalar")
    if isinstance(nocc, tuple):
        h1, eri = validate_unrestricted_integrals(h1, eri)
        frozen = unrestricted_frozen_indices(h1[0].shape[0], nocc, frozen)
        nocc = tuple(map(int, nocc))
    else:
        h1, eri = validate_integrals(h1, eri)
        if isinstance(nocc, bool) or not isinstance(nocc, int) or not 0 <= nocc <= h1.shape[0]:
            raise ValueError("nocc must be an integer between 0 and nmo")
        frozen = frozen_indices(h1.shape[0], nocc, frozen)
    # Tolerances are numeric inputs: preserving their weak/strong dtype matters
    # at float32 boundaries. Only topology and output controls enter the key.
    topology = _compiled_config(config)
    return _run_series(h1, eri, config.denominator_tol, config.canonical_tol,
        nocc=nocc, frozen=frozen, nuclear_repulsion=nuclear_repulsion, config=topology,
        max_memory=max_memory)


def run_from_orbitals(coefficients, focks, nocc, reference_energy, *, frozen=None,
                      config, eri=None, eri_pair_matrix=None, df_factors=None,
                      direct_basis=None, max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Noncanonical finite-order MP from live C/F/total-HF-energy/AO inputs.

    The Fock tuple contains the complete occupied and virtual blocks before
    freezing. One selected CI preparation supplies all normal-order kernels;
    no complete MO or spin ERI is formed. The existing Taylor residual and its
    implicit linear response own every order. Native parameters remain dynamic
    pytree inputs. The static MB budget covers known arrays, not compiler RSS.
    """
    if config.orbital_basis != "noncanonical":
        raise ValueError("Selected MP series requires orbital_basis='noncanonical'")
    maximum = validate_memory_mb(max_memory)
    reserved = validate_reserved_bytes(reserved_bytes)
    unrestricted = isinstance(nocc, tuple)
    cs = tuple(map(jnp.asarray, coefficients)) if unrestricted else (jnp.asarray(coefficients),)
    fs, energy = tuple(map(jnp.asarray, focks)), jnp.asarray(reference_energy)
    if (len(cs) != (2 if unrestricted else 1) or any(c.ndim != 2 for c in cs)
            or any(c.shape != cs[0].shape for c in cs) or len(fs) != len(cs)
            or any(f.shape != (cs[0].shape[1],) * 2 for f in fs) or energy.shape != ()):
        raise ValueError("Supply complete AO/MO frames, one Fock per spin and scalar HF energy")
    if any(jnp.iscomplexobj(x) for x in (*cs, *fs, energy)):
        raise NotImplementedError("Selected MP requires real orbitals, Focks and HF energy")
    if any(not jnp.issubdtype(x.dtype, jnp.floating) for x in (*cs, *fs, energy)):
        raise ValueError("Selected MP frames, Focks and HF energy must be floating point")
    if unrestricted:
        frozen = unrestricted_frozen_indices(cs[0].shape[1], nocc, frozen)
        nocc = tuple(map(int, nocc))
    else:
        if isinstance(nocc, bool) or not isinstance(nocc, int) or not 0 <= nocc <= cs[0].shape[1]:
            raise ValueError("nocc must be an integer between 0 and nmo")
        frozen = frozen_indices(cs[0].shape[1], nocc, frozen)
    if sum(x is not None for x in (eri, eri_pair_matrix, df_factors, direct_basis)) != 1:
        raise ValueError("Supply exactly one AO ERI representation")
    eri, eri_pair_matrix, df_factors = (None if x is None else jnp.asarray(x)
                                      for x in (eri, eri_pair_matrix, df_factors))
    return _run_from_orbitals(cs, fs, energy, config.denominator_tol, config.canonical_tol,
        eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
        direct_basis=direct_basis, nocc=nocc, frozen=frozen, config=_compiled_config(config),
        max_memory=maximum, reserved_bytes=reserved)


@partial(jax.jit, static_argnames=("nocc", "frozen", "config", "max_memory", "reserved_bytes"))
def _run_from_orbitals(cs, focks, energy, denominator_tol, canonical_tol, *, nocc, frozen,
                       config, eri, eri_pair_matrix, df_factors, direct_basis,
                       max_memory, reserved_bytes):
    from ..ci.integrals import from_orbitals
    from ._series_memory import history_bytes, plan_series
    unrestricted = isinstance(nocc, tuple)
    space = make_series_space(cs[0].shape[1], nocc, frozen=frozen, config=config, packed_only=True)
    determinants, indices, to_physical, from_physical = _coefficient_chart(space, config)
    source = dict(eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
                  direct_basis=direct_basis)
    dtype = (jnp.dtype(jnp.float64) if direct_basis is not None else
             jnp.result_type(*cs, *focks, energy, *jax.tree.leaves((eri, eri_pair_matrix, df_factors))))
    energy = energy.astype(dtype)
    focks = tuple(f.astype(dtype) for f in focks)
    history = reserved_bytes + history_bytes(space, config, dtype) + tree_storage_bytes(space)
    workspace_bytes(max_memory, resident_bytes=history + tree_storage_bytes(cs, focks, energy, source))
    data = from_orbitals(cs if unrestricted else cs[0], focks if unrestricted else focks[0],
        energy, determinants, max_memory=max_memory, reserved_bytes=history, **source)
    memory = plan_series((cs, energy, source), (), focks, space, nocc, frozen,
        config, max_memory, prepared=data, reserved_bytes=reserved_bytes)
    from ..ci.direct import from_prepared
    def build(part, limits=None):
        is_fock = lambda key: sum(key[0]) + sum(key[1]) == 2 and key[0] == key[1]
        kernels = {key: value for key, value in data.kernels.items()
                   if is_fock(key) == (part == 'fock')}
        reservation = memory.resident_bytes - tree_storage_bytes(kernels)
        kwargs = {} if limits is None else dict(max_input_rank=limits[0], max_output_rank=limits[1])
        return from_prepared(data, determinants, part=part, max_memory=max_memory,
            reserved_bytes=reservation, **kwargs)
    one_body = build('fock')
    gaps = one_body.diagonal[indices] if config.algorithm == "tensor" else one_body.diagonal
    fock_action = lambda c: from_physical(one_body.apply(to_physical(c)))
    k = config.order // 2 if config.wavefunction_order is None else config.wavefunction_order
    maximum = max(space.ranks)
    limits = {_action_ranks(config, maximum, degree) for degree in range(k + 1)}
    limits.add(_action_ranks(config, maximum, 1))
    operators = {key: build('fluctuation', key) for key in sorted(limits)}
    ranks = jnp.asarray(space.ranks)
    def fluctuation(c, degree):
        left, right = _action_ranks(config, maximum, degree)
        state = jnp.where(ranks <= left, c, 0.)
        value = from_physical(operators[left, right].apply(to_physical(state)))
        return jnp.where(ranks <= right, value, 0.)
    minimum = _minimum_gap(focks, nocc, determinants)
    amplitudes, alpha, beta = _state_outputs(space, determinants, config, unrestricted)
    integrals = tuple(x for x in jax.tree.leaves((cs, energy, source, data))
                      if hasattr(x, 'dtype'))
    return _solve_series(space, focks, energy, nocc=nocc, frozen=frozen,
        nuclear_repulsion=jnp.zeros((), dtype), config=config, denominator_tol=denominator_tol,
        canonical_tol=canonical_tol, memory=memory, max_memory=max_memory,
        actions=(gaps, fock_action, fluctuation, minimum, amplitudes, alpha, beta), integrals=integrals)


@partial(jax.jit, static_argnames=("nocc", "frozen", "config", "max_memory"))
def _run_series(h1, eri, denominator_tol, canonical_tol, *, nocc, frozen,
                nuclear_repulsion, config, max_memory):
    hs, gs, focks, reference_energy = _reference_data(h1, eri, nocc)
    unrestricted = isinstance(nocc, tuple)
    space = make_series_space(hs[0].shape[0], nocc, frozen=frozen, config=config,
                              packed_only=True)
    from ._series_memory import plan_series
    memory = plan_series(hs, gs, focks, space, nocc, frozen, config, max_memory)
    if config.algorithm == "tensor" and not memory.packed and config.orbital_basis == "canonical":
        from ._tensor import _with_dense_maps
        space = _with_dense_maps(space)
    n = space.nmo
    if config.algorithm == "tensor" and config.orbital_basis == "canonical":
        from ._tensor import build_tensor_action
        gaps = space.gaps(focks)
        action = build_tensor_action(gs if unrestricted else gs[0], focks, space)
        maximum = max(sum(block.excitation) for block in space.blocks)
        def fluctuation(c, degree):
            input_rank, output_rank = _action_ranks(config, maximum, degree)
            if memory.packed:
                return action.apply_packed(c, max_input_rank=input_rank,
                    max_output_rank=output_rank, max_tile_elements=memory.tile_elements)
            return action.apply(c, max_input_rank=input_rank, max_output_rank=output_rank)
        alpha_rank, beta_rank = space.spin_ranks()
        amplitudes = space.amplitudes
        fock_action = lambda c: gaps*c
        minimum = jnp.min(jnp.abs(gaps[1:]), initial=jnp.inf)
    else:
        gaps, fock_action, fluctuation, minimum, determinant_space = _direct_actions(
            h1, eri, focks, reference_energy, nocc, frozen, space, config, max_memory, memory)
        amplitudes, alpha_rank, beta_rank = _state_outputs(
            space, determinant_space, config, unrestricted)
    actions = (gaps, fock_action, fluctuation, minimum, amplitudes, alpha_rank, beta_rank)
    return _solve_series(space, focks, reference_energy, nocc=nocc, frozen=frozen,
        nuclear_repulsion=nuclear_repulsion, config=config, denominator_tol=denominator_tol,
        canonical_tol=canonical_tol, memory=memory, max_memory=max_memory,
        actions=actions, integrals=(*hs, *gs))


def _solve_series(space, focks, reference_energy, *, nocc, frozen, nuclear_repulsion,
                  config, denominator_tol, canonical_tol, memory, max_memory,
                  actions, integrals):
    """The one residual/Taylor implementation shared by raw and selected inputs."""
    gaps, fock_action, fluctuation, minimum, amplitudes, alpha_rank, beta_rank = actions
    if memory.checkpoint:
        fluctuation = jax.checkpoint(fluctuation, static_argnums=(1,))
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
    safe = jnp.where(jnp.abs(jacdiag) > denominator_tol, jacdiag, 1.)
    operator = LinearOperator((space.size, space.size), zero.dtype, jacobian, diagonal=jacdiag,
                              exact_diagonal=config.orbital_basis == "canonical")
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
            preconditioner=lambda v: v/safe, transpose_preconditioner=lambda v: v/safe,
            max_memory=max_memory, reserved_bytes=memory.resident_bytes,
            operator_min_bytes=memory.operator_min_bytes)
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
        orbital_basis=config.orbital_basis, frozen_gap_tol=denominator_tol)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in (*integrals, *focks)]))
    valid = (finite & jnp.isfinite(nuclear_repulsion) & (stationary_error < canonical_tol)
             & (minimum > denominator_tol) & solved & jnp.all(jnp.isfinite(corrections)))
    # Multiplicative guard rejects invalid JVP and VJP as well as primal values.
    mask = lambda x: x*jnp.where(valid, jnp.ones_like(x), jnp.full_like(x, jnp.nan))
    corr = jnp.sum(corrections)
    e3 = corrections[1] if config.order >= 3 else jnp.zeros_like(corr)
    t2 = jax.tree.map(mask, amplitudes(first)) if config.with_t2 else None
    return MPResult(mask(reference_energy+nuclear_repulsion+corr), mask(corr),
        reference_energy+nuclear_repulsion, mask(corrections[0]), mask(e3), mask(ss), mask(os), t2,
        minimum, error, valid, mask(corrections),
        mask(jnp.stack(waves)) if config.with_coefficients else None, norm)
