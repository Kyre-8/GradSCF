"""Static working-array plans shared by real eigensolver callers.

These estimates cover known arrays, not allocator/compilation residency or a
user operator's undeclared internals. Response solves have a separate lifetime
and are deliberately not charged to the numerical Ritz solve.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp

from gradscf._memory import validate_reserved_bytes, workspace_bytes
from .eigen.davidson import _davidson_max_subspace, _solver_dtype


def _largest_fitting(lower, upper, fits):
    """Largest legal static capacity, without a linear search over dimensions."""
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if fits(middle):
            lower = middle
        else:
            upper = middle - 1
    return lower


def plan_hermitian(size, dtype, config, max_memory, *, resident_bytes=0,
                   operator_min_bytes=0, operator_bytes_per_column=0,
                   initial_columns=0, device=None):
    """Return existing config, solver reservation bytes and action column width.

    The caller supplies known resident data and its operator's smallest tile.
    It must charge the returned reservation against the SAME max_memory in
    subsequent actions; a remainder in MB would apply the runtime reserve twice.
    Optional capacities are reduced after applying the solver's existing
    mandatory-root minima; a trial cap below q still means q, as before.
    All explicit guesses and the requested roots plus boundary remain mandatory.
    """
    n, q = int(size), min(config.nroots + 1, int(size))
    if config.nroots > n:
        raise ValueError("nroots exceeds operator dimension")
    for name, value in (("resident_bytes", resident_bytes),
                        ("operator_min_bytes", operator_min_bytes),
                        ("operator_bytes_per_column", operator_bytes_per_column),
                        ("initial_columns", initial_columns)):
        validate_reserved_bytes(value, name=name)
    available = workspace_bytes(max_memory, resident_bytes=resident_bytes, device=device)
    solver_dtype = _solver_dtype(dtype) if config.method == "davidson" else jnp.dtype(dtype)
    itemsize = jnp.dtype(solver_dtype).itemsize

    def operator_bytes(columns):
        return operator_min_bytes + operator_bytes_per_column * columns

    if config.method == "dense":
        if initial_columns:
            raise ValueError("initial_vectors applies only to Davidson")
        if n > config.max_dense:
            raise ValueError("Dense eigensolve exceeds max_dense; use Davidson")
        # Identity, matrix, eigenspace and simultaneous decomposition/update
        # arrays. Numerical-library scratch also has the shared 20% reserve.
        reserved = itemsize * (6*n*n + 6*n*q + 8*n)
        if reserved + operator_bytes(1) > available:
            raise ValueError(f"Dense solver minimum exceeds the memory budget: "
                             f"{reserved + operator_bytes(1)} working bytes required, {available} available")
        batch = _largest_fitting(1, n, lambda b: reserved + operator_bytes(b) <= available)
        return config, reserved, batch

    minimum = min(n, q + 2)
    if config.max_subspace is not None and config.max_subspace < minimum:
        raise ValueError("max_subspace must include nroots plus one boundary root "
                         "and two expansion slots (or the full dimension)")
    upper = _davidson_max_subspace(q, n, config.max_subspace)
    minimum = max(minimum, initial_columns)
    if minimum > upper:
        raise ValueError("initial vectors exceed the Davidson subspace capacity")
    guess_limit = max(q, 2*q if config.initial_guess_count is None
                      else config.initial_guess_count, initial_columns)
    trial_limit = max(q, q if config.max_trial_vectors is None else config.max_trial_vectors)
    collapse_limit = max(q, max(2*q, q+4) if config.collapse_subspace is None
                         else config.collapse_subspace)

    def sizes(m, g, t):
        c = min(n, collapse_limit, max(q, m-t))
        # Old/new V and AV, projected eigenspaces, guesses, Ritz/correction
        # blocks and restart QR/action buffers. No iteration history is stored.
        reserved = itemsize * (4*n*m + 8*m*m + n*(4*g + 8*t + 4*c + 6*q + 8))
        return reserved, c

    min_guess = max(q, initial_columns)
    required = sizes(minimum, min_guess, q)[0] + operator_bytes(1)
    if required > available:
        raise ValueError(f"Davidson minimum exceeds the memory budget: "
                         f"{required} working bytes required, {available} available")
    def preferred(m):
        g, t = min(m, guess_limit), min(m, trial_limit)
        reserved, c = sizes(m, g, t)
        return g, t, c, max(q, g, t, c), reserved

    def preferred_fits(m):
        _, _, _, batch, reserved = preferred(m)
        return reserved + operator_bytes(batch) <= available

    if preferred_fits(minimum):
        # Keep the usual guess/correction/restart action blocks before growing
        # V/AV. A larger optional basis can otherwise starve integral tiles.
        m = _largest_fitting(minimum, upper, preferred_fits)
        g, t, c, batch, reserved = preferred(m)
    else:
        # Only a genuinely tight minimum reduces optional blocks. Preserve a
        # complete root batch whenever possible, and all supplied guesses.
        batch_min = q if sizes(minimum, min_guess, q)[0] + operator_bytes(q) <= available else 1
        m = _largest_fitting(minimum, upper,
            lambda v: sizes(v, min_guess, q)[0] + operator_bytes(batch_min) <= available)
        t = _largest_fitting(q, min(m, trial_limit),
            lambda v: sizes(m, min_guess, v)[0] + operator_bytes(batch_min) <= available)
        g = _largest_fitting(min_guess, min(m, guess_limit),
            lambda v: sizes(m, v, t)[0] + operator_bytes(batch_min) <= available)
        reserved, c = sizes(m, g, t)
        batch = _largest_fitting(batch_min, max(q, g, t, c),
            lambda v: reserved + operator_bytes(v) <= available)
    return replace(config, max_subspace=m, initial_guess_count=g,
                   max_trial_vectors=t, collapse_subspace=c), reserved, batch


def plan_nonhermitian(size, dtype, config, max_memory, *, resident_bytes=0,
                      operator_min_bytes=0, operator_bytes_per_column=0,
                      initial_columns=0, device=None):
    """Plan the existing real two-sided primal solve, preserving its minima.

    Only optional Krylov capacity and action columns can shrink. Requested and
    guard roots, iteration limits, seed and response restart remain untouched.
    Real matrices/cluster SVD and complex eigenspaces have separate storage
    widths. This covers known primal arrays; implicit response has a separate
    lifetime and its existing solver configuration is not changed here.
    Return (replaced config, solver reservation bytes, maximum action columns).
    All callers retain the original MB budget, including its one 20% reserve.
    """
    n, k = int(size), config.nroots
    if k > n:
        raise ValueError("nroots exceeds operator dimension")
    for name, value in (("resident_bytes", resident_bytes),
                        ("operator_min_bytes", operator_min_bytes),
                        ("operator_bytes_per_column", operator_bytes_per_column),
                        ("initial_columns", initial_columns)):
        validate_reserved_bytes(value, name=name)
    real = jnp.dtype(dtype).itemsize
    complex_ = jnp.dtype(jnp.result_type(dtype, jnp.complex64)).itemsize
    integer = 8 if jax.config.x64_enabled else 4
    available = workspace_bytes(max_memory, resident_bytes=resident_bytes, device=device)

    def action_bytes(columns):
        # A probe/input and its output coexist with the caller's action scratch.
        return operator_min_bytes+(2*n*real+operator_bytes_per_column)*columns

    if config.method == "dense":
        if initial_columns:
            raise ValueError("initial_vectors applies only to Davidson")
        if n > config.max_dense:
            raise ValueError("Non-Hermitian dense reference exceeds max_dense")
        # Stored blocks plus assembled A; real shifted matrix, U/V, overlap,
        # solve/repair work; two complex eigenspaces and sorted/selected copies.
        reserved = (real*(10*n*n+12*n*k+8*k*k+8*n)
                    +complex_*(4*n*n+4*n*k+8*n)
                    +integer*(8*n+2*k)+4*(n+k))
        required = reserved+action_bytes(k)
        if required > available:
            raise ValueError(f"Non-Hermitian dense minimum exceeds the memory budget: "
                             f"{required} working bytes required, {available} available")
        upper = max(k, min(n, config.block_size))
        batch = _largest_fitting(k, upper, lambda b: reserved+action_bytes(b) <= available)
        return replace(config, block_size=min(config.block_size, batch)), reserved, batch

    q = min(n, k+config.guard_roots)
    upper = min(n, config.max_space)
    minimum = min(n, 4*q+2)
    if upper < minimum:
        raise ValueError("max_space must be at least 4*(nroots+guard_roots)+2 (or the full dimension)")
    if initial_columns > upper:
        raise ValueError("initial vectors exceed the Davidson max_space capacity")
    minimum = max(minimum, initial_columns)

    def reservation(m):
        # Old/new V, AV/ATV; projected real matrix and cluster-repair SVD;
        # real restart/correction/seed blocks; complex projected eigenspaces,
        # Ritz pairs, both residuals and preconditioned correction work.
        return (real*(4*n*m+10*m*m+n*(16*q+2*initial_columns+6*k)+8*k*k+8*n)
                +complex_*(4*m*m+8*n*q+4*m*q+8*m)
                +integer*(8*m+2*q)+4*(m+q))

    required = reservation(minimum)+action_bytes(k)
    if required > available:
        raise ValueError(f"Non-Hermitian Davidson minimum exceeds the memory budget: "
                         f"{required} working bytes required, {available} available")
    capacity = _largest_fitting(minimum, upper,
        lambda m: reservation(m)+action_bytes(k) <= available)
    reserved = reservation(capacity)
    batch = _largest_fitting(k, capacity,
        lambda b: reserved+action_bytes(b) <= available)
    return replace(config, max_space=capacity), reserved, batch


def apply_columns(operator, vectors, batch_size=None):
    """Apply an existing linear operator with bounded simultaneous columns."""
    if vectors.ndim == 1 or batch_size is None or vectors.shape[1] <= batch_size:
        return operator.apply(vectors)
    columns = vectors.shape[1]

    def body(index, output):
        indices = index * batch_size + jnp.arange(batch_size)
        block = jnp.take(vectors, indices, axis=1, mode="fill", fill_value=0.)
        return output.at[:, indices].set(operator.apply(block), mode="drop")

    return jax.lax.fori_loop(0, (columns + batch_size - 1) // batch_size,
        body, jnp.zeros((operator.shape[0], columns),
                        dtype=jnp.result_type(operator.dtype, vectors.dtype)))
