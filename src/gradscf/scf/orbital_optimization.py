"""Fixed-occupation SCF with shared JAX forward and implicit/explicit backward.

Occupations specify a static integer topology. All continuous integral inputs,
including overlap and grid arrays, are explicit differentiable parameters.
"""
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from ._pytree import pytree_dataclass
from ..solvers.nonlinear.minimize import solve_orbitals
from ..solvers.nonlinear import stationary_value
from .autodiff import SCFDifferentiationConfig, attach_scf_backward, normalize_scf_gradient_mode
from .uks import UKSConfig, _raw_fock_and_energy_for_state
from ..dft.xc import hybrid_coeff, xc_type
from .rsh import local_xc_spec, rsh_preset, validate_rsh_inputs
from .._memory import (jaxpr_temporary_bytes, tree_storage_bytes,
    validate_memory_mb, validate_reserved_bytes, workspace_bytes)


@pytree_dataclass(static_fields=('optimizer_message', 'polishing_status', 'gradient_mode'))
@dataclass(frozen=True)
class OrbitalOptimizationResult:
    """Array-valued result usable under JIT/grad; stationarity is not stability.

    Gradient norm is half the real orbital-angle gradient norm in the final
    tangent frame. Histories have fixed shapes and include inactive entries.
    ``evaluations`` counts primary objective/line-search calls; HVP calls are
    not separately counted. Use the returned physical density for observables;
    individual orbitals retain gauge freedom within equal-occupation spaces.
    """
    total_energy: jax.Array
    density_matrix: jax.Array
    mo_coeff: jax.Array
    mo_occ: jax.Array
    fock_matrix: jax.Array
    gradient_norm: jax.Array
    stationary: jax.Array
    optimizer_success: jax.Array
    optimizer_status: jax.Array
    optimizer_message: str
    iterations: jax.Array
    evaluations: jax.Array
    initial_energy: jax.Array
    polishing_steps: jax.Array
    polishing_hvp_evaluations: None
    roundoff_energy_allowance: jax.Array
    polishing_status: str
    stage_history: tuple
    polishing_history: tuple
    gradient_mode: str


def _orthonormalize(coeff, metric):
    """Cholesky transport includes the moving AO metric without eigengauge AD."""
    def transport(c):
        gram = c.conj().T @ metric @ c
        lower = jnp.linalg.cholesky(.5*(gram+gram.conj().T))
        return solve_triangular(lower.conj(), c.T, lower=True).T
    return jax.vmap(transport)(coeff)


def _concrete(value):
    return None if isinstance(value, jax.core.Tracer) else np.asarray(value)


def _native_workspace(provider, workers, cutoff=0.):
    if provider is None:
        return 0
    from ..integrals.backends.native.jk import NativeDirectBasis, native_jk_workspace_bytes
    primitive = getattr(provider, 'primitive_basis', provider)
    if not isinstance(primitive, NativeDirectBasis):
        # Opaque providers remain caller-owned through reserved_bytes.
        return 0
    size = native_jk_workspace_bytes(primitive, workers, cutoff)
    if size is None:
        raise ValueError('Native orbital workspace requires complete shell metadata')
    # The existing native estimate includes worker-private shell/table/ENV
    # storage; primal and the two derivative orientations can overlap.
    return 3*size


def _rotation_pairs(method, occupation_rows):
    """Static unequal-occupation pairs in the orbital-angle packing order."""
    occupations = np.asarray(occupation_rows)
    pairs = []
    for occ in ([occupations] if method == 'roks' else occupations):
        different = occ[..., :, None] != occ[..., None, :]
        if different.ndim == 3:
            different = np.any(different, axis=0)
        pairs.append(np.where(np.triu(different, 1)))
    return tuple(pairs)


def _orbital_gap_diagonal(method, occupation_rows, coeff, fock):
    """Frozen-Fock part of the full energy-angle Hessian, in Hartree."""
    occupations = jnp.asarray(occupation_rows)
    pairs = _rotation_pairs(method, occupation_rows)
    blocks = jnp.broadcast_to(coeff, fock.shape) if method == 'roks' else coeff
    eps = jnp.einsum('spi,spq,sqi->si', blocks.conj(), fock, blocks).real
    if method == 'roks':
        i, a = pairs[0]
        return 2*jnp.sum((occupations[:, i]-occupations[:, a])*(eps[:, a]-eps[:, i]), axis=0)
    gaps = []
    for spin, (i, a) in enumerate(pairs):
        gap = 2*(occupations[spin, i]-occupations[spin, a])*(eps[spin, a]-eps[spin, i])
        gaps.extend((gap, gap) if method == 'gks' else (gap,))
    return jnp.concatenate(gaps)


@lru_cache(maxsize=64)
def _gap_function(method, occupation_rows, density_from, evaluate):
    """Cache only the callback identity; physical arrays remain arguments."""
    def diagonal(coeff, args):
        _, fock = evaluate(density_from(coeff), args)
        return _orbital_gap_diagonal(method, occupation_rows, coeff, fock)
    return diagonal


@lru_cache(maxsize=64)
def _rotation_functions(method, occupation_rows):
    """Shared occupation topology, Cayley rotations and density construction."""
    occupations = np.asarray(occupation_rows)
    common = method == 'roks'
    generalized = method == 'gks'
    n = occupations.shape[-1]
    pairs = _rotation_pairs(method, occupation_rows)
    sizes = [len(i) for i, _ in pairs]
    dimension = sum(sizes)*(2 if generalized else 1)

    def rotate(angles, base):
        matrices, offset = [], 0
        for block, ((i, a), size) in enumerate(zip(pairs, sizes)):
            values = angles[offset:offset+size]
            offset += size
            if generalized:
                values = values+1j*angles[offset:offset+size]
                offset += size
            kappa = jnp.zeros((n,n), dtype=base.dtype)
            kappa = kappa.at[i,a].set(values).at[a,i].set(-values.conj())
            # Cayley retraction is unitary for anti-Hermitian kappa and has
            # the same tangent at zero as exp(kappa). Its linear-solve AD avoids
            # the large higher-order scaling/squaring graph in force training.
            identity=jnp.eye(n,dtype=base.dtype)
            rotation=jnp.linalg.solve(identity-.5*kappa,identity+.5*kappa)
            matrices.append(base[block] @ rotation)
        return jnp.stack(matrices)

    def density_from(coeff):
        blocks = jnp.broadcast_to(coeff, (len(occupations),n,n)) if common else coeff
        return jnp.einsum('spi,si,sqi->spq', blocks, jnp.asarray(occupations), blocks.conj())

    return dimension, rotate, density_from


@lru_cache(maxsize=64)
def _problem_functions(method, xc_spec, collinear, occupation_rows, *, uks_config=None):
    dimension, rotate, density_from = _rotation_functions(method, occupation_rows)
    generalized = method == 'gks'
    if generalized:
        from .gks import GKSConfig, generalized_energy_and_fock
        cfg = GKSConfig(xc_spec=xc_spec, collinear=collinear)
        def evaluate(density, args, *, compute_potential=True):
            if args.get('grid_ao') is not None:
                raise NotImplementedError('Generalized orbital minimization requires dense grid AO data.')
            total, _, fock = generalized_energy_and_fock(density=density[0],
                hcore=args['hcore'], eri=args['eri'], nuclear_repulsion=args['nuclear_repulsion'],
                ao=args['ao'], ao_deriv1=args['ao_deriv1'], grid_weights=args['grid_weights'], config=cfg)
            return total, fock[None]
    else:
        kind = xc_type(local_xc_spec(xc_spec))
        if kind not in {'HF','LDA','GGA'}:
            raise NotImplementedError('Orbital minimization supports HF, LDA and GGA/global hybrids.')
        cfg = uks_config or UKSConfig(xc_spec=xc_spec)
        def evaluate(density, args, *, compute_potential=True):
            total, _, fa, fb = _raw_fock_and_energy_for_state(
                density_a=density[0], density_b=density[1], ao=args['ao'],
                ao_deriv1=args['ao_deriv1'], weights=args['grid_weights'], h=args['hcore'],
                eri=args['eri'], df_factors=None, enuc=args['nuclear_repulsion'],
                alpha=jnp.asarray(1. if rsh_preset(xc_spec) is not None else hybrid_coeff(xc_spec),
                                  dtype=density.dtype), cfg=cfg, xc_kind=kind,
                eri_lr=args.get('eri_lr'), dispersion_energy=args.get('dispersion_energy'),
                compute_potential=compute_potential, direct_basis=args.get('direct_basis'),
                grid_ao=args.get('grid_ao'))
            return total, jnp.stack([fa,fb]) if compute_potential else None

    def energy(angles, base, args):
        return evaluate(density_from(rotate(angles, base)), args, compute_potential=False)[0]
    return dimension, rotate, density_from, evaluate, energy


def _minimize(*, method, overlap, hcore, eri, nuclear_repulsion, ao, ao_deriv1,
              grid_weights, mo_coeff, mo_occ, xc_spec, collinear, max_iterations,
              gradient_tolerance, gradient_mode, differentiation, orthonormalize_initial,
              eri_lr=None, dispersion_energy=None, grid_ao=None,
              direct_basis=None, direct_scf_tol=0.0, direct_scf_workers=1,
              max_memory=None, reserved_bytes=0):
    maximum = None if max_memory is None else validate_memory_mb(max_memory)
    reserved = validate_reserved_bytes(reserved_bytes)
    sources = (overlap, hcore, eri, nuclear_repulsion, ao, ao_deriv1, grid_weights,
               mo_coeff, mo_occ, eri_lr, dispersion_energy, grid_ao, direct_basis)
    jk_backend = 'direct' if direct_basis is not None else 'full'
    if direct_basis is not None:
        if method == 'gks':
            raise NotImplementedError('Direct GKS orbital minimization is not supported.')
        if not callable(getattr(direct_basis, 'get_jk', None)) or not hasattr(direct_basis, 'nao'):
            raise TypeError('direct_basis must provide nao and a callable get_jk.')
        if eri is not None and jnp.size(eri):
            raise ValueError('Direct orbital minimization requires eri=None or an empty ERI array.')
    elif eri is None or not jnp.size(eri):
        raise ValueError('Orbital minimization requires ERIs or a direct_basis provider.')
    rsh = validate_rsh_inputs(xc_spec, eri, eri_lr, dispersion_energy,
                              jk_backend=jk_backend, direct_basis=direct_basis)
    if rsh is not None and method == 'gks':
        raise NotImplementedError('Range-separated GKS orbital minimization is not supported.')
    if max_iterations < 1 or gradient_tolerance <= 0:
        raise ValueError('Optimizer iterations and gradient tolerance must be positive.')
    config = differentiation or SCFDifferentiationConfig(mode=gradient_mode or 'implicit')
    if gradient_mode is not None and normalize_scf_gradient_mode(gradient_mode) != config.mode:
        raise ValueError('gradient_mode and differentiation.mode disagree.')
    s, c = jnp.asarray(overlap), jnp.asarray(mo_coeff)
    if not jax.config.x64_enabled:
        raise ValueError('Orbital minimization requires JAX float64/complex128.')
    n = s.shape[0]
    if s.shape != (n,n):
        raise ValueError('Overlap must be square.')
    if direct_basis is not None and direct_basis.nao != n:
        raise ValueError('direct_basis AO dimensions must match the overlap.')
    try:
        occ = np.asarray(mo_occ)
    except jax.errors.TracerArrayConversionError as exc:
        raise ValueError('mo_occ is a static integer occupation topology; close over it before JIT.') from exc
    if method == 'gks':
        if c.shape != (2*n,2*n) or occ.shape != (2*n,):
            raise ValueError('Expected complete spinor coefficients and occupations.')
        metric, coeff, occ = jnp.kron(jnp.eye(2),s), c.astype(jnp.complex128)[None], occ[None]
    else:
        expected = (n,n) if method=='roks' else (2,n,n)
        if c.shape != expected or occ.shape != (2,n):
            raise ValueError(f'Expected mo_coeff shape {expected} and mo_occ shape {(2,n)}.')
        if jnp.iscomplexobj(c) or jnp.iscomplexobj(s):
            raise ValueError('UKS/ROKS currently requires real inputs.')
        if c.dtype != jnp.float64:
            raise ValueError('Orbital minimization requires float64 coefficients.')
        metric, coeff = s, c[None] if method=='roks' else c
    if not np.all(np.isfinite(occ)) or np.any((occ!=0.) & (occ!=1.)):
        raise ValueError('Orbital minimization requires integer occupations 0/1.')
    if rsh is not None and method == 'roks' and not np.array_equal(occ[0], occ[1]):
        raise NotImplementedError('Range-separated shared orbitals currently require closed-shell occupations.')
    for array in (coeff, metric):
        concrete = _concrete(array)
        if concrete is not None and not np.all(np.isfinite(concrete)):
            raise ValueError('Orbital inputs must be finite.')
    native = 0
    if maximum is not None:
        resident = reserved+tree_storage_bytes(sources, coeff, metric)
        workspace_bytes(maximum, resident_bytes=resident)
        native = _native_workspace(direct_basis, direct_scf_workers, direct_scf_tol)
        # Initial transport, density/Fock outputs and returned coefficients
        # are complete mandatory matrices, including a zero-angle space.
        matrix = jax.ShapeDtypeStruct(coeff.shape, coeff.dtype)
        spin = jax.ShapeDtypeStruct((2, n, n), coeff.dtype)
        outputs = tree_storage_bytes(matrix)+2*tree_storage_bytes(spin)
        transport = jaxpr_temporary_bytes(jax.make_jaxpr(_orthonormalize)(coeff, metric))
        workspace_bytes(maximum, resident_bytes=resident+native+outputs+transport)
    gram = coeff.conj().swapaxes(-1,-2) @ metric @ coeff
    concrete = _concrete(gram)
    if concrete is not None and not orthonormalize_initial:
        if not np.allclose(concrete, np.eye(coeff.shape[-1]), atol=1e-8, rtol=0):
            raise ValueError('Initial orbitals must be metric-orthonormal; set orthonormalize_initial=True to transport a seed.')
    topology = tuple(tuple(float(x) for x in row) for row in occ)
    uks_config = UKSConfig(xc_spec=xc_spec, jk_backend=jk_backend,
                           direct_scf_tol=direct_scf_tol, direct_scf_workers=direct_scf_workers)
    dimension, rotate, density_from, evaluate, energy = _problem_functions(
        method, xc_spec, collinear, topology, uks_config=uks_config)
    diagonal = (_gap_function(method, topology, density_from, evaluate)
        if dimension > 32 and method in {'uks', 'roks'}
        and xc_type(local_xc_spec(xc_spec)) == 'HF' else None)
    args = dict(metric=metric, hcore=jnp.asarray(hcore), eri=None if eri is None else jnp.asarray(eri),
        nuclear_repulsion=jnp.asarray(nuclear_repulsion), ao=jnp.asarray(ao),
        ao_deriv1=jnp.asarray(ao_deriv1), grid_weights=jnp.asarray(grid_weights),
        grid_ao=grid_ao, direct_basis=direct_basis)
    if rsh is not None:
        # Present physical arrays cross the implicit boundary as dynamic leaves.
        # A native direct provider supplies LR exchange without an ERI array.
        args.update(eri_lr=None if eri_lr is None else jnp.asarray(eri_lr),
                    dispersion_energy=None if dispersion_energy is None else jnp.asarray(dispersion_energy))
    initial = _orthonormalize(coeff, metric)
    if maximum is None:
        initial_energy = evaluate(density_from(initial),args)[0]
    solve_args, solve_coeff = args, initial
    if config.mode == 'implicit':
        solve_args, solve_coeff = jax.tree.map(jax.lax.stop_gradient, (args,initial))
    if dimension:
        memory = ({} if maximum is None else dict(max_memory=maximum,
            reserved_bytes=reserved+tree_storage_bytes(sources, coeff, metric,
                exclude=(args, initial)), operator_min_bytes=native))
        forward = solve_orbitals(solve_args, solve_coeff, energy=energy, rotate=rotate,
            dimension=dimension, max_iterations=max_iterations, gradient_tolerance=gradient_tolerance,
            hessian_diagonal=diagonal, **memory)
        final, norm = forward.coeff, forward.gradient_norm
        iterations, evaluations, stages, history = forward.iterations, forward.evaluations, forward.stages, forward.polishing
    else:
        final, norm = solve_coeff, jnp.asarray(0., dtype=coeff.real.dtype)
        iterations, evaluations, stages, history = jnp.asarray(0), jnp.asarray(1), (), ()
    if maximum is not None:
        # The same initial observable follows the fixed-buffer preflight;
        # immutable `initial` and live physical args keep its original AD.
        initial_energy = evaluate(density_from(initial),args)[0]
    stationary = jnp.isfinite(norm) & (norm <= gradient_tolerance)
    if config.mode == 'implicit' and dimension:
        anchor = jax.lax.stop_gradient(final)
        origin = jnp.zeros(dimension, dtype=coeff.real.dtype)
        def tangent_energy(angles, root_inputs):
            parameters, anchor = root_inputs
            return energy(angles, _orthonormalize(anchor,parameters['metric']), parameters)
        memory = ({} if maximum is None else dict(max_memory=maximum,
            reserved_bytes=reserved+tree_storage_bytes(sources, coeff, metric, initial, initial_energy, forward,
                exclude=(args, anchor, origin)), operator_min_bytes=native))
        angles = attach_scf_backward((args, anchor), solution=origin, residual=jax.grad(tangent_energy),
                                     config=config, converged=stationary, **memory)
        final = rotate(angles, _orthonormalize(anchor,metric))
    elif config.mode == 'implicit':
        final = _orthonormalize(jax.lax.stop_gradient(final),metric)
    density = density_from(final)
    total, fock = evaluate(density,args)
    stationary &= jnp.isfinite(total)
    if config.mode == 'implicit' and config.require_converged and dimension:
        # The orbital-angle chart is stationary, not the raw AO coefficients.
        # Its partial input derivative retains metric transport (Pulay terms).
        # Only the reported energy uses this rule; state response and the
        # optimizer/residual callbacks remain unchanged, including failures.
        total = stationary_value(tangent_energy, angles, (args, anchor), converged=stationary)
    polishing_steps = sum((row['accepted'].astype(jnp.int32) for row in history), jnp.asarray(0))
    allowance = (jnp.max(jnp.stack([row['energy_allowance'] for row in history]))
                 if history else jnp.asarray(0., dtype=coeff.real.dtype))
    return OrbitalOptimizationResult(
        total_energy=total, density_matrix=density[0] if method=='gks' else density,
        mo_coeff=final if method=='uks' else final[0],
        mo_occ=jnp.asarray(occ[0] if method=='gks' else occ),
        fock_matrix=fock[0] if method=='gks' else fock, gradient_norm=norm, stationary=stationary,
        optimizer_success=stationary, optimizer_status=jnp.where(stationary,0,1),
        optimizer_message='JAX L-BFGS; inspect stationary and gradient_norm',
        iterations=iterations, evaluations=evaluations, initial_energy=initial_energy,
        polishing_steps=polishing_steps, polishing_hvp_evaluations=None,
        roundoff_energy_allowance=allowance, polishing_status='see polishing_history',
        stage_history=stages, polishing_history=history, gradient_mode=config.mode)


def minimize_uks_from_integrals(*, overlap, hcore, eri, nuclear_repulsion, ao, ao_deriv1,
    grid_weights, mo_coeff, mo_occ, xc_spec='hf', max_iterations=500, gradient_tolerance=1e-7,
    gradient_mode=None, differentiation=None, orthonormalize_initial=False,
    eri_lr=None, dispersion_energy=None, grid_ao=None,
    direct_basis=None, direct_scf_tol=0.0, direct_scf_workers=1,
    max_memory=None, reserved_bytes=0):
    """UHF/UKS fixed occupations; backward mode is ``implicit`` or ``explicit``.

    Supply static ``mo_occ`` of shape (2,nao), and float64 coefficients of shape
    (2,nao,nao). Set ``orthonormalize_initial=True`` when varying an overlap with
    a fixed seed. Both modes use identical forward iterations. Unrolled follows
    finite iterates, including initial-orbital dependence; implicit follows the
    stationary branch and returns nonfinite gradients if it has not converged.
    A JAX-pytree AO J/K ``direct_basis`` replaces ERIs; pass ``eri=None``.
    Native providers retain moving coordinates in the stationary response.
    Geometry AD requires ``direct_scf_tol=0``; primitive exponent
    and coefficient derivatives remain unsupported by native direct J/K.
    Optional decimal-MB planning covers known forward and implicit-response
    arrays; explicit outer AD, compilation and opaque library storage are not
    a process cap. None retains the original numerical path.
    """
    return _minimize(method='uks', collinear='col', **locals())


def minimize_roks_from_integrals(*, overlap, hcore, eri, nuclear_repulsion, ao, ao_deriv1,
    grid_weights, mo_coeff, mo_occ, xc_spec='hf', max_iterations=500, gradient_tolerance=1e-7,
    gradient_mode=None, differentiation=None, orthonormalize_initial=False,
    eri_lr=None, dispersion_energy=None, grid_ao=None,
    direct_basis=None, direct_scf_tol=0.0, direct_scf_workers=1,
    max_memory=None, reserved_bytes=0):
    """ROHF/ROKS shared spatial orbitals; see ``minimize_uks_from_integrals``."""
    return _minimize(method='roks', collinear='col', **locals())


def minimize_gks_from_integrals(*, overlap, hcore, eri, nuclear_repulsion, ao, ao_deriv1,
    grid_weights, mo_coeff, mo_occ, xc_spec='hf', collinear='col', max_iterations=500,
    gradient_tolerance=1e-7, gradient_mode=None, differentiation=None, orthonormalize_initial=False,
    grid_ao=None):
    """GHF/GKS complex spinors with spatial overlap and static 0/1 occupations."""
    return _minimize(method='gks', **locals())


__all__ = ['OrbitalOptimizationResult', 'minimize_uks_from_integrals',
           'minimize_roks_from_integrals', 'minimize_gks_from_integrals']
