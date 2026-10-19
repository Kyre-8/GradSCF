"""A common, finite source-k Hamiltonian for periodic GW methods.

All matrices live in the fixed, orthonormal reference MO frame. The density
accepted by :func:`hartree_exchange` is per spin. Product factors contain the
per-cell Coulomb square root, with no k weight: every physical k sum applies
its own 1/Nk. No q=0 cell correction, orbital AD or JIT is supplied here.
"""
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ...integrals.periodic.fft import get_kpoint_jk
from ...solvers import LinearSolverConfig
from ...solvers.linear import factor_shifted, solve_shifted
from ..freq import scaled_legendre_grid
from ..screened import solve_transition_screening
from .product_basis import kpoint_product_factor_builder, lehmann_weights
from .query import _require_eager, _source, _sigma_imag_full, _sigma_residue_full


@dataclass(frozen=True)
class PeriodicGWModel:
    """Fixed source-k model; factors[q][ki,P,m,n] joins ki to ki-q.

    ``hcore`` and ``reference_potential=F-h-J`` are projected from the actual
    native SCF, without a fitted or frozen Hartree offset. ``inverse[k,q]``
    identifies the internal k+q partner of a self-energy at k. Compression
    changes only the auxiliary frame and records its cross-pair Gram error.
    """
    energy: object
    coeff: object
    hcore: object
    reference_potential: object
    reference_fock: object
    nocc: int
    kpoints: object
    lattice: object
    table: object
    inverse: object
    factors: tuple
    madelung: float = 0.
    metadata: dict | None = None
    nuclear_repulsion: float = 0.
    screening_factors: tuple | None = None

    @property
    def correlation_factors(self):
        """Dielectric vertices; bare Hartree/exchange always use factors."""
        return self.factors if self.screening_factors is None else self.screening_factors

    @property
    def reference_density(self):
        occupation = jnp.arange(self.energy.shape[-1]) < self.nocc
        return jnp.broadcast_to(jnp.diag(occupation.astype(jnp.float64)),
                                self.reference_fock.shape)

    @property
    def chemical_potential(self):
        return (jnp.max(self.energy[:, :self.nocc])+jnp.min(self.energy[:, self.nocc:]))/2

    @property
    def zero_q(self):
        table = np.asarray(self.table)
        zero = np.flatnonzero(np.all(table == np.arange(len(table))[:, None], axis=0))
        if len(zero) != 1:
            raise ValueError('The source model requires one identity (Gamma) transfer')
        return int(zero[0])


def require_eager(*values):
    """Reject unsupported outer AD/JIT before an eager conversion occurs."""
    _require_eager(*values)


def validate_model(model, energy=None):
    """Validate the common finite 3D source model and a global insulating gap."""
    if not isinstance(model, PeriodicGWModel):
        raise TypeError('Use a PeriodicGWModel from build_periodic_model')
    require_eager(model.energy, model.coeff, model.hcore, model.factors, energy)
    values = np.asarray(model.energy if energy is None else energy)
    if (values.ndim != 2 or np.iscomplexobj(values) or not np.isfinite(values).all()
            or not isinstance(model.nocc, Integral) or isinstance(model.nocc, bool)
            or not 0 < model.nocc < values.shape[-1]):
        raise ValueError('Use finite real (nk,nmo) energies with occupied and virtual bands')
    nk, nmo = values.shape
    if values.shape != np.shape(model.energy):
        raise ValueError('The source energy shape cannot change')
    if np.min(values[:, model.nocc:])-np.max(values[:, :model.nocc]) <= 0:
        raise ValueError('The periodic source spectrum requires a common positive insulating gap')
    if not jax.config.x64_enabled:
        raise ValueError('Periodic GW source models require float64')
    for name in ('hcore', 'reference_potential', 'reference_fock'):
        matrix = np.asarray(getattr(model, name))
        if (matrix.shape != (nk, nmo, nmo) or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.conj().swapaxes(-1, -2), atol=1e-9, rtol=0)):
            raise ValueError(f'{name} must contain finite Hermitian source-k matrices')
    if np.shape(model.coeff)[:1] != (nk,) or np.shape(model.coeff)[-1] != nmo:
        raise ValueError('Source coefficients must have shape (nk,nao,nmo)')
    table, inverse = np.asarray(model.table), np.asarray(model.inverse)
    if (table.shape != (nk, nk) or not np.issubdtype(table.dtype, np.integer)
            or np.any(table < 0) or np.any(table >= nk)
            or not np.all(np.sort(table, axis=0) == np.arange(nk)[:, None])
            or not np.array_equal(inverse, np.argsort(table, axis=0))):
        raise ValueError('Use mutually inverse complete momentum-transfer permutations')
    if len(model.factors) != nk:
        raise ValueError('One Coulomb-factor array is required per source transfer')
    if len(model.correlation_factors) != nk:
        raise ValueError('One dielectric-factor array is required per source transfer')
    for factor in (*model.factors, *model.correlation_factors):
        b = np.asarray(factor)
        if (b.ndim != 4 or b.shape[0] != nk or b.shape[1] < 1
                or b.shape[2:] != (nmo, nmo) or not np.isfinite(b).all()):
            raise ValueError('Product factors must have shape (nk,naux_q,nmo,nmo)')
    if not np.isfinite(model.madelung):
        raise ValueError('The Ewald Madelung value must be finite')
    if model.metadata and model.metadata.get('fc', False):
        raise NotImplementedError('The common source-k model currently requires fc=False')
    model.zero_q
    return values


def compress_pair_factors(factors, *, tolerance=1e-12):
    """Compress a whole q block, preserving its complete cross-k pair Gram.

    The relative Frobenius Gram residual is measured after reconstruction.
    ``tolerance=0`` retains the original auxiliary frame exactly. Positive
    tolerances bound a controlled finite-Hamiltonian approximation, not AD.
    """
    require_eager(factors)
    b = np.asarray(factors)
    if b.ndim != 4 or b.shape[0] < 1 or b.shape[1] < 1 or b.shape[2] != b.shape[3]:
        raise ValueError('Use a complete (nk,naux,nmo,nmo) product block')
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1 or not np.isfinite(b).all():
        raise ValueError('Use finite factors and a compression tolerance in [0,1)')
    nk, naux, nmo, _ = b.shape
    if tolerance == 0:
        return jnp.asarray(b), dict(rank=naux, original_rank=naux, relative_gram_residual=0.)
    columns = b.transpose(1, 0, 2, 3).reshape(naux, -1)
    gram = columns.conj().T@columns
    spectrum, vectors = np.linalg.eigh((gram+gram.conj().T)/2)
    norm = np.linalg.norm(gram)
    if norm == 0:
        reduced = np.zeros((1, columns.shape[1]), dtype=columns.dtype)
    else:
        # Discard a tail only when its Frobenius Gram norm is within budget.
        spectrum = np.maximum(spectrum, 0.)
        discarded = np.sqrt(np.cumsum(spectrum**2))
        stop = int(np.searchsorted(discarded, .5*tolerance*norm, side='right'))
        stop = min(stop, len(spectrum)-1)
        reduced = np.sqrt(spectrum[stop:])[:, None]*vectors[:, stop:].conj().T
    error = float(np.linalg.norm(reduced.conj().T@reduced-gram)/max(norm, np.finfo(float).tiny))
    if error > max(tolerance, 100*np.finfo(float).eps):
        raise ArithmeticError(f'Product compression failed its Gram certificate: {error:.3e}')
    result = reduced.reshape(-1, nk, nmo, nmo).transpose(1, 0, 2, 3)
    return jnp.asarray(result), dict(rank=len(reduced), original_rank=naux,
                                     relative_gram_residual=error)


def hartree_exchange(model, density):
    """Common Hartree and negative exchange for a Hermitian ONE-SPIN density.

    J includes the factor two for restricted spin; exchange does not. Ewald
    exchange is ``-madelung*density[k]``, including its off-diagonal entries.
    The q=0 Hartree kernel has its original zero Fourier head removed.
    """
    rho = jnp.asarray(density)
    if rho.shape != np.shape(model.reference_fock):
        raise ValueError('Use one-spin density (nk,nmo,nmo) in the reference MO frame')
    nk = len(model.factors)
    zero = jnp.asarray(model.factors[model.zero_q])
    charge = 2*jnp.einsum('kPnm,kmn->P', zero, rho)/nk
    hartree = jnp.einsum('kPba,P->kab', zero.conj(), charge)
    exchange = jnp.zeros_like(hartree)
    for q, factor in enumerate(model.factors):
        indices = jnp.asarray(model.inverse[:, q])
        b = jnp.asarray(factor)[indices]
        exchange -= jnp.einsum('kPmp,kmn,kPnq->kpq', b.conj(), rho[indices], b)/nk
    exchange -= model.madelung*rho
    return ((hartree+hartree.conj().swapaxes(-1, -2))/2,
            (exchange+exchange.conj().swapaxes(-1, -2))/2)


def rotate_factors(model, rotation, *, factors=None):
    """Rotate both momentum-linked orbital frames; leave auxiliary q frames fixed."""
    u = jnp.asarray(rotation)
    return tuple(jnp.einsum('kma,kPmn,knb->kPab', u.conj(), jnp.asarray(b),
                            u[jnp.asarray(model.table[:, q])])
                 for q, b in enumerate(model.factors if factors is None else factors))


def build_periodic_model(mf, *, tolerance=1e-12, fc=False,
                         max_memory_bytes=2*1024**3, hartree_tolerance=1e-8,
                         screening_cutoff_rydberg=None):
    """Project a native 3D restricted source reference into the common model.

    The model's Hartree is checked against the actual native SCF Hartree.
    A mismatch is rejected; no density-independent correction is inserted.
    General FFT wrapped exchange can differ from native raw-difference K.
    That measured finite-model difference is reported instead of hidden.
    An optional spherical dielectric cutoff |G-q|^2 <= Ecut (Ry) only
    limits correlation vertices; bare Hartree/exchange keep the full FFT
    model. None preserves the full correlation representation.
    """
    require_eager(getattr(mf, 'result', None))
    if fc:
        raise NotImplementedError('The common self-consistent source model requires fc=False; no Gamma-cell map is supplied')
    if screening_cutoff_rydberg is not None and (
            not np.isfinite(screening_cutoff_rydberg) or screening_cutoff_rydberg <= 0):
        raise ValueError('The dielectric cutoff must be finite and positive in Rydberg')
    if getattr(mf.cell, 'dimension', 3) != 3 or getattr(mf.inputs, 'boundary', None) is not None:
        raise NotImplementedError('The common periodic self-consistent source model requires 3D inputs')
    energy, nocc, _, table = _source(mf, False)
    coeff = jnp.asarray(mf.result.mo_coeff_spin[0])
    nk, nmo = energy.shape
    project = lambda matrix: jnp.einsum('kma,kmn,knb->kab', coeff.conj(), matrix, coeff)
    actual_j, _ = get_kpoint_jk(mf.inputs, mf.result.density_spin, mesh=mf.cell.mesh,
                              exxdiv=mf.exxdiv, with_k=False)
    hcore = project(mf.inputs.hcore)
    fock = jax.vmap(jnp.diag)(energy)
    potential = project(mf.result.fock_spin[0]-mf.inputs.hcore-actual_j)
    build = kpoint_product_factor_builder(mf.inputs, coeff, mesh=mf.cell.mesh,
        momentum_table=table, max_block_bytes=max_memory_bytes)
    pairs = np.array([(m, n) for m in range(nmo) for n in range(nmo)], dtype=int)
    factors, compression, screening, screening_info = [], [], [], []
    retained = 0
    for q in range(nk):
        block = build(q, pairs=pairs).reshape(nk, -1, nmo, nmo)
        reduced, certificate = compress_pair_factors(block, tolerance=tolerance)
        retained += reduced.size*reduced.dtype.itemsize
        if screening_cutoff_rydberg is not None:
            transfer = np.asarray(mf.inputs.gvectors)-np.asarray(mf.inputs.kpoints)[q]
            selected = np.sum(transfer**2, axis=1) <= screening_cutoff_rydberg+1e-12
            selected_block = (block[:, jnp.asarray(selected)] if selected.any() else
                              jnp.zeros((nk, 1, nmo, nmo), dtype=block.dtype))
            low, info = compress_pair_factors(selected_block, tolerance=tolerance)
            screening.append(low)
            screening_info.append(dict(info, plane_waves=int(selected.sum())))
            retained += low.size*low.dtype.itemsize
        if retained > max_memory_bytes:
            raise ValueError('Retained source factors exceed max_memory_bytes')
        factors.append(reduced)
        compression.append(certificate)
    metadata = dict(fc=False, compression_tolerance=tolerance,
        compression=compression, mesh=list(mf.cell.mesh),
        screening_cutoff_rydberg=screening_cutoff_rydberg,
        screening_compression=screening_info,
        scope='3D source-k finite Hamiltonian; no Gamma-cell correction or material convergence')
    model = PeriodicGWModel(energy=energy, coeff=coeff, hcore=hcore,
        reference_potential=potential, reference_fock=fock, nocc=nocc,
        kpoints=np.asarray(mf.inputs.kpoints), lattice=np.asarray(mf.inputs.lattice),
        table=table, inverse=np.argsort(table, axis=0), factors=tuple(factors),
        madelung=float(mf.inputs.madelung) if mf.exxdiv == 'ewald' else 0.,
        screening_factors=None if screening_cutoff_rydberg is None else tuple(screening),
        metadata=metadata, nuclear_repulsion=float(mf.inputs.nuclear_repulsion))
    validate_model(model)
    jmodel, xmodel = hartree_exchange(model, model.reference_density)
    difference = float(jnp.max(jnp.abs(jmodel-project(actual_j))))
    metadata['reference_hartree_residual'] = difference
    metadata['reference_fock_residual'] = float(jnp.max(jnp.abs(project(mf.result.fock_spin[0])-fock)))
    metadata['reference_exchange_potential_difference'] = float(jnp.max(jnp.abs(xmodel-potential)))
    if not np.isfinite(hartree_tolerance) or hartree_tolerance <= 0:
        raise ValueError('hartree_tolerance must be finite and positive')
    if difference > hartree_tolerance:
        raise ArithmeticError(f'The source-model Hartree differs from native SCF by {difference:.3e} Ha; no offset is applied')
    return model


@partial(jax.jit, static_argnames=('nk', 'config'))
def _screening_frequencies(gram, differences, signs, projected, frequencies, factorization=None, *, nk, config):
    """Reuse one executable while G/W numbers change between outer steps."""
    flat = projected.reshape(len(differences), -1)
    def at_frequency(frequency):
        if factorization is None:
            response = lehmann_weights(frequency, differences, signs, nk)
            value = solve_transition_screening(gram, response, flat, config=config).solution
        else:
            value = solve_shifted(factorization, -2/nk*signs[:, None]*flat, 1j*frequency,
                                  rtol=config.rtol, atol=config.atol)
        solution = value.reshape(projected.shape)
        return jnp.einsum('rkmp,rkmv->kmpv', projected.conj(), solution)/nk
    return jax.lax.map(at_frequency, frequencies)


def build_correlation(model, energy, *, rotation=None, nw=32, eta=1e-3,
                      linear_config=None, screening_eta=None):
    """Cache complete OV+VO response and both imaginary branches in an MO frame.

    eta is the imaginary-contour G broadening. screening_eta=None preserves
    the legacy time-ordered G-pole susceptibility, with 2i*eta/0 shifts.
    An explicit nonnegative screening_eta instead evaluates real-axis W at
    omega+i*screening_eta*sign(omega), independently of eta. It changes only
    the residues; the complete two-sided imaginary W cache remains eta-free.
    """
    validate_model(model, energy)
    if not isinstance(nw, Integral) or isinstance(nw, bool) or nw < 1:
        raise ValueError('nw must be a positive integer')
    if not np.isfinite(eta) or eta < 0:
        raise ValueError('eta must be finite and nonnegative')
    if screening_eta is not None and (not np.isfinite(screening_eta) or screening_eta < 0):
        raise ValueError('screening_eta must be finite and nonnegative, or None')
    config = linear_config or LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
    e = jnp.asarray(energy)
    nk, nmo = e.shape
    factors = (model.correlation_factors if rotation is None else
               rotate_factors(model, rotation, factors=model.correlation_factors))
    nocc = model.nocc
    pairs = np.array([(i, a) for i in range(nocc) for a in range(nocc, nmo)], dtype=int)
    pairs = np.concatenate((pairs, pairs[:, ::-1]))
    signs = jnp.tile(jnp.concatenate((jnp.ones(len(pairs)//2), -jnp.ones(len(pairs)//2))), nk)
    frequencies, weights = scaled_legendre_grid(nw)
    positive, negative, static, blocks, internal = [], [], [], [], []
    for q, factor in enumerate(factors):
        b = jnp.asarray(factor)
        transitions = jnp.concatenate([b[k][:, pairs[:, 0], pairs[:, 1]] for k in range(nk)], axis=1)
        differences = jnp.concatenate([e[k, pairs[:, 0]]
            -e[int(model.table[k, q]), pairs[:, 1]] for k in range(nk)])
        gram = transitions.conj().T@transitions
        outgoing = b[jnp.asarray(model.inverse[:, q])]
        projected = jnp.einsum('Pr,kPmp->rkmp', transitions.conj(), outgoing)
        imaginary_factor = residue_factor = None
        if config.method == 'direct':
            if 2*len(differences) > config.max_dense:
                raise ValueError('Full signed-transition screening exceeds direct max_dense; use GMRES')
            matrix = -jnp.diag(differences)+2/nk*signs[:, None]*gram
            imaginary_factor = factor_shifted(matrix, symmetric=False)
            residue_factor = (imaginary_factor if eta == 0 or screening_eta is not None else
                factor_shifted(matrix-1j*eta*jnp.diag(signs), symmetric=False))
        sampled = jnp.concatenate((frequencies, -frequencies, jnp.zeros(1, frequencies.dtype)))
        screened = _screening_frequencies(gram, differences, signs, projected, sampled,
                                         imaginary_factor, nk=nk, config=config)
        plus, minus, zero = screened[:nw], screened[nw:2*nw], screened[-1]
        if not all(np.isfinite(x).all() for x in (plus, minus, zero)):
            raise ArithmeticError('Periodic transition screening failed its linear residual certificate')
        positive.append(plus)
        negative.append(minus)
        static.append(zero)
        internal.append(e[jnp.asarray(model.inverse[:, q])])
        block = dict(gram=gram, differences=differences, signs=signs,
                     projected=projected, q_weight=1/nk)
        if screening_eta is not None:
            block['screening_eta'] = screening_eta
        if residue_factor is not None:
            block['shifted_factor'] = residue_factor
        blocks.append(block)
    return dict(positive=tuple(positive), negative=tuple(negative), static=tuple(static),
        blocks=tuple(blocks), internal=tuple(internal), frequencies=frequencies,
        weights=weights, energy=e, ef=(jnp.max(e[:, :nocc])+jnp.min(e[:, nocc:]))/2,
        eta=eta, linear_config=config, nk=nk)


@partial(jax.jit, static_argnames=('nk', 'linear_config'))
def _correlation_kernel(cache, omega, *, k, nk, linear_config):
    """Numerical cache leaves are dynamic; only dimensions/solver controls specialize."""
    positive = jnp.concatenate([value[:, k] for value in cache['positive']], axis=1)
    negative = jnp.concatenate([value[:, k] for value in cache['negative']], axis=1)
    static = jnp.concatenate([value[k] for value in cache['static']], axis=0)
    internal = jnp.concatenate([value[k] for value in cache['internal']])
    blocks = [dict(value, projected=value['projected'][:, k]) for value in cache['blocks']]
    return (_sigma_imag_full(omega, positive, negative, internal, cache['ef'],
        cache['frequencies'], cache['weights'], cache['eta'], static=static)
        +_sigma_residue_full(omega, internal, blocks, cache['ef'], cache['eta'],
                             nk, linear_config, static=static))


def correlation_at(cache, k, omega):
    """Full complex matrix Sigma_c(k,omega) with the shared static subtraction.

    Repeated frequency and self-consistency evaluations pass numerical cache
    arrays dynamically into a module-level kernel, avoiding compiled closures
    that capture a different full screening cache for every iteration.
    """
    numerical = {name: value for name, value in cache.items()
                 if name not in ('nk', 'linear_config')}
    return _correlation_kernel(numerical, jnp.asarray(omega), k=k,
        nk=cache['nk'], linear_config=cache['linear_config'])


def correlation_diagonal(cache, k, omega, bands):
    """Trace-average a complete output cluster before its real-pole solves.

    Internal G bands and source OV/VO transitions stay complete. A singleton
    uses a vector RHS; a larger cluster uses only its principal matrix block.
    Both share the full CD kernels and the original finite-eta convention.
    """
    if not len(bands):
        raise ValueError('A nonempty complete output cluster is required')
    indices = jnp.asarray(bands)
    if len(bands) == 1:
        matrix = lambda value: value[..., indices[0], indices[0]]
        columns = lambda value: value[..., indices[0]]
    else:
        matrix = lambda value: jnp.take(jnp.take(value, indices, axis=-2), indices, axis=-1)
        columns = lambda value: jnp.take(value, indices, axis=-1)
    projected = dict(cache,
        positive=tuple(matrix(value) for value in cache['positive']),
        negative=tuple(matrix(value) for value in cache['negative']),
        static=tuple(matrix(value) for value in cache['static']),
        blocks=tuple(dict(value, projected=columns(value['projected']))
                     for value in cache['blocks']))
    sigma = correlation_at(projected, k, omega)
    return sigma if len(bands) == 1 else jnp.trace(sigma)/len(bands)


def diagonal_dyson_residual(model, energy, *, nw=32, eta=1e-3, linear_config=None):
    """True all-band evGW on-shell residual, rebuilding both G and W."""
    from .evgw import _evaluate_diagonal
    return _evaluate_diagonal(model, energy, nw=nw, eta=eta,
                              linear_config=linear_config).qp_residual


__all__ = ['PeriodicGWModel', 'build_periodic_model', 'compress_pair_factors',
           'validate_model', 'require_eager', 'hartree_exchange', 'rotate_factors',
           'build_correlation', 'correlation_at', 'correlation_diagonal',
           'diagonal_dyson_residual']
