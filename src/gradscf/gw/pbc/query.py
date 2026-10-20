"""Eager restricted G0W0 at arbitrary canonical k points, with fixed SCF W0.

This finite model uses the original complete, equal-weight Gamma-centered
q mesh, full OV and VO response, and shifted internal bands at kappa+q.
Both exchange and correlation use the same G-q product factors. At finite
FFT cutoff this convention can differ at wrapped grid edges from exchange
formed with the raw difference of the original source k representatives.
The older folded-OV periodic GW driver is not a general full-response oracle.

Gapped pure LDA/GGA references and complete selected KS eigenspaces are
supported. In 3D, fc=True replaces the correlation Gamma node by a nonzero
Gauss cell integral with shifted source and internal bands. Reduced-dimensional calls require exxdiv=None, fc=False and
the existing finite-part wire/slab boundary. Their omitted negative bare
exchange head is restored from the actual cross-frame sampled overlaps;
there is no q-cell average or 3D finite-size correction. No geometry/orbital
AD or JIT contract is made.
Imaginary-axis chi is eta-free; G retains eta. Residues evaluate time-ordered
chi at the complex G pole, including its signed displacement. The two sides
of the imaginary axis are retained separately; W(q,z) need not be even in z.
HF is supported only at the original source k points with fc=False. Its
subtracted mean-field potential is the SCF Fock minus core and Hartree;
off-mesh HF, hybrid references and HF Gamma-cell screening are unsupported.
"""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ...dft.xc import hybrid_coeff, parse_recipe, xc_type
from ...integrals.periodic.fft import build_band_inputs, get_kpoint_jk
from ...integrals.periodic.coulomb import ReducedBoundary
from ...pbc._kpoint import grid_xc_potential, project_xc_potential
from ...pbc.kpoints import kmesh_shape
from ...pbc.momentum import momentum_transfer_table, wrap_to_kmesh
from ...pbc.scf import validate_reference
from ...solvers import LinearSolverConfig, LinearOperator, solve_complex
from ...solvers.operators import as_operator
from ...solvers.eigen.dense import dense_vectors
from ...solvers.nonlinear import ScalarRootConfig
from ...solvers.linear import solve_shifted
from ..freq import scaled_legendre_grid
from ..screened import solve_transition_screening
from ..self_energy import sigma_imag_part
from ..types import GWResult
from ..qp import solve_quasiparticle_roots
from .product_basis import (kpoint_product_factor_builder, kpoint_transition_block,
                            lehmann_weights, mixed_product_factors)
from .interaction import finite_head, _check_periodic_directions
from .q0 import gamma_cell_quadrature


def _require_eager(*values):
    # Tracer leaves alone miss jitted closures whose inputs are all constants.
    from jax._src import core
    if (not core.trace_state_clean()
            or any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(values))):
        raise NotImplementedError('g0w0_bands is eager only; JIT and AD are unsupported')


def _sigma_imag_full(omega, positive, negative, energy, ef, freqs, weights, eta,
                     *, occupation=None, static=None):
    """Two-sided contour integral, with shared even-part integration."""
    if static is None and occupation is not None:
        f = jnp.asarray(occupation)
        empty, occupied = omega-energy+1j*eta, omega-energy-1j*eta
        plus = ((1-f)[None, :]/(empty[None, :]-1j*freqs[:, None])
                +f[None, :]/(occupied[None, :]-1j*freqs[:, None]))
        minus = ((1-f)[None, :]/(empty[None, :]+1j*freqs[:, None])
                 +f[None, :]/(occupied[None, :]+1j*freqs[:, None]))
        return -(jnp.tensordot(weights[:, None]*plus, positive, axes=((0, 1), (0, 1)))
                 +jnp.tensordot(weights[:, None]*minus, negative, axes=((0, 1), (0, 1))))/(2*jnp.pi)
    even, odd = (positive+negative)/2, (positive-negative)/2
    if static is None:
        value = sigma_imag_part(omega, even, energy, ef, freqs, weights, eta)
        delta = omega-energy-1j*eta*jnp.sign(ef-energy)
    else:
        sign = jnp.sign(energy-ef) if occupation is None else 1-2*jnp.asarray(occupation)
        value = sigma_imag_part(omega, even, energy, ef, freqs, weights, eta,
                               wmn_static=static, occupation_sign=sign)
        delta = omega-energy
    if odd.ndim == 2:
        return value-1j/jnp.pi*jnp.sum(weights[:, None]*freqs[:, None]*odd
                                      /(delta[None, :]**2+freqs[:, None]**2))
    kernel = weights[:, None]*freqs[:, None]/(delta[None, :]**2+freqs[:, None]**2)
    return value-1j/jnp.pi*jnp.tensordot(kernel, odd, axes=((0, 1), (0, 1)))


def _sigma_residue_full(omega, energy, blocks, ef, eta, nk, config, *, occupation=None, static=None):
    """Strict active poles; projected blocks have (rank,ninternal[,noutput])."""
    above = omega > ef
    prefactor = jax.lax.stop_gradient(jnp.where(above, 1., -1.))
    # The response source has Nk points, but a Gamma-cell quadrature has a
    # different number of outer q samples. Keep these normalizations separate.
    energies = jnp.asarray(energy).reshape(len(blocks), -1)
    occupations = None if occupation is None else jnp.asarray(occupation).reshape(energies.shape)
    config = config or LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
    shape = blocks[0]['projected'].shape[2:]
    output_shape = shape*2
    statics = None if static is None else jnp.asarray(static).reshape((len(blocks), -1)+output_shape)
    total = jnp.zeros(output_shape, dtype=jnp.complex128)
    for q, block in enumerate(blocks):
        q_weight = block.get('q_weight', 1/nk)
        values = energies[q]
        if occupations is None:
            amplitudes = prefactor*jnp.where(above,
                (values > ef) & (values < omega), (values < ef) & (values > omega))
        else:
            amplitudes = jnp.where(omega > values, 1-occupations[q],
                                   jnp.where(omega < values, -occupations[q], 0.))
            # At eta=0 a G pole on the contour contributes half its endpoint
            # residue. Both one-sided limits give (1/2-f)*Wc(0).
            amplitudes = jnp.where((eta == 0) & (omega == values),
                                   .5-occupations[q], amplitudes)
        amplitudes = jax.lax.stop_gradient(amplitudes)
        def at_pole(args):
            amplitude, frequency, projected, index = args
            def evaluate(_):
                def regular(_):
                    if 'screening_eta' in block:
                        z = frequency+1j*block['screening_eta']*jnp.where(frequency >= 0, 1., -1.)
                        weights = 2*block['signs']/(nk*(z+block['differences']))
                    else:
                        weights = lehmann_weights(frequency, block['differences'], block['signs'], nk, eta=eta)
                    return solve_transition_screening(block['gram'], weights, projected, config=config).solution
                def bare_pole(_):
                    # Construct D^-1 directly from its finite denominators,
                    # not from infinite Lehmann weights. At simultaneous
                    # dependent poles this checked system can remain singular.
                    inverse = nk*(frequency+block['differences'])/(2*block['signs'])
                    def apply(x):
                        weighted = inverse*x if x.ndim == 1 else inverse[:, None]*x
                        return weighted-block['gram']@x
                    operator = LinearOperator(block['gram'].shape,
                        jnp.result_type(block['gram'], projected, 1j), apply, matmat=apply)
                    return solve_complex(operator, projected, config=config).solution
                if 'shifted_factor' in block:
                    # Multiplying out the signed Lehmann denominators gives
                    # (z I-K)Z=(2/Nk) S C. K_eta keeps the legacy TO shifts;
                    # explicit screening_eta uses K0 and an independent W
                    # broadening. Both are division-free at a bare pole.
                    broadening = block.get('screening_eta', eta)
                    z = frequency+1j*broadening*jnp.where(frequency >= 0, 1., -1.)
                    rhs = -2/nk*block['signs']
                    rhs = rhs*projected if projected.ndim == 1 else rhs[:, None]*projected
                    solution = solve_shifted(block['shifted_factor'], rhs, z,
                                             rtol=config.rtol, atol=config.atol)
                else:
                    exact_pole = ((block.get('screening_eta', eta) == 0)
                                  & jnp.any(frequency+block['differences'] == 0))
                    solution = jax.lax.cond(exact_pole, bare_pole, regular, operand=None)
                value = (jnp.vdot(projected, solution) if projected.ndim == 1 else
                         projected.conj().T@solution)
                if statics is not None:
                    # Cached W0 already contains this outer q-cell weight.
                    value -= statics[q, index]/q_weight
                return amplitude*value
            return jax.lax.cond(amplitude != 0, evaluate,
                lambda _: jnp.zeros(output_shape, dtype=jnp.complex128), operand=None)
        total += q_weight*jnp.sum(jax.lax.map(at_pole,
            (amplitudes, omega-values, jnp.swapaxes(block['projected'], 0, 1),
             jnp.arange(len(values)))), axis=0)
    return total


def _band_occupations(source_energy, query_energy, shifted_energy, nocc, tolerance,
                      *, band_touching=False):
    """Integer source W; optional T=0 ensemble for complete shifted touchings."""
    source = np.asarray(source_energy)
    if np.min(source[:, nocc])-np.max(source[:, nocc-1]) <= tolerance:
        raise ValueError('The integer screening source must have a positive insulating gap')
    shifted = np.asarray(shifted_energy)
    spectra = np.concatenate([source, np.asarray(query_energy), shifted.reshape(-1, shifted.shape[-1])])
    lower, upper = spectra[:, nocc-1].max(), spectra[:, nocc].min()
    gap = upper-lower
    if gap < -tolerance:
        raise ValueError('The source/query/shifted union has a band overlap; metals are unsupported')
    if gap <= tolerance and not band_touching:
        raise ValueError('The source/query/shifted union has no common insulating gap')
    f = np.broadcast_to((np.arange(shifted.shape[-1]) < nocc).astype(float), shifted.shape).copy()
    if gap <= tolerance:
        flat_f = f.reshape(-1, f.shape[-1])
        for row, occupation in zip(shifted.reshape(flat_f.shape), flat_f):
            if row[nocc]-row[nocc-1] <= tolerance:
                start, stop = nocc-1, nocc+1
                while start > 0 and row[start]-row[start-1] <= tolerance:
                    start -= 1
                while stop < len(row) and row[stop]-row[stop-1] <= tolerance:
                    stop += 1
                occupation[start:stop] = (nocc-start)/(stop-start)
    return (lower+upper)/2, jnp.asarray(f)


def _query_subspaces(energy, selected, tolerance):
    """Partition requested bands, rejecting a cut through any KS cluster."""
    requested = set(map(int, selected))
    groups = []
    for values in np.asarray(energy):
        cuts = np.flatnonzero(np.diff(values) > tolerance)+1
        local = []
        for cluster in np.split(np.arange(len(values)), cuts):
            indices = tuple(map(int, cluster))
            if requested.intersection(indices):
                if not set(indices).issubset(requested):
                    raise ValueError('Query bands must include the complete degenerate subspace self-energy')
                local.append(indices)
        groups.append(tuple(local))
    return groups


def _screened_subspace(projected, solution, positions):
    """Cross contractions for one output cluster, retaining complex Wc."""
    indices = jnp.asarray(positions)
    return jnp.einsum('rmp,rmq->mpq', projected[:, :, indices].conj(),
                      solution[:, :, indices])


def _solve_subspace_qp(energy, exchange, vxc, correlation, *, config, occupied,
                       initial_guess=None):
    """Nonlinear projected Dyson branches; no orbital or geometry AD contract."""
    size = len(energy)
    static = jnp.diag(energy)+exchange-vxc
    shift = jnp.broadcast_to(jnp.where(jnp.asarray(occupied), -.01, .01), energy.shape)

    def spectrum(omega):
        sigma = correlation(omega)
        matrix = static+(sigma+sigma.conj().T)/2
        vectors = dense_vectors(as_operator(matrix), nroots=size, max_dense=size)
        values = jnp.sum(vectors.conj()*(matrix@vectors), axis=0).real
        return values, vectors, matrix, sigma

    roots, sigmas, residuals, projected, valid = [], [], [], [], []
    diagnostics = {name: [] for name in ('weight', 'on_shell', 'linearized',
                                         'candidate_count', 'branch_valid', 'vectors')}
    for branch in range(size):
        def equation(omega):
            values, _, _, _ = spectrum(omega[0])
            return omega-values[branch]
        start = (energy[branch]+shift[branch] if initial_guess is None
                 else initial_guess[branch])
        solved = solve_quasiparticle_roots(equation, energy[branch:branch+1],
            initial_guess=start[None],
            target=None if initial_guess is None else initial_guess[branch:branch+1],
            config=config)
        omega = solved.roots[0]
        values, vectors, matrix, sigma = spectrum(omega)
        vector = vectors[:, branch]
        error = jnp.linalg.norm(matrix@vector-omega*vector)
        sigma_value = jnp.vdot(vector, sigma@vector)
        def hermitian_correlation(frequency):
            value = correlation(frequency)
            return (value+value.conj().T)/2
        _, derivative = jax.jvp(hermitian_correlation, (omega,), (jnp.ones_like(omega),))
        derivative = vectors.conj().T@derivative@vectors
        # Root stopping tolerances do not measure eigenvalue resolution.
        # Combine eigensolve backward error with twice the actual Newton
        # correction. This is a local crossing guard, not a global nonlinear
        # root-isolation bound.
        eigen_error = (jnp.linalg.norm(matrix@vectors-vectors*values)
            +32*size*jnp.finfo(values.dtype).eps*jnp.maximum(1., jnp.linalg.norm(matrix)))
        weight = solved.weight[0]
        finite_weight = jnp.isfinite(weight)
        slope_valid = finite_weight & (weight > 0)
        frequency_error = 2*(jnp.abs(solved.residual[0])+eigen_error)*jnp.where(finite_weight, jnp.abs(weight), jnp.inf)
        coupling = jnp.where(jnp.arange(size) == branch, 0., derivative[branch])
        pair_speed = jnp.sqrt((jnp.diag(derivative).real-derivative[branch, branch].real)**2
                             +4*jnp.abs(coupling)**2)
        crossing_error = jnp.where(pair_speed > 0, pair_speed*frequency_error, 0.)
        degenerate = jnp.abs(values-values[branch]) <= 2*eigen_error+crossing_error
        count = jnp.sum(degenerate)
        restricted = jnp.where(degenerate[:, None]&degenerate[None, :], derivative, 0.)
        average = jnp.trace(restricted).real/count
        projected_derivative = restricted-average*jnp.diag(degenerate)
        # At a frequency crossing, an arbitrary raw eigenvector does not
        # identify a differentiable ordered branch if H' splits the cluster.
        # No inverse eigenvalue gap or clipping enters this certificate.
        unambiguous = (finite_weight & jnp.all(jnp.isfinite(derivative))
            & (jnp.linalg.norm(projected_derivative)
               <= config.slope_floor*jnp.maximum(1., jnp.linalg.norm(restricted))))
        # Preserve a finite negative Z for diagnosis, but never certify it.
        eligible = slope_valid & unambiguous
        roots.append(omega)
        sigmas.append(sigma_value)
        residuals.append(solved.residual[0])
        projected.append(error)
        valid.append(solved.converged[0] & jnp.isfinite(sigma_value) & jnp.isfinite(error)
                     & (error < config.ftol) & eligible)
        for name in diagnostics:
            value = vector if name == 'vectors' else getattr(solved, name)[0]
            if name == 'weight':
                value = jnp.where(unambiguous, value, jnp.nan)
            elif name == 'branch_valid':
                value = value & eligible
            diagnostics[name].append(value)
    result = dict(roots=jnp.stack(roots), sigma=jnp.stack(sigmas), residual=jnp.stack(residuals),
                  projected_residual=jnp.stack(projected), converged=jnp.stack(valid))
    result.update({name: jnp.stack(values, axis=1 if name == 'vectors' else 0)
                   for name, values in diagnostics.items()})
    return result


def _source(mf, fc):
    if not isinstance(fc, (bool, np.bool_)):
        raise TypeError('fc must be a boolean')
    if getattr(mf, 'unrestricted', False):
        raise NotImplementedError('Arbitrary-k GW currently requires a restricted reference')
    boundary = getattr(mf.inputs, 'boundary', None)
    dimension = getattr(mf.cell, 'dimension', 3)
    if fc and (boundary is not None or dimension != 3):
        raise NotImplementedError('Gamma-cell screened integration requires a gapped 3D reference; use fc=False for a finite-part model')
    if boundary is None:
        if dimension != 3:
            raise ValueError('Reduced-dimensional references require their matching boundary')
    elif not isinstance(boundary, ReducedBoundary) or boundary.dimension not in (1, 2):
        raise NotImplementedError('Use 3D inputs or a supported finite-part ReducedBoundary')
    elif boundary.dimension != dimension:
        raise ValueError('The reference cell and boundary dimensions must agree')
    elif mf.exxdiv is not None:
        raise NotImplementedError('Reduced-dimensional GW requires exxdiv=None')
    if mf.exxdiv not in (None, 'ewald'):
        raise NotImplementedError('Arbitrary-k GW requires exxdiv=None or ewald')
    kind = xc_type(mf.xc)
    recipe = parse_recipe(mf.xc)
    recipe.require_complete_support()
    if recipe.exact_exchange.omega is not None:
        raise NotImplementedError('Periodic GW query screened starting potentials are not implemented.')
    if kind != 'HF' and (hybrid_coeff(mf.xc) != 0 or kind not in ('LDA', 'GGA')):
        raise NotImplementedError('GW query supports HF source points or pure LDA/GGA references')
    if kind == 'HF' and fc:
        raise NotImplementedError('HF source-point GW requires fc=False; off-mesh HF Gamma-cell frames are unsupported')
    validate_reference(mf, integer=True, equal_weights=True, low_dimensional=boundary is not None)
    if not jax.config.x64_enabled:
        raise ValueError('Arbitrary-k GW requires float64')
    energy = jnp.asarray(mf.result.mo_energy_spin[0])
    occupation = np.asarray(mf.result.mo_occ_spin)
    nk, nmo = energy.shape
    nocc = int(np.sum(occupation[0, 0]))
    expected = np.broadcast_to((np.arange(nmo) < nocc).astype(float), (2, nk, nmo))
    if not 0 < nocc < nmo or not np.allclose(occupation, expected, atol=1e-10, rtol=0):
        raise NotImplementedError('Use the same nonempty integer occupied space at every k point')
    if not np.allclose(mf.result.mo_energy_spin[0], mf.result.mo_energy_spin[1], atol=1e-10, rtol=0):
        raise NotImplementedError('Use a spin-restricted reference spectrum')
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(mf.inputs.lattice)).T
    _check_periodic_directions(mf.inputs, mf.inputs.kpoints)
    fractions = np.asarray(mf.inputs.kpoints)@np.linalg.inv(reciprocal)
    if not np.allclose(fractions, wrap_to_kmesh(fractions), atol=1e-9, rtol=0):
        raise ValueError('Source k points must use canonical [-0.5,0.5) representatives')
    kmesh_shape(mf.inputs.lattice, mf.inputs.kpoints)
    table = momentum_transfer_table(fractions)
    return energy, nocc, reciprocal, table


def _prepare(mf, kpts, orbs, *, nw, eta, fc, max_memory_bytes, gap_tol, linear_config=None,
             band_touching=False, pole_subtraction=False, self_energy='matrix', q0_order=2):
    source_energy, nocc, reciprocal, table = _source(mf, fc)
    points = np.asarray(kpts)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or np.iscomplexobj(points) or not np.isfinite(points).all()):
        raise ValueError('kpts must be a finite real (nquery,3) array in inverse Bohr')
    _check_periodic_directions(mf.inputs, points)
    is_hf = xc_type(getattr(mf, 'xc', 'pbe')) == 'HF'
    source_rows = None
    if is_hf:
        canonical = wrap_to_kmesh(points@np.linalg.inv(reciprocal))@reciprocal
        matches = np.all(abs(canonical[:, None]-np.asarray(mf.inputs.kpoints)[None]) < 1e-10, axis=-1)
        if np.any(np.sum(matches, axis=1) != 1):
            raise NotImplementedError('HF GW query is restricted to the original source k points')
        source_rows = np.argmax(matches, axis=1)
    nk, nmo = source_energy.shape
    selected = np.asarray(orbs)
    if (selected.ndim != 1 or not len(selected) or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= nmo) or len(np.unique(selected)) != len(selected)):
        raise ValueError('orbs must contain distinct in-range integer band indices')
    if not isinstance(nw, Integral) or isinstance(nw, bool) or nw < 1:
        raise ValueError('nw must be a positive integer')
    if not np.isfinite(eta) or eta < 0 or not np.isfinite(gap_tol) or gap_tol <= 0:
        raise ValueError('eta must be nonnegative and gap_tol must be positive')
    if not isinstance(max_memory_bytes, Integral) or max_memory_bytes < 1:
        raise ValueError('max_memory_bytes must be a positive integer')
    nq, ngrid, nao = len(points), len(mf.inputs.coords), mf.inputs.ao.shape[-1]
    if fc and (isinstance(q0_order, bool) or not isinstance(q0_order, Integral)
               or q0_order < 2 or q0_order % 2):
        raise ValueError('Gamma-cell quadrature order must be an even integer >= 2')
    ncell = q0_order**3 if fc else 0
    qcount = nk+ncell
    static_frequency = int(pole_subtraction or band_touching)
    rank, npair = 2*nk*nocc*(nmo-nocc), nmo*len(selected)
    config = linear_config or LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
    if config.method == 'direct' and 2*rank > config.max_dense:
        raise ValueError('Full transition screening exceeds the real direct max_dense capacity; use GMRES')
    # Dominant retained arrays and live FFT/AO blocks. This is an allocation
    # guard, not a claim about process RSS or compiler memory.
    estimate = 16*(ngrid*(12*nq*nao+8*rank+8*npair+nk*nmo)
        +qcount*(rank*rank+rank*nq*npair+(2*nw+static_frequency)*nq*npair)
        +(qcount+1)*nq*nao*nmo+ncell*nk*nao*(nmo+8*ngrid)+nq*nmo*nmo)
    if self_energy == 'matrix' and len(selected) > 1:
        estimate += 16*(qcount*(2*nw+static_frequency)*nmo+2)*nq*len(selected)**2
    if config.method == 'direct':
        estimate += 48*rank**2  # Shared complex solve's real matrix and workspace.
    if estimate > max_memory_bytes:
        raise ValueError(f'Query arrays exceed max_memory_bytes: estimated {estimate} bytes')
    canonical = wrap_to_kmesh(points@np.linalg.inv(reciprocal))@reciprocal
    if is_hf:
        canonical = np.asarray(mf.inputs.kpoints)[source_rows]
    qpoints = np.asarray(mf.inputs.kpoints)
    qweights = np.full(nk, 1/nk)
    source_shifted = np.empty((0, nk, 3))
    if fc:
        zero = np.flatnonzero(np.all(qpoints == 0., axis=1))
        if len(zero) != 1:
            raise ValueError('Gamma-cell integration requires one exact-zero source q')
        cell_points, cell_weights = gamma_cell_quadrature(mf.inputs.lattice,
            kmesh_shape(mf.inputs.lattice, mf.inputs.kpoints), order=q0_order)
        qweights[zero[0]] = 0.
        qpoints = np.concatenate([qpoints, cell_points])
        qweights = np.concatenate([qweights, cell_weights])
        source_shifted = np.stack([wrap_to_kmesh(
            (np.asarray(mf.inputs.kpoints)-q)@np.linalg.inv(reciprocal))@reciprocal
            for q in cell_points])
    shifted = np.stack([wrap_to_kmesh((canonical+q)@np.linalg.inv(reciprocal))@reciprocal
                        for q in qpoints])
    all_points = np.concatenate([canonical, shifted.reshape(-1, 3), source_shifted.reshape(-1, 3)])
    all_energy, all_coeff = mf.get_bands(all_points)
    all_energy = jnp.asarray(all_energy)
    if not np.isfinite(all_energy).all() or not np.isfinite(all_coeff).all():
        raise ValueError('Band evaluation returned nonfinite arrays')
    target_energy = all_energy[:nq]
    stop = nq*(qcount+1)
    shifted_energy = all_energy[nq:stop].reshape(qcount, nq, nmo)
    source_shifted_energy = all_energy[stop:].reshape(ncell, nk, nmo)
    gap_spectra = jnp.concatenate([target_energy, source_shifted_energy.reshape(-1, nmo)])
    ef, occupation = _band_occupations(source_energy, gap_spectra, shifted_energy,
        nocc, gap_tol, band_touching=band_touching)
    subspaces = _query_subspaces(target_energy, selected, gap_tol)
    if self_energy == 'matrix':
        # A fixed requested window uses one operator at every k point;
        # a KS splitting must not silently remove off-diagonal terms.
        subspaces = [(tuple(sorted(map(int, selected))),)]*nq
    extra = sum(len(group)**2 for groups in subspaces for group in groups if len(group) > 1)
    extra_bytes = 0 if self_energy == 'matrix' else 16*(qcount*(2*nw+static_frequency)*nmo+2)*extra
    if estimate+extra_bytes > max_memory_bytes:
        raise ValueError('Query subspace interactions exceed max_memory_bytes')
    target_coeff = jnp.asarray(all_coeff[:nq])
    target = build_band_inputs(mf.cell, canonical, reference=mf.inputs)
    if is_hf:
        hartree, _ = get_kpoint_jk(mf.inputs, mf.result.density_spin, mesh=mf.cell.mesh,
                                 exxdiv=mf.exxdiv, with_k=False)
        # The subtraction is the actual SCF non-Hartree AO potential,
        # including its exchange boundary; it is not a local XC grid term.
        vxc = (mf.result.fock_spin[0]-mf.inputs.hcore-hartree)[jnp.asarray(source_rows)]
    else:
        _, potential = grid_xc_potential(mf.result.density_spin, mf.inputs, xc=mf.xc)
        vxc = project_xc_potential(target.ao, mf.inputs.weights, potential, xc_type(mf.xc))[0]
    vxc_diagonal = jnp.einsum('kam,kab,kbm->km', target_coeff.conj(), vxc, target_coeff).real
    vxc_blocks = {(k, group): target_coeff[k][:, jnp.asarray(group)].conj().T@vxc[k]
                  @target_coeff[k][:, jnp.asarray(group)]
                  for k, groups in enumerate(subspaces) for group in groups if len(group) > 1}
    target_values = jnp.einsum('kga,kam->kgm', target.ao[:, 0], target_coeff)
    return dict(energy=target_energy, coefficients=target_coeff, target_values=target_values,
        shifted_energy=shifted_energy, shifted_occupation=occupation if band_touching else None,
        shifted_coeff=jnp.asarray(all_coeff[nq:stop]).reshape(qcount, nq, nao, nmo),
        shifted=shifted, canonical=canonical, vxc=vxc_diagonal, ef=ef,
        table=table, nocc=nocc, selected=selected, source_energy=source_energy,
        subspaces=subspaces, vxc_blocks=vxc_blocks, qpoints=qpoints, qweights=qweights,
        source_shifted=source_shifted, source_shifted_energy=source_shifted_energy,
        source_shifted_coeff=jnp.asarray(all_coeff[stop:]).reshape(ncell, nk, nao, nmo))


def _cell_transition_block(mf, data, index, source_values, *, max_memory_bytes):
    """Full response at a nonzero cell point, with shifted source partners."""
    nk = len(data['source_energy'])
    points = data['source_shifted'][index]
    inputs = build_band_inputs(mf.cell, points, reference=mf.inputs)
    values = jnp.einsum('kga,kam->kgm', inputs.ao[:, 0], data['source_shifted_coeff'][index])
    q_cart = data['qpoints'][nk+index]
    def build(q, *, pairs, k_indices):
        return jnp.stack([mixed_product_factors(mf.inputs, source_values[k], values[k],
            q_cart=q_cart, left_k=mf.inputs.kpoints[k], right_k=points[k], pairs=pairs,
            mesh=mf.cell.mesh, max_block_bytes=max_memory_bytes) for k in k_indices])
    # Row k of the separately evaluated partner frame belongs to source k.
    table = np.tile(np.arange(nk)[:, None], (1, nk))
    return kpoint_transition_block(build, data['source_energy'], data['nocc'], table, 0,
        full=True, partner_energy=data['source_shifted_energy'][index])


def _negative_head_exchange(head, left, right, weights, nocc, nk, *, matrix=False, occupation=None):
    """Restore an omitted bare v0/V using actual, possibly non-diagonal overlaps."""
    occupied = left[:, :nocc] if occupation is None else left
    overlap = occupied.conj().T@(weights[:, None]*right)
    fractions = jnp.ones(overlap.shape[0]) if occupation is None else jnp.asarray(occupation)
    value = (overlap.conj().T@(fractions[:, None]*overlap) if matrix else
             jnp.sum(fractions[:, None]*jnp.abs(overlap)**2, axis=0))
    return -jnp.minimum(head, 0.)*value/nk


def g0w0_bands(mf, kpts, orbs, *, nw=32, eta=1e-3, fc=False,
               root_config=None, linear_config=None, max_memory_bytes=512*1024**2,
               gap_tol=1e-7, band_touching=False, pole_subtraction=False,
               self_energy='matrix', initial_guess=None, q0_order=2):
    """Return a GWResult for selected bands at arbitrary cartesian k points.

    ``kpts`` has shape (nquery,3), in inverse Bohr. Points are canonicalized
    into [-0.5,0.5) before any band or product evaluation. ``orbs`` limits only
    output QP bands; all source OV/VO transitions and all shifted internal
    bands remain in the sums. Unrequested energies retain their KS values;
    require both qp_computed_mask and converged_mask when plotting QP bands.
    The returned mo_energy/mo_coeff use the query mesh; screening_energy
    records the original SCF mesh. Those arrays generally have different
    leading dimensions and do not describe a common orbital frame.

    Ewald exchange subtracts inputs.madelung for each occupied query band;
    exxdiv=None omits that constant. fc=True integrates the correlation
    Gamma cell with q0_order**3 nonzero Gauss points. The source OV/VO
    partners and internal G poles are evaluated at their actual shifted
    momenta. Even q0_order values >= 2 are required; order convergence must
    be checked. Bare exchange retains its original mesh and Madelung term.
    This numerical cell integration is restricted to gapped 3D references.
    fc=False retains the finite body-only correlation model.
    Exact HF references require fc=False and source-mesh query points. Their
    non-Hartree potential is reconstructed from the SCF Fock/core/Hartree
    arrays in the matched AO frame. The finite FFT product convention can
    differ from the SCF raw-difference exchange at wrapped grid edges; an
    exact exchange cancellation is not assumed.
    In a 1D/2D finite-part model only exxdiv=None is accepted. Raw momenta
    must vanish in nonperiodic directions before canonical wrapping, and
    the shared factor builder checks sampled orbital orthonormality. A
    removed negative q=0 Coulomb head is restored in bare exchange through
    actual cross-frame overlaps; positive heads are already present once.
    The common QP selector refines several seeds and selects a converged
    positive-slope root nearest the on-shell value. initial_guess, with the
    full query-spectrum shape, supplies both a seed and an explicit
    continuation target. qp_candidate_count counts only tested acceptable
    roots; it is not a global enumeration. Failed/ambiguous branches remain
    flagged. AD slopes, on-shell and linearized energies are returned.

    Every requested KS-degenerate cluster must be complete. The default
    self_energy='matrix' uses the same entire selected window at every k;
    KS splittings do not remove off-diagonal terms. self_energy='diagonal'
    explicitly uses diagonal self-energies and averages contributions over
    complete KS-degenerate clusters. Matrix branches check the projected
    Dyson vector residual as well. sigma_qp is the
    correlation self-energy expectation in each QP eigenvector; mo_coeff
    remains the KS frame and is not a set of updated quasiparticle orbitals.
    qp_vectors stores normalized QP columns in that frame, zero outside
    requested coverage. Columns solve different energy-dependent operators
    and need not be mutually orthogonal.

    band_touching=True permits a T=0 frontier touching within gap_tol; source
    screening still requires integer occupations and a positive sampled gap.
    Complete shifted frontier clusters receive uniform ensemble occupations
    conserving their electron count. G retains both time-ordered poles with
    these weights; bare exchange uses the same occupations. Energies are not
    shifted or gapped. Finite temperature, band-overlap metals, source intraband
    screening and reduced-dimensional q-cell corrections remain unsupported.
    band_touching cannot be combined with fc=True: a gapped sampled source
    does not provide the gapless long-wave screening required for graphene.
    Exact eta=0 bare-transition poles use a division-free checked screening
    limit; simultaneous dependent poles can still fail that linear certificate.

    pole_subtraction=True integrates the static Wc(0) contribution analytically
    and uses G_eta=0 on the imaginary axis, retaining signed eta in real-pole W.
    This avoids quadrature cancellation close to internal G poles. It is exact
    at eta=0; at finite eta it uses the shared static-subtraction convention,
    rather than the default finite-eta imaginary G convention. band_touching
    always enables this stabilization. It does not change the source W.
    """
    _require_eager(kpts, orbs, getattr(mf, 'result', None))
    if self_energy not in ('matrix', 'diagonal'):
        raise ValueError("self_energy must be 'matrix' or 'diagonal'")
    if not isinstance(band_touching, (bool, np.bool_)):
        raise TypeError('band_touching must be a boolean')
    if not isinstance(pole_subtraction, (bool, np.bool_)):
        raise TypeError('pole_subtraction must be a boolean')
    if fc and band_touching:
        raise NotImplementedError('Gamma-cell screening requires a gapped reference; gapless long-wave response is unsupported')
    stable = bool(pole_subtraction or band_touching)
    config = ScalarRootConfig(method='hybrid') if root_config is None else root_config
    if not isinstance(config, ScalarRootConfig):
        raise TypeError('root_config must be a ScalarRootConfig')
    linear_config = (LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
                     if linear_config is None else linear_config)
    if not isinstance(linear_config, LinearSolverConfig):
        raise TypeError('linear_config must be a LinearSolverConfig')
    data = _prepare(mf, kpts, orbs, nw=nw, eta=eta, fc=fc,
                    max_memory_bytes=max_memory_bytes, gap_tol=gap_tol, linear_config=linear_config,
                    band_touching=band_touching, pole_subtraction=stable, self_energy=self_energy,
                    q0_order=q0_order)
    energy, selected, nocc = data['energy'], data['selected'], data['nocc']
    nk, nmo = data['source_energy'].shape
    nq, npout = len(energy), len(selected)
    occupations = data.get('shifted_occupation')
    subspaces = data.get('subspaces') or _query_subspaces(energy, selected, gap_tol)
    if self_energy == 'matrix':
        subspaces = [(tuple(sorted(map(int, selected))),)]*nq
    if initial_guess is not None:
        initial_guess = jnp.asarray(initial_guess)
        if (initial_guess.shape != energy.shape or jnp.iscomplexobj(initial_guess)
                or not np.isfinite(initial_guess[:, selected]).all()):
            raise ValueError('initial_guess must be a real query-shaped spectrum, finite on requested bands')
    positions = {int(band): pos for pos, band in enumerate(selected)}
    matrix_groups = [(k, group, tuple(positions[p] for p in group))
                     for k, groups in enumerate(subspaces) for group in groups if len(group) > 1]
    freqs, weights = scaled_legendre_grid(nw)
    build = kpoint_product_factor_builder(mf.inputs, mf.result.mo_coeff_spin[0],
        mesh=mf.cell.mesh, momentum_table=data['table'], max_block_bytes=max_memory_bytes)
    pairs = np.array([(m, int(p)) for m in range(nmo) for p in selected], dtype=int)
    blocks, interactions = [], []
    exchange = jnp.zeros((nq, npout))
    exchange_blocks = { (k, group): jnp.zeros((len(group), len(group)), dtype=jnp.complex128)
                        for k, group, _ in matrix_groups }
    matrix_interactions = []
    static_interactions = [] if stable else None
    head = float(finite_head(mf.inputs)) if getattr(mf.inputs, 'boundary', None) is not None else 0.
    qpoints = data.get('qpoints', np.asarray(mf.inputs.kpoints))
    qweights = np.asarray(data.get('qweights', np.full(nk, 1/nk)))
    source_values = (jnp.einsum('kga,kam->kgm', mf.inputs.ao[:, 0], mf.result.mo_coeff_spin[0])
                     if fc else None)
    for q, q_cart in enumerate(qpoints):
        block = (kpoint_transition_block(build, data['source_energy'], nocc, data['table'], q, full=True)
                 if q < nk else _cell_transition_block(mf, data, q-nk, source_values,
                                                       max_memory_bytes=max_memory_bytes))
        q_weight = qweights[q]
        shifted = build_band_inputs(mf.cell, data['shifted'][q], reference=mf.inputs)
        values = jnp.einsum('kga,kam->kgm', shifted.ao[:, 0], data['shifted_coeff'][q])
        columns = []
        restore_head = q < nk and head < 0. and np.any(np.all(
            np.asarray(mf.inputs.gvectors)-np.asarray(q_cart) == 0., axis=1))
        for k in range(nq):
            pair = mixed_product_factors(mf.inputs, values[k], data['target_values'][k],
                q_cart=q_cart, left_k=data['shifted'][q, k],
                right_k=data['canonical'][k], pairs=pairs, mesh=mf.cell.mesh,
                max_block_bytes=max_memory_bytes)
            pair_values = pair.reshape(-1, nmo, npout)
            occupation = None if occupations is None else occupations[q, k]
            # Correlation replaces the Gamma node by a cell integral. Bare
            # exchange retains its original mesh plus the Ewald correction;
            # adding exchange at cell samples would double-count that repair.
            if q < nk:
                if occupation is None:
                    bare = jnp.sum(jnp.abs(pair_values[:, :nocc])**2, axis=(0, 1))
                else:
                    bare = jnp.sum(occupation[None, :, None]*jnp.abs(pair_values)**2, axis=(0, 1))
                exchange = exchange.at[k].add(-bare/nk)
                for ki, group, local_positions in matrix_groups:
                    if ki == k:
                        occupied_pairs = pair_values[:, :nocc] if occupation is None else pair_values
                        values_pair = occupied_pairs[:, :, jnp.asarray(local_positions)]
                        weighted = values_pair if occupation is None else occupation[None, :, None]*values_pair
                        exchange_blocks[k, group] -= jnp.einsum('gmp,gmq->pq',
                            values_pair.conj(), weighted)/nk
            if restore_head:
                exchange = exchange.at[k].add(_negative_head_exchange(head, values[k],
                    data['target_values'][k][:, selected], mf.inputs.weights, nocc, nk,
                    occupation=occupation))
                for ki, group, _ in matrix_groups:
                    if ki == k:
                        exchange_blocks[k, group] += _negative_head_exchange(head, values[k],
                            data['target_values'][k][:, jnp.asarray(group)], mf.inputs.weights,
                            nocc, nk, matrix=True, occupation=occupation)
            if q_weight:
                columns.append(block['factors'].conj().T@pair)
        if not q_weight:
            del block, shifted, values, columns, pair
            continue
        projected = jnp.stack(columns, axis=1).reshape(len(block['differences']), -1)
        def at_frequency(frequency):
            response_weights = lehmann_weights(frequency, block['differences'], block['signs'], nk)
            solved = solve_transition_screening(block['gram'], response_weights, projected, config=linear_config)
            diagonal = q_weight*jnp.sum(projected.conj()*solved.solution, axis=0)
            left = projected.reshape(len(block['differences']), nq, nmo, npout)
            right = solved.solution.reshape(left.shape)
            matrices = tuple(q_weight*_screened_subspace(left[:, k], right[:, k], local_positions)
                             for k, _, local_positions in matrix_groups)
            return diagonal, matrices
        (positive, matrix_positive), (negative, matrix_negative) = (
            jax.lax.map(at_frequency, freqs), jax.lax.map(at_frequency, -freqs))
        if not np.isfinite(positive).all() or not np.isfinite(negative).all():
            raise RuntimeError('The full transition screening solve failed its residual certificate')
        interactions.append((positive.reshape(nw, nq, nmo, npout), negative.reshape(nw, nq, nmo, npout)))
        matrix_interactions.append(tuple(zip(matrix_positive, matrix_negative)))
        if stable:
            diagonal_static, matrix_static = at_frequency(jnp.asarray(0.))
            if (not np.isfinite(diagonal_static).all()
                    or any(not np.isfinite(value).all() for value in matrix_static)):
                raise RuntimeError('Static transition screening failed its residual certificate')
            static_interactions.append((diagonal_static.reshape(nq, nmo, npout), matrix_static))
        blocks.append(dict(gram=block['gram'], differences=block['differences'], signs=block['signs'],
                           projected=projected.reshape(len(block['differences']), nq, nmo, npout),
                           q_weight=q_weight))
        del block, shifted, values, columns, pair
    if mf.exxdiv == 'ewald':
        if occupations is not None:
            target_occupation = np.broadcast_to((np.arange(nmo) < nocc).astype(float), energy.shape).copy()
            occupation_clusters = _query_subspaces(energy, np.arange(nmo), gap_tol)
            for k, groups in enumerate(occupation_clusters):
                for group in groups:
                    if group[0] < nocc <= group[-1]:
                        target_occupation[k, list(group)] = (nocc-group[0])/len(group)
        else:
            target_occupation = jnp.broadcast_to(jnp.arange(nmo) < nocc, energy.shape)
        exchange -= jnp.asarray(mf.inputs.madelung)*target_occupation[:, selected]
        for k, group, _ in matrix_groups:
            exchange_blocks[k, group] -= jnp.asarray(mf.inputs.madelung)*jnp.diag(
                jnp.asarray(target_occupation[k])[jnp.asarray(group)])
    qp, residual = energy, jnp.zeros_like(energy)
    sigma = jnp.zeros(energy.shape, dtype=jnp.complex128)
    computed, converged = jnp.zeros(energy.shape, bool), jnp.ones(energy.shape, bool)
    weight, on_shell, linearized = (jnp.full_like(energy, jnp.nan) for _ in range(3))
    candidate_count = jnp.zeros(energy.shape, dtype=jnp.int32)
    branch_valid = jnp.zeros(energy.shape, dtype=bool)
    vectors = jnp.zeros((nq, nmo, nmo), dtype=jnp.complex128)
    for k in range(nq):
        sampled = np.flatnonzero(qweights > 0.)
        internal = data['shifted_energy'][sampled, k, :].reshape(-1)
        internal_occupation = None if occupations is None else occupations[sampled, k, :].reshape(-1)
        for group in subspaces[k]:
            if len(group) > 1:
                group_index = next(i for i, (ki, bands, _) in enumerate(matrix_groups)
                                   if ki == k and bands == group)
                local_positions = jnp.asarray([positions[p] for p in group])
                positive = jnp.concatenate([w[group_index][0] for w in matrix_interactions], axis=1)
                negative = jnp.concatenate([w[group_index][1] for w in matrix_interactions], axis=1)
                static = (jnp.concatenate([w[1][group_index] for w in static_interactions], axis=0)
                          if stable else None)
                local = [dict(b, projected=b['projected'][:, k][:, :, local_positions]) for b in blocks]
                def matrix_correlation(omega):
                    return (_sigma_imag_full(omega, positive, negative, internal, data['ef'],
                        freqs, weights, eta, occupation=internal_occupation, static=static)
                        +_sigma_residue_full(omega, internal, local, data['ef'], eta, nk,
                        linear_config, occupation=internal_occupation, static=static))
                bands = jnp.asarray(group)
                local_exchange = exchange_blocks[k, group]
                local_vxc = data['vxc_blocks'][k, group]
                correlation_function = matrix_correlation
                if self_energy == 'diagonal':
                    # Symmetry-equivalent diagonal self-energies are averaged
                    # over the complete cluster, independently of its gauge.
                    identity = jnp.eye(len(group), dtype=jnp.complex128)
                    local_exchange = identity*jnp.trace(local_exchange)/len(group)
                    local_vxc = identity*jnp.trace(local_vxc)/len(group)
                    correlation_function = lambda omega: identity*jnp.trace(matrix_correlation(omega))/len(group)
                out = _solve_subspace_qp(energy[k, bands], local_exchange,
                    local_vxc, correlation_function, config=config, occupied=bands < nocc,
                    initial_guess=None if initial_guess is None else initial_guess[k, bands])
                qp = qp.at[k, bands].set(out['roots'])
                sigma = sigma.at[k, bands].set(out['sigma'])
                residual = residual.at[k, bands].set(out['residual'])
                computed = computed.at[k, bands].set(True)
                converged = converged.at[k, bands].set(out['converged'])
                weight = weight.at[k, bands].set(out['weight'])
                on_shell = on_shell.at[k, bands].set(out['on_shell'])
                linearized = linearized.at[k, bands].set(out['linearized'])
                candidate_count = candidate_count.at[k, bands].set(out['candidate_count'])
                branch_valid = branch_valid.at[k, bands].set(out['branch_valid'])
                vectors = vectors.at[k, bands[:, None], bands[None, :]].set(out['vectors'])
                continue
            p = group[0]
            pos = positions[p]
            positive = jnp.concatenate([w[0][:, k, :, pos] for w in interactions], axis=1)
            negative = jnp.concatenate([w[1][:, k, :, pos] for w in interactions], axis=1)
            static = (jnp.concatenate([w[0][k, :, pos] for w in static_interactions], axis=0)
                      if stable else None)
            local = [dict(b, projected=b['projected'][:, k, :, pos]) for b in blocks]
            def correlation(omega):
                return (_sigma_imag_full(omega, positive, negative, internal, data['ef'], freqs,
                    weights, eta, occupation=internal_occupation, static=static)
                    +_sigma_residue_full(omega, internal, local, data['ef'], eta, nk,
                    linear_config, occupation=internal_occupation, static=static))
            def equation(omega):
                return jnp.asarray([omega[0]-energy[k, p]-exchange[k, pos]+data['vxc'][k, p]
                                    -correlation(omega[0]).real])
            start = energy[k, p]+(-.01 if p < nocc else .01)
            if initial_guess is not None:
                start = initial_guess[k, p]
            solved = solve_quasiparticle_roots(equation, energy[k, p:p+1],
                initial_guess=start[None],
                target=None if initial_guess is None else initial_guess[k, p:p+1], config=config)
            value = correlation(solved.roots[0])
            success = solved.converged[0] & jnp.isfinite(value)
            qp = qp.at[k, p].set(solved.roots[0])
            sigma = sigma.at[k, p].set(value)
            residual = residual.at[k, p].set(solved.residual[0])
            computed = computed.at[k, p].set(True)
            converged = converged.at[k, p].set(success)
            weight = weight.at[k, p].set(solved.weight[0])
            on_shell = on_shell.at[k, p].set(solved.on_shell[0])
            linearized = linearized.at[k, p].set(solved.linearized[0])
            candidate_count = candidate_count.at[k, p].set(solved.candidate_count[0])
            branch_valid = branch_valid.at[k, p].set(solved.branch_valid[0])
            vectors = vectors.at[k, p, p].set(1.)
    return GWResult(mo_energy=qp, mo_coeff=data['coefficients'], converged=jnp.all(converged),
        sigma_qp=sigma, converged_mask=converged, nw=int(nw), qp_residual=residual,
        qp_computed_mask=computed, screening_energy=data['source_energy'],
        qp_weight=weight, qp_on_shell=on_shell, qp_linearized=linearized,
        qp_candidate_count=candidate_count, qp_branch_valid=branch_valid, qp_vectors=vectors,
        g_orbitals=tuple(range(nmo)), screening_occupied=tuple(range(nocc)),
        screening_virtual=tuple(range(nocc, nmo)),
        method=('periodic_query_full_cd'+('_subtracted' if stable else '')
                +f'_{self_energy}'+(f'_qcell{q0_order}' if fc else '')))


__all__ = ['g0w0_bands']
