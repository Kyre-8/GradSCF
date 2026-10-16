"""Eager restricted G0W0 at arbitrary canonical k points, with fixed SCF W0.

This finite model uses the original complete, equal-weight Gamma-centered
q mesh, full OV and VO response, and shifted internal bands at kappa+q.
Both exchange and correlation use the same G-q product factors. At finite
FFT cutoff this convention can differ at wrapped grid edges from exchange
formed with the raw difference of the original source k representatives.
The older folded-OV periodic GW driver is not a general full-response oracle.

Gapped pure LDA/GGA references, selected nondegenerate output bands, and
fc=False are supported. Reduced-dimensional calls require exxdiv=None and
the existing finite-part wire/slab boundary. Their omitted negative bare
exchange head is restored from the actual cross-frame sampled overlaps;
there is no q-cell average or 3D finite-size correction. No geometry/orbital
AD or JIT contract is made.
Imaginary-axis chi is eta-free; G retains eta. Residues evaluate time-ordered
chi at the complex G pole, including its signed displacement. The two sides
of the imaginary axis are retained separately; W(q,z) need not be even in z.
"""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ...dft.libxc_jax.jax_libxc import hybrid_coeff, xc_type
from ...integrals.periodic.fft import build_band_inputs
from ...integrals.periodic.coulomb import ReducedBoundary
from ...pbc._kpoint import grid_xc_potential, project_xc_potential
from ...pbc.kpoints import kmesh_shape
from ...pbc.momentum import momentum_transfer_table, wrap_to_kmesh
from ...pbc.scf import validate_reference
from ...solvers import LinearSolverConfig
from ...solvers.nonlinear import ScalarRootConfig, solve_scalar_roots
from ..freq import scaled_legendre_grid
from ..screened import solve_transition_screening
from ..self_energy import sigma_imag_part
from ..types import GWResult
from .product_basis import (kpoint_product_factor_builder, kpoint_transition_block,
                            lehmann_weights, mixed_product_factors)
from .interaction import finite_head, _check_periodic_directions


def _require_eager(*values):
    # Tracer leaves alone miss jitted closures whose inputs are all constants.
    from jax._src import core
    if (not core.trace_state_clean()
            or any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(values))):
        raise NotImplementedError('g0w0_bands is eager only; JIT and AD are unsupported')


def _sigma_imag_full(omega, positive, negative, energy, ef, freqs, weights, eta):
    """Two-sided contour integral, with shared even-part integration."""
    even, odd = (positive+negative)/2, (positive-negative)/2
    value = sigma_imag_part(omega, even, energy, ef, freqs, weights, eta)
    delta = omega-energy-1j*eta*jnp.sign(ef-energy)
    return value-1j/jnp.pi*jnp.sum(weights[:, None]*freqs[:, None]*odd
                                   /(delta[None, :]**2+freqs[:, None]**2))


def _sigma_residue_full(omega, energy, blocks, ef, eta, nk, config):
    """Strict active poles only; projected blocks each have (rank,ninternal)."""
    above = omega > ef
    prefactor = jax.lax.stop_gradient(jnp.where(above, 1., -1.))
    energies = jnp.asarray(energy).reshape(nk, -1)
    total = jnp.zeros((), dtype=jnp.complex128)
    for q, block in enumerate(blocks):
        values = energies[q]
        active = jax.lax.stop_gradient(jnp.where(above,
            (values > ef) & (values < omega), (values < ef) & (values > omega)))
        def at_pole(args):
            selected, frequency, projected = args
            def evaluate(_):
                weights = lehmann_weights(frequency, block['differences'], block['signs'], nk, eta=eta)
                solved = solve_transition_screening(block['gram'], weights, projected, config=config)
                return jnp.vdot(projected, solved.solution)
            return jax.lax.cond(selected, evaluate,
                lambda _: jnp.zeros((), dtype=jnp.complex128), operand=None)
        total += prefactor*jnp.sum(jax.lax.map(at_pole,
            (active, omega-values, block['projected'].T)))/nk
    return total


def _source(mf, fc):
    if fc is not False:
        raise NotImplementedError('Arbitrary-k GW currently requires fc=False')
    if getattr(mf, 'unrestricted', False):
        raise NotImplementedError('Arbitrary-k GW currently requires a restricted reference')
    boundary = getattr(mf.inputs, 'boundary', None)
    dimension = getattr(mf.cell, 'dimension', 3)
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
    if hybrid_coeff(mf.xc) != 0 or xc_type(mf.xc) not in ('LDA', 'GGA'):
        raise NotImplementedError('Arbitrary-k GW currently supports pure LDA/GGA references')
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


def _prepare(mf, kpts, orbs, *, nw, eta, fc, max_memory_bytes, gap_tol, linear_config=None):
    source_energy, nocc, reciprocal, table = _source(mf, fc)
    points = np.asarray(kpts)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or np.iscomplexobj(points) or not np.isfinite(points).all()):
        raise ValueError('kpts must be a finite real (nquery,3) array in inverse Bohr')
    _check_periodic_directions(mf.inputs, points)
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
    rank, npair = 2*nk*nocc*(nmo-nocc), nmo*len(selected)
    config = linear_config or LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
    if config.method == 'direct' and 2*rank > config.max_dense:
        raise ValueError('Full transition screening exceeds the real direct max_dense capacity; use GMRES')
    # Dominant retained arrays and live FFT/AO blocks. This is an allocation
    # guard, not a claim about process RSS or compiler memory.
    estimate = 16*(ngrid*(12*nq*nao+8*rank+8*npair+nk*nmo)
        +nk*(rank*rank+rank*nq*npair+2*nw*nq*npair)
        +(nk+1)*nq*nao*nmo)
    if config.method == 'direct':
        estimate += 48*rank**2  # Shared complex solve's real matrix and workspace.
    if estimate > max_memory_bytes:
        raise ValueError(f'Query arrays exceed max_memory_bytes: estimated {estimate} bytes')
    canonical = wrap_to_kmesh(points@np.linalg.inv(reciprocal))@reciprocal
    shifted = np.stack([wrap_to_kmesh((canonical+q)@np.linalg.inv(reciprocal))@reciprocal
                        for q in np.asarray(mf.inputs.kpoints)])
    all_points = np.concatenate([canonical, shifted.reshape(-1, 3)])
    all_energy, all_coeff = mf.get_bands(all_points)
    all_energy = jnp.asarray(all_energy)
    if not np.isfinite(all_energy).all() or not np.isfinite(all_coeff).all():
        raise ValueError('Band evaluation returned nonfinite arrays')
    spectra = np.concatenate([np.asarray(source_energy), np.asarray(all_energy)])
    lower, upper = spectra[:, nocc-1].max(), spectra[:, nocc].min()
    if upper-lower <= gap_tol:
        raise ValueError('The source/query/shifted union has no common insulating gap')
    target_energy = all_energy[:nq]
    for p in selected:
        separations = np.abs(np.asarray(target_energy)[:, p, None]-np.asarray(target_energy))
        separations[:, p] = np.inf
        if separations.min() <= gap_tol:
            raise ValueError('Selected query band is degenerate and requires an unsupported subspace self-energy')
    target_coeff = jnp.asarray(all_coeff[:nq])
    target = build_band_inputs(mf.cell, canonical, reference=mf.inputs)
    _, potential = grid_xc_potential(mf.result.density_spin, mf.inputs, xc=mf.xc)
    vxc = project_xc_potential(target.ao, mf.inputs.weights, potential, xc_type(mf.xc))[0]
    vxc_diagonal = jnp.einsum('kam,kab,kbm->km', target_coeff.conj(), vxc, target_coeff).real
    target_values = jnp.einsum('kga,kam->kgm', target.ao[:, 0], target_coeff)
    return dict(energy=target_energy, coefficients=target_coeff, target_values=target_values,
        shifted_energy=all_energy[nq:].reshape(nk, nq, nmo),
        shifted_coeff=jnp.asarray(all_coeff[nq:]).reshape(nk, nq, nao, nmo),
        shifted=shifted, canonical=canonical, vxc=vxc_diagonal, ef=(lower+upper)/2,
        table=table, nocc=nocc, selected=selected, source_energy=source_energy)


def _negative_head_exchange(head, left, right, weights, nocc, nk):
    """Restore an omitted bare v0/V using actual, possibly non-diagonal overlaps."""
    overlap = left[:, :nocc].conj().T@(weights[:, None]*right)
    return -jnp.minimum(head, 0.)*jnp.sum(jnp.abs(overlap)**2, axis=0)/nk


def g0w0_bands(mf, kpts, orbs, *, nw=32, eta=1e-3, fc=False,
               root_config=None, linear_config=None, max_memory_bytes=512*1024**2,
               gap_tol=1e-7):
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
    exxdiv=None omits that constant. No q->0 head/wing correction is added.
    In a 1D/2D finite-part model only exxdiv=None is accepted. Raw momenta
    must vanish in nonperiodic directions before canonical wrapping, and
    the shared factor builder checks sampled orbital orthonormality. A
    removed negative q=0 Coulomb head is restored in bare exchange through
    actual cross-frame overlaps; positive heads are already present once.
    The root solver returns its best residual candidate if convergence fails;
    no pole-search or branch-selection guarantee is implied by this driver.
    """
    _require_eager(kpts, orbs, getattr(mf, 'result', None))
    config = ScalarRootConfig() if root_config is None else root_config
    if not isinstance(config, ScalarRootConfig):
        raise TypeError('root_config must be a ScalarRootConfig')
    linear_config = (LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
                     if linear_config is None else linear_config)
    if not isinstance(linear_config, LinearSolverConfig):
        raise TypeError('linear_config must be a LinearSolverConfig')
    data = _prepare(mf, kpts, orbs, nw=nw, eta=eta, fc=fc,
                    max_memory_bytes=max_memory_bytes, gap_tol=gap_tol, linear_config=linear_config)
    energy, selected, nocc = data['energy'], data['selected'], data['nocc']
    nk, nmo = data['source_energy'].shape
    nq, npout = len(energy), len(selected)
    freqs, weights = scaled_legendre_grid(nw)
    build = kpoint_product_factor_builder(mf.inputs, mf.result.mo_coeff_spin[0],
        mesh=mf.cell.mesh, momentum_table=data['table'], max_block_bytes=max_memory_bytes)
    pairs = np.array([(m, int(p)) for m in range(nmo) for p in selected], dtype=int)
    blocks, interactions = [], []
    exchange = jnp.zeros((nq, npout))
    head = float(finite_head(mf.inputs)) if getattr(mf.inputs, 'boundary', None) is not None else 0.
    for q in range(nk):
        block = kpoint_transition_block(build, data['source_energy'], nocc, data['table'], q, full=True)
        shifted = build_band_inputs(mf.cell, data['shifted'][q], reference=mf.inputs)
        values = jnp.einsum('kga,kam->kgm', shifted.ao[:, 0], data['shifted_coeff'][q])
        columns = []
        restore_head = head < 0. and np.any(np.all(
            np.asarray(mf.inputs.gvectors)-np.asarray(mf.inputs.kpoints[q]) == 0., axis=1))
        for k in range(nq):
            pair = mixed_product_factors(mf.inputs, values[k], data['target_values'][k],
                q_cart=mf.inputs.kpoints[q], left_k=data['shifted'][q, k],
                right_k=data['canonical'][k], pairs=pairs, mesh=mf.cell.mesh,
                max_block_bytes=max_memory_bytes)
            exchange = exchange.at[k].add(-jnp.sum(jnp.abs(pair.reshape(-1, nmo, npout)[:, :nocc])**2,
                                                   axis=(0, 1))/nk)
            if restore_head:
                exchange = exchange.at[k].add(_negative_head_exchange(head, values[k],
                    data['target_values'][k][:, selected], mf.inputs.weights, nocc, nk))
            columns.append(block['factors'].conj().T@pair)
        projected = jnp.stack(columns, axis=1).reshape(len(block['differences']), -1)
        def at_frequency(frequency):
            response_weights = lehmann_weights(frequency, block['differences'], block['signs'], nk)
            solved = solve_transition_screening(block['gram'], response_weights, projected, config=linear_config)
            return jnp.sum(projected.conj()*solved.solution, axis=0)/nk
        positive, negative = jax.lax.map(at_frequency, freqs), jax.lax.map(at_frequency, -freqs)
        if not np.isfinite(positive).all() or not np.isfinite(negative).all():
            raise RuntimeError('The full transition screening solve failed its residual certificate')
        interactions.append((positive.reshape(nw, nq, nmo, npout), negative.reshape(nw, nq, nmo, npout)))
        blocks.append(dict(gram=block['gram'], differences=block['differences'], signs=block['signs'],
                           projected=projected.reshape(len(block['differences']), nq, nmo, npout)))
        del block, shifted, values, columns, pair
    if mf.exxdiv == 'ewald':
        exchange -= jnp.asarray(mf.inputs.madelung)*(selected[None, :] < nocc)
    qp, residual = energy, jnp.zeros_like(energy)
    sigma = jnp.zeros(energy.shape, dtype=jnp.complex128)
    computed, converged = jnp.zeros(energy.shape, bool), jnp.ones(energy.shape, bool)
    for k in range(nq):
        internal = data['shifted_energy'][:, k, :].reshape(-1)
        for pos, p in enumerate(selected):
            positive = jnp.concatenate([w[0][:, k, :, pos] for w in interactions], axis=1)
            negative = jnp.concatenate([w[1][:, k, :, pos] for w in interactions], axis=1)
            local = [dict(b, projected=b['projected'][:, k, :, pos]) for b in blocks]
            def correlation(omega):
                return (_sigma_imag_full(omega, positive, negative, internal, data['ef'], freqs, weights, eta)
                    +_sigma_residue_full(omega, internal, local, data['ef'], eta, nk, linear_config))
            def equation(omega):
                return jnp.asarray([omega[0]-energy[k, p]-exchange[k, pos]+data['vxc'][k, p]
                                    -correlation(omega[0]).real])
            start = energy[k, p]+(-.01 if p < nocc else .01)
            solved = solve_scalar_roots(equation, start[None], config=config)
            value = correlation(solved.roots[0])
            success = solved.converged[0] & jnp.isfinite(value)
            qp = qp.at[k, p].set(solved.roots[0])
            sigma = sigma.at[k, p].set(value)
            residual = residual.at[k, p].set(solved.residual[0])
            computed = computed.at[k, p].set(True)
            converged = converged.at[k, p].set(success)
    return GWResult(mo_energy=qp, mo_coeff=data['coefficients'], converged=jnp.all(converged),
        sigma_qp=sigma, converged_mask=converged, nw=int(nw), qp_residual=residual,
        qp_computed_mask=computed, screening_energy=data['source_energy'],
        g_orbitals=tuple(range(nmo)), screening_occupied=tuple(range(nocc)),
        screening_virtual=tuple(range(nocc, nmo)), method='periodic_query_full_cd')


__all__ = ['g0w0_bands']
