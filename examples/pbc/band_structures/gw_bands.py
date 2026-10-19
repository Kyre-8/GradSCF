"""Direct PBE -> G0W0 bands in compact, explicitly labelled finite models.

The geometries match the material DFT examples, but the SZV basis, source
k mesh and FFT grids differ. Each QP is evaluated directly by GradSCF;
there is no scissor shift, QP interpolation or artificial graphene gap.
Start with a two-point pilot before requesting a reciprocal-space path.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import dft, gto
from gradscf.pbc.momentum import wrap_to_kmesh
from gradscf.gw.pbc import g0w0_bands
from gradscf.solvers import LinearSolverConfig
from gradscf.solvers.nonlinear import ScalarRootConfig
from examples.pbc.band_structures.systems import structure


OUTPUT = Path('reproducibility/ad_showcase/material_gw_bands')
MESH = {'silicon': (21, 21, 21), 'gaas': (21, 21, 21),
        'gan': (21, 21, 33), 'graphene': (25, 25, 97)}
# Explicit output windows. G and W always retain all bands/transitions.
# A partial degenerate cluster is rejected by the SRC query implementation;
# inspect the saved KS spectrum before deliberately enlarging a window.
ORBS = {'silicon': tuple(range(1, 7)), 'gaas': tuple(range(6, 12)),
        'gan': tuple(range(15, 21)), 'graphene': tuple(range(7))}


def reference(material='silicon', *, checkpoint=None):
    """Return a converged native SCF reference, optionally seeded by density.

    A checkpoint is an initial density, not a reconstructed private SCF state.
    Use one from the same geometry, basis, pseudopotential and source k mesh.
    The historical Si SZV checkpoint is compatible with this Si declaration.
    """
    atoms, _, physical_dimension = structure(material, vacuum=12.)
    dimension = physical_dimension if material == 'graphene' else 3
    cell = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
        a=atoms.cell.array, unit='Angstrom', basis='gth-szv', pseudo='gth-pbe',
        mesh=MESH[material], cart=True, precision=1e-10, dimension=dimension)
    kmesh = (2, 2, 1) if material == 'graphene' else (2, 1, 1)
    mf = dft.KRKS(cell, kpts=cell.make_kpts(kmesh), xc='pbe',
        exxdiv=None if dimension == 2 else 'ewald',
        temperature=None,
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9,
        max_cycle=150, damping=.15)
    density = None
    if checkpoint is not None:
        with np.load(checkpoint) as saved:
            key = 'density_spin' if 'density_spin' in saved else 'density'
            density = saved[key].sum(axis=0)
    mf.kernel(dm0=density)
    if not mf.converged:
        raise RuntimeError('The native SCF reference did not converge')
    return mf


def run(material='silicon', *, case=None, path_points=None, checkpoint=None,
        calculate_gw=True, orbs=None, resume_from=None):
    """Save a two-point pilot or a directly evaluated high-symmetry path.

    ``calculate_gw=False`` saves SCF/query spectra and restart density only.
    ``path_points`` uses the ordinary ASE path bookkeeping; all selected QPs
    are then calculated at those actual points in separate one-point calls.
    ``orbs`` is an explicit output window, shared by all requested points.
    ``resume_from`` reuses saved points in a new case after checking its model,
    physical source hashes, and reconverged KS spectrum. Earlier files remain.
    Graphene starts from a new zero-temperature SCF. ``band_touching=True``
    classifies a numerical Dirac contact without changing any KS energies.
    This is not a finite-temperature or intraband metallic GW calculation.
    """
    label = case or f'{material}_pbe_szv_pilot'
    folder = OUTPUT/label
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/'native.json'
    if destination.exists():
        raise FileExistsError('Use a new case name; preserve earlier results')
    started = perf_counter()
    report = dict(complete=False, passed=False, stage='scf', material=material,
                  case=label, calculate_gw=calculate_gw)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    mf = reference(material, checkpoint=checkpoint)
    jax.block_until_ready(mf.result)
    scf_seconds = perf_counter()-started
    atoms, path_name, _ = structure(material, vacuum=12.)
    if path_points is None:
        fractional = np.array([[0., 0., 0.], [.17, .13, .11]])
        distance, ticks, labels = np.arange(2), [0, 1], ['G', 'generic']
        if material == 'graphene':
            fractional[1] = [1/3, 1/3, 0.]
            labels = ['G', 'K']
    else:
        path = atoms.cell.bandpath(path_name, npoints=path_points, pbc=atoms.pbc)
        fractional = path.kpts
        distance, ticks, labels = path.get_linear_kpoint_axis()
    path_fractional = fractional.copy()
    fractional = wrap_to_kmesh(fractional)
    kpoints = fractional@np.asarray(mf.cell.reciprocal_vectors())
    ks_energy, ks_coeff = mf.get_bands(kpoints, chunk_size=1)
    ks_energy, ks_coeff = np.asarray(ks_energy), np.asarray(ks_coeff)
    selected = tuple(ORBS[material] if orbs is None else orbs)
    nocc, nmo = mf.cell.nelectron//2, ks_energy.shape[1]
    energy_zero = max(float(ks_energy[:, nocc-1].max()), float(np.asarray(mf.mo_energy)[:, nocc-1].max()))
    if mf.chemical_potential is not None:
        energy_zero = float(np.asarray(mf.chemical_potential).mean())
    sources = ['examples/pbc/band_structures/gw_bands.py',
        'examples/pbc/band_structures/systems.py', 'src/gradscf/gw/pbc/query.py',
        'src/gradscf/gw/pbc/product_basis.py', 'src/gradscf/gw/screened.py',
        'src/gradscf/gw/self_energy.py', 'src/gradscf/pbc/scf.py',
        'src/gradscf/pbc/_kpoint.py', 'src/gradscf/pbc/bands.py',
        'src/gradscf/solvers/nonlinear/scalar.py']
    report.update(stage='reference', system=material, scf_converged=bool(mf.converged),
        scf_energy_hartree=float(mf.e_tot), scf_seconds=scf_seconds,
        checkpoint=None if checkpoint is None else str(checkpoint),
        checkpoint_sha256=None if checkpoint is None else hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        xc='pbe', basis='gth-szv', pseudo='gth-pbe',
        source_kmesh=[2, 2, 1] if material == 'graphene' else [2, 1, 1],
        mesh=list(mf.cell.mesh), electrostatic_dimension=mf.cell.dimension,
        exxdiv=mf.exxdiv, temperature_hartree=mf.temperature,
        lattice_bohr=np.asarray(mf.cell.lattice).tolist(),
        coordinates_bohr=np.asarray(mf.cell.coords).tolist(),
        nocc=nocc, nmo=nmo, output_bands=list(selected),
        query_fractional=fractional.tolist(), path_fractional=path_fractional.tolist(),
        path=path_name if path_points else None,
        point_count=len(kpoints), tick_labels=labels, tick_positions=np.asarray(ticks).tolist(),
        energy_zero_hartree=energy_zero, completed_point_indices=[],
        full_internal_bands=list(range(nmo)),
        full_screening_occupied=list(range(nocc)), full_screening_virtual=list(range(nocc, nmo)),
        lehmann_channels='OV+VO', screening_rank=2*len(mf.kpts)*nocc*(nmo-nocc),
        nw=16, eta_hartree=1e-3, fc=False, solver='direct', maxiter=80,
        pole_subtraction=True, imaginary_green_eta_hartree=0.,
        failed_scalar_retry='AD hybrid from original starting energy',
        scalar_retry_maxiter=120,
        root_tolerance_hartree=1e-9, degenerate_subspace_method='projected nonlinear Dyson equation',
        boundary='2D finite-part slab' if mf.cell.dimension == 2 else '3D periodic body only',
        band_touching=material == 'graphene', gap_tolerance_hartree=1e-5 if material == 'graphene' else 1e-7,
        max_query_array_bytes=2*1024**3, query_chunk_size=1,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        scope='Native compact finite model; direct QP points; no material convergence or GW AD claim')
    np.savez_compressed(folder/'scf_reference.npz',
        mo_energy_spin=np.asarray(mf.result.mo_energy_spin),
        mo_coeff_spin=np.asarray(mf.result.mo_coeff_spin),
        mo_occ_spin=np.asarray(mf.result.mo_occ_spin),
        density_spin=np.asarray(mf.result.density_spin),
        source_kpoints=np.asarray(mf.kpts), query_ks_energy=ks_energy,
        query_ks_coeff=ks_coeff, query_kpoints=kpoints,
        fractional_kpoints=fractional, path_fractional_kpoints=path_fractional,
        path_distance=distance, tick_positions=ticks)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    qp, sigma = ks_energy.copy(), np.zeros(ks_energy.shape, dtype=complex)
    residual = np.zeros_like(qp)
    computed, converged = np.zeros_like(qp, dtype=bool), np.zeros_like(qp, dtype=bool)
    if resume_from is not None:
        previous = Path(resume_from)
        old = json.loads((previous/'native.json').read_text())
        if not old.get('pole_subtraction', False):
            raise ValueError('Cannot resume a different contour protocol')
        controls = ('system', 'xc', 'basis', 'pseudo', 'source_kmesh', 'mesh',
            'electrostatic_dimension', 'exxdiv', 'temperature_hartree', 'output_bands',
            'nw', 'eta_hartree', 'fc', 'gap_tolerance_hartree')
        if any(old[key] != report[key] for key in controls):
            raise ValueError('The restart model or output window differs')
        for source, digest in old['source_sha256'].items():
            if source.startswith('src/') and hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise ValueError(f'The restart physical source changed: {source}')
        with np.load(previous/'bands.npz') as stored:
            np.testing.assert_allclose(stored['kpoints'], kpoints, atol=1e-12, rtol=0)
            np.testing.assert_allclose(stored['ks_energy'], ks_energy, atol=1e-9, rtol=0)
            qp[:], sigma[:], residual[:] = stored['qp_energy'], stored['sigma_c'], stored['qp_residual']
            computed[:], converged[:] = stored['computed_mask'], stored['converged_mask']
        report.update(resumed_from=str(previous),
            resumed_bands_sha256=hashlib.sha256((previous/'bands.npz').read_bytes()).hexdigest(),
            completed_point_indices=list(old['completed_point_indices']))
    def save_bands():
        np.savez_compressed(folder/'bands.npz', ks_energy=ks_energy, qp_energy=qp,
            sigma_c=sigma, qp_residual=residual, computed_mask=computed,
            converged_mask=converged, fractional_kpoints=fractional,
            path_fractional_kpoints=path_fractional, kpoints=kpoints,
            path_distance=distance, tick_positions=ticks, energy_zero_hartree=energy_zero,
            source_kpoints=np.asarray(mf.kpts), screening_energy=np.asarray(mf.mo_energy))
    save_bands()
    if not calculate_gw:
        report.update(complete=True, passed=True, elapsed_seconds=perf_counter()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        destination.write_text(json.dumps(report, indent=2)+'\n')
        return mf
    began = perf_counter()
    for index, point in enumerate(kpoints):
        if index in report['completed_point_indices']:
            continue
        report.update(stage='query', current_query=index)
        destination.write_text(json.dumps(report, indent=2)+'\n')
        try:
            result = g0w0_bands(mf, point[None], orbs=selected, nw=16, eta=1e-3,
                fc=False, max_memory_bytes=2*1024**3,
                pole_subtraction=True,
                band_touching=material == 'graphene', gap_tol=1e-5 if material == 'graphene' else 1e-7,
                linear_config=LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13),
                root_config=ScalarRootConfig(method='secant', ftol=1e-9, xtol=1e-9, maxiter=80))
        except (ValueError, NotImplementedError, RuntimeError) as error:
            report.update(stage='query_rejected', error_type=type(error).__name__,
                          error=str(error), elapsed_seconds=perf_counter()-started)
            destination.write_text(json.dumps(report, indent=2)+'\n')
            raise
        qp[index], sigma[index], residual[index] = (np.asarray(result.mo_energy)[0],
            np.asarray(result.sigma_qp)[0], np.asarray(result.qp_residual)[0])
        computed[index], converged[index] = (np.asarray(result.qp_computed_mask)[0],
                                           np.asarray(result.converged_mask)[0])
        report['completed_point_indices'].append(index)
        report['method'] = result.method
        save_bands()
        # Each eager query builds frequency/root closures. Release their JIT
        # executables before the next point in a long CPU demonstration.
        del result
        jax.clear_caches()
        destination.write_text(json.dumps(report, indent=2)+'\n')
    accepted = computed & converged
    report.update(complete=True, passed=bool(accepted.sum() == len(kpoints)*len(selected)),
        stage='complete', computed_count=int(computed.sum()), accepted_count=int(accepted.sum()),
        max_computed_residual_hartree=float(np.max(abs(residual[computed]))),
        gw_seconds=perf_counter()-began, elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        backend=jax.default_backend(), dtype='float64/complex128', hostname=platform.node())
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(material, 'accepted QPs', report['accepted_count'], '/', report['computed_count'],
          'GW seconds', report['gw_seconds'], flush=True)
    from gradscf_tools.periodic_gw_plot import plot_material
    plot_material(folder)
    return report


if __name__ == '__main__':
    run()
