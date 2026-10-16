"""Native periodic KS bands for controlled material/functional comparisons.

Start with one ordinary run call in a bounded fresh process. No CLI.
PBE0 uses spherical exchange quadrature for smooth off-mesh bands; pure
LDA/PBE/r2SCAN retain their usual periodic Coulomb treatment.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.pbc import gto, dft
from gradscf.scf import SCFDifferentiationConfig
from gradscf.tools.spectra import HARTREE_TO_EV

try:
    from .systems import structure, settings
except ImportError:
    from systems import structure, settings


OUTPUT = Path('reproducibility/ad_showcase/PBC_material_bands')
FUNCTIONALS = {'lda': 'svwn', 'pbe': 'pbe', 'r2scan': 'r2scan', 'pbe0': 'pbe0'}


def run(material='silicon', method='pbe', *, case=None, overrides=None):
    if method not in FUNCTIONALS:
        raise ValueError('Choose lda, pbe, r2scan or pbe0.')
    options = dict(settings(material), **({} if overrides is None else overrides))
    label = case or f'{material}_{method}_pilot'
    folder = OUTPUT/label
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/'native.json'
    if destination.exists():
        raise FileExistsError('Use a new case name; preserve the earlier numerical record.')
    started = perf_counter()
    report = dict(complete=False, passed=False, material=material, method=method,
                  settings=options, discretization_converged=False)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    atoms, path_name, dimension = structure(material, vacuum=options['vacuum'])
    path = atoms.cell.bandpath(path_name, npoints=options['path_points'], pbc=atoms.pbc)
    distance, ticks, labels = path.get_linear_kpoint_axis()
    periodic_dimension = options.get('dimension', 3)
    if periodic_dimension != 3 and periodic_dimension != dimension:
        raise ValueError('Reduced periodicity must match the declared material structure.')
    if method == 'pbe0' and periodic_dimension != 3:
        raise NotImplementedError('This hybrid band example currently uses a 3D cell.')
    exxdiv = 'vcut_sph' if method == 'pbe0' else 'ewald'
    cell = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
        a=atoms.cell.array, unit='Angstrom', basis=options['basis'], pseudo=options['pseudo'],
        mesh=options['mesh'], cart=True, precision=1e-10, dimension=periodic_dimension)
    reference = dft.KRKS(cell, xc=FUNCTIONALS[method], kpts=cell.make_kpts(options['kmesh']),
        temperature=options['temperature'], exxdiv=exxdiv, max_cycle=150, damping=.15,
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9,
        differentiation=SCFDifferentiationConfig(mode='implicit', tolerance=1e-10)).run()
    jax.block_until_ready(reference.result)
    scf_seconds = perf_counter()-started
    report.update(stage='scf', scf_converged=bool(reference.converged),
                  energy_hartree=float(reference.e_tot), scf_seconds=scf_seconds)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    assert reference.converged, report
    saved_reference = reference.result
    absolute_k = jnp.asarray(path.kpts)@cell.reciprocal_vectors()
    before = perf_counter()
    bands, coefficients = reference.get_bands(absolute_k, chunk_size=options['chunk_size'])
    jax.block_until_ready((bands, coefficients))
    band_seconds = perf_counter()-before
    assert reference.result is saved_reference
    on_mesh, _ = reference.get_bands(reference.kpts, chunk_size=options['chunk_size'])
    on_mesh_error = float(jnp.max(abs(on_mesh-reference.mo_energy)))
    assert on_mesh_error < 2e-7 and np.isfinite(bands).all()
    nocc = int(cell.nelectron//2)
    mesh_energy = np.asarray(reference.mo_energy)
    energies = np.asarray(bands)
    vbm = max(float(energies[:, nocc-1].max()), float(mesh_energy[:, nocc-1].max()))
    if dimension == 2 and reference.result.chemical_potential is not None:
        chemical = np.asarray(reference.result.chemical_potential)
        zero, alignment = float(chemical.mean()), 'SCF chemical potential at the declared electronic temperature'
    else:
        zero, alignment = vbm, 'Maximum occupied band on the SCF mesh and displayed path'
    path_gap = float(energies[:, nocc].min()-energies[:, nocc-1].max())
    mesh_gap = float(mesh_energy[:, nocc].min()-mesh_energy[:, nocc-1].max())
    files = [Path(__file__), Path(__file__).with_name('systems.py')]
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')}
    np.savez_compressed(folder/'bands.npz', energies_hartree=energies,
        fractional_kpoints=path.kpts, absolute_kpoints=np.asarray(absolute_k),
        path_distance=distance, tick_positions=ticks, energy_zero_hartree=zero,
        scf_energies_hartree=mesh_energy, scf_fractional_kpoints=np.asarray(reference.kpts)
        @np.asarray(cell.lattice).T/(2*np.pi))
    report.update(complete=True, passed=True, stage='bands',
        symbols=atoms.get_chemical_symbols(), coordinates_angstrom=atoms.positions.tolist(),
        lattice_angstrom=atoms.cell.array.tolist(), physical_dimension=dimension,
        electrostatic_dimension=periodic_dimension,
        electrostatic_boundary='2D slab Coulomb cutoff' if periodic_dimension==2 else '3D periodic',
        isolated_low_dimensional_result=False,
        low_dimensional_note=(('Coulomb-truncated slab; vacuum/mesh convergence not established.'
            if periodic_dimension==2 else 'Repeated-slab pilot only; no Coulomb truncation or vacuum convergence.')
            if dimension < 3 else None),
        xc=reference.xc, exxdiv=exxdiv, basis_data=cell.basis_spec,
        pseudopotential_data=cell.pseudopotentials,
        ionic_valence_charges=np.asarray(cell.charges).tolist(), nelectron=cell.nelectron,
        nao=cell.topology.nao, nocc=nocc, path=path_name, tick_labels=labels,
        fractional_kpoints=path.kpts.tolist(), tick_positions=ticks.tolist(),
        energy_alignment=alignment, energy_zero_hartree=zero,
        path_gap_ev=path_gap*HARTREE_TO_EV, scf_mesh_gap_ev=mesh_gap*HARTREE_TO_EV,
        gap_scope='Sampled gaps only; a path minimum is not a converged Brillouin-zone gap.',
        spin_orbit=False, on_mesh_error_hartree=on_mesh_error,
        band_seconds=band_seconds, elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), jax=jax.__version__, backend=jax.default_backend(), dtype='float64',
        source_sha256=source, script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(material, method, 'sampled path gap / eV', report['path_gap_ev'],
          'on-mesh error / Eh', on_mesh_error, 'seconds', report['elapsed_seconds'], flush=True)
    return report


if __name__ == '__main__':
    run()

# Measured compact Si comparison on jy_gpu_1, CPU float64, 2026-10-16:
# overrides={'kmesh': (2, 1, 1), 'mesh': (21, 21, 21), 'path_points': 61}
# PBE  sampled path gap / eV: 0.30048834344726105
# PBE0 sampled path gap / eV: 1.9364575769785288
# Independent PySCF maximum frontier-band errors / Eh:
# PBE: 3.06e-9; PBE0: 3.69e-9.
# These sampled finite-model gaps are not converged material predictions.
