"""Independent PySCF SCF and bands for the native material cases.

Only the physical inputs and explicit k path are shared. No native density,
orbitals, Fock matrix or band energies are used to construct this reference.
"""
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import pyscf
from pyscf.pbc import dft, gto
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.scf import addons


OUTPUT = Path('reproducibility/ad_showcase/PBC_material_bands')


def compare(case='silicon_pbe_pilot'):
    started = perf_counter()
    folder = OUTPUT/case
    path = folder/'native.json'
    native = json.loads(path.read_text())
    assert native['complete'] and native['passed']
    options = native['settings']
    pseudo = dict(zip(native['symbols'], native['pseudopotential_data']))
    cell = gto.M(atom=list(zip(native['symbols'], native['coordinates_angstrom'])),
        a=native['lattice_angstrom'], unit='Angstrom', basis=native['basis_data'],
        pseudo=pseudo, cart=True, mesh=options['mesh'], precision=1e-10, verbose=0,
        dimension=native.get('electrostatic_dimension', 3))
    assert cell.nelectron == native['nelectron'] and cell.nao == native['nao']
    reference = dft.KRKS(cell, kpts=cell.make_kpts(options['kmesh']),
                         exxdiv=native.get('exxdiv', 'ewald'))
    reference.max_memory = 2000
    reference.xc = native['xc']
    reference.grids = UniformGrids(cell)
    reference.grids.mesh = reference.with_df.mesh = np.asarray(options['mesh'])
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-11, 1e-9, 150
    if options['temperature'] is not None:
        reference = addons.smearing_(reference, sigma=options['temperature'], method='fermi')
    reference.kernel()
    assert reference.converged
    fractional = np.asarray(native['fractional_kpoints'])
    energies, _ = reference.get_bands(cell.get_abs_kpts(fractional))
    energies = np.asarray(energies)
    with np.load(folder/'bands.npz') as saved:
        actual = saved['energies_hartree']
    nocc = native['nocc']
    frontier = slice(max(0, nocc-4), min(energies.shape[1], nocc+4))
    raw_error = float(np.max(abs(actual[:, frontier]-energies[:, frontier])))
    energy_error = abs(float(reference.e_tot)-native['energy_hartree'])
    native_gap = float(actual[:, nocc].min()-actual[:, nocc-1].max())
    reference_gap = float(energies[:, nocc].min()-energies[:, nocc-1].max())
    gates = dict(energy_hartree_per_atom=1e-5, frontier_bands_hartree=1e-5)
    passed = energy_error/len(native['symbols']) < gates['energy_hartree_per_atom']
    passed &= raw_error < gates['frontier_bands_hartree']
    report = dict(complete=True, passed=bool(passed), energy_hartree=float(reference.e_tot),
        total_energy_error_hartree=energy_error, frontier_band_error_hartree=raw_error,
        all_band_error_hartree=float(np.max(abs(actual-energies))),
        path_gap_error_hartree=abs(native_gap-reference_gap), gates=gates,
        frontier_indices=list(range(frontier.start, frontier.stop)),
        native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pyscf=pyscf.__version__, elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Independent same-input finite-grid SCF/bands; not material discretization convergence.')
    np.savez_compressed(folder/'pyscf_bands.npz', energies_hartree=energies,
                        scf_energies_hartree=np.asarray(reference.mo_energy))
    (folder/'pyscf.json').write_text(json.dumps(report, indent=2)+'\n')
    print(case, report, flush=True)
    assert passed, report
    return report


if __name__ == '__main__':
    compare()
