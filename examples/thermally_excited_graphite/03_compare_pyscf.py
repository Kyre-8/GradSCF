"""Independent FFT/PBE reference at the same geometry and discretization.

Run 00, 01 and 02 first. PySCF solves its own SCF at every displaced geometry;
neither GradSCF orbitals nor fitted coefficients initialize the reference.
An energy match at one coarse grid is not a material convergence result.
"""
import json
import time
import numpy as np
from pyscf.pbc import gto, dft, df, scf
from pyscf.pbc.dft.gen_grid import UniformGrids
from settings import OUTPUT, BASIS, FFT_MESH, KB_HARTREE, optical_directions, calculation_settings

with np.load(OUTPUT/'reference.npz') as archive:
    reference = dict(archive)
with np.load(OUTPUT/'frozen_mode.npz') as archive:
    scan = dict(archive)
with np.load(OUTPUT/'gamma_300K.npz') as archive:
    gamma = dict(archive)
assert json.loads(str(reference['settings'])) == calculation_settings()
assert str(reference['settings']) == str(scan['settings']), 'Do not mix electronic structure settings.'
assert str(gamma['settings']) == str(scan['settings'])

direction = optical_directions()[0]
results = []
start = time.perf_counter()
step = .0004
for amplitude in (-step, 0., step):
    cell = gto.Cell()
    cell.atom = [('C', r) for r in reference['coordinates']+amplitude*direction]
    cell.a = reference['lattice']
    cell.unit = 'Bohr'
    cell.basis = BASIS
    cell.pseudo = 'gth-pbe'
    cell.cart = True
    cell.mesh = FFT_MESH
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()

    # Match the actual unwrapped k representatives, not only the mesh dimensions.
    # Native S/T use a finite reciprocal box; reciprocal shifts are not an exact
    # finite-discretization equivalence. PySCF still uses its own AO integrals.
    kpoints = np.asarray(reference['kpoints'])
    mf = dft.KRKS(cell, kpoints, xc='pbe', exxdiv=None)
    mf.with_df = df.FFTDF(cell, kpoints)
    mf.grids = UniformGrids(cell)
    mf.grids.mesh = FFT_MESH
    mf = scf.addons.smearing_(mf, sigma=300.*KB_HARTREE, method='fermi')
    mf.conv_tol = 1e-12
    mf.max_cycle = 180
    mf.kernel()
    assert mf.converged
    results.append({'displacement_bohr': amplitude, 'energy_Eh_cell': mf.e_tot,
                    'free_energy_Eh_cell': mf.e_free})
    print('PySCF displacement / free energy:', amplitude, mf.e_free, flush=True)

curvature = (results[0]['free_energy_Eh_cell']-2*results[1]['free_energy_Eh_cell']
             +results[2]['free_energy_Eh_cell'])/step**2
native_fit = float(scan['derivatives4'][2])
center = int(np.argmin(abs(scan['displacements'])))
plus = int(np.argmin(abs(scan['displacements']-step)))
minus = int(np.argmin(abs(scan['displacements']+step)))
assert np.isclose(scan['displacements'][plus], step)
native_difference = float((scan['energies'][plus]-2*scan['energies'][center]
                           +scan['energies'][minus])/step**2)
directions = gamma['directions'].reshape(len(gamma['directions']), -1)
np.testing.assert_allclose(directions @ directions.T, np.eye(len(directions)), atol=1e-12)
components = directions @ scan['direction'].ravel()
native_dfpt = float((components @ gamma['force_constants'] @ components).real)
np.testing.assert_allclose(native_dfpt, scan['derivatives6'][2], rtol=5e-4, atol=1e-6)
print('PySCF finite-displacement curvature (Eh/Bohr^2):', curvature)
print('GradSCF fitted curvature (Eh/Bohr^2):', native_fit)
print('GradSCF matched-step curvature (Eh/Bohr^2):', native_difference)
print('GradSCF AD/DFPT curvature (Eh/Bohr^2):', native_dfpt)
(OUTPUT/'pyscf_reference.json').write_text(json.dumps({
    'calculations': results, 'curvature_Eh_Bohr2': curvature,
    'gradscf_fit_curvature_Eh_Bohr2': native_fit,
    'gradscf_matched_step_curvature_Eh_Bohr2': native_difference,
    'gradscf_dfpt_curvature_Eh_Bohr2': native_dfpt,
    'finite_difference_step_bohr': step,
    'curvature_difference_Eh_Bohr2': native_fit-curvature,
    'seconds': time.perf_counter()-start,
    'scope': 'independent finite displacement; compare discretization before physical claims',
    'settings': calculation_settings(),
}, indent=2)+'\n')

# CPU float64, identical unwrapped k points, finite-difference step 0.0004 Bohr:
# PySCF F(0) = -22.478401245399947 Eh/cell
# PySCF / GradSCF finite-step curvatures = 0.907564223773 / 0.939971855907 Eh/Bohr^2
# GradSCF AD/DFPT curvature = 0.942508323908 Eh/Bohr^2
# The coarse-grid cross-code error is material; see 06_grid_convergence.py.
