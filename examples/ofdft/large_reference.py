"""Independent DFTpy optimizations for the Al WGC supercell example.

DFTpy supplies TF, vW, XC, Hartree, local pseudopotential, Ewald and LBFGS.
wgc_dftpy supplies an independent NumPy WGC adapter, not native DFTpy WGC.
Only geometry, pseudopotential identity and declared potential probes are shared.
No GradSCF energy, derivative or optimized density enters reference optimization.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import numpy as np
from ase import Atoms, units
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.ions import Ions
from dftpy.functional.pseudo import LocalPseudo
from dftpy.functional.hartree import Hartree
from dftpy.functional.kedf.tf import TF
from dftpy.functional.kedf.vw import vW
from dftpy.functional.xc.semilocal_xc import LDA
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.optimization import Optimization
from dftpy.ewald import ewald

try:
    from .wgc_dftpy import WGCReference
except ImportError:
    from wgc_dftpy import WGCReference


REPEAT = 1
OUTPUT = Path('reproducibility/ad_showcase/P08_ofdft_parameter_response/large_system_scaling')
REFERENCE_COMMIT = 'fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0'


def reference_problem(repeat, lattice, mesh, probes, checkpoint=None):
    positions = np.stack(np.meshgrid(*[np.arange(repeat)/repeat]*3, indexing='ij'), axis=-1).reshape(-1, 3)
    atoms = Atoms(['Al']*repeat**3, cell=lattice*units.Bohr, scaled_positions=positions, pbc=True)
    ions = Ions.from_ase(atoms)
    grid = DirectGrid(lattice=ions.cell, nr=mesh, full=True)
    pp = Path(__file__).with_name('data')/'Al_lda.oe01.recpot'
    pseudo = LocalPseudo(grid=grid, ions=ions, PP_list={'Al': str(pp)}, PME=False)
    hartree = Hartree()
    nuclear = ewald(precision=1e-12, ions=ions, grid=grid, PME=False).energy
    vectors = np.asarray(grid.get_reciprocal().g).reshape(3, -1).T
    nonlocal_term = WGCReference(vectors, mesh, grid.volume, ions.get_ncharges()/grid.volume)
    number = float(ions.get_ncharges())

    def solve(theta, initial=None):
        evaluations = 0
        extra = DirectField(grid=grid, griddata_3d=(probes@theta).reshape(mesh))
        def evaluate(rho, calcType=('E', 'V'), **kwargs):
            nonlocal evaluations
            evaluations += 1
            del calcType, kwargs
            value = TF(rho)+vW(rho)+hartree(rho)+LDA(rho)+pseudo(rho)+nonlocal_term(rho)
            value += FunctionalOutput(name='external-probe',
                energy=float(np.sum(rho*extra)*grid.dV), potential=extra)
            value.energy += nuclear
            return value
        guess = (DirectField(grid=grid, griddata_3d=np.full(mesh, number/grid.volume))
                 if initial is None else initial.copy())
        # An independent fixed-N energy/potential check precedes optimization.
        tangent = guess*probes[:, 0].reshape(mesh)
        tangent -= guess*(float(tangent.integral())/number)
        step = 1e-4
        finite = (evaluate(guess+step*tangent).energy-evaluate(guess-step*tangent).energy)/(2*step)
        analytic = float(np.sum(evaluate(guess).potential*tangent)*grid.dV)
        potential_fd_error = abs(finite-analytic)/repeat**3
        assert potential_fd_error < 2e-8, (finite, analytic)
        attempts = []
        for attempt in range(3):
            optimizer = Optimization(optimization_method='LBFGS', EnergyEvaluator=evaluate,
                optimization_options={'maxiter': 1500, 'econv': 1e-14*repeat**3, 'ncheck': 5})
            density = optimizer.optimize_rho(guess_rho=guess)
            result = evaluate(density)
            potential = np.asarray(result.potential)
            chemical = np.sum(density*potential)*grid.dV/number
            euler = np.sqrt(np.sum(density*(potential-chemical)**2)*grid.dV/number)
            attempts.append(dict(attempt=attempt, convergence_code=int(optimizer.converged),
                euler_rms_hartree=float(euler), energy_hartree=float(result.energy),
                density_min=float(density.min()), density_max=float(density.max()),
                objective_evaluations=evaluations, optimizer_iterations=None,
                iteration_note='DFTpy prints iterations to the run log; no public counter is stored'))
            if checkpoint is not None:
                checkpoint(dict(theta=np.asarray(theta).tolist(), **attempts[-1]))
            if optimizer.converged == 0 and euler < 5e-6:
                break
            # Reset only DFTpy's LBFGS history; preserve its own last density.
            guess = density.copy()
        diagnostics = dict(convergence_code=int(optimizer.converged),
            euler_rms_hartree=float(euler), electron_number=float(density.integral()),
            fixed_density_potential_fd_error_hartree_per_atom=float(potential_fd_error), attempts=attempts)
        assert optimizer.converged == 0 and euler < 5e-6, diagnostics
        assert abs(float(density.integral())-number)/number < 1e-11
        return density, result, diagnostics
    return solve, pseudo, nuclear


def compare(repeat=REPEAT):
    started = perf_counter()
    native_path = OUTPUT/f'al_{repeat}.json'
    native = json.loads(native_path.read_text())
    assert native['complete']
    target = OUTPUT/f'al_{repeat}_reference.json'
    progress = dict(passed=False, repeat=repeat, attempts=[], hostname=platform.node(),
        python=platform.python_version(), backend='NumPy CPU', dtype='float64',
        environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        native_source_sha256=native['source_sha256'], reference_commit=REFERENCE_COMMIT,
        reference_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).with_name('wgc_dftpy.py').read_bytes()).hexdigest(),
        scope='Independent comparison in progress; inspect preserved attempts and logs if interrupted')
    def checkpoint(attempt):
        progress['attempts'].append(attempt)
        target.write_text(json.dumps(progress, indent=2)+'\n')
    target.write_text(json.dumps(progress, indent=2)+'\n')
    arrays = np.load(OUTPUT/f'al_{repeat}.npz')
    mesh = tuple(native['mesh'])
    solve, pseudo, nuclear = reference_problem(repeat, arrays['lattice'], mesh, arrays['probes'], checkpoint)
    baseline, result, diagnostics = solve(np.zeros(2))
    rho = np.asarray(baseline).ravel()
    difference = arrays['density']-rho
    number = native['electron_number']
    density_error = float(np.linalg.norm(difference)/np.linalg.norm(rho))
    energy_error = abs(native['energy_hartree']-float(result.energy))/repeat**3
    potential_error = float(np.max(abs(np.asarray(pseudo.vreal).ravel()-arrays['potential'])))
    nuclear_error = abs(float(nuclear)-native['energy_components']['nuclear'])/repeat**3
    assert density_error < 2e-5 and energy_error < 1e-8
    assert potential_error < 1e-10 and nuclear_error < 1e-9
    direction = np.asarray(native['direction'])
    exact_jvp = arrays['density_jvp']
    rows, estimates = [], []
    for step in (3e-3, 1e-3, 3e-4):
        print('REFERENCE STEP', repeat, step, flush=True)
        plus, eplus, dplus = solve(step*direction, baseline)
        minus, eminus, dminus = solve(-step*direction, baseline)
        derivative = (np.asarray(plus).ravel()-np.asarray(minus).ravel())/(2*step)
        curvature = (float(eplus.energy)-2*float(result.energy)+float(eminus.energy))/step**2
        rows.append(dict(step_hartree=step,
            density_jvp_relative_l2_error=float(np.linalg.norm(derivative-exact_jvp)/np.linalg.norm(exact_jvp)),
            observable_directional_derivative=float(arrays['density_cotangent']@derivative),
            vjp_direction_error=float(abs(arrays['density_cotangent']@derivative-arrays['vjp']@direction)),
            energy_curvature_per_atom=curvature/repeat**3,
            hvp_direction_error_per_atom=abs(curvature-native['directional_energy_curvature'])/repeat**3,
            plus_diagnostics=dplus, minus_diagnostics=dminus))
        estimates.append(derivative)
    assert min(row['density_jvp_relative_l2_error'] for row in rows) < 3e-4, rows
    assert min(row['hvp_direction_error_per_atom'] for row in rows) < 3e-5, rows
    np.savez_compressed(OUTPUT/f'al_{repeat}_reference.npz', density=rho,
                        density_jvp_fd=np.stack(estimates))
    record = dict(passed=True, repeat=repeat, atom_count=repeat**3, mesh=list(mesh),
        energy_hartree=float(result.energy), energy_error_hartree_per_atom=energy_error,
        density_relative_l2_error=density_error, potential_max_error_hartree=potential_error,
        nuclear_error_hartree_per_atom=nuclear_error, diagnostics=diagnostics,
        response_finite_differences=rows, elapsed_seconds=perf_counter()-started,
        native_json_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        reference_commit=REFERENCE_COMMIT,
        reference_euler_tolerance_hartree=5e-6,
        reference_energy_tolerance_hartree_per_atom=1e-14,
        reference_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).with_name('wgc_dftpy.py').read_bytes()).hexdigest(),
        method='DFTpy own optimizer and functionals + independent NumPy WGC adapter; central FD response',
        scope='Same discrete OFDFT model agreement; not KEDF-versus-KS physical error',
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(), backend='NumPy CPU', dtype='float64')
    target.write_text(json.dumps(record, indent=2)+'\n')
    print('REFERENCE PASS', repeat, 'E/atom error', energy_error, 'density L2', density_error, flush=True)
    return record


if __name__ == '__main__':
    compare()
