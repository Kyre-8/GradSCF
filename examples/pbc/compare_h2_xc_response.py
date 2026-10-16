"""Independent PySCF gradient differences for the periodic XC-response example.

No GradSCF imports or native optimized orbitals enter this reference. Geometry,
basis, pseudopotential, FFT mesh and the exchange coefficient match the saved
model. Finite differences are an oracle, not the source of the reported AD.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import resource

import numpy as np
import pyscf
from pyscf.pbc import gto, dft
from pyscf.pbc.dft.gen_grid import UniformGrids


OUTPUT = Path('reproducibility/ad_showcase/P07_periodic_xc')


def compare():
    started = perf_counter()
    native_path = OUTPUT/'gradscf.json'
    native = json.loads(native_path.read_text())
    assert native['complete'] and native['passed']
    coordinates = np.asarray(native['coordinates_bohr'])
    controls = native['controls']
    evaluations = []
    checkpoint = OUTPUT/f"pyscf_fft{native['mesh'][0]}.json"
    report = dict(complete=False, passed=False,
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest())

    def save_progress(stage, **values):
        report.update(stage=stage, elapsed_seconds=perf_counter()-started, **values)
        checkpoint.write_text(json.dumps(report, indent=2)+'\n')

    save_progress('starting')

    def gradient(alpha, displacement):
        cell = gto.M(atom=list(zip(['H', 'H'], coordinates+displacement.reshape(2, 3))),
            a=native['lattice_bohr'], unit='Bohr', basis=native['basis'],
            pseudo=native['pseudo'], mesh=native['mesh'], precision=native['precision'],
            cart=True, verbose=0)
        # The one-k-point KRKS FFT gradient supports the same Gamma model;
        # PySCF's RKS gradient entry point requires a different multigrid backend.
        mf = dft.KRKS(cell, kpts=np.zeros((1, 3)), exxdiv=None)
        mf.grids = UniformGrids(cell)
        mf.grids.mesh = mf.with_df.mesh = np.asarray(native['mesh'])
        mf.xc = f'{alpha:.15g}*LDA_X'
        mf.conv_tol = controls['conv_tol']
        mf.conv_tol_grad = controls['conv_tol_grad']
        mf.max_cycle = controls['max_cycle']
        mf.kernel()
        norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
        assert mf.converged and norm < 1e-9
        value = mf.nuc_grad_method().kernel()
        evaluations.append(dict(alpha=alpha, displacement_bohr=displacement.tolist(),
            energy_hartree=float(mf.e_tot), orbital_gradient_norm=norm,
            forces_hartree_bohr=(-value).tolist()))
        return np.asarray(value).ravel()

    def hessian(alpha, step):
        matrix = np.empty((6, 6))
        for index, direction in enumerate(np.eye(6)):
            matrix[:, index] = (gradient(alpha, step*direction)-gradient(alpha, -step*direction))/(2*step)
        return matrix

    gradient(1., np.zeros(6))
    energy_error = abs(evaluations[-1]['energy_hartree']-native['total_energy_hartree'])
    save_progress('forward', energy_error_hartree=energy_error)
    assert energy_error < 2e-6
    exact = np.asarray(native['hessian_hartree_bohr2'])
    curvature_checks = []
    for step in (3e-4, 1e-4):
        matrix = hessian(1., step)
        curvature_checks.append(dict(step_bohr=step,
            hessian_error=float(np.max(abs(matrix-exact))),
            raw_antisymmetry=float(np.max(abs(matrix-matrix.T)))))
        print('Reference Hessian:', curvature_checks[-1], flush=True)
        save_progress('curvature', hessian_checks=curvature_checks)
    assert curvature_checks[-1]['hessian_error'] < 1e-5
    exact_derivative = np.asarray(native['hessian_alpha_derivative'])
    records = []
    for step in (1e-2, 3e-3, 1e-3):
        plus, minus = hessian(1.+step, 1e-4), hessian(1.-step, 1e-4)
        derivative = (plus-minus)/(2*step)
        records.append(dict(parameter_step=step, displacement_step_bohr=1e-4,
            hessian_derivative_error=float(np.max(abs(derivative-exact_derivative)))))
        print('Independent parameter response:', records[-1], flush=True)
        save_progress('parameter_response', parameter_checks=records)
    passed = records[-1]['hessian_derivative_error'] < 1e-5
    report.update(complete=True, passed=passed, energy_error_hartree=energy_error,
        hessian_checks=curvature_checks, parameter_checks=records,
        gates=dict(energy_hartree=2e-6, hessian_hartree_bohr2=1e-5,
                   hessian_alpha_derivative_hartree_bohr2=1e-5),
        evaluations=evaluations, pyscf=pyscf.__version__,
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Independent reconverged periodic LDA_X analytic-gradient differences; fixed Gamma/grid model.')
    save_progress('complete')
    (OUTPUT/'pyscf.json').write_text(json.dumps(report, indent=2)+'\n')
    assert passed, records
    return report


if __name__ == '__main__':
    compare()

# Independent PySCF 2.13.0, c20 CPU, FFT41^3, 2026-10-16:
# Energy error: 6.83e-12 Eh. Hessian errors (h=3e-4,1e-4 bohr):
# 3.95e-8, 1.00e-8 Eh/bohr^2. Parameter-Hessian errors
# (alpha step=1e-2,3e-3,1e-3): 1.63e-10, 2.86e-10, 1.26e-9 Eh/bohr^2.
# Full comparison: 1300.44 s; process peak RSS: 168,512 KiB.
# All original gates passed. FFT9 and FFT31 failures are retained separately.
