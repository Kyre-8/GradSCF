"""Direct off-mesh G0W0 points for small 3D finite models; no CLI.

This deliberately small GTH-SZV/FFT7/three-q model demonstrates the API and
complete OV+VO contour. It is not a converged material band structure.
Every displayed QP point is computed directly; connecting lines guide the eye.
run(system='silicon') selects one three-point generic Si segment, not a
high-symmetry path, interpolation, or bulk-convergence study.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import warnings

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import gto, dft
from gradscf.gw.pbc import g0w0_bands
from gradscf.solvers.nonlinear import ScalarRootConfig
from gradscf_tools.periodic_gw_plot import plot_query

OUTPUT = Path('reproducibility/ad_showcase/periodic_gw_query')


def reference(system='hydrogen'):
    if system == 'hydrogen':
        cell = gto.M(atom='H 3 3 2.3; H 3 3 3.7', a=np.eye(3)*6., unit='Bohr',
            basis='gth-szv', pseudo='gth-pbe', mesh=(7, 7, 7), precision=1e-10)
        return dft.KRKS(cell, kpts=cell.make_kpts((3, 1, 1)), xc='pbe',
            conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150).run()
    if system == 'silicon':
        from examples.pbc.band_structures.systems import structure
        atoms, _, _ = structure('silicon')
        cell = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
            a=atoms.cell.array, unit='Angstrom', basis='gth-szv', pseudo='gth-pbe',
            mesh=(21, 21, 21), cart=True, precision=1e-10)
        return dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='pbe', exxdiv='ewald',
            conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9,
            max_cycle=150, damping=.15).run()
    raise ValueError('Choose hydrogen or silicon')


def run(output=None, *, system='hydrogen'):
    output = Path(output) if output is not None else (OUTPUT if system == 'hydrogen' else OUTPUT/'silicon')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'native.json').exists():
        raise FileExistsError('Use a new output directory; preserve the existing numerical record')
    started = perf_counter()
    report = dict(complete=False, passed=False, system=system, stage='scf')
    (output/'native.json').write_text(json.dumps(report, indent=2)+'\n')
    mf = reference(system)
    report.update(scf_converged=bool(mf.converged), scf_energy_hartree=float(mf.e_tot),
                  elapsed_seconds=perf_counter()-started)
    (output/'native.json').write_text(json.dumps(report, indent=2)+'\n')
    if not mf.converged:
        raise RuntimeError('The demonstration SCF reference did not converge')
    if system == 'hydrogen':
        fractions = np.zeros((5, 3))
        fractions[:, 0] = [0., .12, .24, .36, .48]
        orbs, kmesh, model = [0, 1], [3, 1, 1], '3D periodic H2 array'
    else:
        fractions = np.linspace([.07, .11, .13], [.29, .23, .17], 3)
        orbs, kmesh, model = [3, 4], [2, 1, 1], 'Silicon generic segment'
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(mf.cell.lattice)).T
    kpts = fractions@reciprocal
    ks_energy, _ = mf.get_bands(kpts)
    report.update(stage='query', query_fractional=fractions.tolist(), output_bands=orbs)
    (output/'native.json').write_text(json.dumps(report, indent=2)+'\n')
    try:
        result = g0w0_bands(mf, kpts, orbs=orbs, nw=16, eta=1e-3, fc=False,
            root_config=ScalarRootConfig(ftol=1e-9, xtol=1e-9, maxiter=80))
    except (ValueError, NotImplementedError, RuntimeError) as error:
        report.update(stage='query_rejected', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=perf_counter()-started)
        (output/'native.json').write_text(json.dumps(report, indent=2)+'\n')
        raise
    np.savez(output/'bands.npz', fractional_kpoints=fractions, kpoints=kpts,
        source_kpoints=mf.kpts, ks_energy=ks_energy, qp_energy=result.mo_energy,
        sigma_c=result.sigma_qp, qp_residual=result.qp_residual,
        computed_mask=result.qp_computed_mask, converged_mask=result.converged_mask,
        screening_energy=result.screening_energy)
    sources = [__file__, 'src/gradscf/gw/pbc/query.py', 'src/gradscf/gw/pbc/product_basis.py',
        'src/gradscf/gw/screened.py', 'src/gradscf/gw/self_energy.py',
        'src/gradscf/solvers/nonlinear/scalar.py', 'src/gradscf_tools/periodic_gw_plot.py']
    if system == 'silicon':
        sources.append('examples/pbc/band_structures/systems.py')
    computed = np.asarray(result.qp_computed_mask)
    converged = np.asarray(result.converged_mask)
    report = dict(complete=True, passed=bool(result.converged), model=model, system=system,
        xc='pbe', basis='gth-szv', pseudo='gth-pbe', lattice_bohr=np.asarray(mf.cell.lattice).tolist(),
        coordinates_bohr=np.asarray(mf.cell.coords).tolist(), mesh=list(mf.cell.mesh), source_kmesh=kmesh,
        query_fractional=fractions.tolist(), output_bands=orbs, full_internal_bands=list(result.g_orbitals),
        full_screening_occupied=list(result.screening_occupied),
        full_screening_virtual=list(result.screening_virtual), lehmann_channels='OV+VO',
        nw=16, eta_hartree=1e-3, fc=False, exxdiv='ewald', method=result.method,
        scf_energy_hartree=float(mf.e_tot), scf_converged=bool(mf.converged),
        computed_count=int(computed.sum()), converged_computed_count=int((computed & converged).sum()),
        max_qp_residual_hartree=float(np.max(np.abs(result.qp_residual))),
        elapsed_seconds=perf_counter()-started, backend=jax.default_backend(), dtype='float64/complex128',
        jax_version=jax.__version__, python_version=platform.python_version(),
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        boundary='Canonical G-q finite FFT model, fixed SCF W0, body only, eager forward; not material convergence')
    (output/'native.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        warnings.warn('Some requested QP roots did not converge; inspect masks and residuals', RuntimeWarning)
    plot_query(output)
    print(json.dumps(report, indent=2))
    return result


if __name__ == '__main__':
    run()

# H2 measured 2026-10-16, CPU/float64: 10/10 direct QP points converged;
# max Dyson residual 6.25e-17 Ha. Independent NumPy Sigma error 2.26e-17 Ha.
# The archived H2 input/source manifest predates the optional Si declaration.
# Si fixed generic segment, 2026-10-16: 6/6 direct QP points converged,
# max residual 1.69e-14 Ha; 80.67 s body, 2,846,156 KiB process-tree peak RSS.
