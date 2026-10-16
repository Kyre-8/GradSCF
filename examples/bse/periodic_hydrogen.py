"""PBE -> G0W0 -> static singlet TDA-BSE for a small 3D H2 array.

This is an eager, forward-only functional example: two k points, FFT7 and
GTH-SZV. Both spectra use the same computed QP energies and KS optical
vertices. GW and BSE screening omit q->0 head/wing corrections (body only).
The output is a per-cell oscillator strength, not a material absorption
coefficient, converged binding energy, or experimental prediction. No CLI.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import gto, dft
from gradscf.pbc.optics import HARTREE_TO_EV
from gradscf.gw.pbc import KRGW
from gradscf.bse import pbc as bse

OUTPUT = Path('reproducibility/ad_showcase/periodic_bse_body')


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    cell = gto.M(atom='H 3 3 2.3; H 3 3 3.7', a=np.eye(3)*6., unit='Bohr',
        basis='gth-szv', pseudo='gth-pbe', mesh=(7, 7, 7), precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='pbe',
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150).run()
    assert mf.converged
    coefficients = mf.result.mo_coeff_spin[0]
    ks_energies = mf.result.mo_energy_spin[0]
    nocc = cell.nelec[0]

    # The public periodic GW facade uses the shared transition-screening path.
    # Request all bands at both k points; no uncomputed level is used below.
    mygw = KRGW(mf, nw=16, eta=1e-3, fc=False).run()
    assert bool(mygw.converged)
    assert np.all(np.asarray(mygw.result.qp_computed_mask))
    assert np.all(np.asarray(mygw.result.converged_mask))
    assert float(jnp.max(jnp.abs(mygw.result.qp_residual))) < 2e-8
    assert mygw.result.method == 'periodic_cd_transition'
    if mygw.result.screening_energy is None:
        raise RuntimeError('The GW result must record the spectrum used for W0.')
    qp_energies = mygw.mo_energy
    matrix = bse.build_tda_matrix(mf.inputs, coefficients, mygw.result.screening_energy,
        qp_energies, nocc, mesh=cell.mesh, max_transitions=64)
    excitons = bse.solve_tda(matrix, nstates=len(matrix), tol=1e-10)
    assert np.all(np.asarray(excitons.converged))
    assert np.all(np.asarray(excitons.energies) > 0.)

    # Both spectra retain the same PBE orbitals and complete KS length vertex.
    # All velocity projection, k weights and optical formulas live in src.
    vertices = bse.ks_dipoles(mf)
    ip = bse.independent_particles(qp_energies, nocc, vertices)
    bse_dipoles = bse.transition_dipoles(vertices, excitons.vectors)
    bse_strengths = bse.oscillator_strengths(excitons.energies, bse_dipoles)

    np.savez(output/'arrays.npz', ks_energies=ks_energies, qp_energies=qp_energies,
        screening_energies=mygw.result.screening_energy, qp_residual=mygw.result.qp_residual,
        coefficients=coefficients, kpoints=mf.kpts, lattice=cell.lattice, coordinates=cell.coords,
        kernel=matrix, vectors=excitons.vectors, vertices=vertices,
        ip_energies=ip.energies, ip_dipoles=ip.dipoles, ip_strengths=ip.strengths,
        bse_energies=excitons.energies, bse_dipoles=bse_dipoles, bse_strengths=bse_strengths,
        bse_residuals=excitons.residual_norms)
    sources = [__file__, 'src/gradscf/bse/pbc/kernel.py', 'src/gradscf/bse/pbc/response.py',
        'src/gradscf/bse/pbc/properties.py', 'src/gradscf/gw/pbc/ksigma.py',
        'src/gradscf/gw/pbc/product_basis.py', 'src/gradscf/gw/screened.py',
        'src/gradscf/gw/types.py', 'src/gradscf/pbc/optics.py']
    report = dict(complete=True, model='3D periodic H2 array', xc='pbe', basis='gth-szv',
        pseudo='gth-pbe', lattice_bohr=np.asarray(cell.lattice).tolist(),
        coordinates_bohr=np.asarray(cell.coords).tolist(), mesh=list(cell.mesh), kmesh=[2, 1, 1],
        nocc=nocc, volume_bohr3=float(cell.volume), scf_energy_hartree=float(mf.e_tot),
        gw_method=mygw.result.method, gw_nw=16, gw_eta_hartree=1e-3, gw_fc=False,
        qp_energies_hartree=np.asarray(qp_energies).tolist(),
        qp_max_residual_hartree=float(jnp.max(jnp.abs(mygw.result.qp_residual))),
        bse_max_residual_hartree=float(jnp.max(excitons.residual_norms)),
        hermiticity_error=float(jnp.max(jnp.abs(matrix-matrix.conj().T))),
        ip_energies_ev=np.asarray(ip.energies*HARTREE_TO_EV).tolist(),
        ip_strengths_per_cell=np.asarray(ip.strengths).tolist(),
        bse_energies_ev=np.asarray(excitons.energies*HARTREE_TO_EV).tolist(),
        bse_strengths_per_cell=np.asarray(bse_strengths).tolist(),
        optical_vertex='Length vertex i*v_KS/Delta_KS, including GTH nonlocal velocity',
        normalization='Euclidean x^dagger x=1; singlet sqrt(2), k factor sqrt(1/Nk)',
        bse_static_screening='Full OV+VO Lehmann response; no time-reversal reduction',
        scope='Static TDA body only; no direct q0 head/wing correction, material convergence or AD claim.',
        elapsed_seconds=perf_counter()-started, hostname=platform.node(), python=platform.python_version(),
        jax=jax.__version__, backend=jax.default_backend(), dtype='float64/complex128',
        source_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print('SCF energy / Ha:', report['scf_energy_hartree'])
    print('QP energies / Ha:\n', np.asarray(qp_energies))
    print('QP maximum residual / Ha:', report['qp_max_residual_hartree'])
    print('Independent-particle transitions / eV:', report['ip_energies_ev'])
    print('Static TDA-BSE body transitions / eV:', report['bse_energies_ev'])
    print('IP / BSE oscillator strengths per cell:', report['ip_strengths_per_cell'],
          report['bse_strengths_per_cell'])
    print('BSE maximum residual / Ha:', report['bse_max_residual_hartree'])
    return report


if __name__ == '__main__':
    run()
