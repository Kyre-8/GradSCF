"""Finite-model H2 wire/layer: LDA -> G0W0 -> static singlet TDA-BSE.

The matching reduced-dimensional Coulomb kernel is used throughout. Its
finite-part zero-q gauge is retained in bare exchange and the BSE direct
term; this is not a q-cell integration or a 3D finite-size correction.
The output is line/sheet polarizability, not an absorption coefficient.
These are forward-only functional examples, with one setting per system.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import dft, gto
from gradscf.pbc.optics import HARTREE_TO_EV
from gradscf.gw.pbc import g0w0_bands
from gradscf.gw.pbc.interaction import finite_head
from gradscf.bse import pbc as bse
from gradscf.solvers.nonlinear import ScalarRootConfig
from gradscf_tools import periodic_bse_reference as reference

OUTPUT = Path('reproducibility/ad_showcase/reduced_bse_body')


def mean_field(dimension):
    # Same geometry, Gaussian basis and FFT mesh as pbc/reduced_dimensions.py.
    if dimension == 1:
        lattice = np.diag([5., 16., 16.])
        positions = [[.4, 8., 8.], [1.8, 8., 8.]]
        mesh = (25, 81, 81)
    elif dimension == 2:
        lattice = np.diag([5., 6., 18.])
        positions = [[.4, .5, 8.7], [1.8, .7, 9.2]]
        mesh = (25, 31, 91)
    else:
        raise ValueError('Choose the 1D wire or 2D layer.')
    cell = gto.M(atom=list(zip(['H', 'H'], positions)), a=lattice,
        unit='Bohr', dimension=dimension, basis='gth-szv', pseudo='gth-pade',
        mesh=mesh, cart=True, precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='svwn', exxdiv=None,
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9).run()
    assert mf.converged
    return mf


def run(dimension=1, output=OUTPUT):
    started = perf_counter()
    output = Path(output)/f'dimension_{dimension}'
    output.mkdir(parents=True, exist_ok=True)
    mf = mean_field(dimension)
    cell = mf.cell
    print(f'{dimension}D SCF converged: {float(mf.e_tot):.12f} Ha', flush=True)
    coefficients = mf.result.mo_coeff_spin[0]
    ks_energies = mf.result.mo_energy_spin[0]
    nocc = cell.nelec[0]

    # Query every source k point and band. No uncomputed QP level enters BSE.
    gw = g0w0_bands(mf, mf.kpts, tuple(range(ks_energies.shape[1])),
        nw=16, eta=1e-3, fc=False, root_config=ScalarRootConfig(ftol=1e-9, xtol=1e-9))
    assert bool(gw.converged)
    assert np.all(np.asarray(gw.qp_computed_mask))
    assert np.all(np.asarray(gw.converged_mask))
    assert np.max(abs(np.asarray(gw.qp_residual))) < 2e-8
    print(f'{dimension}D G0W0: all requested roots converged', flush=True)
    matrix = bse.build_tda_matrix(mf.inputs, coefficients, gw.screening_energy,
        gw.mo_energy, nocc, mesh=cell.mesh, max_transitions=64)
    excitons = bse.solve_tda(matrix, nstates=len(matrix), tol=1e-10)
    assert np.all(np.asarray(excitons.converged))
    assert np.all(np.asarray(excitons.energies) > 0.)

    # The NumPy reference shares SCF/Coulomb inputs, but independently forms
    # Fourier pairs, the signed-v Lehmann/Dyson response and the BSE matrix.
    # It solves only a small transition matrix, never a full G-space matrix.
    oracle = reference.matrix(mf.inputs, coefficients, gw.screening_energy,
                              gw.mo_energy, nocc, mesh=cell.mesh)
    np.testing.assert_allclose(matrix, oracle, atol=2e-9, rtol=2e-9)

    # Both optical models use identical QP gaps and fixed KS length vertices.
    vertices = bse.ks_dipoles(mf)
    ip = bse.independent_particles(gw.mo_energy, nocc, vertices)
    moments = bse.transition_dipoles(vertices, excitons.vectors)
    frequency_ev = np.linspace(0., float(max(np.max(ip.energies),
        np.max(excitons.energies)))*HARTREE_TO_EV+2., 1601)
    width_ev = .10  # Imposed eta, not a calculated lifetime.
    optical = dict(eta=width_ev/HARTREE_TO_EV, lattice=cell.lattice, dimension=dimension)
    frequency = frequency_ev/HARTREE_TO_EV
    ip_alpha = bse.polarizability(ip.energies, ip.dipoles, frequency, **optical)
    bse_alpha = bse.polarizability(excitons.energies, moments, frequency, **optical)
    alpha_oracle = reference.polarizability(excitons.energies, moments, frequency, **optical)
    np.testing.assert_allclose(bse_alpha, alpha_oracle, atol=2e-11, rtol=2e-11)

    np.savez(output/'arrays.npz', lattice=cell.lattice, coordinates=cell.coords,
        kpoints=mf.kpts, coefficients=coefficients, ks_energies=ks_energies,
        screening_energies=gw.screening_energy, qp_energies=gw.mo_energy,
        qp_residual=gw.qp_residual, qp_computed_mask=gw.qp_computed_mask,
        qp_converged_mask=gw.converged_mask, kernel=matrix, reference_kernel=oracle,
        vectors=excitons.vectors, vertices=vertices, bse_residuals=excitons.residual_norms,
        ip_energies=ip.energies, ip_dipoles=ip.dipoles,
        bse_energies=excitons.energies, bse_dipoles=moments,
        frequency_ev=frequency_ev, ip_alpha=ip_alpha, bse_alpha=bse_alpha)
    sources = [__file__, 'src/gradscf/gw/pbc/query.py',
        'src/gradscf/gw/pbc/interaction.py', 'src/gradscf/gw/pbc/product_basis.py',
        'src/gradscf/gw/screened.py', 'src/gradscf/bse/pbc/kernel.py',
        'src/gradscf/bse/pbc/properties.py', 'src/gradscf/bse/pbc/response.py',
        'src/gradscf/pbc/optics.py', 'src/gradscf_tools/periodic_bse_reference.py']
    report = dict(complete=True, dimension=dimension, xc='svwn', basis='gth-szv',
        pseudo='gth-pade', exxdiv=None, mesh=list(cell.mesh), kmesh=[2, 1, 1],
        lattice_bohr=np.asarray(cell.lattice).tolist(), coordinates_bohr=np.asarray(cell.coords).tolist(),
        scf_energy_hartree=float(mf.e_tot), raw_finite_head_hartree=float(finite_head(mf.inputs)),
        gw_method=gw.method, gw_nw=16, gw_eta_hartree=1e-3, gw_fc=False,
        qp_energies_hartree=np.asarray(gw.mo_energy).tolist(),
        qp_max_residual_hartree=float(np.max(abs(np.asarray(gw.qp_residual)))),
        bse_max_residual_hartree=float(np.max(excitons.residual_norms)),
        raw_hermiticity_error=float(np.max(abs(np.asarray(matrix)-np.asarray(matrix).conj().T))),
        independent_matrix_error_hartree=float(np.max(abs(np.asarray(matrix)-oracle))),
        independent_alpha_error=float(np.max(abs(np.asarray(bse_alpha)-alpha_oracle))),
        ip_energies_ev=np.asarray(ip.energies*HARTREE_TO_EV).tolist(),
        bse_energies_ev=np.asarray(excitons.energies*HARTREE_TO_EV).tolist(),
        broadening_eta_ev=width_ev, optical_units=f'bohr^{3-dimension}',
        scope='Finite-part Coulomb gauge; no q-cell averaging, 3D finite-size correction, material convergence or AD.',
        reference_scope='Shared SCF/raw Coulomb boundary; independent NumPy Fourier, full Lehmann, signed-v Dyson and BSE assembly.',
        elapsed_seconds=perf_counter()-started, hostname=platform.node(), python=platform.python_version(),
        jax=jax.__version__, backend=jax.default_backend(), dtype='float64/complex128',
        source_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{dimension}D SCF energy / Ha:', report['scf_energy_hartree'])
    print('QP energies / Ha:', report['qp_energies_hartree'])
    print('IP transitions / eV:', report['ip_energies_ev'])
    print('Static TDA-BSE transitions / eV:', report['bse_energies_ev'])
    print('Independent matrix error / Ha:', report['independent_matrix_error_hartree'])
    return report


if __name__ == '__main__':
    run(1)
    run(2)
