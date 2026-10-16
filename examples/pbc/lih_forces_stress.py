"""P01/P02: two-atom rocksalt LiH forces and tensile-positive strain stress.

Periodic Cartesian Gaussian/GTH model, not an all-electron calculation.
Each run uses one declared basis/FFT/k setting; sensitivity axes are separate.
"""
import hashlib
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import dft, gto, scf
from gradscf.scf import SCFDifferentiationConfig


OUTPUT = Path('reproducibility/ad_showcase/P01_P02_lih_periodic')
SETTINGS = dict(a_bohr=7.6, xc='pbe', basis='gth-szv', pseudo='gth-pade',
    mesh=[31, 31, 31], kmesh=[2, 2, 2], precision=1e-10,
    displacement_bohr=[[0., 0., 0.], [.025, -.015, .01]])


def make_mean_field(settings, *, coordinates=None, lattice=None):
    """Public cell/SCF facades with fixed-N, zero-temperature occupations."""
    if coordinates is None and lattice is None:
        lattice = .5*settings['a_bohr']*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        coordinates = np.array([[0., 0., 0.], [.5, .5, .5]])@lattice
        coordinates += np.asarray(settings.get('displacement_bohr', np.zeros((2, 3))))
    elif coordinates is None or lattice is None:
        raise ValueError('Supply both Cartesian coordinates and row lattice, or neither.')
    cell = gto.M(atom=list(zip(['Li', 'H'], np.asarray(coordinates))), a=np.asarray(lattice),
                 unit='Bohr', basis=settings['basis'], pseudo=settings['pseudo'],
                 mesh=tuple(settings['mesh']), precision=settings['precision'], cart=True, spin=0)
    kpts = cell.make_kpts(tuple(settings['kmesh']))
    controls = dict(kpts=kpts, exxdiv='ewald', temperature=None,
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150, damping=.1,
        differentiation=SCFDifferentiationConfig(mode='implicit', tolerance=1e-11, max_iter=100, restart=40))
    return scf.KRHF(cell, **controls) if settings['xc'] == 'hf' else dft.KRKS(cell, xc=settings['xc'], **controls)


def diagnostics(mean_field):
    """Record the actual PP electron count and the selected sampled branch."""
    cell, state = mean_field.cell, mean_field.result
    coefficients, energies = np.asarray(state.mo_coeff_spin[0]), np.asarray(state.mo_energy_spin[0])
    density, overlap = np.asarray(mean_field.make_rdm1()), np.asarray(mean_field.inputs.overlap)
    weights = np.asarray(mean_field.inputs.kweights)
    nocc = cell.nelectron//2
    gap = float(np.min(energies[:, nocc])-np.max(energies[:, nocc-1]))
    occupied, virtual = coefficients[:, :, :nocc], coefficients[:, :, nocc:]
    fov = np.einsum('kpi,kpq,kqa->kia', occupied.conj(), np.asarray(state.fock_spin[0]), virtual)
    number = float(np.einsum('k,kij,kji->', weights, density, overlap).real)
    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    if not mean_field.converged or gap <= 1e-3:
        raise RuntimeError('A resolved, positively gapped sampled closed-shell reference is required.')
    if overlap_eigenvalues.min() <= 1e-8*max(1., overlap_eigenvalues.max()):
        raise RuntimeError('The selected Gaussian basis is too close to linear dependence.')
    np.testing.assert_allclose(number, cell.nelectron, atol=1e-9, rtol=0)
    assert np.linalg.norm(fov) < 2e-8
    return dict(explicit_nelectron=cell.nelectron, pseudopotential_valence_charges=np.asarray(cell.charges).tolist(),
        electron_number=number, nao=coefficients.shape[1], occupied_orbitals=nocc,
        sampled_indirect_gap_hartree=gap, mo_fock_occupied_virtual_norm=float(np.linalg.norm(fov)),
        minimum_overlap_eigenvalue=float(overlap_eigenvalues.min()),
        occupation_values=np.unique(np.asarray(mean_field.mo_occ)).tolist(),
        scf_converged=bool(mean_field.converged), scf_cycles=int(mean_field.cycles))


def run_case(name='pbe_szv_k2_fft31', settings=None):
    """One independent, bounded case; settings are ordinary Python data."""
    started = perf_counter()
    settings = dict(SETTINGS if settings is None else settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT/(name+'.json')
    destination.write_text('{"complete": false}\n')
    mean_field = make_mean_field(settings).run()
    forward_seconds = perf_counter()-started
    state = diagnostics(mean_field)
    driver = mean_field.nuc_grad_method()
    began = perf_counter()
    gradient = np.asarray(driver.kernel())
    stress = np.asarray(driver.stress())
    derivative_seconds = perf_counter()-began
    assert np.isfinite(gradient).all() and np.isfinite(stress).all()
    np.testing.assert_allclose(driver.result.energy, mean_field.e_free, atol=2e-10, rtol=0)
    np.testing.assert_allclose(mean_field.e_free, mean_field.e_tot, atol=1e-12, rtol=0)
    source = hashlib.sha256()
    for path in sorted(Path('src').rglob('*.py')):
        source.update(path.as_posix().encode()); source.update(path.read_bytes())
    report = dict(complete=True, settings=settings, **state,
        energy_hartree=float(driver.result.energy), forces_hartree_per_bohr=(-gradient).tolist(),
        stress_hartree_per_bohr3=stress.tolist(), pressure_hartree_per_bohr3=float(-np.trace(stress)/3),
        raw_net_force_hartree_per_bohr=(-gradient.sum(axis=0)).tolist(),
        coordinates_bohr=np.asarray(mean_field.cell.coords).tolist(), lattice_bohr=np.asarray(mean_field.cell.lattice).tolist(),
        volume_bohr3=float(mean_field.cell.volume),
        kpoints_inverse_bohr=np.asarray(mean_field.kpts).tolist(),
        fractional_kpoints=(np.asarray(mean_field.kpts)@np.asarray(mean_field.cell.lattice).T/(2*np.pi)).tolist(),
        basis_data=mean_field.cell.basis_spec, pseudopotential_data=mean_field.cell.pseudopotentials,
        stress_convention='Tensile-positive symmetric strain at fixed fractional atoms/kpoints; pressure=-trace(stress)/3.',
        exchange_convention='HF uses full-k Ewald exchange divergence correction; pure PBE has no exact exchange.',
        model='GTH-Pade pseudopotentials used for both PBE and HF; T=0 fixed integer occupations, full k mesh.',
        forward_seconds=forward_seconds, force_stress_seconds_includes_reconvergence_and_ad=derivative_seconds,
        elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_python_sha256=source.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype='float64')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(OUTPUT/(name+'_state.npz'), density=np.asarray(mean_field.make_rdm1()),
                        mo_coeff=np.asarray(mean_field.mo_coeff), mo_energy=np.asarray(mean_field.mo_energy),
                        overlap=np.asarray(mean_field.inputs.overlap))
    print(name, 'native force/stress complete:', report['energy_hartree'],
          'gap', report['sampled_indirect_gap_hartree'], 'seconds', report['elapsed_seconds'], flush=True)
    return report


if __name__ == '__main__':
    run_case()

# Recorded on jy_gpu_1, CPU float64 (2026-10-16); see P01_P02_lih_periodic.
# PBE/GTH-Pade/SZV, k2^3/FFT31^3, displaced H, a=7.6 bohr:
# E = -8.007235930221265 Eh; sampled indirect gap = 0.150136911344 Eh.
# Four explicit electrons (Li q3/H q1); raw forces and tensile stress retained.
# This finite-basis/k/grid model is not a converged material prediction.
