"""GaAs: four periodic GW approximations in one declared finite model.

The SZV basis and small source mesh make this a method comparison, not a
converged prediction of the bulk gap. All electrons represented by the
Ga/As GTH pseudopotentials are retained; SOC is absent. Source QP matrices
are interpolated for drawing and BZ integration. Dense plotting points do
not replace source-k or basis convergence. No command-line interface.
"""
from pathlib import Path
import json
from time import perf_counter

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import dft, gto
from gradscf.gw.pbc import (build_periodic_model, g0w0_model,
    evgw_cd_kpoints, qsgw_cd_kpoints, scgw_matsubara_kpoints)
from examples.pbc.band_structures.systems import structure


MAX_MEMORY_BYTES = 16*1024**3


def reference(*, source_kmesh=(2, 2, 2), mesh=(61, 61, 61), exxdiv=None):
    """One scalar-relativistic PBE reference, shared by every GW scheme."""
    atoms, _, _ = structure('gaas')
    cell = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
        a=atoms.cell.array, unit='Angstrom', basis='gth-szv', pseudo='gth-pbe',
        mesh=mesh, cart=True, precision=1e-10, dimension=3)
    mf = dft.KRKS(cell, kpts=cell.make_kpts(source_kmesh), xc='pbe',
        exxdiv=exxdiv, temperature=None, conv_tol=1e-11,
        conv_tol_density=1e-10, conv_tol_grad=1e-9, max_cycle=150, damping=.15).run()
    if not mf.converged:
        raise RuntimeError('The common PBE reference did not converge')
    return mf


def calculate(method, directory, *, source_kmesh=(2, 2, 2), mesh=(61, 61, 61),
              nw=32, beta=400., max_iter=60, tolerance=1e-6, exxdiv=None,
              screening_cutoff_rydberg=None, eta=.001, diis_space=8,
              time_oversampling=2, time_quadrature='uniform'):
    """Compute one source-grid state; save arrays and actual convergence.

    ``nw`` is Legendre nodes for CD methods and positive Matsubara nodes for
    scGW, so these discretizations require separate convergence checks.
    ``time_oversampling`` controls the scGW imaginary-time grid independently
    of the Matsubara frequency cutoff: Nt = 2*nw*time_oversampling.
    ``time_quadrature`` selects uniform midpoint or Gauss-Legendre integration
    for the imaginary-time products; its convergence is checked independently.
    ``eta`` broadens CD response only; Matsubara scGW has no eta parameter.
    G0W0/evGW solve diagonal QP equations; qsGW uses full mode-A matrices;
    scGW solves the full frequency-dependent Dyson problem at explicit beta.
    Every method uses fc=False and the same finite Coulomb-factor model.
    The default omits the bare Gamma head AND its correlation completion.
    A bare Ewald correction alone is not a matched dielectric head treatment;
    it can be requested explicitly as a diagnostic, not a bulk prediction.
    """
    if method not in ('g0w0', 'evgw', 'qsgw', 'scgw'):
        raise ValueError('Choose g0w0, evgw, qsgw or scgw')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/f'{method}.npz').exists():
        raise FileExistsError('Keep prior runs; use a fresh result directory')
    started = perf_counter()
    mf = reference(source_kmesh=source_kmesh, mesh=mesh, exxdiv=exxdiv)
    scf_seconds = perf_counter()-started
    # The tight Ga 3d functions need a resolved AO Fourier grid. These two
    # W-point representatives differ by a reciprocal lattice vector.
    images = np.array([[.5, .25, .75], [-.5, .25, -.25]])@np.asarray(mf.cell.reciprocal_vectors())
    image_energy, _ = mf.get_bands(images, chunk_size=1)
    reference_error = float(np.max(abs(np.asarray(image_energy[0]-image_energy[1]))))
    if not np.isfinite(reference_error) or reference_error > 1e-6:
        raise ArithmeticError(f'FFT mesh {mesh} does not resolve reciprocal-equivalent GaAs bands: '
            f'{reference_error:.3e} Ha > 1e-6 Ha; increase the mesh before GW')
    reference_check_seconds = perf_counter()-started-scf_seconds
    model = build_periodic_model(mf, tolerance=1e-12, fc=False,
                                 max_memory_bytes=MAX_MEMORY_BYTES,
                                 screening_cutoff_rydberg=screening_cutoff_rydberg)
    model_seconds = perf_counter()-started-scf_seconds-reference_check_seconds
    print(f'PBE {scf_seconds:.3f} s; source model {model_seconds:.3f} s', flush=True)
    print('Compressed auxiliary ranks:', [b.shape[1] for b in model.factors], flush=True)
    print('Dielectric ranks:', [b.shape[1] for b in model.correlation_factors], flush=True)
    begin = perf_counter()
    if method == 'g0w0':
        result = g0w0_model(model, nw=nw, eta=eta, self_energy='diagonal')
    elif method == 'evgw':
        def show_ev(iteration, energy, residual):
            print(f'evGW {iteration:3d}: |R|={residual:.3e} Ha', flush=True)
        result = evgw_cd_kpoints(model, nw=nw, eta=eta,
            max_iter=max_iter, tol=tolerance, damping=.3,
            diis_space=diis_space, callback=show_ev)
    elif method == 'qsgw':
        def show_qs(iteration, fock, residual):
            print(f'qsGW {iteration:3d}: |R|={residual:.3e} Ha', flush=True)
        result = qsgw_cd_kpoints(model, nw=nw, eta=eta,
            max_iter=max_iter, tol=tolerance, tol_density=tolerance, damping=.3,
            diis_space=diis_space, callback=show_qs)
    else:
        def show_iteration(iteration, residual, mu, number_error):
            print(f'scGW {iteration:3d}: |R|={residual:.3e} Ha; '
                  f'mu={mu:.8f} Ha; dN={number_error:.2e}', flush=True)
        result = scgw_matsubara_kpoints(model, nw=nw, beta=beta,
            max_iter=max_iter, tol=tolerance, particle_tol=1e-9, mixing=.5,
            time_oversampling=time_oversampling, time_quadrature=time_quadrature,
            callback=show_iteration)
    jax.block_until_ready(result.mo_energy if method != 'scgw' else result.sigma_c_iw)
    seconds = perf_counter()-begin
    data = dict(reference_energy=np.asarray(model.energy),
        reference_coeff=np.asarray(model.coeff), reference_density=np.asarray(mf.result.density_spin),
        source_kpoints=np.asarray(model.kpoints), lattice_bohr=np.asarray(model.lattice),
        coordinates_bohr=np.asarray(mf.cell.coords),
        atomic_numbers=np.asarray(mf.cell.topology.nuclear_charges),
        valence_charges=np.asarray(mf.cell.charges),
        overlap=np.asarray(mf.inputs.overlap), nocc=np.array(model.nocc))
    report = dict(method=result.method, converged=bool(result.converged),
        source_kmesh=list(source_kmesh), fft_mesh=list(mesh), basis='gth-szv',
        pseudo='gth-pbe', xc='pbe', soc=False, fc=False, exxdiv=exxdiv, nw=nw,
        screening_cutoff_rydberg=screening_cutoff_rydberg,
        model_memory_limit_bytes=MAX_MEMORY_BYTES,
        eta_hartree=None if method == 'scgw' else eta,
        scf_seconds=scf_seconds, model_seconds=model_seconds, gw_seconds=seconds,
        reference_check_seconds=reference_check_seconds,
        reference_reciprocal_image_error_hartree=reference_error,
        tolerance_hartree=tolerance, source_model=model.metadata,
        scope='Finite Gaussian/source-k model; no bulk-convergence claim')
    if method == 'scgw':
        data.update(sigma_c_iw=np.asarray(result.sigma_c_iw),
            sigma_moment=np.asarray(result.sigma_moment),
            full_fock=np.asarray(result.full_fock), density_mo=np.asarray(result.density_mo),
            chemical_potential=np.asarray(result.chemical_potential),
            fermion=np.asarray(result.grid.fermion))
        report.update(beta_hartree_inverse=beta, iterations=result.n_iter,
            time_oversampling=result.time_oversampling, time_quadrature=result.time_quadrature,
            ntau=len(result.grid.tau),
            residual_hartree=float(result.fixed_point_residual),
            particle_number_error=float(result.particle_number_error),
            charge_response_electrons_per_hartree=float(result.charge_response))
    else:
        data.update(qp_energy=np.asarray(result.mo_energy), qp_coeff=np.asarray(result.mo_coeff),
            converged_mask=np.asarray(result.converged_mask),
            qp_computed_mask=np.asarray(result.qp_computed_mask), residual=np.asarray(result.qp_residual))
        for name in ('qp_weight', 'qp_branch_valid', 'qp_candidate_count'):
            value = getattr(result, name, None)
            if value is not None:
                data[name] = np.asarray(value)
        report['iterations'] = 1 if method == 'g0w0' else None
        report['residual_hartree'] = float(np.max(abs(data['residual'])))
    np.savez_compressed(directory/f'{method}.npz', **data)
    (directory/f'{method}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{result.method}: converged={result.converged}; GW {seconds:.3f} s', flush=True)
    return mf, model, result


if __name__ == '__main__':
    output = Path('artifacts/gaas_gw_comparison')
    common = dict(source_kmesh=(2, 1, 1), screening_cutoff_rydberg=16.)
    for method in ('g0w0', 'evgw', 'qsgw'):
        calculate(method, output, nw=64, **common)
    # At fixed beta, increase Matsubara nw before a low-temperature check.
    calculate('scgw', output, nw=1024, beta=400., time_oversampling=4,
              time_quadrature='gauss', **common)
