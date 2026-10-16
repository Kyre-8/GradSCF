"""P04: zero-field Born charges of a declared small rocksalt LiH model.

PBE/GTH-Pade/SZV at the saved P02 cubic minimum, FFT31^3, full k2^3. Differentiate the relaxed
projector equations and the moving AO Berry links; no canonical eigenvector AD.
This is a fixed-cell dP/dR example, not a uniform-field/dielectric calculation.
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
from gradscf.pbc import dft, gto
from gradscf.pbc.scf import run_kpoint_scf
from gradscf.pbc.polarization import make_plan, polarization
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.dfpt.electric import born_problem
from gradscf.scf import SCFDifferentiationConfig
from gradscf.solvers import LinearSolverConfig


OUTPUT = Path('reproducibility/ad_showcase/P04_lih_born')
SETTINGS = dict(a_bohr=7.780032276700287, xc='pbe', basis='gth-szv', pseudo='gth-pade',
                mesh=[31, 31, 31], kmesh=[2, 2, 2], precision=1e-10)
CONTROLS = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                max_cycle=150, damping=.1)
DIFFERENTIATION = SCFDifferentiationConfig(mode='implicit', tolerance=1e-11,
                                         max_iter=100, restart=40)
LINEAR = LinearSolverConfig(method='gmres', rtol=1e-10, atol=1e-12,
                            restart=40, maxiter=100)


def mean_field(settings=None):
    settings = SETTINGS if settings is None else settings
    lattice = .5*settings['a_bohr']*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    coordinates = np.array([[0., 0., 0.], [.5, .5, .5]])@lattice
    cell = gto.M(atom=list(zip(['Li', 'H'], coordinates)), a=lattice, unit='Bohr',
                 basis=settings['basis'], pseudo=settings['pseudo'], mesh=tuple(settings['mesh']),
                 precision=settings['precision'], spin=0, cart=True)
    return dft.KRKS(cell, xc=settings['xc'], kpts=cell.make_kpts(tuple(settings['kmesh'])),
                    temperature=None, exxdiv='ewald', differentiation=DIFFERENTIATION,
                    **CONTROLS).run()


def run(name='gradscf', *, directions=None, steps=(3e-4, 1e-4), settings=None):
    """Run selected directions; the default forms the complete 2x3x3 tensor."""
    started = perf_counter()
    settings = dict(SETTINGS if settings is None else settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT/(name+'.json')
    destination.write_text('{"complete": false}\n')
    mf = mean_field(settings)
    cell = mf.cell
    if not mf.converged:
        raise RuntimeError('The LiH SCF reference did not converge.')
    fractional_k = np.asarray(mf.kpts)@np.asarray(cell.lattice).T/(2*np.pi)
    plan = make_plan(fractional_k)
    reference = polarization(mf.inputs, mf.result, plan, nocc=cell.nelec)
    if float(reference.minimum_gap) <= 1e-3:
        raise RuntimeError('The sampled insulating gap is below the declared 1e-3 Eh gate.')
    anchor = jax.tree.map(jax.lax.stop_gradient, reference.phases)
    print('LiH reference', float(mf.e_tot), 'gap', float(reference.minimum_gap),
          'minimum Berry singular value', float(jnp.min(reference.minimum_link_singular_value)), flush=True)
    prepared = perf_counter()
    problem = born_problem(mf)
    response = jax.jit(lambda direction:problem.response(tangent=direction, config=LINEAR))
    directions = tuple((atom, axis) for atom in range(2) for axis in range(3)) if directions is None else tuple(directions)
    records = []
    density_derivatives = []
    volume = float(cell.volume)
    for atom, axis in directions:
        direction = jnp.zeros_like(cell.coords).at[atom, axis].set(1.)
        before = perf_counter()
        result = response(direction)
        jax.block_until_ready(result)
        charge = volume*np.asarray(result.values['polarization'])
        row = dict(atom=atom, axis=axis, charge_e=charge.tolist(),
                   ionic_charge_e=(np.asarray(cell.charges)[atom]*np.eye(3)[:, axis]).tolist(),
                   linear_residual=float(result.residual_norm), converged=bool(result.converged),
                   seconds=perf_counter()-before)
        records.append(row)
        density_derivatives.append(np.asarray(result.density))
        print('Born direction', atom, axis, row, flush=True)
        if not row['converged'] or not np.isfinite(charge).all():
            (OUTPUT/(name+'_failed_response.json')).write_text(json.dumps(records, indent=2)+'\n')
            raise RuntimeError('A Born response failed its actual residual or finite-value check.')

    # Rebuild and reconverge every displaced state. The compiled array driver
    # reuses the existing SCF; only geometry is dynamic, not a changing closure.
    def displaced(coordinates):
        data = build_inputs(cell, parameters=cell.parameters_at(coordinates), kpts=mf.kpts)
        state = run_kpoint_scf(data, mesh=cell.mesh, nelec=cell.nelec, xc=mf.xc,
            exxdiv=mf.exxdiv, differentiation=DIFFERENTIATION, **CONTROLS)
        result = polarization(data, state, plan, nocc=cell.nelec, reference_phases=anchor)
        return result, state.density_spin, state.total_energy
    displaced = jax.jit(displaced)
    finite_differences = []
    for (atom, axis), analytic, density_derivative in zip(directions, records, density_derivatives):
        direction = jnp.zeros_like(cell.coords).at[atom, axis].set(1.)
        for step in steps:
            plus, dp, ep = displaced(cell.coords+step*direction)
            minus, dm, em = displaced(cell.coords-step*direction)
            jax.block_until_ready((plus, minus, dp, dm))
            if not bool(plus.valid & minus.valid & plus.response_valid & minus.response_valid):
                raise RuntimeError('A finite-displacement state or local branch is invalid.')
            finite = volume*np.asarray(plus.total-minus.total)/(2*step)
            error = float(np.max(abs(finite-np.asarray(analytic['charge_e']))))
            finite_differences.append(dict(atom=atom, axis=axis, step_bohr=step,
                charge_e=finite.tolist(), maximum_charge_error_e=error,
                maximum_density_error=float(np.max(abs(np.asarray(dp-dm)/(2*step)-density_derivative))),
                gap_plus_hartree=float(plus.minimum_gap), gap_minus_hartree=float(minus.minimum_gap),
                energy_plus_hartree=float(ep), energy_minus_hartree=float(em),
                minimum_link_singular_value=float(jnp.minimum(jnp.min(plus.minimum_link_singular_value), jnp.min(minus.minimum_link_singular_value)))))
            print('FD', atom, axis, step, 'Born error', error, flush=True)
    full = set(directions) == {(a, i) for a in range(2) for i in range(3)}
    tensor = None
    diagnostics = {}
    if full:
        tensor = np.empty((2, 3, 3))
        for row in records:
            tensor[row['atom'], :, row['axis']] = row['charge_e']
        isotropic = np.trace(tensor, axis1=1, axis2=2)/3
        diagnostics = dict(raw_acoustic_charge_sum_e=tensor.sum(axis=0).tolist(),
            raw_maximum_acoustic_charge_sum_e=float(np.max(abs(tensor.sum(axis=0)))),
            raw_maximum_cubic_deviation_e=float(np.max(abs(tensor-isotropic[:, None, None]*np.eye(3)))),
            raw_maximum_offdiagonal_e=float(np.max(abs(tensor*(1-np.eye(3))))),
            average_diagonal_charges_e=isotropic.tolist())
    passed = all(row['maximum_charge_error_e'] < 2e-5 for row in finite_differences)
    paths = ['src/gradscf/dfpt/electric.py', 'src/gradscf/pbc/polarization.py',
             'src/gradscf/dfpt/periodic.py', 'src/gradscf/dfpt/ks.py', 'src/gradscf/pbc/_kpoint.py']
    report = dict(complete=True, full_tensor=full, validation_passed=passed, settings=settings,
        energy_hartree=float(mf.e_tot), sampled_indirect_gap_hartree=float(reference.minimum_gap),
        minimum_link_singular_value=float(jnp.min(reference.minimum_link_singular_value)),
        principal_response_valid=bool(reference.response_valid), local_phase_anchor=[np.asarray(p).tolist() for p in anchor],
        coordinates_bohr=np.asarray(cell.coords).tolist(), lattice_bohr=np.asarray(cell.lattice).tolist(),
        fractional_kpoints=fractional_k.tolist(), pseudopotential_charges=np.asarray(cell.charges).tolist(),
        explicit_electrons=cell.nelectron, nao=cell.topology.nao, occupations=list(cell.nelec), volume_bohr3=volume,
        basis_data=cell.basis_spec, pseudopotential_data=cell.pseudopotentials,
        born_charge_e=None if tensor is None else tensor.tolist(), directions=records, finite_differences=finite_differences,
        charge_axes='atom, polarization Cartesian alpha, displacement Cartesian beta',
        definition='Z*[atom,alpha,beta] = volume * dP_alpha/dR_atom,beta at zero field and fixed cell; charge in e',
        scf_reference_seconds=prepared-started, elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **diagnostics,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), hostname=platform.node(),
        python=platform.python_version(), jax=jax.__version__, dtype='float64',
        scope='Declared finite FFT/k/basis model. No acoustic/cubic projection, no field/dielectric response, no material convergence or experimental claim.')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(OUTPUT/(name+'_response.npz'), density_derivatives=density_derivatives,
                        reference_density=np.asarray(mf.result.density_spin))
    assert passed, 'Recorded Born finite differences exceed the declared 2e-5 e gate.'
    print('LiH Born completed', report['elapsed_seconds'], 'seconds; peak RSS', report['peak_rss_kib'], 'KiB', flush=True)
    return report


if __name__ == '__main__':
    run()

# Verified on jy_gpu_1, CPU float64, JAX 0.10.0 (2026-10-16):
# E = -8.00763182486617 Eh; sampled gap = 0.14272638774851437 Eh.
# Mean diagonal Z*: Li +0.9966941800957084, H -0.9967546291835733 e.
# Raw max |sum_atom Z*| = 6.04490878663988e-5 e; no ASR projection.
# Six response directions, twelve +/- displacement pairs: all passed.
# Max linear residual 8.53e-13; max native FD error 1.1093e-9 e.
# Independent PySCF max Born error 2.7593e-8 e (separate comparison script).
# Native workflow 123.76 s; peak RSS 5,145,848 KiB, including compilation.
# These are FFT31/k2/SZV finite-model checks, not material convergence.
# Matching P03 at FFT71 (same a/k/basis): Li +0.9967341373414307 e,
# H -0.9967341427539095 e; raw max |sum Z*| = 5.41248e-9 e.
# Six directions/twelve FD pairs passed, max FD error 1.03224e-9 e.
# Full FFT71 workflow 440.49 s, peak RSS 18,810,928 KiB; no projection.
