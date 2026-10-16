"""Small-field reconvergence checks for lih_dielectric.py.

This is a native finite-field check, not an independent electronic-structure
code. The separate dimer and H2 tests provide independent field oracles.
Finite fields use the existing shared nonlinear solver; no SCF loop is copied.
"""
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.dfpt.electric import _electric_model, _solve_field, _field_term
from gradscf.pbc._kpoint import energy_and_fock
from gradscf.pbc.polarization import make_plan, polarization, ao_links, _berry, _projected_frame
from gradscf.solvers.nonlinear.iterate import NonlinearConfig
from lih_born_charges import mean_field, LINEAR, SETTINGS
from lih_dielectric import OUTPUT


def run(name='finite_field', *, reference='gradscf', settings=None):
    started = perf_counter()
    settings = dict(SETTINGS if settings is None else settings)
    linear_path = OUTPUT/(reference+'.json')
    linear = json.loads(linear_path.read_text())
    if not linear['complete'] or not linear['full_tensor'] or linear['settings'] != settings:
        raise RuntimeError('A matching complete zero-field response is required.')
    chi = np.asarray(linear['susceptibility'])
    mf = mean_field(settings)
    if not mf.converged:
        raise RuntimeError('The zero-field LiH reference did not converge.')
    data, counts = mf.inputs, mf.cell.nelec
    evaluate = lambda d,a,p:energy_and_fock(d, a, mesh=mf.cell.mesh, xc=mf.xc,
        exxdiv=mf.exxdiv, functional=mf.functional, params=p)
    model, diagonal = _electric_model(data, mf.result, nocc=counts,
                                     evaluate=evaluate, params=mf.params)
    nonlinear = NonlinearConfig(maxiter=150, residual_tol=1e-11, energy_tol=1e-13, damping=.2)
    solve = jax.jit(lambda field:_solve_field(model, diagonal, field,
                    config=nonlinear, linear_config=LINEAR))
    plan = make_plan(np.asarray(mf.kpts)@np.asarray(mf.cell.lattice).T/(2*np.pi))
    reference = polarization(data, mf.result, plan, nocc=counts)
    anchor = reference.phases
    frames = tuple(mf.result.mo_coeff_spin[s, :, :, :no] for s, no in enumerate(counts))
    empty_frames = tuple(mf.result.mo_coeff_spin[s, :, :, no:] for s, no in enumerate(counts))
    links = ao_links(data.ao, data.weights, data.coords, data.lattice, plan)
    metric = np.asarray(data.overlap)
    inverse_metric = jnp.linalg.inv(data.overlap)
    effective_fock = jax.jit(lambda density, field:evaluate(density, data, mf.params)[1]
        +_field_term(density, data, field, frames, plan, anchor)[1])

    def diagnostics(density, field):
        # This inspector is outside AD. It checks the actual finite-field state,
        # rather than inheriting the zero-field reference's gap or rank flag.
        occupied, empty, grams = [], [], []
        for spin, (ref, ref_empty) in enumerate(zip(frames, empty_frames)):
            c, gram = _projected_frame(density[spin], data.overlap, ref)
            v, gram_empty = _projected_frame(inverse_metric-density[spin], data.overlap, ref_empty)
            occupied.append(c)
            empty.append(v)
            grams.extend([float(jnp.min(jnp.linalg.eigvalsh(gram))),
                          float(jnp.min(jnp.linalg.eigvalsh(gram_empty)))])
        berry = _berry(tuple(occupied), links, data.lattice, plan, reference_phases=anchor)
        fock = np.asarray(effective_fock(density, field))
        gaps, overlaps, ov = [], [], []
        for spin, (c, v, ref) in enumerate(zip(occupied, empty, frames)):
            c, v, ref = map(np.asarray, (c, v, ref))
            adjoint = lambda a:a.conj().swapaxes(-1, -2)
            eo = np.linalg.eigvalsh(adjoint(c)@fock[spin]@c)
            ev = np.linalg.eigvalsh(adjoint(v)@fock[spin]@v)
            gaps.append(float(ev.min()-eo.max()))
            overlaps.append(float(np.linalg.svd(adjoint(ref)@metric@c, compute_uv=False).min()))
            ov.append(float(abs(adjoint(v)@fock[spin]@c).max()))
        d = np.asarray(density)
        ranks = np.trace(d@metric, axis1=-2, axis2=-1)
        number = float(np.sum(ranks*np.asarray(data.kweights)).real)
        phase_step = max(float(jnp.max(jnp.abs(p-a))) for p,a in zip(berry.phases, anchor))
        values = dict(density_hermiticity_error=float(abs(d-d.conj().swapaxes(-1,-2)).max()),
            density_idempotency_error=float(abs(d@metric@d-d).max()),
            maximum_block_occupation_error=float(abs(ranks-np.asarray(counts)[:,None]).max()),
            electron_number=number, electron_number_error=abs(number-mf.cell.nelectron),
            minimum_chart_gram_eigenvalue=min(grams),
            minimum_reference_subspace_overlap=min(overlaps),
            effective_fock_indirect_gap_hartree=min(gaps),
            effective_fock_occupied_empty_residual=max(ov),
            minimum_link_singular_value=float(jnp.min(berry.minimum_link_singular_value)),
            minimum_branch_clearance_radian=float(jnp.min(berry.branch_clearance)),
            maximum_local_phase_change_radian=phase_step,
            berry_valid=bool(berry.valid & berry.response_valid))
        values['passed'] = (values['berry_valid'] and values['density_hermiticity_error'] < 1e-10
            and values['density_idempotency_error'] < 1e-9
            and values['maximum_block_occupation_error'] < 1e-9
            and values['electron_number_error'] < 1e-9
            and min(grams) > .5 and min(overlaps) > .99 and min(gaps) > 1e-3
            and max(ov) < 1e-9 and phase_step < .1)
        return values

    records, endpoints = [], []
    destination = OUTPUT/(name+'.json')
    hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in linear['source_sha256']}
    if hashes != linear['source_sha256']:
        raise RuntimeError('Finite-field and zero-field source snapshots differ.')
    source = dict(complete=False, linear_result_sha256=hashlib.sha256(linear_path.read_bytes()).hexdigest(),
        settings=settings, field_steps=[1e-4, 3e-5],
        gate_chi_absolute=2e-7, gate_chi_relative=2e-5,
        boundary='Small local fields, fixed ions/cell. Effective Fock gap is an endpoint diagnostic, not a quasiparticle gap or a full stability proof.',
        source_sha256=hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        endpoints=endpoints, finite_differences=records)
    destination.write_text(json.dumps(source, indent=2)+'\n')
    for axis in range(3):
        for step in source['field_steps']:
            values_pair = []
            for sign in (1, -1):
                field = jnp.eye(3)[axis]*(sign*step)
                before = perf_counter()
                root, values = solve(field)
                jax.block_until_ready(values)
                endpoint = dict(axis=axis, field_au=np.asarray(field).tolist(),
                    converged=bool(root.converged), residual=float(root.residual_norm),
                    iterations=int(root.iterations), electric_enthalpy=float(values['electric_enthalpy']),
                    polarization=np.asarray(values['polarization']).tolist(),
                    diagnostics=diagnostics(values['density'], field), seconds=perf_counter()-before)
                endpoints.append(endpoint)
                destination.write_text(json.dumps(source, indent=2)+'\n')
                print('Finite field', endpoint, flush=True)
                if not endpoint['converged'] or endpoint['residual'] > 1e-11 or not endpoint['diagnostics']['passed']:
                    raise RuntimeError('A finite-field state failed its actual residual or domain checks.')
                values_pair.append(np.asarray(values['polarization']))
            finite = (values_pair[0]-values_pair[1])/(2*step)
            error = abs(finite-chi[:,axis])
            row = dict(axis=axis, step_au=step, chi_column=finite.tolist(),
                maximum_error=float(error.max()),
                passed=bool(np.all(error < 2e-7+2e-5*abs(chi[:,axis]))))
            records.append(row)
            destination.write_text(json.dumps(source, indent=2)+'\n')
            if not row['passed']:
                raise RuntimeError('A finite-field difference exceeds the declared chi tolerance.')
    source.update(complete=True, validation_passed=True, elapsed_seconds=perf_counter()-started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    destination.write_text(json.dumps(source, indent=2)+'\n')
    print('All finite-field endpoints and differences passed', source['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    run()
