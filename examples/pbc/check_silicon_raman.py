"""Check the raw Si Raman column before interpreting its selection rules.

Keep the initial model, optical displacement and Berry anchor fixed. Test
zero/translation tangents, physical occupied-frame gauge, and two independent
cold-SCF finite differences. Finite differences validate AD; they do not
generate the reported Raman derivative.
"""
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.pbc.scf import PeriodicSCFResult, run_kpoint_scf
from gradscf.pbc._kpoint import energy_and_fock
from gradscf.pbc.polarization import make_plan, polarization
from gradscf.dfpt.electric import _electric_model, _solve_electric
from silicon_raman import (OUTPUT, SETTINGS, MASSES_AMU, AMU_TO_ME, CONTROLS,
    LINEAR, NONLINEAR, mean_field, optical_displacement, mixed_problem)


def save_reference(path, state, overlap):
    """Save optional None fields as names, never pickled object arrays."""
    none_fields = [key for key in state._fields if getattr(state, key) is None]
    arrays = {key: np.asarray(getattr(state, key)) for key in state._fields
              if key not in none_fields}
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays, overlap=np.asarray(overlap),
                        none_fields=np.asarray(none_fields, dtype=str))
    temporary.replace(path)


def load_reference(path, *, legacy_zero_temperature=False):
    """Read numeric data; the original zero-T checkpoint omitted None metadata.

    Only that documented legacy format may skip chemical_potential. Its hash
    is verified by the caller, and its object payload is never deserialized.
    All physical arrays are restored unchanged; invalid objects in them fail.
    """
    with np.load(path, allow_pickle=False) as archive:
        if 'none_fields' in archive:
            none_fields = set(archive['none_fields'].tolist())
        elif legacy_zero_temperature:
            none_fields = {'chemical_potential'}
        else:
            raise ValueError('Explicitly identify a legacy zero-temperature checkpoint.')
        allowed = {key for key, value in PeriodicSCFResult._field_defaults.items()
                   if value is None}
        if not none_fields <= allowed:
            raise ValueError('A required SCF field cannot be None.')
        values = {key: None if key in none_fields else jnp.asarray(archive[key])
                  for key in PeriodicSCFResult._fields}
        overlap = np.asarray(archive['overlap'])
    return PeriodicSCFResult(**values), overlap


def inputs(name='gamma_fft15', settings=None):
    settings = SETTINGS if settings is None else settings
    original = json.loads((OUTPUT/(name+'_probe.json')).read_text())
    if original['settings'] != settings or not original['probe_passed']:
        raise RuntimeError('Use the same model as the accepted original probe.')
    anchor = tuple(jnp.asarray(p) for p in original['phase_anchor'])
    return original,anchor


def save(name, values, started, settings=None):
    settings = SETTINGS if settings is None else settings
    sources = ['src/gradscf/dfpt/electric.py','src/gradscf/pbc/polarization.py',
               'examples/pbc/silicon_raman.py','examples/pbc/check_silicon_raman.py']
    values.update(settings=settings,elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (OUTPUT/(name+'.json')).write_text(json.dumps(values,indent=2)+'\n')


def run_ad_checks(name='gamma_fft15', settings=None):
    settings = SETTINGS if settings is None else settings
    started = perf_counter()
    original,anchor = inputs(name, settings)
    mf = mean_field(settings)
    model,_,respond = mixed_problem(mf,reference_phases=anchor)
    u = jnp.asarray(optical_displacement())
    direction = jnp.array([0.,1.,0.])
    # Preserve this exact physical state for the later frame-gauge test.
    reference_path = OUTPUT/('validation_reference.npz' if name == 'gamma_fft15' else name+'_reference.npz')
    save_reference(reference_path, mf.result, mf.inputs.overlap)
    translation = jnp.zeros_like(u).at[:,2].set(1/np.sqrt(np.sum(MASSES_AMU)*AMU_TO_ME))
    rows = {}
    for label,displacement in [('optical',u),('zero',jnp.zeros_like(u)),('translation',translation)]:
        result = respond(displacement,direction)
        jax.block_until_ready(result)
        row = {key:np.asarray(value).tolist() for key,value in result.items()}
        rows[label] = row
        print(label,row,flush=True)
        residuals = [row[k] for k in ['reference_residual','field_residual','mixed_residual']]
        if not np.isfinite(row['derivative']).all() or not np.isfinite(residuals).all() or max(residuals) >= 1e-10:
            raise RuntimeError('An AD check failed its actual differentiated residual.')
    parity = float(np.max(abs(np.asarray(rows['optical']['derivative'])-original['dchi_column_dQ'])))
    zero_error = float(np.max(abs(np.asarray(rows['zero']['derivative']))))
    translation_error = float(np.max(abs(np.asarray(rows['translation']['derivative']))))
    passed = parity < 1e-10 and zero_error < 1e-8 and translation_error < 1e-8
    save(name+'_ad_checks',dict(complete=True,validation_passed=passed,checks=rows,
        probe_parity_error=parity,zero_error=zero_error,translation_error=translation_error,
        baseline_polarization=np.asarray(model.quantities(model.solution,model.inputs)['polarization']).tolist(),
        reference_checkpoint_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()),started,settings)
    assert passed,'Raw zero/translation/probe-parity checks failed; do not project the tensor.'


def run_gauge_check(name='gamma_fft15', settings=None):
    settings = SETTINGS if settings is None else settings
    started = perf_counter()
    _,anchor = inputs(name, settings)
    baseline = json.loads((OUTPUT/(name+'_ad_checks.json')).read_text())
    assert baseline['validation_passed']
    path = OUTPUT/('validation_reference.npz' if name == 'gamma_fft15' else name+'_reference.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest() != baseline['reference_checkpoint_sha256']:
        raise RuntimeError('The physical reference checkpoint changed.')
    mf = mean_field(settings)  # Rebuild the same integral model; use the saved state below.
    state, overlap = load_reference(path, legacy_zero_temperature=(name == 'gamma_fft15'))
    np.testing.assert_allclose(mf.inputs.overlap,overlap,atol=1e-13,rtol=0)
    coefficients = np.array(state.mo_coeff_spin, dtype=np.complex128, copy=True)
    rng = np.random.default_rng(20261016)
    for spin,no in enumerate(mf.cell.nelec):
        for start,stop in [(0,no),(no,coefficients.shape[-1])]:
            n = stop-start
            unitary = np.linalg.qr(rng.normal(size=(len(mf.kpts),n,n))+
                                   1j*rng.normal(size=(len(mf.kpts),n,n)))[0]
            coefficients[spin,:,:,start:stop] = coefficients[spin,:,:,start:stop]@unitary
    density = (coefficients*np.asarray(state.mo_occ_spin)[...,None,:])@coefficients.conj().swapaxes(-1,-2)
    density_error = float(np.max(abs(density-np.asarray(state.density_spin))))
    assert density_error < 1e-12
    rotated = state._replace(mo_coeff_spin=jnp.asarray(coefficients))
    model,_,respond = mixed_problem(mf,state=rotated,reference_phases=anchor)
    observed = respond(jnp.asarray(optical_displacement()),jnp.array([0.,1.,0.]))
    jax.block_until_ready(observed)
    raw = {key:np.asarray(value).tolist() for key,value in observed.items()}
    error = float(np.max(abs(np.asarray(raw['derivative'])-baseline['checks']['optical']['derivative'])))
    p_error = float(np.max(abs(np.asarray(model.quantities(model.solution,model.inputs)['polarization'])-
                               baseline['baseline_polarization'])))
    residuals = [raw[k] for k in ['reference_residual','field_residual','mixed_residual']]
    passed = error < 1e-8 and p_error < 1e-10 and np.isfinite(residuals).all() and max(residuals) < 1e-10
    save(name+'_gauge',dict(complete=True,validation_passed=passed,check=raw,
        maximum_gauge_error=error,baseline_polarization_error=p_error,density_error=density_error,
        same_reference_checkpoint_sha256=baseline['reference_checkpoint_sha256'],
        anchor='Exactly the saved original phase anchor; no new principal branch selection.'),started,settings)
    print('Gauge error',error,'P0 error',p_error,'density error',density_error,flush=True)
    assert passed,'Fixed-anchor physical-gauge invariance failed.'


def run_finite_difference(maximum_displacement_bohr, name='gamma_fft15', settings=None,
                          *, use_probe=False):
    """One +/- pair; an optional probe comparison cannot certify full tensors.

    Probe-only comparisons use separate output names, so they do not satisfy
    the full-tensor AD/gauge/FD prerequisites. They diagnose a surprising raw
    column before committing to the complete validation of a larger model.
    """
    settings = SETTINGS if settings is None else settings
    started = perf_counter()
    if maximum_displacement_bohr not in (3e-4,1e-4):
        raise ValueError('Use one of the two predeclared maximum Cartesian steps.')
    original,anchor = inputs(name, settings)
    if use_probe:
        analytic = np.asarray(original['dchi_column_dQ'])
    else:
        baseline = json.loads((OUTPUT/(name+'_ad_checks.json')).read_text())
        assert baseline['validation_passed']
        analytic = np.asarray(baseline['checks']['optical']['derivative'])
    mf = mean_field(settings)
    _,arguments,_ = mixed_problem(mf,reference_phases=anchor)
    u = jnp.asarray(optical_displacement())
    step = maximum_displacement_bohr/float(jnp.max(jnp.abs(u)))
    evaluate = lambda d,a,p:energy_and_fock(d,a,mesh=mf.cell.mesh,xc=mf.xc,
        exxdiv=mf.exxdiv,functional=mf.functional,params=p)
    cold = jax.jit(lambda q:run_kpoint_scf(arguments(q,jnp.zeros(3),u)[0],mesh=mf.cell.mesh,
        nelec=mf.cell.nelec,xc=mf.xc,exxdiv=mf.exxdiv,**CONTROLS))
    endpoints = []
    for sign in (1.,-1.):
        q = jnp.asarray(sign*step)
        data = arguments(q,jnp.zeros(3),u)[0]
        state = cold(q)
        assert state.converged
        model,diagonal = _electric_model(data,state,nocc=mf.cell.nelec,evaluate=evaluate,
                                         params=mf.params,reference_phases=anchor)
        def branch(field):
            result,values = _solve_electric(model,diagonal,(data,mf.params,field),
                                            config=NONLINEAR,linear_config=LINEAR)
            return result.solution,values['polarization']
        direction = jnp.array([0.,1.,0.])
        (x0,_),(x_e,chi) = jax.jit(lambda:jax.jvp(branch,(jnp.zeros(3),),(direction,)))()
        residual = jax.jvp(lambda e:model.residual(x0+e*x_e,(data,mf.params,e*direction)),
                           (jnp.array(0.),),(jnp.array(1.),))[1]
        plan = make_plan(np.asarray(mf.kpts)@np.asarray(mf.cell.lattice).T/(2*np.pi))
        checked = polarization(data,state,plan,nocc=mf.cell.nelec,reference_phases=anchor)
        record = dict(q_sqrt_me_bohr=float(q),chi_column=np.asarray(chi).tolist(),
            linear_residual=float(jnp.max(jnp.abs(residual))),energy_hartree=float(state.total_energy),
            gap_hartree=float(checked.minimum_gap),minimum_link=float(jnp.min(checked.minimum_link_singular_value)),
            maximum_phase_change=max(float(jnp.max(jnp.abs(p-a))) for p,a in zip(checked.phases,anchor)))
        endpoints.append(record)
        print('Cold endpoint',record,flush=True)
        assert record['linear_residual'] < 1e-10 and record['gap_hartree'] > 1e-3
        assert record['minimum_link'] > 1e-4 and record['maximum_phase_change'] < .1
    finite = (np.asarray(endpoints[0]['chi_column'])-endpoints[1]['chi_column'])/(2*step)
    error = abs(finite-analytic)
    passed = bool(np.all(error < 2e-8+5e-4*abs(analytic)))
    name = name+('_probe_fd_' if use_probe else '_fd_')+str(maximum_displacement_bohr)
    save(name,dict(complete=True,validation_passed=passed,endpoints=endpoints,
        maximum_displacement_bohr=maximum_displacement_bohr,q_step_sqrt_me_bohr=step,
        comparison_source='single-column probe' if use_probe else 'complete AD invariant checks',
        finite_difference=finite.tolist(),absolute_error=error.tolist(),
        gate='abs(error) < 2e-8 + 5e-4*abs(AD dchi/dQ)'),started,settings)
    print('FD',maximum_displacement_bohr,'error',error.tolist(),flush=True)
    assert passed,'Cold-SCF finite difference disagrees with the AD derivative.'


if __name__ == '__main__':
    run_ad_checks()

# Measured default Gamma/FFT15 checks on jy_gpu_1, CPU float64 (2026-10-16):
# optical parity = 0; zero tangent = 0; max translation derivative = 8.09e-17.
# Cold-SCF FD max errors: 1.19161e-9 (3e-4 bohr), 1.22696e-10 (1e-4 bohr).
# The raw forbidden zy term remains; derivative agreement is not convergence.
