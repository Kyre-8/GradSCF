"""Bounded MP2 efficiency diagnostics using only the shared source algorithms.

The state-only mode checks the physical forward solution before attempting
response. Component timings separate the fixed-orbital AO2MO/MP2 derivative
from the SCF state adjoint, using the actual MP2 coefficient/Fock cotangents.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import resource
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from gradscf import gto, scf
from gradscf.ao2mo.reference import transform_fock
from gradscf.mp.integrals import from_orbitals
from gradscf.mp.mp2 import evaluate_mp2
from gradscf_tools.correlated_response import CorrelatedEnergy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--mode', choices=('full', 'direct'), required=True)
    parser.add_argument('--state-only', action='store_true')
    parser.add_argument('--diagnose-forward', action='store_true')
    parser.add_argument('--count-forward', action='store_true')
    parser.add_argument('--count-response', action='store_true')
    parser.add_argument('--capture-response', action='store_true')
    parser.add_argument('--whole-gradient', action='store_true')
    parser.add_argument('--response-restart', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.count_forward and (not args.state_only or args.count_response
                              or args.capture_response or args.whole_gradient):
        parser.error('--count-forward requires only --state-only')
    if args.diagnose_forward and (not args.state_only or args.count_response
                                 or args.capture_response or args.whole_gradient):
        parser.error('--diagnose-forward requires only --state-only')
    data = json.loads(args.geometry.read_text())
    report = dict(complete=False, mode=args.mode, basis='6-31g*', cartesian=True,
                  jax=jax.__version__, input_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
                  affinity=sorted(os.sched_getaffinity(0)), timings={}, state={})

    def save():
        report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)

    def timed(name, function, *inputs):
        timing = report['timings'][name] = {}
        report['phase'] = f'{name}.lower'
        save()
        t0 = perf_counter()
        lowered = jax.jit(function).lower(*inputs)
        timing['lower_seconds'] = perf_counter()-t0
        report['phase'] = f'{name}.compile'
        save()
        t0 = perf_counter()
        executable = lowered.compile()
        timing['compile_only_seconds'] = perf_counter()-t0
        timing['compile_seconds'] = timing['lower_seconds']+timing['compile_only_seconds']
        report['phase'] = f'{name}.compiled'
        save()
        analysis = executable.memory_analysis()
        timing['xla_temp_bytes'] = analysis.temp_size_in_bytes
        report['phase'] = f'{name}.first_execute'
        save()
        t0 = perf_counter()
        result = jax.block_until_ready(executable(*inputs))
        timing['first_seconds'] = perf_counter()-t0
        report['phase'] = f'{name}.first_complete'
        save()
        times = []
        for _ in range(0 if args.state_only else 3):
            t0 = perf_counter()
            result = jax.block_until_ready(executable(*inputs))
            times.append(perf_counter()-t0)
        timing.update(warm_seconds=times,
            warm_median_seconds=float(np.median(times)) if times else None)
        save()
        return result

    mol = gto.M(atom=list(zip(data['symbols'], data['coords_angstrom'])),
                charge=data.get('charge', 0), spin=data.get('spin', 0),
                basis='6-31g*', unit='Angstrom', cart=True)
    report['phase'] = 'seed'
    save()
    mf = scf.RHF(mol, integral_backend='native', execution_device='cpu',
        jk_backend=args.mode, direct_scf_tol=0., conv_tol=1e-12,
        conv_tol_density=1e-11, conv_tol_grad=1e-10,
        max_cycle=250, max_memory=60000.).run()
    report['seed'] = dict(converged=bool(mf.converged), energy=float(mf.e_tot))
    report['phase'] = 'stationary_state'
    save()
    model = CorrelatedEnergy(mf, 'mp2', jk_backend=args.mode,
                              gradient_tolerance=1e-10, max_memory=60000.)
    if args.response_restart is not None:
        model.reference.differentiation = replace(model.reference.differentiation,
                                                  restart=args.response_restart)
    report['response'] = asdict(model.reference.differentiation)
    r = model.coordinates
    if args.count_response or args.capture_response:
        from importlib import import_module
        from gradscf.solvers.linear import implicit as linear
        original = linear.gmres_solve
        gmres_module = import_module('gradscf.solvers.linear.gmres')
        original_gmres = gmres_module.gmres
        counts = dict(solves=0, matvecs=0, rhs_norms=[], methods=[], candidates=[])
        report['response_counts'] = counts
        report['phase'] = 'response'
        save()
        def record_method(method):
            counts['methods'].append(method)
        def counted_gmres(*values, **options):
            method = options.get('solve_method', 'batched')
            jax.debug.callback(lambda: record_method(method), ordered=True)
            candidate, info = original_gmres(*values, **options)
            action = getattr(values[0], '_diagnostic_matvec', values[0])
            rhs = values[1]
            residual = jnp.linalg.norm(action(candidate)-rhs)
            norm_rhs = jnp.linalg.norm(rhs)
            def record_candidate(residual, norm_rhs, info, solve_rtol, solve_atol):
                counts['candidates'].append(dict(method=method,
                    cumulative_matvecs=counts['matvecs'],
                    true_residual_norm=float(residual), rhs_norm=float(norm_rhs),
                    solve_rtol=float(solve_rtol), solve_atol=float(solve_atol),
                    solve_target=float(solve_atol+solve_rtol*norm_rhs),
                    physical_rtol=model.reference.differentiation.tolerance,
                    physical_target=float(model.reference.differentiation.tolerance*norm_rhs),
                    info=int(info)))
                save()
            jax.debug.callback(record_candidate, residual, norm_rhs, info,
                               options['tol'], options['atol'], ordered=True)
            return candidate, info
        gmres_module.gmres = counted_gmres
        def record_rhs(rhs):
            counts['solves'] += 1
            counts['rhs_norms'].append(float(np.linalg.norm(rhs)))
        def record_matvec():
            counts['matvecs'] += 1
        def counted_solve(matvec, rhs, **options):
            jax.debug.callback(record_rhs, rhs, ordered=True)
            if args.capture_response:
                if rhs.size > 96:
                    raise ValueError('Diagnostic operator capture is restricted to at most96 variables.')
                matrix = jax.lax.map(matvec, jnp.eye(rhs.size, dtype=rhs.dtype)).T
                rng = np.random.default_rng(2104)
                u, v = jnp.asarray(rng.normal(size=(2, rhs.size)))
                probes = jnp.stack((jnp.zeros_like(u), u, v, u+v, 2*u))
                actions = jax.lax.map(matvec, probes)
                def save_capture(matrix, rhs, probes, actions):
                    np.savez(args.output.with_suffix('.npz'), matrix=matrix, rhs=rhs,
                             probes=probes, actions=actions)
                jax.debug.callback(save_capture, matrix, rhs, probes, actions, ordered=True)
            def counted_action(value):
                jax.debug.callback(record_matvec, ordered=True)
                return matvec(value)
            counted_action._diagnostic_matvec = matvec
            return original(counted_action, rhs, **options)
        linear.gmres_solve = counted_solve
        result = jax.block_until_ready(jax.jit(jax.value_and_grad(model))(r))
        jax.effects_barrier()
        report['response_counts'] = counts
        report['energy'] = float(result[0])
        report['gradient_finite'] = bool(jnp.isfinite(result[1]).all())
        linear.gmres_solve = original
        gmres_module.gmres = original_gmres
        if args.capture_response:
            from gradscf.solvers import LinearSolverConfig, solve_linear
            captured = np.load(args.output.with_suffix('.npz'))
            matrix, rhs = captured['matrix'], captured['rhs']
            probes, actions = captured['probes'], captured['actions']
            direct = jax.block_until_ready(solve_linear(jnp.asarray(matrix), jnp.asarray(rhs),
                config=LinearSolverConfig(method='direct', rtol=1e-10, atol=0., max_dense=96)))
            iterative = jax.block_until_ready(solve_linear(jnp.asarray(matrix), jnp.asarray(rhs),
                config=LinearSolverConfig(rtol=1e-10, atol=0., restart=40, maxiter=100)))
            spectrum = np.linalg.eigvalsh(.5*(matrix+matrix.T))
            scaled = matrix/np.sqrt(np.abs(np.diag(matrix)[:, None]*np.diag(matrix)[None, :]))
            action_scale = max(np.linalg.norm(actions), np.finfo(float).tiny)
            report['captured_operator'] = dict(shape=matrix.shape,
                symmetry_relative_error=float(np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix)),
                condition=float(np.linalg.cond(matrix)), diagonal_scaled_condition=float(np.linalg.cond(scaled)),
                eigenvalue_extrema=[float(spectrum[0]), float(spectrum[-1])],
                zero_action_norm=float(np.linalg.norm(actions[0])),
                matrix_action_relative_error=float(np.linalg.norm(actions-probes@matrix.T)/action_scale),
                additivity_relative_error=float(np.linalg.norm(actions[3]-actions[1]-actions[2])/action_scale),
                homogeneity_relative_error=float(np.linalg.norm(actions[4]-2*actions[1])/action_scale),
                dense_solution_valid=bool(direct.converged), dense_gmres_valid=bool(iterative.converged),
                dense_residual_norm=float(direct.residual_norm), dense_gmres_residual_norm=float(iterative.residual_norm),
                dense_vs_gmres_solution_error=float(np.linalg.norm(np.asarray(direct.solution-iterative.solution))))
        report['complete'] = True
        save()
        return
    if args.whole_gradient:
        value, gradient = timed('whole_gradient', jax.value_and_grad(model), r)
        report.update(complete=True, energy=float(value),
                      gradient_finite=bool(jnp.isfinite(gradient).all()))
        save()
        return
    if args.count_forward:
        from importlib import import_module
        minimize_module = import_module('gradscf.solvers.nonlinear.minimize')
        original_forward_gmres = minimize_module.gmres
        forward_counts = dict(solves=0, matvecs=0, records=[],
            per_solve_true_residual_recorded=False)
        report['forward_counts'] = forward_counts

        def counted_forward_gmres(operator, rhs, **options):
            def record_start(norm, rtol, atol, restart, maxiter):
                forward_counts['solves'] += 1
                forward_counts['records'].append(dict(
                    solve=forward_counts['solves'], rhs_norm=float(norm),
                    tol=float(rtol), atol=float(atol), restart=int(restart),
                    maxiter=int(maxiter), solve_method=options['solve_method'], matvecs=0))
                save()
            jax.debug.callback(record_start, jnp.linalg.norm(rhs), options['tol'],
                options['atol'], options['restart'], options['maxiter'], ordered=True)

            def counted_action(vector):
                def record_action():
                    forward_counts['matvecs'] += 1
                    forward_counts['records'][-1]['matvecs'] += 1
                    if forward_counts['matvecs'] % 32 == 0:
                        save()
                jax.debug.callback(record_action, ordered=True)
                return operator(vector)

            result, info = original_forward_gmres(counted_action, rhs, **options)
            def record_end(info):
                forward_counts['records'][-1]['info'] = int(info)
                save()
            jax.debug.callback(record_end, info, ordered=True)
            return result, info

        minimize_module.gmres = counted_forward_gmres
    try:
        state = timed('stationary_state', model.reference.solve, r)
    finally:
        if args.count_forward:
            try:
                jax.effects_barrier()
            finally:
                minimize_module.gmres = original_forward_gmres
    # The source computes these residuals already; no diagnostic HVP is added.
    # Failed shift candidates' residuals are not present in this history.
    report['forward_polishing_history'] = [
        {key: np.asarray(value).item() for key, value in row.items()}
        for row in state.polishing_history]
    c, f = state.mo_coeff, state.fock_matrix.mean(axis=0)
    report['state'] = dict(stationary=bool(state.stationary),
        energy=float(state.total_energy), gradient_norm=float(state.gradient_norm),
        iterations=int(state.iterations), evaluations=int(state.evaluations),
        polishing_steps=int(state.polishing_steps),
        ao_fock_skew_norm=float(jnp.linalg.norm(f-f.T)),
        raw_mo_fock_skew_norm=float(jnp.linalg.norm(c.T @ f @ c-(c.T @ f @ c).T)),
        transported_mo_fock_skew_norm=float(jnp.linalg.norm(
            transform_fock(f, c)-transform_fock(f, c).T)),
        stages=[dict((key, float(value)) for key, value in stage.items())
                for stage in state.stage_history])
    save()
    if not report['state']['stationary']:
        report['complete'] = True
        report['response_skipped'] = 'Forward state is not stationary at the unchanged threshold.'
        save()
        return

    def integral_data(coeff, fock, coordinates):
        inputs = model.reference.inputs(coordinates)
        return from_orbitals(coeff, (transform_fock(fock, coeff),), model.nocc,
            jnp.asarray(0., dtype=coeff.dtype), eri=inputs['eri'],
            direct_basis=inputs.get('direct_basis'), max_memory=60000.)

    def reference_evaluation(coeff, fock, coordinates):
        ints = integral_data(coeff, fock, coordinates)
        return ints, evaluate_mp2(ints, model.config)

    if args.diagnose_forward:
        from importlib import import_module
        tensor_sum = import_module('gradscf.solvers.linear.tensor_sum')
        original_residual = tensor_sum.linear_residual
        report['tensor_residual_checks'] = []
        report['phase'] = 'mp2_forward'
        save()
        def record_residual(norm, target, accepted, initial_valid, rhs_finite,
                            candidate_finite, rhs_norm, rtol, atol, roundoff):
            report['tensor_residual_checks'].append(dict(
                invocation=len(report['tensor_residual_checks'])+1,
                residual_norm=float(norm), target=float(target), accepted=bool(accepted),
                initial_valid=bool(initial_valid), rhs_finite=bool(rhs_finite),
                candidate_finite=bool(candidate_finite), rhs_norm=float(rhs_norm),
                rtol=float(rtol), atol=float(atol), roundoff_allowance=float(roundoff)))
            save()
        def observed_residual(operator, solution, rhs, *, rtol, atol=0., converged=True):
            norm, accepted = original_residual(operator, solution, rhs,
                                              rtol=rtol, atol=atol, converged=converged)
            rhs_norm = jnp.linalg.norm(rhs)
            roundoff = 32*jnp.finfo(rhs.dtype).eps*(
                rhs_norm+jnp.linalg.norm(operator(solution)))
            target = atol+rtol*rhs_norm+roundoff
            jax.debug.callback(record_residual, norm, target, accepted, converged,
                jnp.all(jnp.isfinite(rhs)), jnp.all(jnp.isfinite(solution)),
                rhs_norm, rtol, atol, roundoff, ordered=True)
            return norm, accepted
        tensor_sum.linear_residual = observed_residual
    try:
        ints, result = jax.block_until_ready(jax.jit(reference_evaluation)(c, f, r))
        jax.effects_barrier()
    finally:
        if args.diagnose_forward:
            tensor_sum.linear_residual = original_residual
    report['state'].update(mp_valid=bool(result.valid),
        integral_data_finite=bool(ints.finite),
        reference_error=float(ints.reference_error), canonical_error=float(ints.canonical_error),
        min_abs_denominator=float(result.min_abs_denominator), mp_correlation=float(result.correlation_energy))
    if args.diagnose_forward:
        no = model.nocc
        fock = ints.focks[0]
        factors = (fock[:no, :no], fock[:no, :no], -fock[no:, no:], -fock[no:, no:])
        report['tensor_factors'] = [dict(shape=list(factor.shape),
            finite=bool(jnp.isfinite(factor).all()), norm=float(jnp.linalg.norm(factor)),
            skew_norm=float(jnp.linalg.norm(factor-factor.T)),
            symmetry_tolerance=float(32*jnp.finfo(factor.dtype).eps
                                      *jnp.maximum(1., jnp.linalg.norm(factor))))
            for factor in factors]
        factor_path = args.output.with_suffix('.factors.npz')
        np.savez(factor_path, **{f'factor_{i}':np.asarray(factor)
                                for i, factor in enumerate(factors)})
        report['tensor_factor_file'] = str(factor_path)
    save()
    if not report['state']['stationary'] or not report['state']['mp_valid']:
        report['complete'] = True
        report['response_skipped'] = 'Forward state is invalid at the unchanged physical thresholds.'
        save()
        return
    if args.state_only:
        report['complete'] = True
        save()
        return

    def fixed_orbital_correlation(coeff, fock, coordinates):
        return evaluate_mp2(integral_data(coeff, fock, coordinates), model.config).correlation_energy

    _, (gc, gf, gr) = timed('fixed_orbital_ao2mo_mp2_gradient',
        jax.value_and_grad(fixed_orbital_correlation, argnums=(0, 1, 2)), c, f, r)
    def state_adjoint(coordinates):
        solution = model.reference.solve(coordinates)
        return (solution.total_energy+jnp.vdot(solution.mo_coeff, gc)
                +jnp.vdot(solution.fock_matrix.mean(axis=0), gf))
    scf_gradient = timed('hf_plus_state_adjoint_gradient', jax.grad(state_adjoint), r)
    combined = scf_gradient+gr
    report['assembled_gradient_finite'] = bool(jnp.isfinite(combined).all())
    report['assembled_gradient_norm'] = float(jnp.linalg.norm(combined))
    report['translation_residual'] = float(jnp.linalg.norm(combined.sum(axis=0)))
    if args.mode == 'full' and c.shape[0] <= 25:
        from gradscf.scf.orbital_optimization import _problem_functions
        from gradscf.scf.uks import UKSConfig
        rows = tuple(tuple(float(x) for x in row) for row in model.reference.occupations)
        dimension, rotate, density_from, evaluate, energy = _problem_functions(
            'roks', 'hf', None, rows, uks_config=UKSConfig(xc_spec='hf'))
        inp = model.reference.inputs(r)
        inp['metric'] = inp.pop('overlap')
        origin = jnp.zeros(dimension, dtype=c.dtype)
        tangent_energy = lambda angle: energy(angle, c[None], inp)
        hessian = np.asarray(timed('diagnostic_orbital_hessian',
            jax.hessian(tangent_energy), origin))
        hessian = .5*(hessian+hessian.T)
        spectrum = np.linalg.eigvalsh(hessian)
        diagonal = np.diag(hessian)
        normalized = hessian/np.sqrt(diagonal[:, None]*diagonal[None, :])
        scaled = np.linalg.eigvalsh(normalized)
        report['response_spectrum'] = dict(dimension=dimension,
            min_eigenvalue=float(spectrum[0]), max_eigenvalue=float(spectrum[-1]),
            condition=float(spectrum[-1]/spectrum[0]),
            diagonal_scaled_condition=float(scaled[-1]/scaled[0]))
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
