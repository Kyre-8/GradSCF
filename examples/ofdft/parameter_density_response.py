"""P08: fixed-electron TF/vW model, external KEDF parameters and density response.

An analytic periodic model, not a material calculation. All quantities use
atomic units; the Hartree zero mode is removed with a neutralizing background.
Run on jy_gpu_1, followed by the separate independent comparison and plot.
"""
from dataclasses import replace
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

from gradscf import dfpt, ofdft, scf
from gradscf.solvers import LinearSolverConfig


OUTPUT = Path('reproducibility/ad_showcase/P08_ofdft_parameter_response')
LENGTH, NUMBER, CHEMICAL_POTENTIAL = 5., 2., .17
AMPLITUDE_MODULATION = (.18, .12)
TF_WEIGHT, VW_WEIGHT = .8, .6
BASE_KINETIC = ofdft.KineticFunctional('tfvw')
CONFIG = ofdft.OFDFTConfig(xc=None, maxiter=600, tolerance=1e-10,
    differentiation=scf.SCFDifferentiationConfig(mode='implicit', tolerance=1e-11,
                                                max_iter=120, restart=60))
LINEAR = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13, max_dense=1024)


def external_kinetic(parameters, features):
    """An ordinary external callback; its nested coefficients stay explicit."""
    return BASE_KINETIC(parameters['weights'], features)


def make_model(mesh=(9, 9, 3)):
    """Manufacture a positive fixed-N stationary density and its external field.

    phi=A(1+b cos(kx)+c cos(ky)), rho=phi². The amplitude/density Fourier
    support fits every requested odd x/y mesh >=5. The z direction is uniform.
    The potential is fixed after construction when KEDF coefficients vary.
    """
    data = ofdft.periodic_inputs(jnp.eye(3)*LENGTH, mesh, nelectron=NUMBER)
    b, c = AMPLITUDE_MODULATION
    wave_number = 2*jnp.pi/LENGTH
    x, y = wave_number*data.coordinates[:, 0], wave_number*data.coordinates[:, 1]
    scale = jnp.sqrt((NUMBER/LENGTH**3)/(1+(b*b+c*c)/2))
    phi = scale*(1+b*jnp.cos(x)+c*jnp.cos(y))
    density = phi**2
    laplacian_phi = -scale*wave_number**2*(b*jnp.cos(x)+c*jnp.cos(y))
    hartree = 4*jnp.pi*scale**2/wave_number**2*(
        2*b*jnp.cos(x)+2*c*jnp.cos(y)+b*b*jnp.cos(2*x)/8+c*c*jnp.cos(2*y)/8
        +b*c*jnp.cos(x+y)/2+b*c*jnp.cos(x-y)/2)
    c_tf = .3*(3*jnp.pi**2)**(2/3)
    potential = (CHEMICAL_POTENTIAL-(5/3)*TF_WEIGHT*c_tf*density**(2/3)
                 +.5*VW_WEIGHT*laplacian_phi/phi-hartree)
    data = replace(data, external_potential=potential)
    parameters = {'weights': {'tf': jnp.asarray(TF_WEIGHT), 'vw': jnp.asarray(VW_WEIGHT)}}
    return data, parameters, dict(density=density, amplitude=phi, hartree=hartree,
        laplacian_amplitude=laplacian_phi, chemical_potential=CHEMICAL_POTENTIAL)


def calculate(mesh, *, detailed=False):
    """Reuse the stationary optimizer and KKT response on one declared grid."""
    started = perf_counter()
    data, parameters, exact = make_model(mesh)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:, 0]/LENGTH)+.4*jnp.sin(2*jnp.pi*data.coordinates[:, 1]/LENGTH)
    theta0 = jnp.asarray([0., TF_WEIGHT, VW_WEIGHT])

    def inputs(theta):
        # The manufactured external field stays fixed as weights vary.
        moved = replace(data, external_potential=data.external_potential+theta[0]*probe)
        return moved, {'weights': {'tf': theta[1], 'vw': theta[2]}}, {}

    def solve(theta):
        moved, kinetic_parameters, _ = inputs(theta)
        return ofdft.run_ofdft(moved, kinetic=external_kinetic,
                              kinetic_params=kinetic_parameters, config=CONFIG)

    evaluate = jax.jit(solve)
    ground = evaluate(theta0)
    assert ground.converged and ground.iterations > 0
    np.testing.assert_allclose(ground.electron_number, NUMBER, atol=3e-12, rtol=0)
    np.testing.assert_allclose(ground.density, exact['density'], atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(ground.chemical_potential, CHEMICAL_POTENTIAL, atol=2e-8, rtol=0)
    problem = dfpt.ofdft_problem(data, ground, kinetic=external_kinetic,
                                 kinetic_params=parameters, config=CONFIG)
    potential = problem.response(potential=probe, config=LINEAR)
    assert potential.converged
    np.testing.assert_allclose(potential.values['electron_number'], 0., atol=2e-10, rtol=0)
    arrays = dict(coordinates_bohr=np.asarray(data.coordinates), weights=np.asarray(data.weights),
        density=np.asarray(ground.density), exact_density=np.asarray(exact['density']),
        external_potential=np.asarray(data.external_potential), probe=np.asarray(probe),
        potential_density_response=np.asarray(potential.density))
    record = dict(mesh=list(mesh), total_energy_hartree=float(ground.total_energy),
        energy_components={name: float(value) for name, value in zip(ground.components._fields, ground.components)},
        chemical_potential_hartree=float(ground.chemical_potential), electron_number=float(ground.electron_number),
        forward_residual=float(ground.residual_norm), forward_iterations=int(ground.iterations),
        maximum_manufactured_density_error=float(jnp.max(abs(ground.density-exact['density']))),
        potential_response_residual=float(potential.residual_norm),
        potential_energy_curvature=float(jnp.sum(data.weights*probe*potential.density)))

    if detailed:
        jacobian = jax.jit(jax.jacfwd(lambda theta: solve(theta).density))(theta0)
        assert np.isfinite(jacobian).all()
        axis_responses = []
        for axis in jnp.eye(3):
            tangent = jax.jvp(inputs, (theta0,), (axis,))[1]
            response = problem.response(tangent, config=LINEAR)
            assert response.converged
            axis_responses.append(np.asarray(response.density))
        dfpt_jacobian = np.stack(axis_responses, axis=1)
        np.testing.assert_allclose(jacobian, dfpt_jacobian, atol=2e-8, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(data.weights)@jacobian, 0., atol=2e-10, rtol=0)
        energy_gradient = jax.jit(jax.grad(lambda theta: solve(theta).total_energy))(theta0)
        density_cotangent = data.weights*(probe+.3*jnp.cos(4*jnp.pi*data.coordinates[:, 0]/LENGTH))
        vjp = jax.jit(jax.grad(lambda theta: jnp.dot(density_cotangent, solve(theta).density)))(theta0)
        np.testing.assert_allclose(vjp, density_cotangent@jacobian, atol=2e-9, rtol=2e-7)
        differences = []
        for index, name in enumerate(['potential', 'tf', 'vw']):
            for step in [1e-3, 3e-4, 1e-4]:
                direction = jnp.eye(3)[index]
                plus, minus = evaluate(theta0+step*direction), evaluate(theta0-step*direction)
                assert plus.converged & minus.converged
                finite = (plus.density-minus.density)/(2*step)
                energy_fd = (plus.total_energy-minus.total_energy)/(2*step)
                differences.append(dict(parameter=name, step=step,
                    density_max_error=float(jnp.max(abs(finite-jacobian[:, index]))),
                    energy_gradient_error=float(abs(energy_fd-energy_gradient[index]))))
            np.testing.assert_allclose(jacobian[:, index], finite, atol=3e-7, rtol=3e-5)
            np.testing.assert_allclose(energy_gradient[index], energy_fd, atol=2e-7, rtol=2e-5)

        path_direction = jnp.asarray([.03, .05, -.04])
        series = problem.series(path=lambda t: inputs(theta0+t*path_direction), order=2, config=LINEAR)
        assert series.converged and np.isfinite(series.density).all() and np.isfinite(series.energy).all()
        np.testing.assert_allclose(series.density[0], jacobian@path_direction, atol=2e-9, rtol=2e-6)
        np.testing.assert_allclose(series.energy[0], energy_gradient@path_direction, atol=2e-9, rtol=2e-6)
        np.testing.assert_allclose(series.density@data.weights, 0., atol=2e-9, rtol=0)
        path_checks, curves = [], []
        for step in [.02, .01, .005]:
            plus, minus = evaluate(theta0+step*path_direction), evaluate(theta0-step*path_direction)
            assert plus.converged & minus.converged
            density_second = (plus.density-2*ground.density+minus.density)/(2*step**2)
            energy_second = (plus.total_energy-2*ground.total_energy+minus.total_energy)/(2*step**2)
            path_checks.append(dict(step=step,
                density_coefficient2_error=float(jnp.max(abs(density_second-series.density[1]))),
                energy_coefficient2_error=float(abs(energy_second-series.energy[1]))))
        np.testing.assert_allclose(series.density[1], density_second, atol=3e-7, rtol=3e-4)
        np.testing.assert_allclose(series.energy[1], energy_second, atol=2e-7, rtol=2e-4)
        for t in [-.2, -.1, -.05, 0., .05, .1, .2]:
            actual = evaluate(theta0+t*path_direction)
            assert actual.converged
            linear = ground.total_energy+t*series.energy[0]
            quadratic = linear+t*t*series.energy[1]
            curves.append(dict(t=t, energy_hartree=float(actual.total_energy),
                linear_energy_hartree=float(linear), quadratic_energy_hartree=float(quadratic),
                density_probe=float(jnp.sum(data.weights*probe*actual.density)),
                linear_density_probe=float(jnp.sum(data.weights*probe*(ground.density+t*series.density[0]))),
                quadratic_density_probe=float(jnp.sum(data.weights*probe*(ground.density+t*series.density[0]+t*t*series.density[1])))))
        arrays.update(density_jacobian=np.asarray(jacobian), dfpt_density_jacobian=dfpt_jacobian,
                      path_density_coefficients=np.asarray(series.density))
        record.update(parameter_order=['potential_amplitude_Eh', 'tf_weight', 'vw_weight'],
            base_parameters=np.asarray(theta0).tolist(), energy_gradient=np.asarray(energy_gradient).tolist(),
            potential_parameter_fd_checks=differences, density_vjp=np.asarray(vjp).tolist(),
            density_adjoint_error=float(jnp.max(abs(vjp-density_cotangent@jacobian))),
            implicit_dfpt_jacobian_error=float(jnp.max(abs(jacobian-dfpt_jacobian))),
            taylor_path_direction=np.asarray(path_direction).tolist(),
            taylor_energy_coefficients=np.asarray(series.energy).tolist(),
            taylor_coefficient_convention='coefficient[k-1] = derivative(order=k)/k!, k=1,2; path t dimensionless',
            taylor_response_residuals=np.asarray(series.residual_norm).tolist(),
            taylor_fd_checks=path_checks, parameter_path=curves)
    for name, value in arrays.items():
        assert np.isfinite(value).all(), name
    record['elapsed_seconds'] = perf_counter()-started
    np.savez_compressed(OUTPUT/f'mesh_{mesh[0]}.npz', **arrays)
    print('P08 mesh', mesh, 'E', record['total_energy_hartree'], 'residual', record['forward_residual'], flush=True)
    return record


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'gradscf.json').write_text('{"complete": false}\n')
    records = [calculate((size, size, 3), detailed=size == 9) for size in [5, 7, 9, 11]]
    source = hashlib.sha256()
    for path in sorted(Path('src').rglob('*.py')):
        source.update(path.as_posix().encode()); source.update(path.read_bytes())
    report = dict(complete=True, model='Fixed-N periodic TF+lambda-vW+Hartree, no XC; manufactured local potential',
        length_bohr=LENGTH, lattice_bohr=(np.eye(3)*LENGTH).tolist(), nelectron=NUMBER,
        amplitude_modulation=list(AMPLITUDE_MODULATION), tf_weight=TF_WEIGHT, vw_weight=VW_WEIGHT,
        target_chemical_potential_hartree=CHEMICAL_POTENTIAL, cases=records,
        units=dict(density='electron/bohr^3', potential='Eh', energy='Eh', kinetic_weights='dimensionless'),
        hartree_convention='4*pi/G^2 for G!=0, zero at G=0; fixed uniform neutralizing background',
        spatial_scope='3D periodic box, exactly uniform in z; x/y collocation grids 5,7,9,11; not a material benchmark',
        source_python_sha256=source.hexdigest(), elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype='float64')
    (OUTPUT/'gradscf.json').write_text(json.dumps(report, indent=2)+'\n')
    print('P08 native fixed-number, parameter, implicit/DFPT/Taylor and FD gates PASS', flush=True)


if __name__ == '__main__':
    main()
