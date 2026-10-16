"""Independent P08 TF/vW density Hessian and fixed-number Fourier/KKT oracle.

Only NumPy/SciPy and analytic expressions are used. No GradSCF physics, AD
derivatives or optimizer participates in this comparison.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve
from numpy.polynomial.legendre import leggauss


OUTPUT = Path('reproducibility/ad_showcase/P08_ofdft_parameter_response')
C_TF = .3*(3*np.pi**2)**(2/3)


def fourier_operators(lattice, mesh):
    """Independent dense matrices for the declared small odd collocation grid."""
    fractions = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    integers = np.stack(np.meshgrid(*[np.fft.fftfreq(n)*n for n in mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    coordinates = fractions@lattice
    vectors = integers@(2*np.pi*np.linalg.inv(lattice).T)
    squared = np.sum(vectors*vectors, axis=1)
    transform = np.exp(1j*coordinates@vectors.T)/np.sqrt(len(coordinates))
    kernel = np.divide(4*np.pi, squared, out=np.zeros_like(squared), where=squared > 0)
    laplacian = ((transform*(-squared))@transform.conj().T).real
    hartree = ((transform*kernel)@transform.conj().T).real
    return laplacian, hartree


def independent_response(lattice, mesh, density, tf, vw, *, kind, potential=None):
    """Return d(rho), d(mu), residual for a potential or TF/vW-weight tangent."""
    laplacian, hartree = fourier_operators(lattice, mesh)
    phi = np.sqrt(density)
    lap_phi = laplacian@phi
    hessian = (np.diag((10/9)*tf*C_TF*density**(-1/3))+hartree
               -.25*vw*laplacian/phi[:, None]/phi[None, :]
               +np.diag(.25*vw*lap_phi/phi**3))
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-11, rtol=0)
    if kind == 'potential':
        source = np.asarray(potential)
    elif kind == 'tf':
        source = (5/3)*C_TF*density**(2/3)
    elif kind == 'vw':
        source = -.5*lap_phi/phi
    else:
        raise ValueError('Choose potential, tf or vw.')
    size = density.size
    bordered = np.zeros((size+1, size+1))
    bordered[:size, :size] = hessian
    bordered[:size, size] = -1.
    bordered[size, :size] = 1.
    solution = solve(bordered, np.r_[-source, 0.], assume_a='gen')
    response, chemical = solution[:-1], solution[-1]
    residual = float(np.linalg.norm(hessian@response+source-chemical))
    assert residual < 1e-9 and abs(np.sum(response)) < 1e-11
    return response, chemical, residual


def continuum_energy(record, quadrature_order):
    """Analytic Fourier Hartree plus independent 2D Gauss-Legendre TF integral."""
    length, number = record['length_bohr'], record['nelectron']
    b, c = record['amplitude_modulation']
    volume, k2 = length**3, (2*np.pi/length)**2
    scale2 = (number/volume)/(1+(b*b+c*c)/2)
    nodes, weights = leggauss(quadrature_order)
    phase = np.pi*(nodes+1)
    q = 1+b*np.cos(phase)[:, None]+c*np.cos(phase)[None, :]
    tf = C_TF*volume*np.sum(np.outer(weights, weights)*(scale2*q*q)**(5/3))/4
    amplitudes = scale2*np.array([2*b, 2*c, b*b/2, c*c/2, b*c, b*c])
    hartree = np.pi*volume*np.sum(amplitudes**2/(k2*np.array([1, 1, 4, 4, 2, 2])))
    # For this manufactured potential the vW kinetic/external terms cancel.
    return record['target_chemical_potential_hartree']*number-(2/3)*record['tf_weight']*tf-hartree


def main():
    started = perf_counter()
    native_path = OUTPUT/'gradscf.json'
    record = json.loads(native_path.read_text())
    assert record['complete']
    continuum = [continuum_energy(record, order) for order in [48, 80]]
    assert abs(continuum[1]-continuum[0]) < 1e-12
    tf, vw = record['tf_weight'], record['vw_weight']
    lattice = np.asarray(record['lattice_bohr'])
    cases, map_grid_increments = [], []
    previous_coefficients = None
    for case in record['cases']:
        mesh = tuple(case['mesh'])
        arrays = np.load(OUTPUT/f'mesh_{mesh[0]}.npz')
        assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
        coordinates, weights = arrays['coordinates_bohr'], arrays['weights']
        length, number = record['length_bohr'], record['nelectron']
        b, c = record['amplitude_modulation']
        scale = np.sqrt((number/length**3)/(1+(b*b+c*c)/2))
        phi = scale*(1+b*np.cos(2*np.pi*coordinates[:, 0]/length)+c*np.cos(2*np.pi*coordinates[:, 1]/length))
        rho = phi*phi
        np.testing.assert_allclose(arrays['exact_density'], rho, atol=1e-15, rtol=0)
        laplacian, hartree = fourier_operators(lattice, mesh)
        vh, lap_phi = hartree@rho, laplacian@phi
        expected_potential = (record['target_chemical_potential_hartree']-(5/3)*tf*C_TF*rho**(2/3)
                              +.5*vw*lap_phi/phi-vh)
        np.testing.assert_allclose(arrays['external_potential'], expected_potential, atol=2e-12, rtol=0)
        np.testing.assert_allclose(arrays['density'], rho, atol=2e-8, rtol=2e-7)
        tf_energy = C_TF*np.sum(weights*rho**(5/3))
        vw_energy = -.5*np.sum(weights*phi*lap_phi)
        hartree_energy = .5*np.sum(weights*rho*vh)
        external_energy = np.sum(weights*rho*expected_potential)
        total = tf*tf_energy+vw*vw_energy+hartree_energy+external_energy
        np.testing.assert_allclose(case['total_energy_hartree'], total, atol=2e-9, rtol=0)
        expected, chemical, residual = independent_response(lattice, mesh, rho, tf, vw,
                                                            kind='potential', potential=arrays['probe'])
        np.testing.assert_allclose(arrays['potential_density_response'], expected, atol=3e-8, rtol=3e-6)
        row = dict(mesh=list(mesh), exact_discrete_energy_hartree=float(total),
            energy_error=float(abs(case['total_energy_hartree']-total)),
            continuum_energy_error=float(abs(case['total_energy_hartree']-continuum[-1])),
            maximum_potential_response_error=float(np.max(abs(arrays['potential_density_response']-expected))),
            independent_kkt_residual=residual, chemical_potential_response=float(chemical),
            potential_energy_curvature=float(np.sum(weights*arrays['probe']*expected)))
        tf_response = independent_response(lattice, mesh, rho, tf, vw, kind='tf')[0]
        vw_response = independent_response(lattice, mesh, rho, tf, vw, kind='vw')[0]
        jacobian = np.stack([expected, tf_response, vw_response], axis=1)
        coefficients = np.fft.fftshift(np.fft.fftn(jacobian.reshape(*mesh, 3), axes=(0, 1, 2)), axes=(0, 1, 2))/rho.size
        if previous_coefficients is not None:
            width = (mesh[0]-previous_coefficients.shape[0])//2
            padded = np.pad(previous_coefficients, ((width, width), (width, width), (0, 0), (0, 0)))
            # Parseval compares entire periodic trigonometric interpolants on
            # the same cell, despite nonnested sample coordinates. The volume
            # factor cancels in these relative L2 norms; this is an increment,
            # not a rigorous bound on the unknown continuum response error.
            change = np.linalg.norm((coefficients-padded).reshape(-1, 3), axis=0)/np.linalg.norm(coefficients.reshape(-1, 3), axis=0)
            map_grid_increments.append(dict(target_mesh=list(mesh), relative_l2_by_parameter=change.tolist()))
        previous_coefficients = coefficients
        if 'density_jacobian' in arrays:
            np.testing.assert_allclose(arrays['density_jacobian'], jacobian, atol=3e-8, rtol=3e-6)
            gradient = np.array([np.sum(weights*rho*arrays['probe']), tf_energy, vw_energy])
            np.testing.assert_allclose(case['energy_gradient'], gradient, atol=2e-8, rtol=2e-7)
            source = np.stack([arrays['probe'], (5/3)*C_TF*rho**(2/3), -.5*lap_phi/phi], axis=1)
            hessian = source.T@(weights[:, None]*jacobian)
            np.testing.assert_allclose(hessian, hessian.T, atol=2e-10, rtol=0)
            assert np.linalg.eigvalsh(hessian)[-1] < 1e-9
            direction = np.asarray(case['taylor_path_direction'])
            expected_series = np.array([gradient@direction, .5*direction@hessian@direction])
            np.testing.assert_allclose(case['taylor_energy_coefficients'], expected_series, atol=3e-9, rtol=3e-6)
            row.update(maximum_parameter_jacobian_error=float(np.max(abs(arrays['density_jacobian']-jacobian))),
                energy_gradient_error=float(np.max(abs(np.asarray(case['energy_gradient'])-gradient))),
                taylor_energy_coefficients=expected_series.tolist(),
                taylor_energy_coefficient_error=float(np.max(abs(np.asarray(case['taylor_energy_coefficients'])-expected_series))),
                constrained_energy_hessian=hessian.tolist())
            np.savez_compressed(OUTPUT/'independent_response.npz', density_jacobian=jacobian)
        cases.append(row)
    response_increments = [abs(b['potential_energy_curvature']-a['potential_energy_curvature'])/abs(b['potential_energy_curvature'])
                           for a, b in zip(cases[:-1], cases[1:])]
    assert response_increments[-1] < 1e-5
    assert max(map_grid_increments[-1]['relative_l2_by_parameter']) < 2e-5
    assert cases[-1]['continuum_energy_error'] < 1e-8
    assert cases[-1]['continuum_energy_error'] < cases[0]['continuum_energy_error']*.01
    result = dict(passed=True, native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        method='Analytic TF/vW density Hessian; independent NumPy Fourier matrices and SciPy fixed-N bordered solve',
        continuum_energy_hartree=continuum[-1], continuum_quadrature_orders=[48, 80],
        continuum_quadrature_difference=abs(continuum[1]-continuum[0]), cases=cases,
        relative_curvature_grid_increments=response_increments,
        density_map_grid_increments=map_grid_increments,
        density_map_grid_definition='Relative L2 change of complete Fourier interpolants; parameter order potential,tf,vw; not a continuum error bound',
        input_npz_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(OUTPUT.glob('mesh_*.npz'))},
        elapsed_seconds=perf_counter()-started, python=platform.python_version(), hostname=platform.node())
    (OUTPUT/'independent.json').write_text(json.dumps(result, indent=2)+'\n')
    print('P08 independent analytic/Fourier/KKT/continuum/grid gates PASS', flush=True)
    print('last density-response grid increment:', response_increments[-1], flush=True)


if __name__ == '__main__':
    main()
