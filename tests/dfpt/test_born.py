"""Zero-field Berry Born response through the existing projector DFPT chart."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_born_problem_includes_density_metric_ao_and_ionic_response():
    from gradscf.dfpt.electric import born_problem
    from gradscf.pbc import gto, scf
    from gradscf.pbc.polarization import make_plan, polarization
    from gradscf.solvers import LinearSolverConfig
    coordinates = np.array([[.4, .5, .6], [1.8, .8, .9]])
    def calculate(r):
        cell = gto.M(atom=[('H', p) for p in r], a=np.eye(3)*6., unit='Bohr', mesh=(9,)*3)
        return scf.KRHF(cell, kpts=cell.make_kpts((3, 1, 1))).run(
            conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150, damping=.1)
    mf = calculate(coordinates)
    assert mf.converged
    plan = make_plan(np.asarray(mf.kpts)@np.asarray(mf.cell.lattice).T/(2*np.pi))
    original = polarization(mf.inputs, mf.result, plan, nocc=mf.cell.nelec)
    problem = born_problem(mf)
    assert problem._base()[1]
    np.testing.assert_allclose(problem.quantities(problem.solution, problem.inputs)['polarization'], original.total, atol=2e-12)
    direction = jnp.zeros_like(mf.cell.coords).at[1, 0].set(1.)
    config = LinearSolverConfig(method='direct', rtol=1e-10, atol=1e-12, max_dense=128)
    response = jax.jit(lambda d:problem.response(tangent=d, config=config))(direction)
    assert response.converged
    actual = np.asarray(response.values['polarization'])
    assert np.max(abs(actual)) > 1e-5
    for step in (1e-4, 3e-5):
        values, densities = [], []
        for sign in (1., -1.):
            shifted = calculate(coordinates+sign*step*np.asarray(direction))
            assert shifted.converged
            result = polarization(shifted.inputs, shifted.result, plan, nocc=mf.cell.nelec,
                                  reference_phases=original.phases)
            values.append(np.asarray(result.total))
            densities.append(np.asarray(shifted.result.density_spin))
        finite = (values[0]-values[1])/(2*step)
        finite_density = (densities[0]-densities[1])/(2*step)
        print('H2 Berry dP/dR', 'step', step, 'polarization error', np.max(abs(actual-finite)),
              'density error', np.max(abs(np.asarray(response.density)-finite_density)))
        np.testing.assert_allclose(actual, finite, atol=2e-7, rtol=2e-5)
        np.testing.assert_allclose(response.density, finite_density, atol=3e-6, rtol=1e-5)
    mf.temperature = .01
    with pytest.raises(NotImplementedError, match='zero'):
        born_problem(mf)
