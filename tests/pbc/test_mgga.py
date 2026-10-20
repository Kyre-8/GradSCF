"""Tau-dependent periodic XC: LibXC values and variational response checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.pbc import _kpoint, gto
from gradscf.integrals.periodic.fft import build_inputs


def test_unrestricted_mgga_bundle_preserves_cross_spin_gradient_against_libxc():
    pytest.importorskip('jax_xc')
    libxc = pytest.importorskip('pyscf.dft.libxc')
    from gradscf.dft.xc import RestrictedFeatureBundle, eval_xc_energy_density_unrestricted
    # Hold both spin-gradient norms fixed while changing their relative angle.
    # Both Gram matrices are positive definite and nonparallel.
    rho_a, rho_b = jnp.full(2, .4), jnp.full(2, .3)
    aa, bb = jnp.full(2, .01), jnp.full(2, .02)
    ab = jnp.array([-.004, .006])
    tau_a, tau_b = jnp.full(2, .25), jnp.full(2, .19)
    def energy(cross):
        features = RestrictedFeatureBundle(rho_a, rho_b, aa, cross, bb, tau_a, tau_b)
        return eval_xc_energy_density_unrestricted('r2scan', features)
    alpha_gradient = np.zeros((2, 3));alpha_gradient[:, 0] = np.sqrt(np.asarray(aa))
    beta_gradient = np.zeros((2, 3));beta_gradient[:, 0] = np.asarray(ab)/alpha_gradient[:, 0]
    beta_gradient[:, 1] = np.sqrt(np.asarray(bb)-beta_gradient[:, 0]**2)
    fields = np.zeros((2, 6, 2))
    fields[:, 0] = np.asarray([rho_a, rho_b])
    fields[0, 1:4] = alpha_gradient.T;fields[1, 1:4] = beta_gradient.T
    fields[:, 5] = np.asarray([tau_a, tau_b])
    epsilon, potential, _, _ = libxc.eval_xc('R2SCAN', fields, spin=1, deriv=1)
    actual = energy(ab)
    np.testing.assert_allclose(actual, epsilon*np.asarray(rho_a+rho_b), rtol=2e-9, atol=1e-11)
    assert abs(actual[0]-actual[1]) > 1e-8
    derivative = jax.grad(lambda cross: jnp.sum(energy(cross)))(ab)
    np.testing.assert_allclose(derivative, potential[1][:, 1], rtol=2e-8, atol=1e-10)


def test_r2scan_spin_fields_match_libxc():
    pytest.importorskip('jax_xc')
    libxc = pytest.importorskip('pyscf.dft.libxc')
    from gradscf.scf.uks import _point_unrestricted_xc_value_and_grad_kernel
    # Nonparallel spin gradients expose adapters that retain only their norms.
    variables = jnp.array([[.4, .3, .03, -.02, .01, -.01, .04, .02, .25, .19],
                           [.1, .2, -.01, .02, .01, .02, -.01, .03, .08, .15]])
    evaluate = _point_unrestricted_xc_value_and_grad_kernel('r2scan', 'MGGA')
    value, potential = evaluate(variables)
    rho = np.zeros((2, 6, len(variables)))
    for spin in range(2):
        rho[spin, 0] = variables[:, spin]
        rho[spin, 1:4] = variables[:, 2+3*spin:5+3*spin].T
        rho[spin, 5] = variables[:, 8+spin]
    exc, vxc, _, _ = libxc.eval_xc('R2SCAN', rho, spin=1, deriv=1)
    np.testing.assert_allclose(value, exc*np.asarray(variables[:, :2].sum(axis=1)), rtol=2e-9, atol=1e-11)
    expected = np.zeros_like(variables)
    expected[:, :2] = vxc[0]
    ga, gb = np.asarray(variables[:, 2:5]), np.asarray(variables[:, 5:8])
    expected[:, 2:5] = 2*vxc[1][:, 0, None]*ga+vxc[1][:, 1, None]*gb
    expected[:, 5:8] = 2*vxc[1][:, 2, None]*gb+vxc[1][:, 1, None]*ga
    expected[:, 8:] = vxc[3]
    np.testing.assert_allclose(potential, expected, rtol=2e-8, atol=1e-10)
    direction = jnp.full(variables.shape, .013)
    hvp = jax.jvp(lambda v: evaluate(v)[1], (variables,), (direction,))[1]
    step = 1e-4
    finite = (evaluate(variables+step*direction)[1]-evaluate(variables-step*direction)[1])/(2*step)
    np.testing.assert_allclose(hvp, finite, rtol=2e-6, atol=2e-9)


def test_r2scan_periodic_potential_and_zero_q_kernel_are_variational():
    from gradscf.pbc.response import q_kernel
    from gradscf.pbc.momentum import momentum_transfer
    cell = gto.M(atom='H .2 .3 .4; H 1.6 .6 .8', a=np.eye(3)*6,
                 unit='Bohr', mesh=(9,)*3)
    kpoints = cell.make_kpts((3, 1, 1))
    inputs = build_inputs(cell, kpts=kpoints)
    density = jnp.broadcast_to(jnp.eye(2)*.2, (2, 3, 2, 2)).astype(complex)
    tangent = jnp.broadcast_to(jnp.array([[.1, .03+.02j], [.03-.02j, -.07]]), density.shape)
    energy = lambda d: _kpoint.grid_xc_potential(d, inputs, xc='r2scan')[0]
    _, fields = _kpoint.grid_xc_potential(density, inputs, xc='r2scan')
    potential = _kpoint.project_xc_potential(inputs.ao, inputs.weights, fields, 'MGGA')
    derivative = jax.jvp(energy, (density,), (tangent,))[1]
    expected = jnp.einsum('k,skij,skji->', inputs.kweights, potential, tangent).real
    np.testing.assert_allclose(derivative, expected, atol=1e-11)
    transfer = momentum_transfer(np.asarray(kpoints)@np.asarray(cell.lattice).T/(2*np.pi), [0., 0., 0.])
    action = q_kernel(inputs, density, transfer, mesh=cell.mesh, xc='r2scan')
    expected = jax.jvp(lambda d: _kpoint.energy_and_fock(d, inputs, mesh=cell.mesh, xc='r2scan')[1],
                       (density,), (tangent,))[1]
    np.testing.assert_allclose(action(tangent), expected, atol=2e-10)


@pytest.mark.parametrize('nk,temperature', [(1, None), (3, .05)])
def test_r2scan_periodic_scf_and_bands(nk, temperature):
    from gradscf.pbc import dft
    cell = gto.M(atom='H .2 .3 .4; H 1.6 .6 .8', a=np.eye(3)*6,
                 unit='Bohr', mesh=(11,)*3)
    cls = dft.RKS if nk == 1 else dft.KRKS
    mf = cls(cell, kpts=cell.make_kpts((nk, 1, 1)), xc='r2scan', temperature=temperature).run(
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9, damping=.2)
    assert mf.converged
    energies, _ = mf.get_bands(mf.kpts)
    np.testing.assert_allclose(energies, mf.mo_energy, atol=2e-8)
