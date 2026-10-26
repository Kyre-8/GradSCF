"""LiH showcase gate: public periodic forces, strain convention and branch checks."""
import jax.numpy as jnp
import numpy as np


def test_lih_periodic_force_stress_and_displaced_branch():
    from examples.pbc.lih_forces_stress import make_mean_field

    settings = dict(a_bohr=7.6, xc='pbe', basis='gth-szv', pseudo='gth-pade',
        mesh=[21, 21, 21], kmesh=[1, 1, 1], precision=1e-10,
        displacement_bohr=[[0., 0., 0.], [.025, -.015, .01]])
    mean_field = make_mean_field(settings).run()
    assert mean_field.converged
    assert mean_field.cell.nelectron == 4
    np.testing.assert_array_equal(mean_field.cell.charges, [3., 1.])
    gradient = mean_field.nuc_grad_method()
    derivative = gradient.kernel()
    assert np.isfinite(derivative).all() and np.isfinite(gradient.stress()).all()
    np.testing.assert_allclose(gradient.result.energy, mean_field.e_tot, atol=1e-10, rtol=0)
    np.testing.assert_allclose(derivative, -gradient.result.forces, atol=0, rtol=0)
    assert np.max(abs(derivative)) > 1e-4
    coordinates, lattice = np.asarray(mean_field.cell.coords), np.asarray(mean_field.cell.lattice)
    direction = np.zeros((2, 3)); direction[1] = [.3, -.2, .1]
    strain = np.array([[.1, .2, 0.], [.2, -.15, .08], [0., .08, .05]])

    def energy(r, a, seed=None):
        calculation = make_mean_field(settings, coordinates=r, lattice=a)
        calculation.kernel(dm0=seed)
        assert calculation.converged
        return float(calculation.e_tot), np.asarray(calculation.make_rdm1())

    for step in [3e-4, 1e-4]:
        plus, _ = energy(coordinates+step*direction, lattice)
        minus, _ = energy(coordinates-step*direction, lattice)
        np.testing.assert_allclose(np.sum(derivative*direction), (plus-minus)/(2*step), atol=3e-6, rtol=0)
        positive, negative = np.eye(3)+step*strain, np.eye(3)-step*strain
        plus, _ = energy(coordinates@positive.T, lattice@positive.T)
        minus, _ = energy(coordinates@negative.T, lattice@negative.T)
        np.testing.assert_allclose(np.sum(gradient.stress()*strain)*np.linalg.det(lattice),
                                   (plus-minus)/(2*step), atol=5e-6, rtol=0)
    displaced = coordinates+1e-4*direction
    cold_energy, cold_density = energy(displaced, lattice)
    warm_energy, warm_density = energy(displaced, lattice, mean_field.make_rdm1())
    np.testing.assert_allclose(warm_energy, cold_energy, atol=1e-9, rtol=0)
    np.testing.assert_allclose(warm_density, cold_density, atol=2e-7, rtol=0)


def test_cubic_fractional_positions_and_kpoints_follow_lattice():
    from examples.pbc.lih_forces_stress import make_mean_field

    settings = dict(a_bohr=7.6, xc='hf', basis='gth-szv', pseudo='gth-pade',
                    mesh=[11]*3, kmesh=[2]*3, precision=1e-10)
    reference = make_mean_field(settings)
    strained = make_mean_field(dict(settings, a_bohr=7.6*1.03))
    fractions = np.asarray(reference.cell.coords)@np.linalg.inv(np.asarray(reference.cell.lattice))
    moved = np.asarray(strained.cell.coords)@np.linalg.inv(np.asarray(strained.cell.lattice))
    np.testing.assert_allclose(moved, fractions, atol=2e-15, rtol=0)
    original_k = reference.kpts@reference.cell.lattice.T/(2*jnp.pi)
    moved_k = strained.kpts@strained.cell.lattice.T/(2*jnp.pi)
    np.testing.assert_allclose(moved_k, original_k, atol=2e-15, rtol=0)
