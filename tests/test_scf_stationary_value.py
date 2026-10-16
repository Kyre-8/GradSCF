"""Stationary SCF energy keeps the metric/Pulay response and higher orders."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import minimize_uks_from_integrals


@pytest.mark.parametrize("mode,order", [("implicit", 1), ("implicit", 2),
                                       ("implicit", 3), ("explicit", 1)])
def test_metric_only_energy_derivatives_match_generalized_eigenvalue(mode, order):
    angle = .3
    c = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    h = jnp.array([[-1., .15], [.15, .5]])

    def energy(t):
        result = minimize_uks_from_integrals(
            overlap=jnp.array([[1., .1*t], [.1*t, 1.+.2*t]]), hcore=h,
            eri=jnp.zeros((2,)*4), nuclear_repulsion=0., ao=jnp.zeros((0, 2)),
            ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
            mo_coeff=np.stack([c, c]), mo_occ=np.array([[1., 0.], [0., 0.]]),
            gradient_mode=mode, orthonormalize_initial=True,
            max_iterations=80, gradient_tolerance=1e-10)
        return jnp.where(result.stationary, result.total_energy, jnp.nan)

    # det(h-E*S)=det(S)*E^2-b*E+det(h); only S varies.
    def oracle(t):
        determinant = 1.+.2*t-.01*t*t
        b = -.5-.23*t
        return (b-jnp.sqrt(b*b+4*determinant*.5225))/(2*determinant)

    np.testing.assert_allclose(energy(.17), oracle(.17), atol=1e-11, rtol=0)
    actual, expected = energy, oracle
    for _ in range(order):
        actual, expected = jax.grad(actual), jax.grad(expected)
    np.testing.assert_allclose(jax.jit(actual)(.17), expected(.17), atol=3e-8, rtol=2e-7)


def test_oh_pi_zero_mode_does_not_invalidate_first_energy_force():
    from gradscf import gto, scf
    from gradscf_tools.molecular_response import MolecularEnergy
    pytest.importorskip("pyscf")
    from pyscf import gto as reference_gto, scf as reference_scf

    atom = "O 0 0 0; H 0 0 1.85"
    mol = gto.M(atom=atom, basis="6-31g*", spin=1, unit="Bohr", cart=True)
    mf = scf.UHF(mol, conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                 max_cycle=250, integral_backend="native").run()
    stability = mf.stability(solver="dense", eigensolver_tol=1e-10)
    assert stability.stationary and abs(stability.minimum_curvature) < 1e-8
    energy = MolecularEnergy(mf, gradient_tolerance=1e-10)
    coordinates = energy.coordinates
    gradient = jax.jit(jax.grad(energy))(coordinates)
    assert np.isfinite(np.asarray(gradient)).all()
    reference = reference_scf.UHF(reference_gto.M(atom=atom, basis="6-31g*", spin=1,
                                                  unit="Bohr", cart=True, verbose=0))
    reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-11
    reference.kernel(dm0=np.asarray(mf.make_rdm1()))
    assert reference.converged
    np.testing.assert_allclose(gradient, reference.nuc_grad_method().kernel(), atol=1e-7, rtol=0)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    forward = jax.jit(energy)
    for step in (3e-4, 1e-4):
        finite = (forward(coordinates+step*direction)-forward(coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(finite, jnp.sum(gradient*direction), atol=2e-6, rtol=0)
