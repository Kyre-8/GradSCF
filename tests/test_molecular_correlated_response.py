"""Full reference and amplitude response precedes correlated spectroscopy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["mp2", "ccsd"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_correlated_geometry_matches_pyscf_and_gradient_differences(method, unrestricted):
    from gradscf import gto, scf
    from gradscf_tools.correlated_response import CorrelatedEnergy
    pytest.importorskip("pyscf")
    from pyscf import cc as pcc, gto as pgto, mp as pmp, scf as pscf

    atom = "H 0 0 0; H 0 0 1.5" if not unrestricted else "H 0 0 0; H 0 0 1.5; H 1.4 0 .2"
    basis = "6-31g" if not unrestricted else "sto-3g"
    spin = int(unrestricted)
    mol = gto.M(atom=atom, basis=basis, spin=spin, unit="Bohr", cart=True)
    mf = (scf.UHF if unrestricted else scf.RHF)(
        mol, integral_backend="native", conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, max_cycle=250).run()
    model = CorrelatedEnergy(mf, method)
    coordinates = model.coordinates
    value, gradient = jax.jit(jax.value_and_grad(model))(coordinates)
    assert np.isfinite(np.asarray(gradient)).all()
    reference = (pscf.UHF if unrestricted else pscf.RHF)(
        pgto.M(atom=atom, basis=basis, spin=spin, unit="Bohr", cart=True, verbose=0))
    reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-11
    reference.kernel(dm0=np.asarray(mf.make_rdm1()))
    assert reference.converged
    correlated = (pmp.MP2 if method == "mp2" else pcc.CCSD)(reference)
    if method == "ccsd":
        correlated.conv_tol, correlated.conv_tol_normt = 1e-12, 1e-11
        correlated.max_cycle = 200
    correlated.kernel()
    if method == "ccsd":
        assert correlated.converged
    np.testing.assert_allclose(value, correlated.e_tot, atol=3e-9, rtol=0)
    np.testing.assert_allclose(gradient, correlated.nuc_grad_method().kernel(), atol=2e-7, rtol=0)
    direction = np.random.default_rng(51).normal(size=coordinates.shape)
    direction /= np.linalg.norm(direction)
    derivative = jax.jit(jax.grad(model))
    hessian_vector = jax.jit(lambda r: jax.jvp(jax.grad(model), (r,), (jnp.array(direction),))[1])(coordinates)
    for step in (3e-4, 1e-4):
        finite = (derivative(coordinates+step*direction)-derivative(coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(hessian_vector, finite, atol=3e-6, rtol=2e-5)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8, rtol=0)


@pytest.mark.parametrize("method", ["mp2", "ccsd"])
def test_correlated_mixed_raman_response_matches_displaced_polarizability(method):
    from gradscf import gto, scf
    from gradscf_tools.correlated_response import CorrelatedEnergy

    mol = gto.M(atom="H 0 0 -.75; H 0 0 .75", basis="6-31g", unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    model = CorrelatedEnergy(mf, method)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    def energy(length_shift, field):
        return model(model.coordinates+length_shift*direction, jnp.array([0., 0., field]))
    polarizability = jax.jit(lambda length_shift: -jax.grad(jax.grad(energy, 1), 1)(length_shift, 0.))
    raman_derivative = jax.jit(jax.grad(polarizability))(0.)
    assert np.isfinite(float(raman_derivative))
    # Exercise the tensor JVP used by the production spectrum, independently
    # of the scalar reverse-over-reverse-over-reverse contraction above.
    def alpha_tensor(shift):
        return jax.jacfwd(model.dipole, argnums=1)(
            model.coordinates+shift*direction, jnp.zeros(3))
    tensor_derivative = jax.jit(lambda shift: jax.jvp(alpha_tensor, (shift,), (1.,))[1])(0.)
    assert np.isfinite(np.asarray(tensor_derivative)).all()
    np.testing.assert_allclose(tensor_derivative, tensor_derivative.T, atol=1e-9, rtol=0)
    np.testing.assert_allclose(tensor_derivative[2, 2], raman_derivative, atol=1e-8, rtol=0)
    for step in (3e-4, 1e-4):
        finite = (polarizability(step)-polarizability(-step))/(2*step)
        np.testing.assert_allclose(raman_derivative, finite, atol=3e-5, rtol=2e-5)
