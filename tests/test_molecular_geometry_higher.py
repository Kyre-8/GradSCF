"""A complete relaxed molecular cubic/quartic gate beyond integral products."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf import gto, scf
from gradscf_tools.molecular_response import MolecularEnergy


def test_h2_relaxed_cubic_quartic_against_independent_hessian_differences():
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="6-31g", unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-13,
                 conv_tol_density=1e-12, conv_tol_grad=1e-11).run()
    energy = MolecularEnergy(mf, gradient_tolerance=1e-11)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    first = jax.grad(lambda q: energy(energy.coordinates+q*direction))
    second = lambda q: jax.jvp(first, (q,), (jnp.ones_like(q),))[1]
    third = lambda q: jax.jvp(second, (q,), (jnp.ones_like(q),))[1]
    fourth = lambda q: jax.jvp(third, (q,), (jnp.ones_like(q),))[1]
    actual = np.array([jax.jit(third)(jnp.asarray(0.)), jax.jit(fourth)(jnp.asarray(0.))])
    assert np.isfinite(actual).all()
    # Independent PySCF 2.13.0 analytic relaxed Hessians, R=1.4 +/- .0003 bohr;
    # CPHF tol1e-12, true orbital-gradient norm <1.1e-12. These are finite-
    # difference references, not exact cubic/quartic analytic PySCF outputs.
    # The full three-step convergence table lives in the G08 h2/pyscf.json.
    expected = [-1.3194751255899788, 4.428751247213351]
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
