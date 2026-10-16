"""CIS transition moments retain relaxed orbital and eigenvector response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry, singlet_cis_densities


@pytest.fixture(scope="module")
def hydrogen():
    pytest.importorskip("pyscf")
    coordinates = np.array([[.2, -.1, -.7], [.2, -.1, .7]])
    molecule = gto.M(atom=list(zip(("H", "H"), coordinates)), basis="6-31g*", unit="Bohr", cart=True)
    mf = scf.RHF(molecule, integral_backend="native", conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    assert mf.converged
    return coordinates, mf


def test_cis_transition_geometry_response_matches_independent_energy_state_fd(hydrogen):
    from pyscf import gto as pgto, scf as pscf
    from scipy.linalg import eigh

    coordinates, mf = hydrogen
    policy = EigenResponseConfig(target="eigenpairs",
        linear_config=LinearSolverConfig(method="direct", max_dense=8, rtol=1e-11))
    model = CISGeometry(mf, nroots=2, response=policy)
    values, jacobian, state = model.transition_derivatives(coordinates, 0)
    anchor = model.anchor(coordinates, state, 0)
    direction = np.array([[.1, -.2, -.6], [-.1, .2, .6]])
    direction /= np.linalg.norm(direction)
    q = float(values[:3]@values[:3])
    dq = 2*np.einsum("x,xai,ai", values[:3], jacobian[:3], direction)
    df = np.sum(jacobian[3]*direction)
    assert q > 1e-4 and abs(dq) > 1e-4

    def oracle(r):
        mol = pgto.M(atom=list(zip(("H", "H"), r)), basis="6-31g*", unit="Bohr", cart=True, verbose=0)
        ref = pscf.RHF(mol)
        ref.conv_tol, ref.conv_tol_grad = 1e-13, 1e-12
        ref.kernel()
        assert ref.converged
        td = ref.TDA()
        a, _ = td.get_ab()
        energies, vectors = eigh(a.reshape(3, 3))
        assert energies[1]-energies[0] > .01
        td.e = energies
        td.xy = [(x.reshape(1, 3)/np.sqrt(2), np.zeros((1, 3))) for x in vectors.T]
        mu = td.transition_dipole()[0]
        return np.array([mu@mu, td.oscillator_strength()[0]])

    np.testing.assert_allclose([q, values[3]], oracle(coordinates), atol=2e-8, rtol=0)
    for step in (1e-3, 3e-4):
        displaced = []
        for sign in (1, -1):
            r = coordinates+sign*step*direction
            value, current = model.transition_properties(r, 0)
            assert model.match(r, current, anchor)["root"] == 0
            displaced.append([value[:3]@value[:3], value[3]])
        fd = (np.asarray(displaced[0])-displaced[1])/(2*step)
        external = (oracle(coordinates+step*direction)-oracle(coordinates-step*direction))/(2*step)
        np.testing.assert_allclose([dq, df], fd, atol=2e-6, rtol=0)
        np.testing.assert_allclose([dq, df], external, atol=2e-6, rtol=0)


def test_transition_derivative_rejects_stopped_amplitudes(hydrogen):
    coordinates, mf = hydrogen
    model = CISGeometry(mf, nroots=2)
    with pytest.raises(ValueError, match="eigenpairs"):
        model.transition_derivatives(coordinates, 0)


def test_normalized_transition_probe_has_covariant_amplitude_and_orbital_tangents():
    rng = np.random.default_rng(106)
    c = jnp.asarray(np.linalg.qr(rng.normal(size=(5, 5)))[0])
    x = jnp.asarray(rng.normal(size=(2, 3)))
    position = jnp.asarray(rng.normal(size=(3, 5, 5)))
    position = .5*(position+position.transpose(0, 2, 1))

    def probe(coefficients, amplitude):
        transition = singlet_cis_densities(coefficients, amplitude, nocc=2).transition
        mu = -jnp.einsum("pq,xqp->x", transition, position)
        return mu@mu

    value, gradient = jax.value_and_grad(lambda amplitude: probe(c, amplitude))(x)
    np.testing.assert_allclose(jnp.sum(gradient*x), 0., atol=1e-11)
    np.testing.assert_allclose(probe(c, -x), value, atol=1e-12)
    u = jnp.asarray(np.linalg.qr(rng.normal(size=(2, 2)))[0])
    v = jnp.asarray(np.linalg.qr(rng.normal(size=(3, 3)))[0])
    rotated_c = jnp.column_stack([c[:, :2]@u, c[:, 2:]@v])
    rotated_x = u.T@x@v
    rotated_gradient = jax.grad(lambda a: probe(rotated_c, a))(rotated_x)
    np.testing.assert_allclose(probe(rotated_c, rotated_x), value, atol=1e-12)
    np.testing.assert_allclose(rotated_gradient, u.T@gradient@v, atol=1e-11)
    step = 1e-5
    direction = jnp.asarray(rng.normal(size=x.shape))
    np.testing.assert_allclose(jnp.sum(gradient*direction),
        (probe(c, x+step*direction)-probe(c, x-step*direction))/(2*step), atol=2e-8)

    def rotate(theta):
        # One occupied-virtual reference rotation, with X held fixed.
        rot = jnp.eye(5).at[1, 1].set(jnp.cos(theta)).at[3, 3].set(jnp.cos(theta))
        rot = rot.at[1, 3].set(jnp.sin(theta)).at[3, 1].set(-jnp.sin(theta))
        return probe(c@rot, x)

    np.testing.assert_allclose(jax.grad(rotate)(0.), (rotate(step)-rotate(-step))/(2*step), atol=2e-8)
