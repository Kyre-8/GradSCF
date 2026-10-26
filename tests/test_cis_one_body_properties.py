"""Unrelaxed CIS densities and NTOs: fermionic normalization and gauge gates."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf_tools import excited


def test_singlet_cis_densities_match_independent_fermionic_rdm_and_transition():
    pytest.importorskip("pyscf")
    from pyscf.ci import cisd
    from pyscf.fci import cistring, direct_spin1

    x = np.array([[.7, -.2], [.1, .5]])
    x /= np.linalg.norm(x)
    result = excited.singlet_cis_densities(jnp.eye(4), jnp.asarray(x), nocc=2)
    cisd_vector = cisd.amplitudes_to_cisdvec(0., x/np.sqrt(2), np.zeros((2, 2, 2, 2)))
    full = cisd.to_fcivec(cisd_vector, 4, (2, 2))
    np.testing.assert_allclose(np.sum(full**2), 1., atol=1e-14)
    reference = np.zeros_like(full)
    reference[0, 0] = 1.
    density = direct_spin1.make_rdm1(full, 4, (2, 2))
    transition = direct_spin1.trans_rdm1(full, reference, 4, (2, 2))
    np.testing.assert_allclose(result.unrelaxed, density, atol=1e-13)
    np.testing.assert_allclose(result.transition, transition, atol=1e-13)
    # Reuse the existing GradSCF fermionic-density API as another convention check.
    space = ci.make_ci_space(4, 2, max_excitation=1)
    strings = list(map(int, cistring.make_strings(range(4), 2)))
    vector = jnp.asarray([full[strings.index(d & 15), strings.index(d >> 4)]
                          for d in space.determinants])
    np.testing.assert_allclose(result.unrelaxed, ci.make_rdm1(vector, space), atol=1e-13)
    np.testing.assert_allclose(np.trace(result.attachment), 1., atol=1e-13)
    np.testing.assert_allclose(np.trace(result.detachment), 1., atol=1e-13)
    np.testing.assert_allclose(np.sum(result.transition**2), 2., atol=1e-13)
    np.testing.assert_allclose(np.trace(result.transition), 0., atol=1e-13)


def test_cis_one_body_fields_are_invariant_to_orbital_gauge_and_state_scale():
    rng = np.random.default_rng(105)
    c, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    u, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    v, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    x = rng.normal(size=(2, 3))
    original = excited.singlet_cis_densities(jnp.asarray(c), jnp.asarray(x), nocc=2)
    rotated_c = np.column_stack([c[:, :2]@u, c[:, 2:]@v])
    rotated = excited.singlet_cis_densities(jnp.asarray(rotated_c), jnp.asarray(u.T@x@v), nocc=2)
    for left, right in zip(original, rotated):
        np.testing.assert_allclose(left, right, atol=1e-12)
    flipped = excited.singlet_cis_densities(jnp.asarray(c), -3*jnp.asarray(x), nocc=2)
    for name in ("ground", "unrelaxed", "attachment", "detachment"):
        np.testing.assert_allclose(getattr(original, name), getattr(flipped, name), atol=1e-12)
    np.testing.assert_allclose(original.transition, -flipped.transition, atol=1e-12)
    probe = jnp.asarray(rng.normal(size=(5, 5)))
    scalar = lambda t: jnp.sum(excited.singlet_cis_densities(
        jnp.asarray(c), jnp.asarray(x)+t*jnp.ones_like(jnp.asarray(x)), nocc=2).transition*probe)**2
    actual = jax.jit(jax.grad(scalar))(0.)
    step = 1e-5
    np.testing.assert_allclose(actual, (scalar(step)-scalar(-step))/(2*step), atol=1e-8)


@pytest.mark.parametrize("degenerate", [False, True])
def test_cis_ntos_reconstruct_transition_and_weighted_densities(degenerate):
    x = np.eye(2) if degenerate else np.array([[.8, .2], [-.1, .3]])
    c = np.eye(4)
    fields = excited.singlet_cis_densities(jnp.asarray(c), jnp.asarray(x), nocc=2)
    ntos = excited.singlet_cis_ntos(c, x, nocc=2)
    np.testing.assert_allclose(np.sum(ntos.weights), 1., atol=1e-14)
    np.testing.assert_allclose(ntos.holes.T@ntos.holes, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(ntos.particles.T@ntos.particles, np.eye(2), atol=1e-14)
    reconstructed = np.sqrt(2)*(ntos.holes*np.sqrt(ntos.weights))@ntos.particles.T
    np.testing.assert_allclose(reconstructed, fields.transition, atol=1e-13)
    np.testing.assert_allclose((ntos.holes*ntos.weights)@ntos.holes.T, fields.detachment, atol=1e-13)
    np.testing.assert_allclose((ntos.particles*ntos.weights)@ntos.particles.T, fields.attachment, atol=1e-13)
    assert ntos.degenerate_groups == (((0, 1),) if degenerate else ())
    # A degenerate pair has no unique individual orbital frame. Only the
    # reconstructed operator and complete weighted projectors are compared.


def test_cis_one_body_properties_reject_zero_norm_or_complex_amplitudes():
    invalid = excited.singlet_cis_densities(jnp.eye(4), jnp.zeros((2, 2)), nocc=2)
    assert np.isnan(invalid.unrelaxed).all()
    with pytest.raises(ValueError, match="nonzero"):
        excited.singlet_cis_ntos(np.eye(4), np.zeros((2, 2)), nocc=2)
    with pytest.raises(NotImplementedError, match="real"):
        excited.singlet_cis_densities(jnp.eye(4), jnp.ones((2, 2), dtype=jnp.complex128), nocc=2)
