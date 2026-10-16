"""Scalar orbital objectives reuse the physical energy without unused Vxc AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.uks import UKSConfig, _raw_fock_and_energy_for_state


@pytest.mark.parametrize("xc", ["hf", "pbe", "wb97xd"])
def test_energy_only_matches_full_energy_gradient_and_hvp(xc):
    rng = np.random.default_rng(941)
    ao = jnp.asarray(rng.normal(scale=.2, size=(8, 3))).at[:, 0].add(1.)
    dao = jnp.concatenate([ao[None], jnp.asarray(rng.normal(scale=.1, size=(3, 8, 3)))])
    factor = jnp.array([[.2, .03, .02], [.03, .12, -.01], [.02, -.01, .09]])
    eri = jnp.einsum("pq,rs->pqrs", factor, factor)
    h = jnp.diag(jnp.array([-2.3, -.7, .9]))
    config = UKSConfig(xc_spec=xc)
    extra = dict(eri_lr=.4*eri, dispersion_energy=jnp.asarray(-.002)) if xc == "wb97xd" else {}

    def evaluate(p, compute_potential):
        da = jnp.diag(jnp.array([.7, .3, .1])).at[0, 1].set(p[0]).at[1, 0].set(p[0])
        db = jnp.diag(jnp.array([.5, .2, .08])).at[0, 2].set(p[1]).at[2, 0].set(p[1])
        return _raw_fock_and_energy_for_state(density_a=da, density_b=db, ao=ao,
            ao_deriv1=dao, weights=jnp.linspace(.03, .06, 8), h=h, eri=eri, df_factors=None,
            enuc=jnp.asarray(.23), alpha=jnp.asarray(float(xc == "hf")), cfg=config,
            xc_kind="HF" if xc == "hf" else "GGA", compute_potential=compute_potential, **extra)

    point, direction = jnp.array([.08, .04]), jnp.array([.3, -.2])
    full, scalar = evaluate(point, True), evaluate(point, False)
    assert scalar[2] is scalar[3] is None
    assert full[2].shape == full[3].shape == (3, 3)
    np.testing.assert_allclose(scalar[:2], full[:2], atol=2e-14, rtol=2e-14)
    derivatives = []
    for compute_potential in (True, False):
        energy = lambda p: evaluate(p, compute_potential)[0]
        gradient = jax.jit(jax.grad(energy))(point)
        hvp = jax.jit(lambda p, d: jax.jvp(jax.grad(energy), (p,), (d,))[1])(point, direction)
        derivatives.append((gradient, hvp))
    for full_value, scalar_value in zip(*derivatives):
        np.testing.assert_allclose(scalar_value, full_value, atol=3e-12, rtol=3e-11)
