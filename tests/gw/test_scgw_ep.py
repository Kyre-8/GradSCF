"""Fixed-phonon scGW: matrix residuals and reconverged implicit response."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw import scgw_matsubara_restricted
from gradscf.scf.autodiff import SCFDifferentiationConfig


def inputs():
    return dict(mo_energy=jnp.array([-.6, .7]), mo_coeff=jnp.eye(2), nocc=1,
                hcore_matrix=jnp.array([[-.55, .03], [.03, .65]]),
                df_factors=jnp.array([[[.12, .05], [.05, .09]]]),
                nw=8, beta=5., max_iter=100, tol=2e-10, particle_tol=1e-12)


def phonons(scale=1.):
    from gradscf.gw.ep_coupling import PhononModel
    return PhononModel(jnp.array([.12, .25]),
        scale * jnp.array([[[.025, .018], [.018, -.013]], [[.01, -.02], [-.02, .017]]]))


def test_zero_coupling_recovers_electronic_scgw():
    kw = inputs()
    bare = scgw_matsubara_restricted(**kw)
    coupled = scgw_matsubara_restricted(**kw, phonons=phonons(0.))
    for name in ('green_iw', 'self_energy_iw', 'sigma_moment', 'density_mo',
                 'fock_mo', 'chemical_potential', 'correlation_energy'):
        np.testing.assert_allclose(getattr(coupled, name), getattr(bare, name), atol=1e-12)
    assert coupled.total_energy is None  # No coupled electron-phonon energy functional supplied.
    np.testing.assert_allclose(coupled.electronic_energy + kw.get('nuclear_repulsion', 0.),
                               bare.total_energy, atol=1e-12)


def test_debye_waller_static_density_and_tail():
    from gradscf.gw.ep_coupling import PhononModel
    kw = inputs()
    kw['df_factors'] = jnp.zeros((1, 2, 2))
    kw['hcore_matrix'] = jnp.diag(jnp.array([-.5, .7]))
    quadratic = jnp.array([[[[.006, 0.], [0., -.004]]]])
    model = PhononModel(jnp.array([.2]), jnp.zeros((1, 2, 2)), quadratic)
    out = scgw_matsubara_restricted(**kw, phonons=model)
    dw = .5 * quadratic[0, 0] / np.tanh(kw['beta'] * .2 / 2)
    levels = jnp.diag(kw['hcore_matrix'] + dw)
    mu = levels.mean()
    occupations = 2 / (1 + np.exp(kw['beta'] * (levels - mu)))
    np.testing.assert_allclose(out.fock_mo, kw['hcore_matrix'] + dw, atol=1e-9)
    np.testing.assert_allclose(jnp.diag(out.density_mo), occupations, atol=1e-9)
    np.testing.assert_allclose(out.debye_waller, dw, atol=1e-12)
    np.testing.assert_allclose(out.sigma_moment, 0., atol=1e-12)


def observables(out):
    return jnp.array([out.chemical_potential, out.density_mo[0, 1],
                      out.self_energy_iw[-1, 0, 0].imag, out.sigma_moment[0, 0]])


@pytest.mark.parametrize('variable', ['coupling', 'frequency', 'quadratic'])
def test_joint_response_matches_reconverged_finite_difference(variable):
    from dataclasses import replace
    kw = inputs()
    config = SCFDifferentiationConfig(tolerance=1e-10, max_iter=60, restart=30)
    base = phonons()
    def run(t, response):
        if variable == 'coupling':
            model = replace(base, couplings=base.couplings.at[0, 0, 1].add(t).at[0, 1, 0].add(t))
        elif variable == 'frequency':
            model = replace(base, energies=base.energies.at[0].add(t))
        else:
            model = replace(base, quadratic=jnp.zeros((2, 2, 2, 2)).at[0, 0].set(
                t * jnp.array([[.2, .1], [.1, -.2]])))
        return observables(scgw_matsubara_restricted(**kw, phonons=model, differentiation=response))
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    value, tangent = jax.jit(lambda t: jax.jvp(lambda x: run(x, config), (t,), (jnp.ones_like(t),)))(jnp.array(0.))
    np.testing.assert_allclose(value, run(0., None), atol=1e-9)
    np.testing.assert_allclose(tangent, fd, atol=5e-7, rtol=1e-4)
    backward = jax.jit(jax.grad(lambda t: jnp.sum(run(t, config))))(0.)
    np.testing.assert_allclose(backward, tangent.sum(), atol=1e-8)


def test_joint_equations_and_fixed_frame_covariance():
    from dataclasses import replace
    from gradscf.gw.scgw import _scgw_step
    kw, model = inputs(), phonons()
    out = scgw_matsubara_restricted(**kw, phonons=model)
    mapped = _scgw_step(out.green_iw, out.fock_mo, out.chemical_potential,
        kw['df_factors'], kw['hcore_matrix'], out.density_mo, out.grid,
        out.sigma_moment, model)
    np.testing.assert_allclose(mapped['sigma_iw'], out.self_energy_iw, atol=kw['tol'])
    np.testing.assert_allclose(mapped['sigma_moment'], out.sigma_moment, atol=kw['tol'])
    np.testing.assert_allclose(jnp.trace(out.density_mo), 2., atol=1e-11)
    assert jnp.max(jnp.abs(out.phonon_self_energy_iw)) > 1e-4
    angle = .31
    u = jnp.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = replace(model, couplings=jnp.einsum('pi,lpq,qj->lij', u, model.couplings, u))
    other = scgw_matsubara_restricted(**{**kw, 'mo_coeff': u}, phonons=rotated)
    np.testing.assert_allclose(other.density_matrix, out.density_matrix, atol=2e-9)
    np.testing.assert_allclose(other.chemical_potential, out.chemical_potential, atol=2e-9)


def test_ep_response_rejects_nonconvergence_and_invalid_modes():
    from dataclasses import replace
    kw = inputs()
    config = SCFDifferentiationConfig()
    with pytest.raises(Exception, match='scGW.*converge'):
        jax.jit(lambda g: scgw_matsubara_restricted(**{**kw, 'max_iter': 1},
            phonons=phonons(g), differentiation=config).density_mo)(1.).block_until_ready()
    with pytest.raises(Exception, match='positive'):
        jax.jit(lambda w: scgw_matsubara_restricted(**kw,
            phonons=replace(phonons(), energies=jnp.array([w, .25])),
            differentiation=config).density_mo)(-.1).block_until_ready()
