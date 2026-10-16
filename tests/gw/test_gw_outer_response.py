"""Reconverged finite differences for molecular GW outer response (CPU/float64)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw import evgw_cd_restricted, evgw_cd_unrestricted, qsgw_cd_restricted
from gradscf.scf.autodiff import SCFDifferentiationConfig

_AD = SCFDifferentiationConfig(tolerance=1e-10, max_iter=30, restart=10)


@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('variable', ['mo_coeff', 'df_factors', 'mo_energy'])
def test_molecular_outer_response_rejects_complex_inputs(unrestricted, variable):
    kw = _inputs()
    if unrestricted:
        kw.update(mo_energy=(kw['mo_energy'], kw['mo_energy']),
                  mo_coeff=(kw['mo_coeff'], kw['mo_coeff']), nocc=(1, 1),
                  fock_matrix=None, hcore_matrix=None,
                  density_matrix=(jnp.eye(2), jnp.eye(2)))
    kw[variable] = jax.tree.map(lambda x: x.astype(complex) + .01j, kw[variable])
    driver = evgw_cd_unrestricted if unrestricted else evgw_cd_restricted
    with pytest.raises(NotImplementedError, match='real'):
        driver(**kw, differentiation=_AD)


def _inputs():
    e = jnp.array([-.6, .7])
    return dict(mo_energy=e, mo_coeff=jnp.eye(2), nocc=1,
                df_factors=jnp.array([[[.16, .08], [.08, .12]], [[.07, -.04], [-.04, .11]]]),
                fock_matrix=jnp.diag(e), hcore_matrix=jnp.diag(e),
                density_matrix=jnp.diag(jnp.array([2., 0.])),
                nw=12, max_iter=60, tol=1e-11)


@pytest.mark.parametrize('update_w', [True, False])
@pytest.mark.parametrize('variable', ['factor', 'initial_energy'])
def test_evgw_response_matches_reconverged_fd(update_w, variable):
    kw = _inputs()
    def run(t, config):
        args = dict(kw)
        if variable == 'factor':
            args['df_factors'] = kw['df_factors'].at[0, 0, 1].add(t).at[0, 1, 0].add(t)
        else:
            args['mo_energy'] = kw['mo_energy'].at[0].add(t)
        return evgw_cd_restricted(**args, update_w=update_w, differentiation=config).mo_energy
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    weights = jnp.array([.3, -.7])
    value, grad = jax.jit(jax.value_and_grad(lambda t: weights @ run(t, _AD)))(0.)
    np.testing.assert_allclose(value, weights @ run(0., None), atol=1e-10)
    np.testing.assert_allclose(grad, weights @ fd, atol=2e-8, rtol=2e-5)


def _open_shell_inputs():
    return dict(
        mo_energy=(jnp.array([-.8, -.35, .7]), jnp.array([-.7, .3, .85])),
        mo_coeff=(jnp.eye(3), jnp.array([[.98, -.1989974874, 0.], [.1989974874, .98, 0.], [0., 0., 1.]])),
        nocc=(2, 1),
        df_factors=jnp.array([[[.16, .04, .06], [.04, .12, -.03], [.06, -.03, .13]],
                              [[.05, -.03, .07], [-.03, .09, .02], [.07, .02, .08]]]),
        density_matrix=(jnp.diag(jnp.array([1., 1., 0.])),
                        jnp.outer(jnp.array([.98, .1989974874, 0.]), jnp.array([.98, .1989974874, 0.]))),
        nw=12, max_iter=60, tol=1e-11)


@pytest.mark.parametrize('spin', [0, 1])
def test_unrestricted_evgw_individual_spin_response(spin):
    kw = _open_shell_inputs()
    def run(t, config):
        energies = list(kw['mo_energy'])
        energies[spin] = energies[spin].at[0].add(t)
        return evgw_cd_unrestricted(**{**kw, 'mo_energy': tuple(energies)}, differentiation=config).mo_energy
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    weights = jnp.array([[.2, -.7, .1], [.4, .1, -.3]])
    grad = jax.jit(jax.grad(lambda t: jnp.sum(weights * run(t, _AD))))(0.)
    np.testing.assert_allclose(grad, jnp.sum(weights * fd), atol=2e-8, rtol=2e-5)


def test_qsgw_orbital_and_energy_response_matches_reconverged_fd():
    kw = _inputs()
    kw.pop('fock_matrix')
    kw.pop('density_matrix')
    kw.update(tol_density=1e-11)
    def run(t, config):
        h = kw['hcore_matrix'] + t * jnp.array([[.1, .2], [.2, -.03]])
        result = qsgw_cd_restricted(**{**kw, 'hcore_matrix': h}, differentiation=config)
        density = 2 * result.mo_coeff[:, :1] @ result.mo_coeff[:, :1].T
        return jnp.array([result.mo_energy[0], density[0, 1]])
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    _, tangent = jax.jit(lambda t: jax.jvp(lambda x: run(x, _AD), (t,), (jnp.ones_like(t),)))(jnp.array(0.))
    np.testing.assert_allclose(tangent, fd, atol=2e-8, rtol=2e-5)
    grad = jax.jit(jax.grad(lambda t: jnp.sum(run(t, _AD))))(0.)
    np.testing.assert_allclose(grad, jnp.sum(fd), atol=2e-8, rtol=2e-5)


def test_unrestricted_evgw0_keeps_initial_screening_and_its_response():
    kw = _inputs()
    kw.update(mo_energy=(kw['mo_energy'], kw['mo_energy'] + jnp.array([.04, -.07])),
              mo_coeff=(jnp.eye(2), jnp.eye(2)), nocc=(1, 1),
              fock_matrix=None, hcore_matrix=None,
              density_matrix=(jnp.diag(jnp.array([1., 0.])), jnp.diag(jnp.array([1., 0.]))))
    def run(t, config):
        energies = (kw['mo_energy'][0].at[0].add(t), kw['mo_energy'][1])
        result = evgw_cd_unrestricted(**{**kw, 'mo_energy': energies},
                                     update_w=False, differentiation=config)
        return jnp.sum(result.mo_energy) + .3 * result.screening_energy[0, 0]
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    grad = jax.jit(jax.grad(lambda t: run(t, _AD)))(0.)
    np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=2e-5)


def test_unrestricted_gw_rejects_empty_or_full_spin_before_fermi_indexing():
    from gradscf.gw import g0w0_cd_unrestricted
    kw = _inputs()
    for nocc in ((1, 0), (2, 1)):
        with pytest.raises(ValueError, match='occupied and virtual'):
            g0w0_cd_unrestricted(mo_energy=(kw['mo_energy'], kw['mo_energy']),
                mo_coeff=(jnp.eye(2), jnp.eye(2)), nocc=nocc, df_factors=kw['df_factors'],
                density_matrix=(jnp.eye(2), jnp.eye(2)), nw=4)


def test_evgw_selected_orbitals_keep_mf_response():
    kw = _inputs()
    def run(t, config):
        return evgw_cd_restricted(**{**kw, 'mo_energy': kw['mo_energy'].at[1].add(t)},
            orbs=(0,), differentiation=config).mo_energy
    _, tangent = jax.jit(lambda t: jax.jvp(lambda x: run(x, _AD), (t,), (jnp.ones_like(t),)))(jnp.array(0.))
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    np.testing.assert_allclose(tangent, fd, atol=2e-8, rtol=2e-5)
    np.testing.assert_allclose(tangent[1], 1., atol=1e-12)


def test_outer_response_rejects_nonconvergence():
    kw = {**_inputs(), 'max_iter': 1}
    with pytest.raises(Exception, match='evGW outer solve failed'):
        jax.jit(lambda e: evgw_cd_restricted(**{**kw, 'mo_energy': e}, differentiation=_AD).mo_energy)(kw['mo_energy']).block_until_ready()


def test_qsgw_response_rejects_degenerate_orbitals():
    with pytest.raises(Exception, match='isolated orbital spectrum required'):
        qsgw_cd_restricted(mo_energy=jnp.array([-.5, .7, .7]), mo_coeff=jnp.eye(3),
            nocc=1, hcore_matrix=jnp.diag(jnp.array([-.5, .7, .7])),
            df_factors=jnp.zeros((1, 3, 3)), nw=4, differentiation=_AD)


def test_evgw_singular_outer_response_is_nan():
    # A residual oracle isolates the outer solver's singular-Jacobian policy
    # from the physical self-energy kernels. At t=0 every energy is a root.
    from gradscf.gw.evgw import _evgw_loop
    from gradscf.gw.types import GWResult
    def loss(t):
        def driver(mo_energy_poles, **kwargs):
            return GWResult(mo_energy=mo_energy_poles, mo_coeff=jnp.eye(1),
                converged=True, qp_residual=jnp.ones(1) * t)
        return _evgw_loop(driver, jnp.ones(1), differentiation=_AD).mo_energy.sum()
    assert jnp.isnan(jax.jit(jax.grad(loss))(0.))


@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('update_w', [False, True])
def test_shared_diis_preserves_evgw_equation_and_response(unrestricted, update_w):
    kw = _open_shell_inputs() if unrestricted else _inputs()
    driver = evgw_cd_unrestricted if unrestricted else evgw_cd_restricted
    def values(t, history):
        result = driver(**{**kw, 'df_factors': kw['df_factors']*(1+t)},
                        update_w=update_w, damping=.3, diis_space=history,
                        differentiation=_AD)
        # The unrestricted driver currently exposes energies/W, but no QP
        # residue field. Do not invent an unimplemented observable for this gate.
        observables = ((result.mo_energy, result.screening_energy) if unrestricted else
                       (result.mo_energy, result.qp_weight, result.screening_energy))
        return jnp.concatenate([jnp.ravel(value) for value in observables])
    primal, tangent = jax.jit(lambda t: jax.jvp(lambda x: values(x, 6),
        (t,), (jnp.ones_like(t),)))(jnp.array(0.))
    baseline = values(0., 0)
    np.testing.assert_allclose(primal, baseline, atol=3e-9, rtol=1e-8)
    evaluate = jax.jit(lambda t: values(t, 6))
    step = 1e-4
    finite = (evaluate(step)-evaluate(-step))/(2*step)
    np.testing.assert_allclose(tangent, finite, atol=3e-8, rtol=2e-5)
    weights = jnp.linspace(.2, .8, primal.size)
    reverse = jax.jit(jax.grad(lambda t: weights@values(t, 6)))(0.)
    np.testing.assert_allclose(reverse, weights@tangent, atol=3e-9, rtol=2e-7)
    final = driver(**kw, update_w=update_w, damping=.3, diis_space=6)
    assert bool(final.converged) and np.asarray(final.converged_mask).all()
    assert np.max(abs(np.asarray(final.qp_residual))) < kw['tol']


def test_shared_diis_selected_orbitals_and_failed_forward():
    kw = _inputs()
    def energy(t):
        return evgw_cd_restricted(**{**kw, 'mo_energy': kw['mo_energy'].at[1].add(t)},
            orbs=(0,), diis_space=6, differentiation=_AD).mo_energy
    _, tangent = jax.jit(lambda t: jax.jvp(energy, (t,), (jnp.ones_like(t),)))(jnp.array(0.))
    np.testing.assert_allclose(tangent[1], 1., atol=1e-12, rtol=0)
    with pytest.raises(Exception, match='evGW outer solve failed'):
        jax.jit(lambda e: evgw_cd_restricted(**{**kw, 'mo_energy': e, 'max_iter': 1},
            diis_space=6, differentiation=_AD).mo_energy)(kw['mo_energy']).block_until_ready()


@pytest.mark.parametrize('history', [-1, True, 1.5])
def test_shared_diis_invalid_history_rejected_before_driver(history):
    from gradscf.gw.evgw import _evgw_loop
    def never_called(**kwargs):
        pytest.fail('Invalid history must be rejected before evaluating GW')
    with pytest.raises(ValueError, match='diis_space'):
        _evgw_loop(never_called, jnp.ones(2), diis_space=history, differentiation=_AD)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_outer_seed_is_numerical_and_unselected_levels_keep_mf(unrestricted):
    kw = _open_shell_inputs() if unrestricted else _inputs()
    driver = evgw_cd_unrestricted if unrestricted else evgw_cd_restricted
    mean_field = jnp.asarray(kw['mo_energy'])
    def energy(value):
        result = driver(**kw, orbs=(0,), update_w=False, diis_space=6,
            initial_guess=mean_field+.03+value, differentiation=_AD)
        return result.mo_energy
    value, tangent = jax.jit(lambda t: jax.jvp(energy, (t,), (jnp.ones_like(t),)))(jnp.array(0.))
    np.testing.assert_allclose(tangent, 0., atol=2e-10, rtol=0)
    np.testing.assert_allclose(value[..., 1:], mean_field[..., 1:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(value, energy(.01), atol=3e-10, rtol=0)
    assert np.isfinite(value).all()


def test_outer_seed_rejects_invalid_shapes_and_complex_values():
    kw = _inputs()
    for seed in (jnp.zeros(3), jnp.ones(2, dtype=complex)):
        with pytest.raises(ValueError, match='initial_guess'):
            evgw_cd_restricted(**kw, initial_guess=seed, diis_space=6)


@pytest.mark.parametrize('variable', ['factor', 'frame_rotation'])
def test_qsgw_fixed_frame_response_matches_reconverged_fd(variable):
    kw = _inputs()
    kw.pop('fock_matrix')
    kw.pop('density_matrix')
    # C0 is S-orthonormal, deliberately not Euclidean-orthogonal.
    kw['mo_coeff'] = jnp.diag(jnp.array([.8, 1.2]))
    kw['tol_density'] = 1e-11
    def run(t, config):
        args = dict(kw)
        if variable == 'factor':
            args['df_factors'] = kw['df_factors'].at[0, 0, 1].add(t).at[0, 1, 0].add(t)
        else:
            rotation = jnp.array([[jnp.cos(t), -jnp.sin(t)], [jnp.sin(t), jnp.cos(t)]])
            args['mo_coeff'] = kw['mo_coeff'] @ rotation
        result = qsgw_cd_restricted(**args, differentiation=config)
        density = 2 * result.mo_coeff[:, :1] @ result.mo_coeff[:, :1].T
        return result.mo_energy[0] + .7 * density[0, 1]
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    grad = jax.jit(jax.grad(lambda t: run(t, _AD)))(0.)
    np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=2e-5)
    if variable == 'frame_rotation':
        np.testing.assert_allclose(grad, 0., atol=2e-8)


@pytest.mark.parametrize('method', ['g0w0', 'evgw', 'evgw0'])
def test_unrestricted_gw_bse_optical_chain_reconverged_fd(method):
    from gradscf import bse
    from gradscf.gw import g0w0_cd_unrestricted
    from gradscf.gw.g0w0 import _mo_factors
    kw = _open_shell_inputs()
    space = bse.make_bse_space(3, kw['nocc'])
    cfg = bse.BSEConfig(singlet=None, solver='dense', nroots=1,
                        conv_tol=1e-11, adjoint_tol=1e-11)
    dipole_ao = jnp.array([[[0., .2, .4], [.2, 0., -.3], [.4, -.3, 0.]],
                          [[.1, -.3, .1], [-.3, -.1, .2], [.1, .2, .0]],
                          [[.1, .1, -.2], [.1, .0, .1], [-.2, .1, .2]]])
    def run(t, differentiation):
        # One AO off-diagonal factor changes GW, static BSE screening, and
        # the electron-hole kernel through both distinct orbital frames.
        factors = kw['df_factors'].at[0, 0, 1].add(t).at[0, 1, 0].add(t)
        args = {**kw, 'df_factors': factors}
        if method == 'g0w0':
            args.pop('max_iter')
            args.pop('tol')
            result = g0w0_cd_unrestricted(**args)
        else:
            result = evgw_cd_unrestricted(**args, update_w=method == 'evgw',
                                         differentiation=differentiation)
        mo_factors = jnp.stack([_mo_factors(factors, c) for c in result.mo_coeff])
        dipole = jnp.stack([_mo_factors(dipole_ao, c) for c in result.mo_coeff])
        excited = bse.run_bse(result.mo_energy, result.screening_energy, mo_factors,
            space, config=cfg, qp_computed_mask=result.qp_computed_mask,
            qp_converged_mask=result.converged_mask)
        strength = bse.oscillator_strengths(excited, dipole, space)[0]
        return excited.excitation_energies[0] + .3 * strength
    step = 1e-4
    fd = (run(step, None) - run(-step, None)) / (2 * step)
    grad = jax.jit(jax.grad(lambda t: run(t, _AD)))(0.)
    assert jnp.isfinite(grad) and abs(float(grad)) > 1e-7
    np.testing.assert_allclose(grad, fd, atol=3e-8, rtol=3e-5)
