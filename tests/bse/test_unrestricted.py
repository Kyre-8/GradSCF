"""Spin-conserving BSE: scalar-loop oracle, closed-shell limit and response."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import bse
from gradscf.gw.screened import build_static_screening
from gradscf.solvers import LinearSolverConfig


def model():
    rng = np.random.default_rng(140)
    l = rng.normal(size=(2, 4, 4, 4)) * .06
    l = (l + l.swapaxes(-1, -2)) * .5
    e = np.array([[-1.2, -.6, .4, 1.1], [-1.0, .1, .7, 1.4]])
    d = rng.normal(size=(2, 3, 4, 4)) * .2
    d = (d + d.swapaxes(-1, -2)) * .5
    return tuple(map(jnp.asarray, (e, l, d)))


def oracle(e, l, nocc):
    """Independent full spin-orbital contractions, without kernel helpers."""
    e, l = np.asarray(e), np.asarray(l)
    pairs = [(s, i, a) for s in range(2) for i in range(nocc[s])
             for a in range(nocc[s], e.shape[1])]
    eps = np.eye(l.shape[1])
    for s, i, a in pairs:
        eps += 2 * np.outer(l[s, :, i, a], l[s, :, i, a]) / (e[s, a] - e[s, i])
    a, b = np.zeros((2, len(pairs), len(pairs)))
    for p, (s, i, v) in enumerate(pairs):
        for q, (t, j, w) in enumerate(pairs):
            a[p, q] = b[p, q] = l[s, :, i, v] @ l[t, :, j, w]
            if s == t:
                a[p, q] -= l[s, :, i, j] @ np.linalg.solve(eps, l[s, :, v, w])
                b[p, q] -= l[s, :, i, w] @ np.linalg.solve(eps, l[s, :, v, j])
            if p == q:
                a[p, q] += e[s, v] - e[s, i]
    return eps, a, b


@pytest.mark.parametrize('method', ['direct', 'gmres'])
def test_spin_kernel_matches_independent_oracle(method):
    e, l, _ = model()
    space = bse.make_bse_space(4, (2, 1))
    state = build_static_screening(e, l, occupied=space.occupied, virtual=space.virtual,
                                  config=LinearSolverConfig(method=method, rtol=1e-12))
    expected_eps, expected_a, expected_b = oracle(e, l, (2, 1))
    np.testing.assert_allclose(state.operator().apply(jnp.eye(4)), expected_eps, atol=2e-12)
    a, b = bse.build_bse_operators(e, l, space, state, singlet=None, block_size=1)
    for op, expected in ((a, expected_a), (b, expected_b)):
        np.testing.assert_allclose(op.apply(jnp.eye(space.size)), expected, atol=2e-12)
        np.testing.assert_allclose(op.diagonal, np.diag(expected), atol=2e-12)


@pytest.mark.parametrize('tda', [True, False])
def test_unrestricted_closed_shell_roots_and_strengths(tda):
    e, l, d = [x[0] for x in model()]
    space = bse.make_bse_space(4, 2)
    cfg = dict(nroots=space.size, solver='dense', tda=tda, conv_tol=1e-11)
    singlet = bse.run_bse(e, e, l, space, config=bse.BSEConfig(**cfg))
    triplet = bse.run_bse(e, e, l, space, config=bse.BSEConfig(**cfg, singlet=False))
    spin_space = bse.make_bse_space(4, (2, 2))
    unrestricted = bse.run_bse(jnp.stack([e, e]), jnp.stack([e, e]), jnp.stack([l, l]),
                              spin_space, config=bse.BSEConfig(**{**cfg, 'nroots': 8}, singlet=None))
    expected = np.concatenate((singlet.excitation_energies, triplet.excitation_energies))
    order = np.argsort(expected)
    np.testing.assert_allclose(unrestricted.excitation_energies, expected[order], atol=2e-11)
    expected_f = np.concatenate((bse.oscillator_strengths(singlet, d, space), np.zeros(4)))
    np.testing.assert_allclose(bse.oscillator_strengths(unrestricted, jnp.stack([d, d]), spin_space),
                               expected_f[order], atol=2e-10)


@pytest.mark.parametrize('tda,method', [(True, 'dense'), (False, 'davidson')])
def test_open_shell_optical_jvp_vjp_includes_both_screening_channels(tda, method):
    e, l, d = model()
    space = bse.make_bse_space(4, (2, 1))
    cfg = bse.BSEConfig(nroots=2, singlet=None, solver=method, tda=tda,
                        conv_tol=1e-11, adjoint_tol=1e-11)
    def value(t):
        # Perturb only beta screening while QP spectrum and factors stay fixed.
        screen = e.at[1].add(t * jnp.arange(4) * .3)
        out = bse.run_bse(e, screen, l, space, config=cfg)
        return out.excitation_energies.sum() + .2 * bse.oscillator_strengths(out, d, space).sum()
    gradient = jax.jit(jax.grad(value))(0.)
    fd = (value(1e-3) - value(-1e-3)) / 2e-3
    assert abs(float(gradient)) > 1e-8
    np.testing.assert_allclose(gradient, fd, atol=2e-8, rtol=2e-4)
    np.testing.assert_allclose(jax.jvp(value, (0.,), (1.,))[1], gradient, atol=1e-10)


def test_spin_reference_facade_windows_coverage_and_invalid_modes():
    from gradscf.gw.types import GWResult
    e, l, d = model()
    masks = jnp.ones_like(e, dtype=bool)
    result = GWResult(e, jnp.stack([jnp.eye(4)] * 2), True,
                      screening_energy=e, qp_computed_mask=masks, converged_mask=masks)
    ref = bse.BSEReference.from_gw_result(result, mo_factors=l, nocc=(2, 1), dipole_mo=d)
    job = bse.BSE(ref, occupied=((1,), (0,)), virtual=((2, 3), (1, 3)),
                  nroots=2, solver='dense').run()
    assert job.result.singlet is None and np.all(job.converged)
    assert job.result.x_amplitudes[0].shape == (2, 1, 2)
    assert np.all(np.isfinite(job.oscillator_strength()))
    with pytest.raises(ValueError, match='singlet'):
        bse.BSE(ref, singlet=False, nroots=1).run()
    from dataclasses import replace
    bad = replace(ref, qp_computed_mask=masks.at[1, 1].set(False))
    with pytest.raises(ValueError, match='actually computed'):
        bse.BSE(bad, nroots=1).run()
    with pytest.raises(ValueError, match='real|complex'):
        bse.BSEReference.from_gw_result(replace(result, mo_energy=e.astype(complex)),
                                       mo_factors=l, nocc=(2, 1))


def test_empty_spin_transition_channel():
    e, l, d = model()
    space = bse.make_bse_space(4, (1, 0))
    cfg = bse.BSEConfig(nroots=1, singlet=None, solver='dense')
    out = bse.run_bse(e, e, l, space, config=cfg)
    assert out.x_amplitudes[1].shape == (1, 0, 4)
    assert np.all(out.converged)
    assert np.all(np.isfinite(bse.oscillator_strengths(out, d, space)))


@pytest.mark.parametrize('method', ['g0w0', 'evgw0'])
def test_oh_ugw_to_bse_facade_and_stale_source(method):
    from gradscf import gto, scf, gw
    mol = gto.M(atom='O 0 0 0; H 0 0 .9697', basis='sto-3g', spin=1)
    mf = scf.UHF(mol, conv_tol=1e-10).run()
    qp = gw.UGW(mf, nw=32, method=method, max_cycle=60, damp=.3).run(orbs=(3, 4, 5))
    job = bse.BSE(qp, occupied=((4,), (3,)), virtual=((5,), (4, 5)),
                  nroots=2, solver='dense').run()
    assert job.result.singlet is None and np.all(job.converged)
    assert np.all(np.isfinite(job.oscillator_strength()))
    with pytest.raises(ValueError, match='actually computed'):
        bse.BSE(qp, nroots=1).run()
    occupations = qp.mo_occ
    qp.mo_occ = jnp.asarray(occupations).at[0, 4].set(0)
    with pytest.raises(RuntimeError, match='changed'):
        job.oscillator_strength()
    qp.mo_occ = occupations
    qp.eta *= 2
    with pytest.raises(RuntimeError, match='changed'):
        job.oscillator_strength()
