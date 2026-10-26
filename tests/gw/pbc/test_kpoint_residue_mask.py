"""Strict contour masks must avoid evaluating exactly zero residue terms."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import LinearSolverConfig


@pytest.mark.parametrize("omega,p", [(-.9, 0), (-.8, 0), (0., 1), (.4, 1), (.6, 1), (1.1, 1)])
def test_residue_mask_skips_inactive_solves_and_keeps_dense_value(monkeypatch, omega, p):
    driver = importlib.import_module("gradscf.gw.pbc.ksigma")
    solve = driver.solve_transition_screening
    head_wing = driver.transition_head_wing
    body_calls, head_calls = [], []

    def observed_solve(*args, **kwargs):
        jax.debug.callback(lambda _: body_calls.append(1), jnp.array(0), ordered=True)
        return solve(*args, **kwargs)

    def observed_head(*args, **kwargs):
        jax.debug.callback(lambda _: head_calls.append(1), jnp.array(0), ordered=True)
        return head_wing(*args, **kwargs)

    monkeypatch.setattr(driver, "solve_transition_screening", observed_solve)
    monkeypatch.setattr(driver, "transition_head_wing", observed_head)
    rng = np.random.default_rng(941)
    nk, nmo, rank, ngrid = 2, 3, 4, 5
    energies = np.array([[-.8, .4, .9], [-.7, .5, 1.]])
    table = np.array([[0, 1], [1, 0]])
    differences = np.array([-1.2, -1.7, -1.3, -1.8])
    eta, ef, volume = .003, 0., 27.
    q_abs = np.array([1e-3, 2e-3, 3e-3])
    qij = .01*(rng.normal(size=rank)+1j*rng.normal(size=rank))
    blocks, factors, pairs = [], [], []
    for _ in range(nk):
        b = .04*(rng.normal(size=(ngrid, rank))+1j*rng.normal(size=(ngrid, rank)))
        x = .03*(rng.normal(size=(ngrid, nmo))+1j*rng.normal(size=(ngrid, nmo)))
        factors.append(b)
        pairs.append(x)
        blocks.append(dict(gram=jnp.asarray(b.conj().T@b),
            differences=jnp.asarray(differences),
            projected=jnp.asarray(b.conj().T@x)[:, None, :, None]))
    config = LinearSolverConfig(method="direct", rtol=1e-11, atol=1e-13)
    actual = driver._sigma_residue_transition(jnp.asarray(omega), p, 0, 0, 0,
        jnp.asarray(energies), blocks, table, ef, eta, nk, config,
        q0_data=(qij.reshape(2, 1, 2), q_abs, volume))
    jax.block_until_ready(actual)
    jax.effects_barrier()

    expected, expected_body_calls = 0j, 0
    above = omega > ef
    sign = 1 if above else -1
    for q, (b, x) in enumerate(zip(factors, pairs)):
        for m, energy in enumerate(energies[q]):
            active = ef < energy < omega if above else omega < energy < ef
            if not active:
                continue
            expected_body_calls += 1
            frequency = abs(energy-omega)
            weights = (2/nk)*(1/(frequency+differences+2j*eta)+1/(-frequency+differences))
            dielectric = np.eye(ngrid)-(b*weights)@b.conj().T
            wc_x = np.linalg.solve(dielectric, x[:, m])-x[:, m]
            expected += sign*np.vdot(x[:, m], wc_x)/nk
    energy = energies[0, p]
    active_head = ef < energy < omega if above else omega < energy < ef
    if active_head:
        weights = driver._transition_weights(abs(energy-omega), differences, nk, eta)
        correction = head_wing(blocks[0]["gram"], weights, qij,
            blocks[0]["projected"][:, 0, p, 0], q_abs, volume, nkpts=nk, config=config)
        expected += sign*complex(correction.head+correction.wings)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    assert len(body_calls) == expected_body_calls
    assert len(head_calls) == int(active_head)
