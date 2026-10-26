"""Neutral excitation–phonon model: independent oracles, cancellation and AD."""

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import bse
from gradscf.gw.ep_coupling import PhononModel as MOPhonons
from gradscf.solvers import LinearSolverConfig


def ep():
    from gradscf.bse import ep_coupling

    return ep_coupling


def case():
    rng = np.random.default_rng(72)
    g = rng.normal(size=(2, 3, 3)) * 0.015 + 1j * rng.normal(size=(2, 3, 3)) * 0.01
    g = (g + g.swapaxes(-1, -2).conj()) * 0.5
    lam = np.stack([np.eye(3) * 0.001, np.eye(3) * -0.0003])
    model = ep().PhononModel(jnp.array([0.04, 0.08]), jnp.asarray(g), jnp.asarray(lam))
    return (
        jnp.array([0.42, 0.55, 0.71]),
        model,
        jnp.asarray(rng.normal(size=(3, 3)) * 0.2),
    )


def oracle(h, model, w, beta, eta):
    h = np.diag(h) if np.ndim(h) == 1 else np.asarray(h)
    result = []
    for frequency in w:
        sigma = np.zeros_like(h, dtype=complex)
        for om, g in zip(model.energies, model.couplings):
            n = 1 / np.expm1(beta * om)
            z = (frequency + 1j * eta) * np.eye(len(h)) - h
            sigma += (
                g
                @ (
                    (n + 1) * np.linalg.inv(z - om * np.eye(len(h)))
                    + n * np.linalg.inv(z + om * np.eye(len(h)))
                )
                @ g.conj().T
            )
        result.append(sigma)
    return np.asarray(result)


def test_projected_vertices_match_electron_minus_hole_dense_oracle():
    space = bse.make_bse_space(5, 2, occupied=(0, 1), virtual=(2, 4))
    rng = np.random.default_rng(8)
    x, _ = np.linalg.qr(rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3)))
    x = jnp.asarray(x.T.reshape(3, 2, 2))
    g = rng.normal(size=(2, 5, 5)) + 1j * rng.normal(size=(2, 5, 5))
    g = (g + g.swapaxes(-1, -2).conj()) * 0.5
    expected = []
    for vertex in g:
        lifted = np.kron(np.eye(2), vertex[np.ix_(space.virtual, space.virtual)])
        lifted -= np.kron(vertex[np.ix_(space.occupied, space.occupied)].T, np.eye(2))
        expected.append(
            np.asarray(x).reshape(3, 4).conj() @ lifted @ np.asarray(x).reshape(3, 4).T
        )
    actual = jax.jit(lambda values: ep().project_couplings(x, space, values))(
        jnp.asarray(g)
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    common = jnp.stack([jnp.eye(5) * 0.3, jnp.eye(5) * -0.1])
    np.testing.assert_allclose(
        ep().project_couplings(x, space, common), 0.0, atol=1e-14
    )
    kernel = jnp.stack([jnp.eye(4) * 0.1, jnp.eye(4) * 0.2])
    action = lambda mode, v: kernel[mode] @ v
    dense = ep().project_couplings(x, space, jnp.asarray(g), kernel=kernel)
    matrix_free = ep().project_couplings(x, space, jnp.asarray(g), kernel=action)
    np.testing.assert_allclose(dense, matrix_free, atol=2e-12)


def solved_case(mode="implicit_eigenvector"):
    qp = jnp.array([-0.9, -0.5, 0.3, 0.75])
    l = jnp.array(
        [
            [
                [0.13, 0.02, 0.01, 0.02],
                [0.02, 0.1, 0.015, 0.01],
                [0.01, 0.015, 0.14, 0.02],
                [0.02, 0.01, 0.02, 0.12],
            ]
        ]
    )
    space = bse.make_bse_space(4, 2)
    result = bse.run_bse(
        qp,
        qp,
        l,
        space,
        config=bse.BSEConfig(nroots=4, solver="dense", gradient_mode=mode),
    )
    dipole = jnp.arange(48, dtype=float).reshape(3, 4, 4) * 0.002
    dipole = (dipole + dipole.swapaxes(-1, -2)) * 0.5
    return space, result, dipole


def test_bse_adapter_quadratic_projection_and_zero_coupling_optics():
    space, result, dipole = solved_case()
    source = MOPhonons(
        jnp.array([0.05, 0.08]), jnp.zeros((2, 4, 4)), jnp.zeros((2, 4, 4))
    )
    model = ep().project_phonons(result, space, source)
    assert model.quadratic.shape == (2, 4, 4)
    d = bse.transition_dipoles(result, dipole, space)
    w = jnp.array([0.0, 0.2, 0.8, 1.1])
    actual = ep().polarizability(
        result.excitation_energies, model, d, w, beta=100.0, eta=0.015
    )
    expected = bse.polarizability(result, dipole, space, w, eta=0.015)
    np.testing.assert_allclose(actual, expected, atol=2e-11)
    np.testing.assert_allclose(
        ep().absorption_cross_section(
            result.excitation_energies, model, d, w, beta=100.0, eta=0.015, unit="Mb"
        ),
        bse.absorption_cross_section(result, dipole, space, w, eta=0.015, unit="Mb"),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        ep().polarizability(
            result.excitation_energies, model, d, 0.2, beta=100.0, eta=0.015
        ),
        expected[1],
        atol=2e-11,
    )


@pytest.mark.parametrize("matrix", [False, True])
def test_fan_matches_independent_bosonic_resolvents(matrix):
    e, model, _ = case()
    h = jnp.diag(e) if matrix else e
    w = jnp.array([0.1, 0.42, 0.58, 0.9])
    actual = jax.jit(lambda x: ep().fan_retarded(x, model, w, beta=35.0, eta=0.013))(h)
    np.testing.assert_allclose(actual, oracle(h, model, w, 35.0, 0.013), atol=3e-12)
    imag = (actual - actual.swapaxes(-1, -2).conj()) / (2j)
    assert np.linalg.eigvalsh(-np.asarray(imag)).min() > -1e-12
    expected = 0.5 * np.einsum(
        "lij,l->ij", model.quadratic, 1 + 2 / np.expm1(35.0 * model.energies)
    )
    np.testing.assert_allclose(ep().debye_waller(model, 35.0), expected, atol=1e-14)
    spectrum = ep().spectral_function(h, model, w, beta=35.0, eta=0.013)
    for z, sig, a in zip(w, actual, spectrum):
        green = np.linalg.inv(
            (z + 1j * 0.013) * np.eye(3) - np.diag(e) - expected - sig
        )
        np.testing.assert_allclose(
            a, -(green - green.conj().T) / (2j * np.pi), atol=3e-11
        )
    assert np.linalg.eigvalsh(spectrum).min() > -1e-12


def test_matrix_response_covariance_and_derivative_at_degeneracy():
    e, model, d = case()
    h = jnp.eye(3) * 0.5
    rng = np.random.default_rng(18)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    u = jnp.asarray(u)
    rotated = ep().PhononModel(
        model.energies,
        u.conj().T @ model.couplings @ u,
        u.conj().T @ model.quadratic @ u,
    )
    w = jnp.array([0.44, 0.52, 0.67])
    alpha = ep().polarizability(h, model, d, w, beta=50.0, eta=0.018)
    other = ep().polarizability(
        u.conj().T @ h @ u, rotated, u.conj().T @ d, w, beta=50.0, eta=0.018
    )
    np.testing.assert_allclose(alpha, other, atol=2e-10)
    perturb = jnp.array([[0.1, 0.03, 0.01], [0.03, -0.08, 0.02], [0.01, 0.02, 0.04]])
    loss = (
        lambda t: ep()
        .absorption_cross_section(h + t * perturb, model, d, w, beta=50.0, eta=0.018)
        .sum()
    )
    derivative = jax.jit(jax.grad(loss))(0.0)
    step = 1e-5
    np.testing.assert_allclose(
        derivative, (loss(step) - loss(-step)) / (2 * step), rtol=2e-5, atol=2e-7
    )


def test_optical_response_gradients_and_shared_gmres():
    e, model, d = case()
    w = jnp.array([0.2, 0.47, 0.8])

    def loss(t):
        moving = ep().PhononModel(
            model.energies * t[0], model.couplings * t[1], model.quadratic * t[2]
        )
        return (
            ep()
            .absorption_cross_section(
                e * t[3], moving, d * t[4], w, beta=40.0, eta=0.012
            )
            .sum()
        )

    t = jnp.ones(5)
    observed = jax.jit(jax.grad(loss))(t)
    step = 1e-5
    fd = jnp.stack(
        [(loss(t + step * v) - loss(t - step * v)) / (2 * step) for v in jnp.eye(5)]
    )
    np.testing.assert_allclose(observed, fd, rtol=2e-5, atol=3e-7)
    cfg = LinearSolverConfig(method="gmres", rtol=1e-11, restart=6)
    direct = ep().polarizability(jnp.diag(e), model, d, w, beta=40.0, eta=0.012)
    iterative = ep().polarizability(
        jnp.diag(e), model, d, w, beta=40.0, eta=0.012, solver_config=cfg
    )
    np.testing.assert_allclose(iterative, direct, atol=3e-10)


def test_projection_guards_full_bse_and_stopped_amplitudes():
    space, result, _ = solved_case()
    source = MOPhonons(jnp.array([0.05]), jnp.eye(4)[None] * 0.01)
    with pytest.raises(ValueError, match="TDA"):
        ep().project_phonons(
            replace(result, y_amplitudes=jnp.ones_like(result.x_amplitudes) * 0.01),
            space,
            source,
        )
    with pytest.raises(ValueError, match="converged"):
        ep().project_phonons(
            replace(result, converged=jnp.zeros_like(result.converged)), space, source
        )
    _, stopped, _ = solved_case("eigenvalue_only")
    loss = (
        lambda scale: ep()
        .project_phonons(
            stopped, space, replace(source, couplings=source.couplings * scale)
        )
        .couplings.sum()
    )
    assert np.isnan(jax.grad(loss)(1.0))


def test_quadratic_projection_and_kernel_callbacks_preserve_mode_layouts():
    space, result, _ = solved_case()
    omega = jnp.array([0.03, 0.07])
    g = jnp.zeros((2, 4, 4))
    diagonal = jnp.stack(
        [jnp.diag(jnp.array([0.02, 0.03, 0.04, 0.05])), jnp.eye(4) * 0.01]
    )
    full = jnp.zeros((2, 2, 4, 4)).at[jnp.arange(2), jnp.arange(2)].set(diagonal)
    full = full.at[0, 1].set(jnp.eye(4) * 0.02).at[1, 0].set(jnp.eye(4) * 0.02)
    kernel = jnp.stack([jnp.eye(space.size) * 0.003, jnp.eye(space.size) * -0.002])
    action = lambda mode, v: kernel[mode] @ v
    a = ep().project_phonons(
        result, space, MOPhonons(omega, g, diagonal), kernel_quadratic=action
    )
    b = ep().project_phonons(
        result, space, MOPhonons(omega, g, full), kernel_quadratic=kernel
    )
    np.testing.assert_allclose(
        a.quadratic, b.quadratic[jnp.arange(2), jnp.arange(2)], atol=1e-13
    )
    np.testing.assert_allclose(
        ep().debye_waller(a, 40.0), ep().debye_waller(b, 40.0), atol=1e-13
    )
    expected = ep().project_couplings(result.x_amplitudes, space, diagonal) + kernel
    np.testing.assert_allclose(a.quadratic, expected, atol=1e-13)


def test_actual_bse_kernel_derivative_projects_to_exciton_energy_response():
    from gradscf.gw.screened import build_static_screening

    space, result, _ = solved_case()
    qp = jnp.array([-0.9, -0.5, 0.3, 0.75])
    factors = jnp.array(
        [
            [
                [0.13, 0.02, 0.01, 0.02],
                [0.02, 0.1, 0.015, 0.01],
                [0.01, 0.015, 0.14, 0.02],
                [0.02, 0.01, 0.02, 0.12],
            ]
        ]
    )

    def action(scale, cols):
        screen = build_static_screening(
            qp, factors * scale, occupied=space.occupied, virtual=space.virtual
        )
        return bse.build_tda_operator(qp, factors * scale, space, screen).apply(cols)

    derivative = lambda mode, cols: jax.jvp(
        lambda value: action(value, cols), (1.0,), (1.0,)
    )[1]
    source = MOPhonons(jnp.array([0.05]), jnp.zeros((1, 4, 4)))
    projected = ep().project_phonons(
        result, space, source, kernel_derivative=derivative
    )

    def energies(scale):
        return bse.run_bse(
            qp,
            qp,
            factors * scale,
            space,
            config=bse.BSEConfig(nroots=4, solver="dense"),
        ).excitation_energies

    expected = jax.jvp(energies, (1.0,), (1.0,))[1]
    np.testing.assert_allclose(jnp.diag(projected.couplings[0]), expected, atol=3e-11)


def test_failed_iterative_response_and_traced_bad_modes_are_rejected():
    e, model, d = case()
    w = jnp.array([0.4, 0.6])
    cfg = LinearSolverConfig(method="gmres", maxiter=1, restart=1, rtol=1e-12)
    with pytest.raises(ValueError, match="converge"):
        ep().polarizability(e, model, d, w, beta=40.0, eta=0.012, solver_config=cfg)
    checked = jax.jit(
        lambda values: ep().fan_retarded(
            e, replace(model, energies=values), w, beta=40.0, eta=0.012
        )
    )
    with pytest.raises(Exception, match="positive"):
        checked(-model.energies).block_until_ready()


def test_resolved_negative_thermal_satellite_is_not_reported_as_absorption():
    model = ep().PhononModel(jnp.array([0.4]), jnp.array([[[0.01]]]))
    e = jnp.array([0.3])
    d = jnp.array([[1.0, 0.0, 0.0]])
    w = jnp.array([0.1])
    # The raw low-population approximation remains available for diagnosis.
    alpha = ep().polarizability(e, model, d, w, beta=40.0, eta=1e-12)
    assert float(alpha[0, 0, 0].imag) < -0.08
    with pytest.raises(ValueError, match="passive"):
        ep().absorption_cross_section(
            e,
            model,
            d,
            w,
            beta=40.0,
            eta=1e-12,
            polarization=jnp.array([1.0, 0.0, 0.0]),
        )


def test_invalid_polarization_preserves_bse_nan_contract():
    e, model, d = case()
    out = ep().absorption_cross_section(
        e, model, d, jnp.array([0.4]), beta=40.0, eta=0.01, polarization=jnp.zeros(3)
    )
    assert np.isnan(out).all()


def test_zero_temperature_skips_unoccupied_absorption_resolvent():
    model = ep().PhononModel(jnp.array([0.1]), jnp.array([[[0.03]]]))
    cfg = LinearSolverConfig(method="gmres", rtol=0.01, maxiter=1, restart=1)
    # The emission solve meets this requested tolerance. The absorption
    # solve is resonant and would fail, but its Bose weight is exactly zero.
    observed = ep().fan_retarded(
        jnp.array([[0.5]]),
        model,
        jnp.array([0.4]),
        beta=1e8,
        eta=0.001,
        solver_config=cfg,
    )
    expected = 0.03**2 / (0.4 + 0.001j - 0.5 - 0.1)
    np.testing.assert_allclose(observed[0, 0, 0], expected, atol=3e-5)


@pytest.mark.parametrize("matrix", [False, True])
def test_mixed_precision_promotes_before_resolvent_arithmetic(matrix):
    e, model, _ = case()
    e = e.astype(jnp.float32)
    w = jnp.array([0.1, 0.42, 0.6], dtype=jnp.float32)
    h = jnp.diag(e) if matrix else e
    actual = ep().fan_retarded(h, model, w, beta=35.0, eta=0.013)
    expected = oracle(
        np.asarray(h, dtype=np.float64),
        model,
        np.asarray(w, dtype=np.float64),
        35.0,
        0.013,
    )
    assert actual.dtype == jnp.complex128
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=1e-10)


def test_literal_zero_vertices_have_same_vector_and_matrix_response():
    model = ep().PhononModel(jnp.array([0.05]), jnp.array([[[0]]]))
    value = ep().fan_retarded(
        jnp.array([[0.4]]), model, jnp.array([0.3]), beta=30.0, eta=0.01
    )
    np.testing.assert_array_equal(value, 0.0)


def test_mixed_precision_dw_is_contracted_in_the_optical_compute_dtype():
    model = ep().PhononModel(
        jnp.array([0.05], dtype=jnp.float32),
        jnp.zeros((1, 1, 1), jnp.float32),
        jnp.array([[[0.01]]], dtype=jnp.float32),
    )
    dw = (
        0.5
        * float(model.quadratic[0, 0, 0])
        / np.tanh(0.5 * 30.0 * float(model.energies[0]))
    )
    energy = 0.5 + dw
    eta = 1e-8
    observed = ep().polarizability(
        jnp.array([0.5]),
        model,
        jnp.array([[1.0, 0.0, 0.0]]),
        energy,
        beta=30.0,
        eta=eta,
    )
    expected = 1 / (-1j * eta) + 1 / (2 * energy + 1j * eta)
    np.testing.assert_allclose(observed[0, 0].real, expected.real, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(observed[0, 0].imag, expected.imag, rtol=1e-10)


@pytest.mark.parametrize(
    "kind", ["negative_mode", "negative_exciton", "eta", "nonhermitian"]
)
def test_invalid_physical_inputs_rejected(kind):
    e, model, _ = case()
    if kind == "negative_mode":
        model = replace(model, energies=-model.energies)
    if kind == "negative_exciton":
        e = -e
    if kind == "nonhermitian":
        model = replace(model, couplings=model.couplings.at[0, 0, 1].add(0.1))
    with pytest.raises(ValueError):
        ep().fan_retarded(
            e, model, jnp.array([0.5]), beta=40.0, eta=0.0 if kind == "eta" else 0.01
        )
