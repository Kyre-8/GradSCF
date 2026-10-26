"""RPA pole expansion must reproduce the existing screened interaction."""
import numpy as np
import jax.numpy as jnp

from gradscf.gw.freq import scaled_legendre_grid
from gradscf.gw.polarizability import rho_response_iw
from gradscf.gw.screened import screened_w_imag_axis
from gradscf.gw.poles import rpa_pole_expansion, screened_w_imag_poles, screened_w_real_poles


def test_rpa_poles_match_imaginary_axis_screening():
    rng = np.random.default_rng(42)
    factors = jnp.asarray(rng.normal(size=(4, 5, 5)) * .2)
    factors = .5 * (factors + factors.swapaxes(1, 2))
    energy = jnp.array([-.8, -.4, .3, .9, 1.5])
    b_ov = factors[:, :2, 2:]
    frequencies = jnp.array([0., .2, 1.1])
    poles, couplings = rpa_pole_expansion(energy, b_ov, factors)
    expected = screened_w_imag_axis(
        factors, lambda w: rho_response_iw(w, (energy[:2], energy[2:]), b_ov), frequencies
    )
    actual = screened_w_imag_poles(poles, couplings, frequencies)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    real = screened_w_real_poles(poles, couplings, jnp.array([.1, .7]), eta=1e-5)
    assert real.shape == (2, 5, 5) and np.all(np.isfinite(real))


def test_rpa_poles_are_differentiable_in_factors():
    energy = jnp.array([-.8, -.4, .3, .9, 1.5])
    base = jnp.ones((4, 5, 5)) * .1
    base = .5 * (base + base.swapaxes(1, 2))
    b_ov = base[:, :2, 2:]
    def loss(scale):
        poles, couplings = rpa_pole_expansion(energy, b_ov * scale, base * scale)
        return jnp.sum(screened_w_imag_poles(poles, couplings, jnp.array([.2])) ** 2)
    value = loss(1.)
    gradient = __import__('jax').grad(loss)(1.)
    assert np.isfinite(value) and np.isfinite(gradient)


def test_rpa_resolvent_matches_poles_and_is_degenerate_safe():
    from gradscf.gw.poles import rpa_resolvent, screened_w_imag_resolvent
    rng = np.random.default_rng(7)
    factors = jnp.asarray(rng.normal(size=(3, 4, 4)) * .1)
    factors = .5 * (factors + factors.swapaxes(1, 2))
    energy = jnp.array([-.7, -.2, .4, 1.1])
    ov = factors[:, :2, 2:]
    model = rpa_resolvent(energy, ov, factors)
    poles, couplings = rpa_pole_expansion(energy, ov, factors)
    frequencies = jnp.array([0., .3, 1.])
    np.testing.assert_allclose(
        screened_w_imag_resolvent(model, frequencies),
        screened_w_imag_poles(poles, couplings, frequencies), atol=2e-11, rtol=2e-11,
    )
    # Duplicate a transition gap: the resolvent remains finite and its AD is
    # independent of any eigenvector ordering inside the degenerate block.
    energy_deg = jnp.array([-.5, -.5, .5, .5])
    def loss(scale):
        b = factors * scale
        return jnp.sum(jnp.real(screened_w_imag_resolvent(
            rpa_resolvent(energy_deg, b[:, :2, 2:], b), jnp.array([.2])
        )) ** 2)
    grad = __import__('jax').grad(loss)(1.)
    assert np.isfinite(grad)


def test_resolvent_exact_degeneracy_jit_joint_response():
    import jax
    from gradscf.gw.poles import rpa_resolvent, screened_w_imag_resolvent, screened_w_real_resolvent
    # Two uncoupled transitions with equal oscillator coupling: M=lambda I.
    # A perturbation splits the poles and changes their eigenvectors.
    energy = jnp.array([-.5, .5, .5])
    factors = jnp.array([[[.3, .2, 0.], [.2, .1, .05], [0., .05, .2]],
                         [[.1, 0., .2], [0., .2, .03], [.2, .03, .4]]])
    direction = jnp.array([[[.02, .03, .01], [.03, -.01, .02], [.01, .02, -.02]],
                           [[.01, -.02, .01], [-.02, .04, 0.], [.01, 0., -.03]]])
    shifts = jnp.array([-.01, .02, -.03])
    def loss(t, reference=False):
        f, e = factors+t*direction, energy+t*shifts
        frequency = jnp.array([.2, .7])+.05*t
        eta = .03+.002*t
        model = rpa_resolvent(e, f[:, :1, 1:], f, eta=eta)
        if reference:
            def at(part, z):
                m, g = part['matrix'], part['coupling']
                x = jnp.linalg.solve(m-z*z*jnp.eye(m.shape[0]), g)
                return -4*jnp.sum(g*x, axis=0)
            imag = jax.vmap(lambda z: at(model, z))(1j*frequency)
            real = jax.vmap(lambda z: at(model['retarded'], z))(frequency+1j*eta)
        else:
            imag = screened_w_imag_resolvent(model, frequency)
            real = screened_w_real_resolvent(model, frequency)
        return jnp.real(jnp.vdot(imag, imag)+jnp.vdot(real, real))
    model = rpa_resolvent(energy, factors[:, :1, 1:], factors)
    np.testing.assert_allclose(model['matrix'], jnp.eye(2)*1.16, atol=1e-13)
    np.testing.assert_allclose(jax.jit(loss)(0.), loss(0., True), atol=2e-12)
    derivative = jax.jit(jax.grad(loss))(0.)
    np.testing.assert_allclose(derivative, jax.grad(lambda t: loss(t, True))(0.), atol=2e-11)
    np.testing.assert_allclose(jax.jit(lambda t: jax.jvp(loss, (t,), (1.,))[1])(0.), derivative, atol=2e-11)
    np.testing.assert_allclose(derivative, (loss(1e-5)-loss(-1e-5))/2e-5, atol=2e-9)


def test_selected_resolvent_pairs_match_full_tensor():
    import jax
    from gradscf.gw.poles import rpa_resolvent, screened_w_real_resolvent
    rng = np.random.default_rng(5)
    b = jnp.asarray(rng.normal(size=(3, 4, 4))*.1)
    b = (b+b.swapaxes(1,2))*.5
    w, eta = jnp.array([.1, .3, .5]), .03
    model = rpa_resolvent(jnp.array([-.8, -.4, .3, .9]), b[:,:2,2:], b, eta=eta)
    pairs = jnp.array([3, 9, 12])  # deliberately nonconsecutive MO-pair columns
    full = screened_w_real_resolvent(model, w).reshape(3, -1)
    selected = jax.jit(lambda m,z: screened_w_real_resolvent(m,z,pairs=pairs))(model,w)
    np.testing.assert_allclose(selected, full[jnp.arange(3), pairs], atol=2e-12)


def test_residue_resolvent_respects_nonconsecutive_g_window():
    from gradscf.gw.poles import rpa_resolvent
    from gradscf.gw.self_energy import sigma_residue_part
    rng = np.random.default_rng(32)
    b = jnp.asarray(rng.normal(size=(3, 5, 5))*.1)
    b = (b+b.swapaxes(1,2))*.5
    energy = jnp.array([-.8, -.4, .3, .7, 1.1])
    ov = b[:,:2,2:]
    model = rpa_resolvent(energy, ov, b)
    p, omega = 4, 1.2
    indices = jnp.array([0, 2, 4])
    params = dict(omega=omega, mo_energy=energy[indices], b_pm=b[:,p,indices],
                  b_mp=b[:,indices,p], channels=((energy, ov, 2.),), ef=-.05, eta=0.)
    direct = sigma_residue_part(**params)
    fast = sigma_residue_part(**params, resolvent_data=model, p_index=p, g_indices=indices)
    np.testing.assert_allclose(fast, direct, atol=2e-12)


def test_finite_eta_resolvent_matches_legacy_near_resonance_and_response():
    import jax
    from gradscf.gw.polarizability import rho_response_real
    from gradscf.gw.poles import rpa_resolvent, screened_w_real_resolvent
    energy = jnp.array([-.5, .5, .9])
    base = jnp.array([[[.5, .2, .1], [.2, .3, .08], [.1, .08, .4]],
                      [[.1, .03, .15], [.03, .2, .04], [.15, .04, .25]]])
    def values(t, reference=False):
        e = energy + t*jnp.array([.02, -.03, .01])
        b = base*(1+.07*t)
        eta = .001+.0002*t
        frequencies = jnp.array([0., .7, 1.076, 1.5])+.01*t
        if reference:
            def at(w):
                pi = rho_response_real(w, e, b[:, :1, 1:], eta=eta)
                wc = jnp.linalg.solve(jnp.eye(2)-pi, pi)
                return jnp.einsum('Pmn,PQ,Qmn->mn', b, wc, b)
            return jax.vmap(at)(frequencies)
        return screened_w_real_resolvent(rpa_resolvent(e, b[:, :1, 1:], b, eta=eta), frequencies)
    np.testing.assert_allclose(jax.jit(values)(0.), values(0., True), atol=2e-10, rtol=2e-11)
    tangent = jax.jit(lambda t: jax.jvp(values, (t,), (1.,))[1])(0.)
    reference = jax.jvp(lambda t: values(t, True), (0.,), (1.,))[1]
    np.testing.assert_allclose(tangent, reference, atol=2e-8, rtol=2e-10)
    loss = lambda t: jnp.real(jnp.vdot(values(t), values(t)))
    exact = lambda t: jnp.real(jnp.vdot(values(t, True), values(t, True)))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), jax.grad(exact)(0.), atol=2e-7, rtol=2e-10)
