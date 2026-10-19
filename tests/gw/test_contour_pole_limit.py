"""Analytic one-screening-pole checks at the Green-function contour boundary."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.gw.freq import scaled_legendre_grid
from gradscf.gw.self_energy import sigma_cd


@pytest.mark.parametrize('explicit_occupation',[True,False])
@pytest.mark.parametrize('occupied',[True,False])
def test_own_pole_value_and_frequency_derivative(occupied,explicit_occupation):
    freq,weight=scaled_legendre_grid(80)
    gap=1.; transition=.2; vertex=.5
    coupling=4*transition**2*gap
    excitation=np.sqrt(gap**2+coupling)
    spectral_weight=vertex**2*coupling/(2*excitation)
    energy=-.3 if occupied else .3
    ctx=dict(mo_energy=jnp.array([energy]),ef=0.,eta=0.,freqs=freq,wts=weight,
             wmn_p=(-vertex**2*coupling/(freq**2+excitation**2))[:,None],
             wmn_static=jnp.array([-vertex**2*coupling/excitation**2]),
             b_pm=jnp.array([[vertex]]),b_mp=jnp.array([[vertex]]),
             channels=((jnp.array([-.5,.5]),jnp.array([[[transition]]]),2.),))
    if explicit_occupation:
        ctx['ef']=energy  # Empty/full spin channels need no artificial gap.
        ctx['occupation_sign']=jnp.array([-1. if occupied else 1.])
    fun=lambda w:jnp.real(sigma_cd(w,ctx))
    sign=1 if occupied else -1
    value,slope=jax.jit(jax.value_and_grad(fun))(energy)
    np.testing.assert_allclose(value,sign*spectral_weight/excitation,atol=1e-12)
    np.testing.assert_allclose(slope,-spectral_weight/excitation**2,atol=1e-11)
    for delta in (-.03,.03):
        np.testing.assert_allclose(fun(energy+delta),spectral_weight/(delta+sign*excitation),atol=2e-10)


@pytest.mark.parametrize('occupied',[True,False])
def test_own_pole_parameter_response_matches_spectral_model(occupied):
    freq,weight=scaled_legendre_grid(80)
    transition=.2
    coupling=4*transition**2
    excitation=jnp.sqrt(1+coupling)
    energy=-.3 if occupied else .3
    def value(t):
        vertex=.5*(1+.1*t)
        pole=energy+.03*t
        ctx=dict(mo_energy=jnp.array([pole]),ef=0.,eta=0.,freqs=freq,wts=weight,
            wmn_p=(-vertex**2*coupling/(freq**2+excitation**2))[:,None],
            wmn_static=jnp.array([-vertex**2*coupling/excitation**2]),
            b_pm=jnp.array([[vertex]]),b_mp=jnp.array([[vertex]]),
            channels=((jnp.array([-.5,.5]),jnp.array([[[transition]]]),2.),))
        return jnp.real(sigma_cd(pole,ctx))
    expected=(1 if occupied else -1)*.1*.5**2*coupling/excitation**2
    np.testing.assert_allclose(jax.jit(jax.grad(value))(0.),expected,atol=1e-12)
    np.testing.assert_allclose(jax.jvp(value,(0.,),(1.,))[1],expected,atol=1e-12)


def _paired_gamma_rpa_context(p, head, *, omega=None, nw=80):
    """One-auxiliary paired RPA with nonzero diagonal and OV vertices."""
    from gradscf.gw.pbc.q0 import screened_w_imag_axis_head_wing
    from gradscf.gw.polarizability import rho_response_iw
    from gradscf.gw.screened import screened_w_imag_axis

    energy = jnp.array([-.4, .6])
    factors = jnp.array([[[.27, .18], [.18, .31]]], dtype=jnp.complex128)
    bov = factors[:, :1, 1:]
    frequencies, quadrature = scaled_legendre_grid(nw)
    response = lambda frequency: rho_response_iw(frequency, energy, bov,
                                                  spin_factor=4., conjugate=True)
    interaction = screened_w_imag_axis(factors, response, frequencies, conjugate=True)
    static = screened_w_imag_axis(factors, response, jnp.zeros(1), conjugate=True)[0]
    ctx = dict(mo_energy=energy, ef=.1, eta=0., freqs=frequencies, wts=quadrature,
        wmn_p=interaction[:, :, p], wmn_static=static[:, p],
        b_pm=factors[:, :, p], b_mp=factors[:, :, p],
        channels=((energy, bov, 2.),), conjugate=True, p_index=p,
        occupation_sign=jnp.array([-1., 1.]))
    gap, transition = 1., .18
    body_strength = 4*gap*transition**2
    body_pole = np.sqrt(gap**2+body_strength)
    vertices = np.asarray(factors)[0, :, p]
    frequency = float(energy[p]) if omega is None else omega
    expected = np.sum(vertices**2*body_strength/(2*body_pole)*
        np.array([1/(frequency-float(energy[0])+body_pole),
                  1/(frequency-float(energy[1])-body_pole)]))
    if head:
        qij, q_abs, volume = jnp.array([[.00009]]), jnp.array([.001, 0., 0.]), 30.
        _, del00, wing = screened_w_imag_axis_head_wing(
            factors, bov, energy, qij, q_abs, volume, frequencies)
        _, head0, wing0 = screened_w_imag_axis_head_wing(
            factors, bov, energy, qij, q_abs, volume, jnp.zeros(1))
        ctx.update(del_w=del00+wing[:, p], del_w_static=head0[0]+wing0[0, p],
            q0=dict(qij=qij, q_abs=q_abs, volume=volume, nkpts=1, p_index=p))
        # The Schur head/wing has a single augmented RPA pole. Its strength
        # follows from the rank-one dielectric, independently of sigma_cd.
        head_strength = 16*np.pi*gap*float(qij[0, 0])**2/float(q_abs[0])**2
        augmented_pole2 = gap**2+body_strength+head_strength
        head_scale = 2/np.pi*(6*np.pi**2/volume)**(1/3)
        wing_scale = np.sqrt(volume/(4*np.pi**3))*(6*np.pi**2/volume)**(2/3)
        cross = np.sqrt(4*np.pi)/float(q_abs[0])*4*gap*transition*float(qij[0, 0])
        strength = head_scale*head_strength+2*wing_scale*vertices[p].real*cross
        correction0 = -strength/augmented_pole2
        np.testing.assert_allclose(ctx['del_w_static'], correction0, atol=2e-14)
        augmented_pole = np.sqrt(augmented_pole2)
        expected += strength/(2*augmented_pole)/(frequency-float(energy[p])
                    +(augmented_pole if p == 0 else -augmented_pole))
    return energy, factors, ctx, expected


@pytest.mark.parametrize('p', [0, 1])
@pytest.mark.parametrize('head', [False, True])
def test_paired_gamma_body_and_head_have_the_own_green_pole_half_residue(p, head):
    energy, _, ctx, expected = _paired_gamma_rpa_context(p, head)
    actual = sigma_cd(energy[p], ctx)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize('p', [0, 1])
@pytest.mark.parametrize('delta', [-1e-5, 1e-5])
def test_gamma_head_static_subtraction_is_continuous_on_both_sides_of_green_pole(p, delta):
    omega = (-.4 if p == 0 else .6)+delta
    # Resolve the narrow nu ~ |delta| region after static subtraction;
    # 320 nodes converge this 1e-5 Ha endpoint probe below the same tolerance.
    _, _, ctx, expected = _paired_gamma_rpa_context(p, True, omega=omega, nw=320)
    np.testing.assert_allclose(sigma_cd(omega, ctx), expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize('head', [False, True])
def test_gamma_driver_evaluate_only_supplies_body_and_matching_head_static(head, monkeypatch):
    from types import SimpleNamespace
    import importlib
    gamma = importlib.import_module('gradscf.gw.pbc.krgw')
    energy, factors, ctx0, expected0 = _paired_gamma_rpa_context(0, head)
    _, _, _, expected1 = _paired_gamma_rpa_context(1, head)
    monkeypatch.setattr(gamma, 'gamma_product_factors', lambda *a, **k: factors)
    monkeypatch.setattr(gamma, 'get_jk', lambda *a, **k:
        (jnp.zeros((2, 2)), jnp.zeros((2, 2, 2))))
    if head:
        monkeypatch.setattr(gamma, 'transition_moments_gamma', lambda *a:
            (ctx0['q0']['qij'], ctx0['q0']['q_abs']))
    result = gamma.g0w0_cd_gamma(inputs=SimpleNamespace(weights=jnp.array([30.])),
        mo_energy=energy, mo_coeff=jnp.eye(2), nocc=1, fock_matrix=jnp.zeros((2, 2)),
        hcore_matrix=jnp.zeros((2, 2)), density_spin=jnp.zeros((2, 2, 2)), mesh=(1, 1, 1),
        nw=80, eta=0., fc=head, evaluate_only=True)
    np.testing.assert_allclose(result.sigma_qp, [expected0, expected1], atol=2e-12, rtol=0)


@pytest.mark.parametrize('p', [0, 1])
def test_matrix_contour_endpoint_skips_inactive_bare_poles_and_matches_spectral_rpa(p):
    from gradscf.gw.self_energy import sigma_imag_matrix, sigma_residue_matrix
    energy, factors, ctx, diagonal_reference = _paired_gamma_rpa_context(p, False)
    coupling = 4*.18**2
    excitation = np.sqrt(1+coupling)
    vertices = np.asarray(factors)[0]
    tensor = np.einsum('mq,qn->qmn', vertices.conj(), vertices)
    interaction = (-coupling/(ctx['freqs']**2+excitation**2))[:, None, None, None]*tensor[None]
    static = -coupling/excitation**2*tensor
    arguments = dict(wmn_static=static, occupation_sign=ctx['occupation_sign'])
    actual = sigma_imag_matrix(energy[p], interaction, energy, ctx['ef'],
        ctx['freqs'], ctx['wts'], 0., **arguments)
    actual += sigma_residue_matrix(energy[p], energy, factors, ctx['channels'],
        ctx['ef'], 0., conjugate=True, **arguments)
    denominators = float(energy[p])-np.asarray(energy)+np.array([excitation, -excitation])
    expected = coupling/(2*excitation)*np.einsum('qmn,q->mn', tensor, 1/denominators)
    assert np.isfinite(np.asarray(actual)).all()
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(actual[p, p], diagonal_reference, atol=2e-12, rtol=0)
