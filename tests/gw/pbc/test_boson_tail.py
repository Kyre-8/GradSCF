"""Independent dressed auxiliary-bath endpoint and bosonic second moments."""
import importlib
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from test_dynamic_tail import _case
from test_matsubara import _model, _fermi


def _module():
    module = importlib.import_module('gradscf.gw.pbc.matsubara')
    assert callable(getattr(module, '_bubble_second_moment', None)), \
        'Missing dressed bosonic second-moment tail capability'
    return module


def _complex_auxiliary_model():
    model = _model()
    rotation = jnp.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    return replace(model, factors=tuple(jnp.einsum('PQ,kQab->kPab', rotation, b)
                                       for b in model.factors))


def _lehmann(model, spectra, frame, beta):
    occupation = _fermi(beta*spectra)
    first, second, transitions = [], [], []
    for q, factor in enumerate(model.factors):
        b = np.asarray(factor)
        m1 = np.zeros((b.shape[1], b.shape[1]), complex)
        m2 = np.zeros_like(m1)
        records = []
        for k in range(len(spectra)):
            partner = model.table[k, q]
            for r in range(spectra.shape[1]):
                for s in range(spectra.shape[1]):
                    vertex = np.einsum('a,Pab,b->P', frame[k, :, r].conj(),
                                       b[k], frame[partner, :, s])
                    gap = spectra[partner, s]-spectra[k, r]
                    residue = (2/len(spectra)*(occupation[k, r]-occupation[partner, s])
                               *np.outer(vertex, vertex.conj()))
                    m1 += residue
                    m2 += gap*residue
                    if abs(gap) > 1e-12:
                        records.append((gap, residue))
        first.append(m1)
        second.append(m2)
        transitions.append(records)
    return tuple(first), tuple(second), tuple(transitions)


@pytest.mark.parametrize('beta', [5., 37.])
def test_dressed_auxiliary_reference_endpoint_derivatives_and_lehmann_pi2(beta):
    module = _module()
    (grid, fock, sigma, moment, green, density, _, d0, dbeta, spectra, frame) = _case(
        module, nw=5, beta=beta, endpoints=True)
    reference = module._green_tail_reference(jnp.asarray(fock), jnp.asarray(sigma[:, -1]),
                                             .04, grid, jnp.asarray(moment))
    assert len(reference) == 5, 'The numerical G reference must expose its endpoint derivatives'
    np.testing.assert_allclose(reference[3], d0, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(reference[4], dbeta, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(d0+dbeta, fock-.04*np.eye(2), atol=2e-14)
    derivative_beta = reference[4]+np.einsum('w,kwij->kij',
        1j*np.asarray(grid.fermion), green-np.asarray(reference[1]))/grid.beta
    derivative_zero = fock-.04*np.eye(2)-derivative_beta
    model = _complex_auxiliary_model()
    first, second, _ = _lehmann(model, spectra, frame, beta)
    actual_first = module._bubble_moment(jnp.asarray(density), model.factors, model.table)
    actual_second = module._bubble_second_moment(jnp.asarray(density), jnp.asarray(derivative_zero),
        jnp.asarray(derivative_beta), model.factors, model.table)
    for left, right in zip(actual_first, first):
        np.testing.assert_allclose(left, right, atol=3e-13, rtol=3e-12)
    for left, right in zip(actual_second, second):
        np.testing.assert_allclose(left, right, atol=3e-13, rtol=3e-12)
    # Complex Bloch vertices and noncommuting endpoint matrices are retained.
    assert np.max(abs(np.asarray(second[0]).imag)) > 1e-5
    assert np.max(abs(density@d0-d0@density)) > 1e-4


def test_screened_second_moment_matches_independent_large_frequency_lehmann_limit():
    module = _module()
    (grid, fock, _, moment, green, density, _, _, _, spectra, frame) = _case(module, endpoints=True)
    model = _complex_auxiliary_model()
    first, second, transitions = _lehmann(model, spectra, frame, grid.beta)
    step = module.gw_matsubara_step(jnp.asarray(green), jnp.asarray(fock), .04, model, grid,
        sigma_moment=jnp.asarray(moment), density=jnp.asarray(density))
    assert 'screened_second_moment' in step
    z = 1j*1e4
    for q, records in enumerate(transitions):
        pi = sum(residue/(z-gap) for gap, residue in records)
        screened = np.linalg.solve(np.eye(len(pi))-pi, pi)
        large_frequency = z*z*(screened-first[q]/z)
        large_frequency = .5*(large_frequency+large_frequency.conj().T)
        np.testing.assert_allclose(step['polarizability_second_moment'][q], second[q], atol=3e-12)
        np.testing.assert_allclose(step['screened_second_moment'][q], large_frequency, atol=3e-8, rtol=3e-8)
        np.testing.assert_allclose(step['screened_second_moment'][q], second[q]+first[q]@first[q], atol=3e-12)
    siv, stau = module._even_reference(grid, .7)
    np.testing.assert_allclose(siv, -1/(np.asarray(grid.boson)**2+.7**2), atol=1e-15)
    assert np.max(np.asarray(stau)) < 0
    np.testing.assert_allclose(stau, np.asarray(stau)[::-1], atol=2e-15)


def test_dressed_second_moment_is_invariant_under_complex_orbital_frame_rotation():
    module = _module()
    (_, _, _, _, _, density, _, d0, dbeta, _, _) = _case(module, endpoints=True)
    model = _complex_auxiliary_model()
    source = importlib.import_module('gradscf.gw.pbc.source')
    rotation = jnp.array([[[1., 1j], [1j, 1.]], [[1., -1j], [-1j, 1.]]])/np.sqrt(2.)
    transform = lambda value: rotation.conj().swapaxes(-1, -2)@value@rotation
    before = module._bubble_second_moment(jnp.asarray(density), jnp.asarray(d0), jnp.asarray(dbeta),
                                          model.factors, model.table)
    after = module._bubble_second_moment(transform(density), transform(d0), transform(dbeta),
                                         source.rotate_factors(model, rotation), model.table)
    for left, right in zip(before, after):
        np.testing.assert_allclose(left, right, atol=3e-13, rtol=3e-12)


def _many_bath_case(module, nw):
    """Three bath orbitals cannot be represented by a two-orbital pole A."""
    grid = module.matsubara_grid(nw=nw, beta=8.)
    fock = np.array([[[-.6, .13+.07j], [.13-.07j, .7]],
                     [[-.4, -.09+.11j], [-.09-.11j, .9]]])
    coupling = np.array([[[.31+.08j, .04-.09j, .13+.06j],
                          [.11+.07j, .23-.02j, -.08+.11j]],
                         [[.27-.04j, -.08+.06j, .09-.03j],
                          [.09+.03j, .35+.07j, .12+.04j]]])
    bath = np.array([[[-1.1, .15+.03j, -.07+.06j],
                      [.15-.03j, .4, .12-.04j], [-.07-.06j, .12+.04j, 1.4]],
                     [[-.8, -.09+.05j, .11+.02j],
                      [-.09-.05j, .6, .08-.03j], [.11-.02j, .08+.03j, 1.7]]])
    moment = coupling@coupling.conj().swapaxes(-1, -2)
    density, derivative, spectra, frames, green, sigma = [], [], [], [], [], []
    for f, v, a in zip(fock, coupling, bath):
        h = np.block([[f, v], [v.conj().T, .04*np.eye(3)+a]])
        energy, coefficient = np.linalg.eigh(h)
        xi, physical = energy-.04, coefficient[:2]
        occupation = _fermi(grid.beta*xi)
        density.append((physical*occupation)@physical.conj().T)
        derivative.append((physical*(xi*occupation))@physical.conj().T)
        spectra.append(xi)
        frames.append(physical)
        green.append(np.einsum('as,ws,bs->wab', physical,
            1/(1j*np.asarray(grid.fermion)[:, None]-xi), physical.conj()))
        sigma.append(np.array([v@np.linalg.solve(1j*w*np.eye(3)-a, v.conj().T)
                               for w in np.asarray(grid.fermion)]))
    return (grid, fock, np.array(sigma), moment, np.array(green), np.array(density),
            np.array(derivative), np.array(spectra), np.array(frames))


def test_endpoint_finite_residual_and_pi2_converge_for_a_larger_independent_bath():
    module = _module()
    model = _complex_auxiliary_model()
    errors = []
    for nw in (8, 16, 32):
        grid, fock, sigma, moment, green, density, dbeta, spectra, frame = _many_bath_case(module, nw)
        reference = module._green_tail_reference(jnp.asarray(fock), jnp.asarray(sigma[:, -1]),
                                                 .04, grid, jnp.asarray(moment))
        d1 = reference[4]+np.einsum('w,kwij->kij', 1j*np.asarray(grid.fermion),
                                    green-np.asarray(reference[1]))/grid.beta
        _, second, _ = _lehmann(model, spectra, frame, grid.beta)
        step = module.gw_matsubara_step(jnp.asarray(green), jnp.asarray(fock), .04, model, grid,
            sigma_moment=jnp.asarray(moment), density=jnp.asarray(density))
        errors.append((np.linalg.norm(d1-dbeta),
            np.linalg.norm(np.asarray(step['polarizability_second_moment'])-np.asarray(second))))
    errors = np.asarray(errors)
    print('Larger-bath endpoint/Pi2 errors nw8/16/32:', errors)
    assert np.all(errors[1] < errors[0]/4), errors
    assert np.all(errors[2] < errors[1]/4), errors
    assert np.max(errors[2]) < 1e-5, errors
