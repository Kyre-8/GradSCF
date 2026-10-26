"""Arbitrary-k GW: non-even contour, eager domain, and a small physical reference."""
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.freq import scaled_legendre_grid


def query_module():
    spec = importlib.util.find_spec('gradscf.gw.pbc.query')
    assert spec is not None, 'The eager arbitrary-k GW entry point is missing'
    return importlib.import_module(spec.name)


def test_public_query_api():
    query = query_module()
    from gradscf.gw.pbc import g0w0_bands
    assert g0w0_bands is query.g0w0_bands


def test_two_sided_integral_retains_odd_response():
    query = query_module()
    frequencies, weights = scaled_legendre_grid(24)
    energy, omega, ef, eta = jnp.array([-.7, .4]), .18, 0., .007
    positive = (jnp.array([.2, .3])[None, :]/(1+frequencies[:, None]**2)
                +1j*frequencies[:, None]*jnp.array([.08, -.12])[None, :]
                /(1+frequencies[:, None]**2)**2)
    negative = positive.conj()
    delta = omega-np.asarray(energy)-1j*eta*np.sign(ef-np.asarray(energy))
    direct = -np.sum(np.asarray(weights)[:, None]*(np.asarray(positive)/(delta-1j*np.asarray(frequencies)[:, None])
        +np.asarray(negative)/(delta+1j*np.asarray(frequencies)[:, None])))/(2*np.pi)
    actual = query._sigma_imag_full(omega, positive, negative, energy, ef, frequencies, weights, eta)
    np.testing.assert_allclose(actual, direct, atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize('a,b', [(.14, .14), (.17, .11)])
def test_contour_matches_independent_rpa_poles(a, b):
    """Pole-residue spectral theorem, independent of the implemented contour."""
    query = query_module()
    gap = .8
    vector = jnp.sqrt(jnp.array([a/2, b/2]))
    pairs = jnp.array([.15, .2])
    block = dict(gram=vector[:, None]*vector[None, :],
        differences=jnp.array([-gap, gap]), signs=jnp.array([1., -1.]),
        projected=vector[:, None]*pairs[None, :])
    energy = jnp.array([-.4, .5])
    freqs, weights = scaled_legendre_grid(160)
    def screened(z):
        return ((a-b)*z+gap*(a+b))/(z*z+(b-a)*z-gap*gap-gap*(a+b))
    positive = screened(1j*freqs[:, None])*pairs[None, :]**2
    negative = screened(-1j*freqs[:, None])*pairs[None, :]**2
    center = (a-b)/2
    radius = np.sqrt(center*center+gap*gap+gap*(a+b))
    pole_pos, pole_neg = center+radius, center-radius
    residue_pos = ((a-b)*pole_pos+gap*(a+b))/(2*pole_pos+b-a)
    residue_neg = ((a-b)*pole_neg+gap*(a+b))/(2*pole_neg+b-a)
    for omega in (-.65, -.1, .2, .8):
        expected = (-float(pairs[0])**2*residue_neg/(omega-float(energy[0])-pole_neg)
                    +float(pairs[1])**2*residue_pos/(omega-float(energy[1])-pole_pos))
        actual = query._sigma_imag_full(omega, positive, negative, energy, 0., freqs, weights, 0.)
        actual += query._sigma_residue_full(omega, energy, [block], 0., 0., 1, None)
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-10)


def test_signed_residue_matches_complex_green_pole():
    query = query_module()
    vector, pairs = jnp.array([.25, .17]), jnp.array([.15, .2])
    differences, signs = jnp.array([-.8, .8]), jnp.array([1., -1.])
    block = dict(gram=vector[:, None]*vector[None, :], differences=differences,
        signs=signs, projected=vector[:, None]*pairs[None, :])
    energies, eta, ef = jnp.array([-.4, .5]), .013, 0.
    for omega in (-.65, -.1, .2, .8):
        expected = 0j
        for m, energy in enumerate(np.asarray(energies)):
            if not (ef < energy < omega or omega < energy < ef):
                continue
            green_pole = omega-energy+1j*eta*np.sign(energy-ef)
            chi = np.sum(np.asarray(vector)**2*2*np.asarray(signs)
                         /(green_pole+np.asarray(differences)+1j*eta*np.asarray(signs)))
            expected += np.sign(omega-ef)*float(pairs[m])**2*(1/(1-chi)-1)
        actual = query._sigma_residue_full(omega, energies, [block], ef, eta, 1, None)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)


def test_query_rejects_jit_and_ad_including_closed_constants():
    query = query_module()
    def call(points):
        return query.g0w0_bands(None, points, [0]).mo_energy.sum()
    for operation in (lambda: jax.jit(call)(jnp.zeros((1, 3))),
                      lambda: jax.jit(lambda: call(np.zeros((1, 3))))(),
                      lambda: jax.grad(lambda x: call(x.reshape(1, 3)))(jnp.zeros(3))):
        with pytest.raises(NotImplementedError, match='eager'):
            operation()


@pytest.mark.parametrize('field,value,message', [
    ('boundary', object(), '3D'), ('unrestricted', True, 'restricted'),
    ('exxdiv', 'vcut_sph', 'exxdiv'), ('xc', 'pbe0', 'LDA/GGA')])
def test_unsupported_domain_rejected_before_build(field, value, message):
    query = query_module()
    mf = SimpleNamespace(unrestricted=False, inputs=SimpleNamespace(boundary=None),
                         cell=SimpleNamespace(dimension=3), xc='pbe', exxdiv='ewald')
    kwargs = dict(fc=False)
    if field == 'fc':
        kwargs['fc'] = value
    elif field == 'boundary':
        mf.inputs.boundary = value
    else:
        setattr(mf, field, value)
    with pytest.raises(NotImplementedError, match=message):
        query.g0w0_bands(mf, np.zeros((1, 3)), [0], **kwargs)


@pytest.fixture(scope='module')
def hydrogen():
    query_module()  # RED must fail before doing an SCF calculation.
    from gradscf.pbc import gto, dft
    cell = gto.M(atom='H 3 3 2.3; H 3 3 3.7', a=np.eye(3)*6., unit='Bohr',
        basis='gth-szv', pseudo='gth-pbe', mesh=(7, 7, 7), precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((3, 1, 1)), xc='pbe',
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150).run()
    assert mf.converged
    return mf


def test_allocation_guard_precedes_band_build(hydrogen, monkeypatch):
    query = query_module()
    def forbidden(*args, **kwargs):
        pytest.fail('Band inputs must not be built after a failed byte guard')
    monkeypatch.setattr(hydrogen, 'get_bands', forbidden)
    with pytest.raises(ValueError, match='max_memory_bytes'):
        query.g0w0_bands(hydrogen, np.zeros((1, 3)), [0], max_memory_bytes=1)


def test_query_checks_joint_gap(hydrogen, monkeypatch):
    query = query_module()
    original = hydrogen.get_bands
    def degenerate(points, **kwargs):
        energy, coefficients = original(points, **kwargs)
        return jnp.zeros_like(energy), coefficients
    monkeypatch.setattr(hydrogen, 'get_bands', degenerate)
    with pytest.raises(ValueError, match='gap|degener'):
        query.g0w0_bands(hydrogen, np.zeros((1, 3)), [0])


@pytest.mark.parametrize('spectrum,selected', [([-.5, -.5, .3, .6], 0),
                                             ([-.5, -.2, .3, .3], 2)])
def test_gapped_selected_degeneracy_requires_subspace(spectrum, selected, monkeypatch):
    query = query_module()
    source = jnp.array([spectrum])
    assert float(source[0, 2]-source[0, 1]) > .1
    inputs = SimpleNamespace(coords=np.zeros((1, 3)), ao=np.zeros((1, 4, 1, 4)),
                             kpoints=np.zeros((1, 3)))
    def bands(points):
        return jnp.tile(source, (len(points), 1)), jnp.tile(jnp.eye(4)[None], (len(points), 1, 1))
    mf = SimpleNamespace(inputs=inputs, get_bands=bands)
    monkeypatch.setattr(query, '_source', lambda *args: (source, 2, np.eye(3), np.zeros((1, 1), int)))
    with pytest.raises(ValueError, match='degenerate.*subspace self-energy'):
        query._prepare(mf, np.zeros((1, 3)), [selected], nw=8, eta=1e-3,
                       fc=False, max_memory_bytes=1024**2, gap_tol=1e-7)


def test_direct_real_capacity_checked_before_band_build(monkeypatch):
    query = query_module()
    from gradscf.solvers import LinearSolverConfig
    source = jnp.array([[-.7, -.4, .3, .6]])
    inputs = SimpleNamespace(coords=np.zeros((1, 3)), ao=np.zeros((1, 4, 1, 4)),
                             kpoints=np.zeros((1, 3)))
    def forbidden(*args, **kwargs):
        pytest.fail('Direct real-space capacity must be checked before building bands')
    mf = SimpleNamespace(inputs=inputs, get_bands=forbidden)
    monkeypatch.setattr(query, '_source', lambda *args: (source, 2, np.eye(3), np.zeros((1, 1), int)))
    with pytest.raises(ValueError, match='max_dense'):
        query.g0w0_bands(mf, np.zeros((1, 3)), [0],
                        linear_config=LinearSolverConfig(method='direct', max_dense=8))


def test_real_offmesh_and_onmesh_full_numpy_reference(hydrogen):
    query = query_module()
    from gradscf_tools.periodic_gw_reference import evaluate_query
    from gradscf.solvers.nonlinear import ScalarRootConfig
    fractions = np.array([[0., 0., 0.], [.13, 0., 0.], [.31, 0., 0.]])
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(hydrogen.cell.lattice)).T
    points = fractions@reciprocal
    options = dict(nw=16, eta=1e-3, self_energy='diagonal',
        root_config=ScalarRootConfig(ftol=1e-9, xtol=1e-9, maxiter=80))
    result = query.g0w0_bands(hydrogen, points, [0, 1], **options)
    reference = evaluate_query(hydrogen, points, [0, 1], np.asarray(result.mo_energy), nw=16, eta=1e-3)
    assert bool(result.converged)
    assert np.asarray(result.qp_computed_mask).all()
    np.testing.assert_allclose(result.qp_residual, reference['residual'], atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(result.sigma_qp, reference['sigma_c'], atol=2e-10, rtol=2e-8)
    assert np.max(np.abs(reference['residual'])) < 2e-9
    assert result.g_orbitals == (0, 1)
    assert result.screening_occupied == (0,)
    assert result.screening_virtual == (1,)
    np.testing.assert_array_equal(result.screening_energy, hydrogen.mo_energy)
    shifted = query.g0w0_bands(hydrogen, points+reciprocal[0], [0], **options)
    np.testing.assert_allclose(shifted.mo_energy[:, 0], result.mo_energy[:, 0], atol=2e-10, rtol=0)
    np.testing.assert_array_equal(shifted.qp_computed_mask, [[True, False]]*3)
    np.testing.assert_allclose(shifted.mo_energy[:, 1], reference['ks_energy'][:, 1], atol=2e-12)


def test_source_domain_boundaries(hydrogen, monkeypatch):
    query = query_module()
    with monkeypatch.context() as patch:
        patch.setattr(hydrogen, 'temperature', .01)
        patch.setattr(hydrogen, '_computed_temperature', .01)
        with pytest.raises(NotImplementedError, match='integer'):
            query.g0w0_bands(hydrogen, np.zeros((1, 3)), [0])
    with monkeypatch.context() as patch:
        weights = jnp.array([.2, .3, .5])
        patch.setattr(hydrogen, 'kweights', weights)
        patch.setattr(hydrogen, '_computed_kweights', weights)
        patch.setattr(hydrogen, 'inputs', hydrogen.inputs._replace(kweights=weights))
        with pytest.raises(NotImplementedError, match='equal k'):
            query.g0w0_bands(hydrogen, np.zeros((1, 3)), [0])
