"""Full complex Lehmann channels against literal NumPy Fourier/response sums."""
import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.pbc import product_basis as products
from gradscf.gw.pbc.ksigma import _transition_weights
from gradscf.gw.screened import solve_transition_screening


@pytest.fixture
def model():
    mesh = (3, 3, 3)
    lattice = np.diag([5., 6., 7.])
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    fractions = np.array([[0., 0., 0.], [1/3, 0., 0.], [-1/3, 0., 0.]])
    table = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    indices = np.stack(np.meshgrid(*[np.fft.fftfreq(n)*n for n in mesh], indexing='ij'), -1).reshape(-1, 3)
    coordinates = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), -1).reshape(-1, 3)@lattice
    volume, nk, nmo = np.linalg.det(lattice), len(fractions), 4
    rng = np.random.default_rng(20261016)
    orbitals = np.stack([np.linalg.qr(rng.normal(size=(len(indices), nmo))
        +1j*rng.normal(size=(len(indices), nmo)))[0] for _ in range(nk)])
    orbitals *= np.sqrt(len(indices)/volume)
    ao = np.zeros((nk, 4, len(indices), nmo), dtype=complex)
    ao[:, 0] = orbitals
    inputs = SimpleNamespace(ao=ao, kpoints=fractions@reciprocal,
        gvectors=indices@reciprocal, coords=coordinates, lattice=lattice,
        weights=np.full(len(indices), volume/len(indices)),
        overlap=np.broadcast_to(np.eye(nmo), (nk, nmo, nmo)),
        kweights=np.full(nk, 1/nk), boundary=None)
    energy = np.array([[-1.1, -.6, .4, 1.2], [-1., -.5, .6, 1.3], [-1.2, -.7, .3, 1.1]])
    coefficients = jnp.broadcast_to(jnp.eye(nmo), (nk, nmo, nmo))
    return inputs, coefficients, energy, table, mesh, 2


def numpy_factors(model, q):
    inputs, _, energy, table, mesh, _ = model
    volume, ngrid = np.linalg.det(inputs.lattice), len(inputs.coords)
    norm = np.sum((inputs.gvectors-inputs.kpoints[q])**2, axis=1)
    kernel = np.divide(4*np.pi, norm, out=np.zeros_like(norm), where=norm>1e-24)
    result = []
    for k, partner in enumerate(table[:, q]):
        left, right = inputs.ao[k, 0], inputs.ao[partner, 0]
        wrap = inputs.kpoints[partner]-inputs.kpoints[k]+inputs.kpoints[q]
        pair = left.conj()[:, :, None]*right[:, None, :]
        pair *= np.exp(1j*(inputs.coords@wrap))[:, None, None]
        transformed = np.fft.fftn(pair.reshape(*mesh, energy.shape[1], energy.shape[1]), axes=(0, 1, 2))
        result.append(transformed.reshape(ngrid, energy.shape[1], energy.shape[1])
                      *np.sqrt(volume*kernel)[:, None, None]/ngrid)
    return np.asarray(result)


def numpy_response(model, factors, q, frequency, eta):
    _, _, energy, table, _, nocc = model
    nk, nmo = energy.shape
    occupation = np.arange(nmo) < nocc
    result = np.zeros((factors.shape[1],)*2, dtype=complex)
    for k, partner in enumerate(table[:, q]):
        for m in range(nmo):
            for n in range(nmo):
                difference = int(occupation[m])-int(occupation[n])
                if not difference:
                    continue
                gap = energy[k, m]-energy[partner, n]
                pole = frequency+1j*eta*(1 if frequency >= 0 else -1) if eta is not None else 1j*frequency
                denominator = pole+gap+(1j*eta*difference if eta is not None else 0.)
                vector = factors[k, :, m, n]
                result += (2*difference/(nk*denominator))*np.outer(vector, vector.conj())
    return result


def test_full_lehmann_api():
    assert 'full' in inspect.signature(products.kpoint_transition_block).parameters
    assert callable(getattr(products, 'lehmann_weights', None))
    assert callable(getattr(products, 'mixed_product_factors', None))


@pytest.mark.parametrize('frequency,eta', [(0., None), (.37, None), (-.37, None),
                                         (.13, .017), (-.13, .017)])
def test_full_channels_match_numpy(model, frequency, eta):
    inputs, c, energy, table, mesh, nocc = model
    q = 1
    build = products.kpoint_product_factor_builder(inputs, c, mesh=mesh, momentum_table=table)
    block = products.kpoint_transition_block(build, energy, nocc, table, q, full=True)
    assert block['factors'].shape[1] == 2*len(c)*nocc*(energy.shape[1]-nocc)
    weights = products.lehmann_weights(frequency, block['differences'], block['signs'], len(c), eta=eta)
    actual = np.asarray((block['factors']*weights)@block['factors'].conj().T)
    expected = numpy_response(model, numpy_factors(model, q), q, frequency, eta)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
    rng = np.random.default_rng(8)
    rhs = rng.normal(size=(len(inputs.coords), 2))+1j*rng.normal(size=(len(inputs.coords), 2))
    solved = solve_transition_screening(block['gram'], weights, block['factors'].conj().T@rhs)
    assert bool(solved.converged)
    reference = np.linalg.solve(np.eye(len(expected))-expected, rhs)-rhs
    np.testing.assert_allclose(block['factors']@solved.solution, reference, atol=2e-12, rtol=2e-11)


def test_default_ov_contract_is_unchanged(model):
    inputs, c, energy, table, mesh, nocc = model
    build = products.kpoint_product_factor_builder(inputs, c, mesh=mesh, momentum_table=table)
    original = products.kpoint_transition_block(build, energy, nocc, table, 1)
    explicit = products.kpoint_transition_block(build, energy, nocc, table, 1, full=False)
    assert set(original) == {'factors', 'gram', 'differences'}
    for name in original:
        np.testing.assert_array_equal(original[name], explicit[name])
    expected = np.concatenate([a[:, :nocc, nocc:].reshape(len(inputs.coords), -1)
                               for a in numpy_factors(model, 1)], axis=1)
    np.testing.assert_allclose(original['factors'], expected, atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize('frequency,eta', [(0., None), (.37, None), (.13, 0.), (.13, .017)])
def test_paired_channels_recover_old_frequency_convention(frequency, eta):
    differences = jnp.array([-.7, -1.3])
    full = products.lehmann_weights(frequency, jnp.r_[differences, -differences],
        jnp.array([1., 1., -1., -1.]), 3, eta=eta)
    np.testing.assert_allclose(full[:2]+full[2:],
        _transition_weights(frequency, differences, 3, eta), atol=2e-14, rtol=2e-13)


def test_residue_weights_follow_both_green_pole_displacements():
    gap, eta = .7, .013
    differences, signs = jnp.array([-gap, gap]), jnp.array([1., -1.])
    positive = products.lehmann_weights(gap, differences, signs, 1, eta=eta)
    negative = products.lehmann_weights(-gap, differences, signs, 1, eta=eta)
    np.testing.assert_allclose(positive[0], 2/(2j*eta), atol=1e-13)
    np.testing.assert_allclose(negative[1], -2/(-2j*eta), atol=1e-13)
    np.testing.assert_allclose(positive.sum(), negative.sum(), atol=1e-13)
    for frequency in (.13, -.13):
        pole = frequency+1j*eta*(1 if frequency > 0 else -1)
        expected = 2*np.asarray(signs)/(pole+np.asarray(differences)+1j*eta*np.asarray(signs))
        np.testing.assert_allclose(products.lehmann_weights(frequency, differences, signs, 1, eta=eta),
                                   expected, atol=2e-14, rtol=2e-13)


def test_imaginary_branches_are_adjoint_but_not_assumed_even(model):
    inputs, c, energy, table, mesh, nocc = model
    build = products.kpoint_product_factor_builder(inputs, c, mesh=mesh, momentum_table=table)
    block = products.kpoint_transition_block(build, energy, nocc, table, 1, full=True)
    responses = []
    for frequency in (.37, -.37):
        weights = products.lehmann_weights(frequency, block['differences'], block['signs'], len(c))
        responses.append(np.asarray((block['factors']*weights)@block['factors'].conj().T))
    np.testing.assert_allclose(responses[0], responses[1].conj().T, atol=2e-13, rtol=2e-12)
    assert not np.allclose(*responses, atol=1e-10, rtol=1e-10)


def test_complex_orbital_phases_preserve_full_response(model):
    inputs, c, energy, table, mesh, nocc = model
    phases = np.exp(1j*np.random.default_rng(19).normal(size=energy.shape))
    responses = []
    for coefficients in (c, c*phases[:, None, :]):
        build = products.kpoint_product_factor_builder(inputs, coefficients, mesh=mesh, momentum_table=table)
        block = products.kpoint_transition_block(build, energy, nocc, table, 1, full=True)
        weights = products.lehmann_weights(.37, block['differences'], block['signs'], len(c))
        responses.append((block['factors']*weights)@block['factors'].conj().T)
    np.testing.assert_allclose(*responses, atol=2e-13, rtol=2e-12)


def test_mixed_pairs_match_independent_fourier_sum(model):
    inputs, _, _, table, mesh, _ = model
    pairs = np.array([[0, 2], [1, 3], [3, 0]])
    for k in range(len(table)):
        partner = table[k, 1]
        actual = products.mixed_product_factors(inputs, inputs.ao[k, 0], inputs.ao[partner, 0],
            q_cart=inputs.kpoints[1], left_k=inputs.kpoints[k], right_k=inputs.kpoints[partner],
            pairs=pairs, mesh=mesh)
        expected = np.column_stack([numpy_factors(model, 1)[k, :, m, n] for m, n in pairs])
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)


def test_static_bse_matches_full_lehmann_limit(model):
    from gradscf.bse.pbc import build_tda_matrix
    inputs, c, energy, table, mesh, no = model
    actual = build_tda_matrix(inputs, c, energy, energy, no, mesh=mesh)
    factors = [numpy_factors(model, q) for q in range(len(c))]
    inverse = [np.linalg.inv(np.eye(len(inputs.coords))
        -numpy_response(model, f, q, 0., None)) for q, f in enumerate(factors)]
    space = [(k, v, a) for k in range(len(c)) for v in range(no) for a in range(no, energy.shape[1])]
    expected = np.empty(actual.shape, dtype=complex)
    for i, (k, v, a) in enumerate(space):
        for j, (kp, vp, ap) in enumerate(space):
            q = np.flatnonzero(table[k] == kp)[0]
            exchange = np.vdot(factors[0][k, :, v, a], factors[0][kp, :, vp, ap])
            direct = np.vdot(factors[q][k, :, v, vp], inverse[q]@factors[q][k, :, a, ap])
            expected[i, j] = (2*exchange-direct)/len(c)+(energy[k, a]-energy[k, v] if i == j else 0.)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
