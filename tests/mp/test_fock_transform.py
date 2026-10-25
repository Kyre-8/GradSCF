"""Physical SCF Fock transport preserves symmetry and its full response."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _transform_fock():
    module = importlib.import_module('gradscf.ao2mo.reference')
    transform = getattr(module, 'transform_fock', None)
    assert transform is not None, 'SCF Fock transport needs a shared Hermitian transform'
    return transform


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('nmo', [3, 5])
def test_physical_fock_transform_is_symmetric_and_preserves_inputs(dtype, nmo):
    transform = _transform_fock()
    rng = np.random.default_rng(722)
    raw = rng.normal(size=(7, 7)).astype(dtype)
    fock = (raw+raw.T)*dtype(.5)
    coefficients = rng.normal(size=(7, nmo)).astype(dtype)
    saved_fock, saved_coefficients = fock.copy(), coefficients.copy()
    actual = jax.jit(transform)(fock, coefficients)
    expected = coefficients.T @ fock @ coefficients
    assert actual.shape == (nmo, nmo) and actual.dtype == dtype
    np.testing.assert_array_equal(actual, actual.T)
    tolerance = 32*np.finfo(dtype).eps
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(fock, saved_fock)
    np.testing.assert_array_equal(coefficients, saved_coefficients)
    # Both occupied/occupied and virtual/virtual off-diagonal entries survive.
    if nmo == 5:
        assert abs(float(actual[0, 1])) > .01
        assert abs(float(actual[3, 4])) > .01


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_physical_fock_coefficient_and_geometry_gradients_and_hvp(dtype):
    transform = _transform_fock()
    rng = np.random.default_rng(142)
    point = (jnp.asarray(rng.normal(scale=.2, size=(5, 5)), dtype),
             jnp.asarray(rng.normal(scale=.2, size=(5, 4)), dtype),
             jnp.asarray([.2, -.1, .3], dtype))
    directions = tuple(jnp.asarray(rng.normal(scale=.1, size=x.shape), dtype)
                       for x in point)
    ao_response = jnp.asarray(rng.normal(scale=.1, size=(3, 5, 5)), dtype)
    coefficient_response = jnp.asarray(rng.normal(scale=.1, size=(3, 5, 4)), dtype)
    weight = jnp.asarray(rng.normal(size=(4, 4)), dtype)

    def energy(operation):
        def evaluate(raw_fock, coefficients, geometry):
            ao = raw_fock+jnp.einsum('k,kpq->pq', jnp.sin(geometry), ao_response)
            fock = .5*(ao+ao.T)
            c = coefficients+jnp.einsum('k,kpq->pq', geometry, coefficient_response)
            mo = operation(fock, c)
            return jnp.sum(weight*mo**2)
        return evaluate

    reference = energy(lambda fock, c: c.T @ fock @ c)
    physical = energy(transform)
    expected = jax.jit(jax.value_and_grad(reference, argnums=(0, 1, 2)))(*point)
    actual = jax.jit(jax.value_and_grad(physical, argnums=(0, 1, 2)))(*point)
    expected_hvp = jax.jit(lambda *p: jax.jvp(
        jax.grad(reference, argnums=(0, 1, 2)), p, directions)[1])(*point)
    actual_hvp = jax.jit(lambda *p: jax.jvp(
        jax.grad(physical, argnums=(0, 1, 2)), p, directions)[1])(*point)
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-13
    for got, want in zip(jax.tree.leaves((actual, actual_hvp)),
                         jax.tree.leaves((expected, expected_hvp))):
        np.testing.assert_allclose(got, want, atol=tolerance, rtol=tolerance)
    assert all(np.linalg.norm(np.asarray(row)) > 1e-5 for row in actual[1])


@pytest.mark.parametrize('unrestricted', [False, True], ids=['rhf', 'uhf'])
def test_scf_and_correlated_mp2_adapters_supply_symmetric_focks(unrestricted, monkeypatch):
    from gradscf import gto, mp, scf
    from gradscf.mp.integrals import from_scf
    from gradscf.mp.mp2 import evaluate_mp2
    from gradscf_tools.correlated_response import CorrelatedEnergy

    atom = ('H 0 0 0; H 0 0 1.5; H 1.4 0 .2' if unrestricted
            else 'O 0 0 0; H 0 -1.4 1.1; H 0 1.4 1.1')
    mol = gto.M(atom=atom, basis='sto-3g', spin=int(unrestricted), unit='Bohr', cart=True)
    source = (scf.UHF if unrestricted else scf.RHF)(mol, integral_backend='native',
        execution_device='cpu', conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, max_cycle=250).run()
    assert source.converged
    ints = from_scf(source)
    for fock in ints.focks:
        np.testing.assert_array_equal(fock, fock.T)
    reference = evaluate_mp2(ints, mp.MPConfig(orbital_basis='noncanonical', with_t2=False))
    assert reference.valid

    workflow = importlib.import_module('gradscf_tools.correlated_response')
    original = workflow.from_orbitals
    observed = []

    def checked(coefficients, focks, *args, **kwargs):
        observed.extend(focks)
        return original(coefficients, focks, *args, **kwargs)

    monkeypatch.setattr(workflow, 'from_orbitals', checked)
    model = CorrelatedEnergy(source, 'mp2')
    value = model(model.coordinates)
    assert np.isfinite(value) and len(observed) == (2 if unrestricted else 1)
    for fock in observed:
        np.testing.assert_array_equal(fock, fock.T)
    np.testing.assert_allclose(value, reference.total_energy, atol=3e-10, rtol=0)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_explicit_nonhermitian_orbital_fock_remains_invalid(unrestricted):
    from gradscf.mp import MPConfig
    from gradscf.mp.integrals import from_orbitals
    from gradscf.mp.mp2 import evaluate_mp2

    fock = jnp.diag(jnp.array([-1.2, -.8, .4, .7])).at[2, 3].set(.02)
    identity = jnp.eye(4)
    if unrestricted:
        coefficients = (identity, identity)
        focks, nocc = (fock, jnp.diag(jnp.array([-1.1, .2, .5, .8]))), (2, 1)
    else:
        coefficients, focks, nocc = identity, (fock,), 2
    ints = from_orbitals(coefficients, focks, nocc, 0., eri=jnp.zeros((4,)*4))
    assert ints.finite
    np.testing.assert_array_equal(ints.focks[0], fock)
    result = evaluate_mp2(ints, MPConfig(orbital_basis='noncanonical', with_t2=False))
    assert not result.valid and np.isnan(result.correlation_energy)


def test_general_integral_transform_keeps_raw_nonsymmetric_ad_chart():
    from gradscf.ao2mo.reference import transform_integrals

    rng = np.random.default_rng(411)
    hcore, tangent = (jnp.asarray(rng.normal(size=(4, 4))) for _ in range(2))
    coefficients = jnp.asarray(rng.normal(size=(4, 3)))
    eri = jnp.zeros((4,)*4)
    operation = lambda h: transform_integrals(h, coefficients, eri=eri)[0]
    value, response = jax.jvp(operation, (hcore,), (tangent,))
    np.testing.assert_allclose(value, coefficients.T @ hcore @ coefficients, atol=2e-14, rtol=0)
    np.testing.assert_allclose(response, coefficients.T @ tangent @ coefficients, atol=2e-14, rtol=0)
    assert np.linalg.norm(np.asarray(value-value.T)) > .1
