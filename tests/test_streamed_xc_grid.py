"""AO-grid streaming preserves quadrature and its coordinate/parameter AD.

The tiny Cartesian basis and fixed seven-point quadrature avoid native integral
dependencies.  The SCF assembly regression separately rejects eager full-grid
AO evaluation before an SCF iteration has consumed a block.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.basis import CartesianAO, CartesianBasis
from gradscf.integrals.grids.ao import evaluate_cartesian_ao
from gradscf.integrals.grids.blocks import AOGrid


def _quadrature(parameters):
    center = jnp.asarray([-.3, .1, .2]) + parameters[0] * jnp.asarray([.3, -.2, .1])
    basis = CartesianBasis(aos=(
        CartesianAO(center=center, angular=(0, 0, 0),
                    exponents=jnp.asarray([.7, 1.9]) * jnp.exp(parameters[1]),
                    coefficients=jnp.asarray([.6, .4]) * (1. + parameters[2])),
        CartesianAO(center=jnp.asarray([.4, -.2, .1]), angular=(1, 0, 0),
                    exponents=jnp.asarray([.9]), coefficients=jnp.asarray([.8])),
    ), precompute_eri_groups=False)
    coords = jnp.asarray([[0., 0., 0.], [.2, -.1, .3], [-.4, .5, -.2],
                          [.8, -.7, .1], [.1, .3, -.4], [-.2, -.1, .6],
                          [.5, .2, .1]])
    coords = coords + parameters[3] * jnp.asarray([-.1, .2, .3])
    weights = jnp.asarray([.2, .4, .1, .3, .15, .25, .35]) * jnp.exp(parameters[4])
    return basis, coords, weights


def _integrand(ao, dao, lap, density):
    """A smooth semilocal test energy with nonzero GGA and Laplacian response."""
    rho = jnp.einsum("gp,pq,gq->g", ao, density, ao)
    gradient = 2 * jnp.einsum("xgp,pq,gq->xg", dao[1:4], density, ao)
    laplacian = 2 * (jnp.einsum("gp,pq,gq->g", lap, density, ao)
                    + jnp.einsum("xgp,pq,xgq->g", dao[1:4], density, dao[1:4]))
    return .13 * rho**2 + .03 * jnp.sum(gradient**2, axis=0) + .007 * laplacian**2


def _energy(parameters, density, *, streamed):
    basis, coords, weights = _quadrature(parameters)
    if not streamed:
        values = evaluate_cartesian_ao(basis, coords, deriv=2)
        return jnp.dot(weights, _integrand(values[0], values[:4], values[4], density))
    grid = AOGrid.from_basis(basis, coords, weights, block_size=3)

    def accumulate(energy, ao, dao, lap, block_coords, block_weights):
        return energy + jnp.dot(block_weights, _integrand(ao, dao, lap, density))

    return grid.reduce(accumulate, jnp.asarray(0., dtype=weights.dtype), deriv=2)


@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_ao_grid_blocks_match_dense_with_zero_weight_tail(deriv):
    basis, coords, weights = _quadrature(jnp.zeros(5))
    grid = AOGrid.from_basis(basis, coords, weights, block_size=3)
    assert (grid.nao, grid.ngrids, grid.block_size) == (2, 7, 3)
    dense = evaluate_cartesian_ao(basis, coords, deriv=deriv)
    for start in (0, 3, 6):
        ao, dao, lap, block_coords, block_weights = grid.block(start, 3, deriv=deriv)
        count = min(3, grid.ngrids - start)
        assert ao.shape == (3, basis.nao)
        assert block_coords.shape == (3, 3)
        assert block_weights.shape == (3,)
        np.testing.assert_allclose(block_coords[:count], coords[start:start+count], atol=0, rtol=0)
        np.testing.assert_allclose(block_weights[:count], weights[start:start+count], atol=0, rtol=0)
        np.testing.assert_array_equal(block_weights[count:], 0.)
        values = dense if deriv == 0 else dense[0]
        np.testing.assert_allclose(ao[:count], values[start:start+count], atol=2e-13, rtol=2e-13)
        if deriv:
            assert dao.shape == (4, 3, basis.nao)
            np.testing.assert_allclose(dao[:, :count], dense[:4, start:start+count],
                                       atol=2e-13, rtol=2e-13)
        if deriv == 2:
            assert lap.shape == (3, basis.nao)
            np.testing.assert_allclose(lap[:count], dense[4, start:start+count],
                                       atol=2e-13, rtol=2e-13)


def test_grid_reduce_matches_dense_energy_potential_and_response():
    parameters = jnp.asarray([.03, -.02, .01, .04, -.03])
    density = jnp.asarray([[1., .12], [.12, .7]])
    direction = jnp.asarray([[.2, -.07], [-.07, -.1]])
    results = []
    for streamed in (False, True):
        function = lambda dm: _energy(parameters, dm, streamed=streamed)
        potential = jax.grad(function)
        results.append(jax.jit(lambda dm: (
            function(dm), potential(dm), jax.jvp(potential, (dm,), (direction,))[1]
        ))(density))
    for actual, expected in zip(results[1], results[0]):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_grid_reduce_coordinate_and_basis_parameter_jvp_vjp_hvp():
    parameters = jnp.asarray([.03, -.02, .01, .04, -.03])
    direction = jnp.asarray([.2, -.3, .1, .4, -.2])
    density = jnp.asarray([[1., .12], [.12, .7]])
    results = []
    for streamed in (False, True):
        function = lambda p: _energy(p, density, streamed=streamed)

        def derivatives(point):
            value, pullback = jax.vjp(function, point)
            gradient = pullback(jnp.asarray(1., dtype=point.dtype))[0]
            tangent = jax.jvp(function, (point,), (direction,))[1]
            hvp = jax.jvp(jax.grad(function), (point,), (direction,))[1]
            return value, gradient, tangent, hvp

        results.append(jax.jit(derivatives)(parameters))
    for actual, expected in zip(results[1], results[0]):
        assert np.isfinite(np.asarray(actual)).all()
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)
    # Coordinate, exponent, contraction, grid-coordinate and weight dependence
    # must all survive the block/provider boundary.
    assert np.all(np.abs(np.asarray(results[1][1])) > 1e-8)


def test_grid_map_returns_only_small_feature_pytree_without_tail_points():
    parameters = jnp.asarray([.03, -.02, .01, .04, -.03])
    density = jnp.asarray([[1., .12], [.12, .7]])

    def features(ao, dao, lap, coords, weights):
        rho = jnp.einsum("gp,pq,gq->g", ao, density, ao)
        grad = 2 * jnp.einsum("xgp,pq,gq->gx", dao[1:4], density, ao)
        return {"rho": rho, "local": (grad, jnp.sum(dao[1:4]**2, axis=(0, 2))),
                "coordinates": coords, "weights": weights}

    def mapped(point):
        basis, coords, weights = _quadrature(point)
        return AOGrid.from_basis(basis, coords, weights, block_size=3).map(features, deriv=1)

    basis, coords, weights = _quadrature(parameters)
    values = evaluate_cartesian_ao(basis, coords, deriv=1)
    expected = features(values[0], values, None, coords, weights)
    actual = jax.jit(mapped)(parameters)
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert result.shape[0] == 7
        np.testing.assert_allclose(result, reference, atol=2e-13, rtol=2e-13)


def _graph_shapes(value):
    if hasattr(value, "jaxpr"):
        return _graph_shapes(value.jaxpr)
    if hasattr(value, "eqns"):
        shapes = {item.aval.shape for item in (*value.invars, *value.constvars, *value.outvars)
                  if hasattr(getattr(item, "aval", None), "shape")}
        for equation in value.eqns:
            shapes.update(item.aval.shape for item in equation.outvars
                          if hasattr(getattr(item, "aval", None), "shape"))
            shapes.update(_graph_shapes(equation.params))
        return shapes
    if isinstance(value, dict):
        return set().union(*(_graph_shapes(item) for item in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(_graph_shapes(item) for item in value))
    return set()


def test_reduction_reverse_graph_does_not_retain_every_blocks_ao():
    """Scalar quadrature must recompute block AO instead of stacking its tape.

    This is an array-shape contract, not an assertion about exact XLA RSS.
    Coordinates, weights, small features and basis data may remain global.
    """
    point = jnp.asarray([.03, -.02, .01, .04, -.03])
    density = jnp.asarray([[1., .12], [.12, .7]])

    def function(parameters):
        basis, coords, weights = _quadrature(parameters)
        coords = jnp.concatenate((coords, coords + .1, coords[:5] - .2))
        weights = jnp.concatenate((weights, .8 * weights, .6 * weights[:5]))
        provider = AOGrid.from_basis(basis, coords, weights, block_size=3)

        def accumulate(energy, ao, dao, lap, block_coords, block_weights):
            return energy + jnp.dot(block_weights, _integrand(ao, dao, lap, density))

        return provider.reduce(accumulate, jnp.asarray(0.), deriv=2)

    forward = jax.make_jaxpr(function)(point)
    reverse = jax.make_jaxpr(jax.grad(function))(point)
    shapes = _graph_shapes(forward) | _graph_shapes(reverse)
    # G=19, B=3, nao=2; seven blocks distinguish stacked AO from the three
    # spatial derivative channels.  Global AO or stacked block residuals
    # would negate streaming.
    for forbidden in ((19, 2), (4, 19, 2), (5, 19, 2),
                      (7, 3, 2), (7, 4, 3, 2), (7, 5, 3, 2)):
        assert forbidden not in shapes


@pytest.mark.parametrize("unrestricted", [False, True])
def test_streamed_scf_assembly_does_not_evaluate_full_grid_ao(monkeypatch, unrestricted):
    from gradscf import gto
    from gradscf.scf.inputs import assembly, grid
    from gradscf.scf.rks import RKSConfig
    from gradscf.scf.uks import UKSConfig

    def reject_eager_ao(*args, **kwargs):
        raise AssertionError("streamed SCF assembly must not evaluate full-grid AO")

    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    monkeypatch.setattr(assembly, "evaluate_cartesian_ao_with_derivatives", reject_eager_ao)
    config_type = UKSConfig if unrestricted else RKSConfig
    builder = assembly.build_uks_integral_inputs if unrestricted else assembly.build_rks_integral_inputs
    config = config_type(xc_spec="pbe", grid_block_size=3, grid_max_memory=16.)
    spec = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True).to_spec()
    inputs = builder(atom=spec, basis="sto-3g", config=config,
                     grids_level=0, integral_backend="jax")
    assert inputs.grid_ao is not None
    assert inputs.grid_ao.block_size == 3
    assert inputs.grid_ao.ngrids == inputs.grid_weights.shape[0] > 3
    assert inputs.ao.shape == (0, inputs.overlap.shape[0])
    assert inputs.ao_deriv1.shape == (4, 0, inputs.overlap.shape[0])
    assert all(bundle.ao.size == bundle.ao_deriv1.size == 0
               for bundle in grid._GRID_AO_INPUT_CACHE.values())
    kwargs = inputs.as_uks_kwargs() if unrestricted else inputs.as_rks_kwargs()
    assert kwargs["grid_ao"] is inputs.grid_ao


def test_grid_ao_cache_evicts_by_total_array_bytes(monkeypatch):
    """A count limit must not retain many individually small dense AO grids."""
    from dataclasses import replace
    from gradscf.scf.inputs import grid

    assert grid._GRID_AO_INPUT_CACHE_MAXBYTES == 256_000_000
    dense = grid._GridAOInputBundle(
        basis=None, coords=np.zeros((2, 3)), grid_weights=np.zeros(2),
        ao=np.zeros((2, 2)), ao_deriv1=np.zeros((4, 2, 2)),
        ao_laplacian=np.zeros((2, 2)))
    without_laplacian = replace(dense, ao_laplacian=None)

    def size(bundle):
        return sum(value.nbytes for value in (bundle.coords, bundle.grid_weights,
            bundle.ao, bundle.ao_deriv1, bundle.ao_laplacian) if value is not None)

    # 224 + 256 bytes fit exactly; including every array matters at this edge.
    budget = size(without_laplacian) + size(dense)
    monkeypatch.setattr(grid, '_GRID_AO_INPUT_CACHE_MAXBYTES', budget)
    cache = {}
    grid._cache_bounded(cache, ('first',), without_laplacian, maxsize=32)
    grid._cache_bounded(cache, ('second',), dense, maxsize=32)
    assert tuple(cache) == (('first',), ('second',))
    monkeypatch.setattr(grid, '_GRID_AO_INPUT_CACHE_MAXBYTES', budget - 1)
    grid._cache_bounded(cache, ('third',), without_laplacian, maxsize=32)
    assert tuple(cache) == (('third',),)
    assert sum(map(size, cache.values())) <= budget - 1

    oversized = replace(dense, ao=np.zeros((40, 2)))
    grid._cache_bounded(cache, ('oversized',), oversized, maxsize=32)
    assert ('oversized',) not in cache
    assert sum(map(size, cache.values())) <= budget - 1
