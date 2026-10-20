"""Semilocal grid actions retain feature algebra without grid-by-OV arrays."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.tddft import response


# Keep an independent small-system oracle after forbidding the production call.
_DENSE_FEATURES = response._restricted_response_features
_FEATURE_COUNTS = {"LDA": 1, "GGA": 4, "MGGA": 5, "MGGA_LAPL": 6}


def _inputs(kind):
    rng = np.random.default_rng(926)
    ngrids, nao, no, nv, batch = 7, 5, 2, 3, 8
    nf = _FEATURE_COUNTS[kind]
    arrays = tuple(jnp.asarray(value) for value in (
        rng.normal(size=(ngrids, nao)),
        rng.normal(size=(4, ngrids, nao)),
        rng.normal(size=(ngrids, nao)),
        rng.normal(size=(nao, no)),
        rng.normal(size=(nao, nv)),
        rng.normal(size=(nf, nf, ngrids)) * .03,
    ))
    # AO values and derivative[0] are deliberately independent dynamic leaves.
    values = jnp.asarray(rng.normal(size=(batch, no, nv)))
    return arrays, values


def _molecule(arrays):
    ao, derivative, laplacian = arrays[:3]
    return SimpleNamespace(
        ao=ao, ao_deriv1=derivative, ao_laplacian=laplacian,
        grid=SimpleNamespace(weights=jnp.ones(ao.shape[0], dtype=ao.dtype)))


def _features(arrays, kind):
    return _DENSE_FEATURES(_molecule(arrays), arrays[3], arrays[4],
                           feature_kind=kind, dtype=jnp.float64)


def _dense(arrays, values, kind):
    features = _features(arrays, kind)
    projected = jnp.einsum("fgia,bia->bfg", features, values)
    weighted = jnp.einsum("fhg,bhg->bfg", arrays[5], projected)
    result = 2 * jnp.einsum("fgia,bfg->bia", features, weighted)
    diagonal = 2 * jnp.einsum("fhg,fgia,hgia->ia", arrays[5], features, features)
    return result, diagonal


def _blocked(arrays, values, kind):
    action, diagonal = response._restricted_grid_xc_response(
        _molecule(arrays), arrays[3], arrays[4], arrays[5],
        feature_kind=kind, dtype=jnp.float64, block_size=3)
    return action(values), diagonal


def _jaxpr_shapes(value):
    if hasattr(value, "jaxpr"):
        return _jaxpr_shapes(value.jaxpr)
    if hasattr(value, "eqns"):
        shapes = {variable.aval.shape
                  for variable in (*value.constvars, *value.invars, *value.outvars)
                  if hasattr(variable.aval, "shape")}
        for equation in value.eqns:
            shapes.update(variable.aval.shape for variable in equation.outvars
                          if hasattr(variable.aval, "shape"))
            shapes.update(_jaxpr_shapes(equation.params))
        return shapes
    if isinstance(value, dict):
        return set().union(*(_jaxpr_shapes(item) for item in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(_jaxpr_shapes(item) for item in value))
    return set()


def _forbid_features(*args, **kwargs):
    raise AssertionError("The tensor grid action must not construct grid-by-OV features")


def test_tensor_grid_default_path_never_constructs_transition_features(monkeypatch):
    arrays, values = _inputs("GGA")
    expected = _dense(arrays, values, "GGA")
    monkeypatch.setattr(response, "_restricted_response_features", _forbid_features)
    action, diagonal = response._restricted_grid_xc_response(
        _molecule(arrays), arrays[3], arrays[4], arrays[5],
        feature_kind="GGA", dtype=jnp.float64)
    np.testing.assert_allclose(action(values), expected[0], atol=3e-11, rtol=2e-12)
    np.testing.assert_allclose(diagonal, expected[1], atol=3e-11, rtol=2e-12)


@pytest.mark.parametrize("kind", _FEATURE_COUNTS)
def test_blocked_grid_action_matches_dense_values_and_leaf_response(monkeypatch, kind):
    arrays, values = _inputs(kind)
    expected = _dense(arrays, values, kind)
    monkeypatch.setattr(response, "_restricted_response_features", _forbid_features)
    actual = jax.jit(lambda data, x: _blocked(data, x, kind))(arrays, values)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, atol=3e-11, rtol=2e-12)

    def observable(build, data, x):
        output, _ = build(data, x, kind)
        weights = jnp.arange(output.size, dtype=x.dtype).reshape(output.shape) / 17
        return jnp.sum(weights * output + .07 * output**2)

    actual_grad = jax.jit(jax.grad(lambda data, x: observable(_blocked, data, x),
                                  argnums=(0, 1)))(arrays, values)
    expected_grad = jax.jit(jax.grad(lambda data, x: observable(_dense, data, x),
                                    argnums=(0, 1)))(arrays, values)
    for value, reference in zip(jax.tree.leaves(actual_grad),
                                 jax.tree.leaves(expected_grad), strict=True):
        np.testing.assert_allclose(value, reference, atol=3e-9, rtol=3e-12)
    direction = jnp.flip(values, axis=-1) / 7
    tangent = jax.jit(lambda x: jax.jvp(lambda x: _blocked(arrays, x, kind)[0],
                                      (x,), (direction,))[1])(values)
    np.testing.assert_allclose(tangent, _dense(arrays, direction, kind)[0],
                               atol=3e-11, rtol=2e-12)

    forward = jax.make_jaxpr(lambda data, x: _blocked(data, x, kind))(arrays, values)
    reverse = jax.make_jaxpr(jax.grad(
        lambda data, x: observable(_blocked, data, x), argnums=(0, 1)))(arrays, values)
    shapes = _jaxpr_shapes(forward) | _jaxpr_shapes(reverse)
    # Full grid, fixed blocks and the one-point tail must all avoid GIA.
    for grids in (7, 3, 1):
        assert (grids, 2, 3) not in shapes
        assert (_FEATURE_COUNTS[kind], grids, 2, 3) not in shapes
    # Reduction must not retain a separate OV contribution for every grid block.
    assert (2, 8, 2, 3) not in shapes


@pytest.mark.parametrize("kind", _FEATURE_COUNTS)
def test_factor_projections_match_dense_features_without_outer_products(kind):
    arrays, values = _inputs(kind)
    features = _features(arrays, kind)
    factors = response._restricted_response_factors(
        _molecule(arrays), arrays[3], arrays[4], feature_kind=kind, dtype=jnp.float64)
    projected = response._project_restricted_transition_to_grid(factors, values)
    np.testing.assert_allclose(projected, jnp.einsum("fgia,bia->bfg", features, values),
                               atol=3e-12, rtol=2e-12)
    weighted = jnp.asarray(np.random.default_rng(927).normal(size=projected.shape))
    back = response._project_grid_response_to_restricted_transition(factors, weighted)
    np.testing.assert_allclose(back, jnp.einsum("fgia,bfg->bia", features, weighted),
                               atol=3e-12, rtol=2e-12)

    graph = jax.make_jaxpr(lambda x, w: (
        response._project_restricted_transition_to_grid(factors, x),
        response._project_grid_response_to_restricted_transition(factors, w),
    ))(values, weighted)
    # Eight vectors exceed each orbital channel, so a three-operand einsum can
    # choose GIA as its cheapest path. Explicit two-step contractions must not.
    assert (7, 2, 3) not in _jaxpr_shapes(graph)


def test_custom_grid_hvp_keeps_cross_grid_coupling_and_full_grid_call():
    kind = "MGGA"
    arrays, values = _inputs(kind)
    arrays = (arrays[1][0], *arrays[1:])
    molecule = _molecule(arrays)
    features = _features(arrays, kind)

    def hvp(mol, tangent):
        assert tangent.shape[-1] == 7
        return (jnp.einsum("fhg,bhg->bfg", arrays[5], tangent)
                + .13 * jnp.sum(tangent, axis=-1, keepdims=True))

    action = response._restricted_grid_xc_response_hvp(
        molecule, arrays[3], arrays[4], hvp, feature_kind=kind, dtype=jnp.float64)

    def dense(x):
        projected = jnp.einsum("fgia,bia->bfg", features, x)
        return 2 * jnp.einsum("fgia,bfg->bia", features, hvp(molecule, projected))

    np.testing.assert_allclose(jax.jit(action)(values), dense(values), atol=3e-11, rtol=2e-12)
    gradient = lambda apply: jax.grad(lambda x: jnp.sum(apply(x)**2))(values)
    np.testing.assert_allclose(gradient(action), gradient(dense), atol=3e-9, rtol=3e-12)


def test_joint_grid_budget_keeps_one_vector_and_one_grid_point_at_minimum():
    from gradscf.tddft.response_options import plan_response_memory

    nao, dim, budget = 4, 4, 12_000
    plan = plan_response_memory(nao, dim, 1, np.float64, budget / 1_000_000,
                                integral_cache="none", grid_block_size=4096)
    vector_bytes = 8 * (12 * nao**2 + 4 * dim)
    grid_point_bytes = 8 * (96 * nao + 384)
    action_bytes = max(vector_bytes + grid_point_bytes, budget // 4)
    assert plan.batch_size == plan.grid_block_size == 1
    assert plan.eri_cache_bytes == 0
    assert plan.batch_size * vector_bytes + plan.grid_block_size * grid_point_bytes <= action_bytes
    with pytest.raises(ValueError, match="memory|Memory"):
        plan_response_memory(nao, dim, 1, np.float64, .01,
                             integral_cache="none", grid_block_size=4096)


def _dimension_only_native_provider(nao):
    from gradscf.integrals.backends.native.jk import NativeDirectBasis

    # Planner-only topology: this never constructs integrals or runs SCF.
    return NativeDirectBasis(SimpleNamespace(topology=SimpleNamespace(nao=nao)), None)


def _topology_only_native_provider(shells, primitives):
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    from gradscf.integrals.basis import BasisTopology

    topology = BasisTopology((0,) * shells, (primitives,) * shells,
                             (1,) * shells, (1,), cart=False)
    return NativeDirectBasis(SimpleNamespace(topology=topology), None)


@pytest.mark.parametrize("rpa", [False, True])
def test_native_optimizer_primitive_pair_storage_cannot_escape_budget(rpa):
    from gradscf.tddft.response_options import plan_response_memory

    # 800 primitives fit libcint's pairdata threshold. The optimizer alone
    # owns 40*800**2 bytes, regardless of the density microbatch width.
    provider = _topology_only_native_provider(80, 10)
    with pytest.raises(ValueError, match="memory|Memory"):
        plan_response_memory(80, 1200, 1, np.float64, 10.,
                             direct_basis=provider, integral_cache="none",
                             direct_workers=8, rpa=rpa)


def test_owned_row_native_workers_do_not_replicate_full_batch_matrices():
    from gradscf.tddft.response_options import plan_response_memory

    provider = _topology_only_native_provider(80, 10)
    controls = dict(direct_basis=provider, integral_cache="none")
    one = plan_response_memory(80, 1200, 1, np.float64, 512.,
                               direct_workers=1, **controls)
    many = plan_response_memory(80, 1200, 1, np.float64, 512.,
                                direct_workers=32, **controls)
    roots = plan_response_memory(80, 1200, 20, np.float64, 512.,
                                 direct_workers=32, **controls)
    assert one.batch_size == many.batch_size == roots.batch_size == 32
    assert many.grid_block_size == roots.grid_block_size == 0


def test_native_primitive_pair_storage_stops_at_libcint_threshold():
    from gradscf.tddft.response_options import plan_response_memory

    # Native shells permit 32 primitives each, but their total exceeds the
    # vendor's 2048-primitive pairdata limit. Shell scratch still applies.
    provider = _topology_only_native_provider(65, 32)
    plan = plan_response_memory(65, 1000, 1, np.float64, 64.,
                                direct_basis=provider, integral_cache="none",
                                direct_workers=8)
    assert plan.eri_cache_bytes == 0
    assert plan.batch_size >= 1


@pytest.mark.parametrize("rpa", [False, True])
def test_joint_budget_accounts_for_32_native_workers_at_full_dabna_dimensions(rpa):
    from gradscf.tddft.response_options import plan_response_memory

    nao, no, nv, roots, workers = 1060, 194, 866, 20, 32
    dim, budget, grid_cap = no * nv, 8_000_000_000, 4096
    provider = _dimension_only_native_provider(nao)
    kwargs = dict(direct_basis=provider, integral_cache="none", rpa=rpa,
                  grid_block_size=grid_cap)
    plan = plan_response_memory(nao, dim, roots, np.float64, budget / 1_000_000,
                                direct_workers=workers, **kwargs)
    one_worker = plan_response_memory(nao, dim, roots, np.float64, budget / 1_000_000,
                                      direct_workers=1, **kwargs)
    builder = plan_response_memory(nao, dim, 1, np.float64, budget / 1_000_000,
                                   direct_workers=workers, **kwargs)
    vector_bytes = 8 * ((12 + 4 * workers) * nao**2 + (8 if rpa else 4) * dim)
    grid_point_bytes = 8 * (96 * nao + 384)
    action_bytes = max(vector_bytes + grid_point_bytes, budget // 4)
    assert plan.eri_cache_bytes == 0
    assert 1 <= plan.batch_size < one_worker.batch_size <= 32
    assert 1 <= plan.grid_block_size <= grid_cap
    assert plan.batch_size * vector_bytes + plan.grid_block_size * grid_point_bytes <= action_bytes
    # The solver's actual root count and the builder's single-root estimate
    # must reserve the same joint action layout, while retaining all roots.
    assert (plan.batch_size, plan.grid_block_size) == (builder.batch_size, builder.grid_block_size)
    assert roots <= plan.max_subspace < dim


def test_hf_zero_grid_plan_releases_grid_quota_for_native_vectors():
    from gradscf.tddft.response_options import plan_response_memory

    nao, dim, budget = 1060, 194 * 866, 8_000_000_000
    kwargs = dict(direct_basis=_dimension_only_native_provider(nao),
                  integral_cache="none", direct_workers=1)
    hf = plan_response_memory(nao, dim, 20, np.float64, budget / 1_000_000,
                              grid_block_size=0, **kwargs)
    grid = plan_response_memory(nao, dim, 20, np.float64, budget / 1_000_000,
                                grid_block_size=4096, **kwargs)
    assert hf.grid_block_size == 0
    assert hf.eri_cache_bytes == 0
    assert 1 <= grid.batch_size < hf.batch_size <= 32
    vector_bytes = 8 * ((12 + 4) * nao**2 + 4 * dim)
    assert hf.batch_size * vector_bytes <= max(vector_bytes, budget // 4)
