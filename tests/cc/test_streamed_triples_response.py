"""Bounded triples workspaces must retain the full noncanonical response.

The small six-index tensor implementation is an independent AD oracle here.
These tests exercise JVPs, VJPs, HVPs, and scalar second derivatives, including
changes inside exactly degenerate Fock blocks. Third-order AD fails explicitly.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def moments():
    from gradscf.cc.uccsd import SpinCCIntegrals

    rng = np.random.default_rng(832)
    no, nv = 4, 4
    n = no + nv
    g = rng.normal(size=(n, n, n, n)) * .025
    g += g.transpose(2, 3, 0, 1)
    g = g - g.swapaxes(0, 1)
    g = g - g.swapaxes(2, 3)
    f = np.diag([-1.4, -1.4, -1., -.9, .3, .3, .6, 1.1])
    f[:no, no:] = rng.normal(size=(no, nv)) * .03
    f[no:, :no] = f[:no, no:].T
    t1 = rng.normal(size=(no, nv)) * .03
    t2 = rng.normal(size=(no, no, nv, nv)) * .02
    t2 = t2 - t2.swapaxes(0, 1)
    t2 = t2 - t2.swapaxes(2, 3)
    o, v = slice(None, no), slice(no, None)
    ints = SpinCCIntegrals(*map(jnp.asarray, (
        f, g[o, o, o, o], g[o, o, o, v], g[o, o, v, v],
        g[o, v, o, v], g[o, v, v, v], g[v, v, v, v], 0.,
    )))
    return ints, jnp.asarray(t1), jnp.asarray(t2)


def _kernel():
    from gradscf.cc import _spin_triples

    kernel = getattr(_spin_triples, "_streamed_general_correction", None)
    assert callable(kernel), "A bounded, full-Fock-response triples kernel is missing"
    return kernel


def _value(kernel, ints, t1, t2, weights=(1., 1.)):
    connected, singles, _, _ = kernel(ints, t1, t2, denominator_tol=1e-10)
    return weights[0] * connected + weights[1] * singles


def _joint_response_direction(ints, t1, t2, seed):
    no, nv = t1.shape
    rng = np.random.default_rng(seed)
    fock = rng.normal(size=ints.fock.shape) * .03
    fock = fock + fock.T
    # Both perturbations split an exactly degenerate block at the origin.
    fock[0, 1] = fock[1, 0] = .13
    fock[no, no + 1] = fock[no + 1, no] = -.11
    g = rng.normal(size=(no + nv,) * 4) * .02
    g = g + g.transpose(2, 3, 0, 1)
    g = g - g.swapaxes(0, 1)
    g = g - g.swapaxes(2, 3)
    a1 = rng.normal(size=t1.shape) * .02
    a2 = rng.normal(size=t2.shape) * .02
    a2 = a2 - a2.swapaxes(0, 1)
    a2 = a2 - a2.swapaxes(2, 3)
    o, v = slice(None, no), slice(no, None)
    return tuple(map(jnp.asarray, (
        fock, g[o, v, v, v], g[o, o, o, v], g[o, o, v, v], a1, a2,
    )))


def _response_scalar(kernel, ints, weights=(1., 1.)):
    def scalar(*values):
        fock, ovvv, ooov, oovv, a1, a2 = values
        changed = ints._replace(fock=fock, ovvv=ovvv, ooov=ooov, oovv=oovv)
        return _value(kernel, changed, a1, a2, weights)
    return scalar


@pytest.mark.parametrize("weights", [(1., 1.), (1., 0.), (0., 1.)])
def test_streamed_moments_and_vjp_match_six_index_oracle(moments, weights):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    kernel = _kernel()

    def scalar(fn, fock, ovvv, ooov, oovv, a1, a2):
        changed = ints._replace(fock=fock, ovvv=ovvv, ooov=ooov, oovv=oovv)
        return _value(fn, changed, a1, a2, weights)

    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    differentiate = lambda fn: jax.jit(jax.value_and_grad(
        lambda *xs: scalar(fn, *xs), argnums=tuple(range(len(args)))))(*args)
    expected, expected_vjp = differentiate(_general_correction)
    actual, actual_vjp = differentiate(kernel)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
    for x, y in zip(actual_vjp, expected_vjp):
        assert np.all(np.isfinite(x))
        np.testing.assert_allclose(x, y, atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("operation", ["forward", "vjp"])
def test_streamed_mixed_integral_amplitude_dtypes(moments, operation):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    args = tuple(a.astype(jnp.float32) for a in
                 (ints.fock, ints.ovvv, ints.ooov, ints.oovv)) + (t1, t2)
    assert tuple(a.dtype for a in args) == (jnp.float32,) * 4 + (jnp.float64,) * 2

    def scalar(kernel, *values):
        fock, ovvv, ooov, oovv, a1, a2 = values
        changed = ints._replace(fock=fock, ovvv=ovvv, ooov=ooov, oovv=oovv)
        return _value(kernel, changed, a1, a2)

    # Promote only the already-rounded inputs in the independent tensor oracle.
    promoted = tuple(a.astype(jnp.float64) for a in args)
    reference = lambda *xs: scalar(_general_correction, *xs)
    actual = lambda *xs: scalar(_kernel(), *xs)
    if operation == "forward":
        expected = jax.jit(reference)(*promoted)
        value = jax.jit(actual)(*args)
    else:
        differentiate = lambda fn, xs: jax.jit(jax.value_and_grad(
            fn, argnums=tuple(range(len(xs)))))(*xs)
        expected, expected_cotangents = differentiate(reference, promoted)
        value, cotangents = differentiate(actual, args)
        for primal, cotangent, expected_cotangent in zip(args, cotangents, expected_cotangents):
            assert cotangent.dtype == primal.dtype
            np.testing.assert_allclose(cotangent, expected_cotangent.astype(primal.dtype),
                                       atol=2e-12, rtol=2e-6)
    assert value.dtype == jnp.result_type(*args) == jnp.float64
    np.testing.assert_allclose(value, expected, atol=2e-13, rtol=2e-12)


def test_restricted_streamed_bridge_preserves_mixed_integral_amplitude_dtypes():
    from gradscf.cc._streamed_triples import _restricted_inputs
    from gradscf.cc.integrals import prepare_integrals

    rng = np.random.default_rng(834)
    no, nv = 2, 2
    h = jnp.diag(jnp.array([-1.2, -.9, .5, .8], dtype=jnp.float32))
    ints = prepare_integrals(h, jnp.zeros((4,) * 4, dtype=jnp.float32), nocc=no)
    t1 = jnp.asarray(rng.normal(size=(no, nv)) * .03, dtype=jnp.float64)
    t2 = jnp.asarray(rng.normal(size=(no, no, nv, nv)) * .02, dtype=jnp.float64)
    t2 = .5 * (t2 + t2.transpose(1, 0, 3, 2))
    for amplitude in (t1, t2):
        assert np.any(np.asarray(amplitude) != np.asarray(amplitude.astype(jnp.float32)))

    spin_ints, spin_t1, spin_t2 = _restricted_inputs(ints, t1, t2)
    assert spin_ints.fock.dtype == jnp.float32
    assert spin_t1.dtype == spin_t2.dtype == jnp.result_type(ints.fock, t1, t2)
    np.testing.assert_array_equal(spin_t1[:no, :nv], t1)
    np.testing.assert_array_equal(spin_t1[no:, nv:], t1)
    np.testing.assert_array_equal(spin_t2[:no, no:, :nv, nv:], t2)


@pytest.mark.parametrize("block", ["occupied", "virtual"])
def test_offdiagonal_fock_response_inside_exact_degenerate_block(moments, block):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    kernel = _kernel()
    start = 0 if block == "occupied" else t1.shape[0]
    direction = jnp.zeros_like(ints.fock)
    direction = direction.at[start, start + 1].set(.13)
    direction = direction.at[start + 1, start].set(.13)

    def correction(fn, x):
        return _value(fn, ints._replace(fock=ints.fock + x * direction), t1, t2)

    expected = jax.grad(lambda x: correction(_general_correction, x))(0.)
    actual = jax.jit(jax.grad(lambda x: correction(kernel, x)))(0.)
    forward = jax.jvp(lambda x: correction(kernel, x), (0.,), (1.,))[1]
    finite_difference = (correction(kernel, 1e-4) - correction(kernel, -1e-4)) / 2e-4
    assert abs(expected) > 1e-9, "The chosen direction must expose lost offdiagonal response"
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(forward, expected, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(actual, finite_difference, atol=2e-11, rtol=2e-7)


def _all_array_shapes(jaxpr):
    if hasattr(jaxpr, "jaxpr"):
        yield from _all_array_shapes(jaxpr.jaxpr)
    elif hasattr(jaxpr, "eqns") and hasattr(jaxpr, "invars"):
        for variable in (*jaxpr.constvars, *jaxpr.invars, *jaxpr.outvars):
            shape = getattr(getattr(variable, "aval", None), "shape", None)
            if shape is not None:
                yield tuple(shape)
        for equation in jaxpr.eqns:
            for variable in (*equation.invars, *equation.outvars):
                shape = getattr(getattr(variable, "aval", None), "shape", None)
                if shape is not None:
                    yield tuple(shape)
            for value in equation.params.values():
                yield from _all_array_shapes(value)
    elif isinstance(jaxpr, (tuple, list)):
        for value in jaxpr:
            yield from _all_array_shapes(value)
    elif isinstance(jaxpr, dict):
        for value in jaxpr.values():
            yield from _all_array_shapes(value)


def test_forward_and_reverse_do_not_materialize_six_index_moments(moments):
    ints, t1, t2 = moments
    kernel = _kernel()
    scalar = lambda fock: _value(kernel, ints._replace(fock=fock), t1, t2)
    for fn in (scalar, jax.grad(scalar)):
        shapes = list(_all_array_shapes(jax.make_jaxpr(fn)(ints.fock)))
        assert shapes
        assert all(len(shape) < 6 for shape in shapes), shapes
        # A flattened/partly packed history can have fewer than six axes;
        # inspect size as well so that stacking all panels cannot pass.
        largest_allowed = max(t1.size, t2.size, ints.fock.size, ints.ovvv.size,
                              ints.ooov.size, ints.oovv.size,
                              t1.shape[1] * t1.shape[0]**3)
        assert all(np.prod(shape) <= largest_allowed for shape in shapes), shapes


def test_restricted_public_path_retains_implicit_amplitude_response(monkeypatch):
    from gradscf import cc
    from gradscf.cc import _spin_triples
    from gradscf.cc.integrals import prepare_integrals

    rng = np.random.default_rng(833)
    n, no = 4, 2
    factors = rng.normal(size=(5, n, n)) * .06
    factors += factors.transpose(0, 2, 1)
    eri = jnp.asarray(np.einsum("Lpq,Lrs->pqrs", factors, factors))
    mean_field = prepare_integrals(jnp.zeros((n, n)), eri, nocc=no).fock
    h = jnp.diag(jnp.array([-1.2, -1.2, .5, .5])) - mean_field
    direction = jnp.array([
        [.03, .04, .01, -.02], [.04, -.02, .03, .01],
        [.01, .03, .02, .07], [-.02, .01, .07, -.03],
    ])
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    initial = cc.run_cc(h, eri, nocc=no, config=cfg)
    assert initial.converged

    def tensor_reference(x):
        shifted = h + x * direction
        state = cc.run_cc(shifted, eri, nocc=no, config=cfg)
        return cc.triples_correction(shifted, eri, state, nocc=no,
                                      orbital_basis="semicanonical")
    expected, expected_derivative = jax.jit(jax.value_and_grad(tensor_reference))(0.)

    def reject_full_spin_integrals(*args, **kwargs):
        raise AssertionError("Restricted streamed triples must not build full spin-orbital ERIs")
    monkeypatch.setattr(_spin_triples, "prepare_ucc_integrals", reject_full_spin_integrals)

    def correction(x, relax=True):
        shifted = h + x * direction
        state = cc.run_cc(shifted, eri, nocc=no, config=cfg) if relax else initial
        return cc.triples_correction(
            shifted, eri, state, nocc=no, orbital_basis="semicanonical",
            # The spin-orbital panel has 4 * 4**3 elements, below the
            # previous full tensor's (4 * 4)**3 elements.
            algorithm="streamed", max_workspace_elements=256, residual_tol=1.,
        )

    value, actual = jax.jit(jax.value_and_grad(correction))(0.)
    fixed = jax.jit(jax.grad(lambda x: correction(x, relax=False)))(0.)
    fd = (correction(1e-4) - correction(-1e-4)) / 2e-4
    assert np.isfinite(actual)
    np.testing.assert_allclose(value, expected, atol=2e-13, rtol=2e-10)
    np.testing.assert_allclose(actual, expected_derivative, atol=2e-11, rtol=2e-7)
    np.testing.assert_allclose(actual, fd, atol=2e-10, rtol=2e-5)
    assert abs(actual - fixed) > 1e-10, "The correction must retain CC amplitude response"


def test_streamed_workspace_limit_is_separate_from_tensor_capacity(moments):
    from gradscf import cc

    _kernel()
    n, no = 4, 2
    eri = jnp.zeros((n,) * 4)
    h = jnp.diag(jnp.array([-1.2, -1.2, .5, .5]))
    state = cc.run_cc(h, eri, nocc=no)
    kwargs = dict(nocc=no, orbital_basis="semicanonical", algorithm="streamed",
                  max_triples_elements=1)
    correction = cc.triples_correction(h, eri, state, max_workspace_elements=256,
                                       **kwargs)
    assert correction == 0.
    with pytest.raises(ValueError, match="max_workspace_elements"):
        cc.triples_correction(h, eri, state, max_workspace_elements=255, **kwargs)


def test_streamed_second_order_retains_uniform_fock_shift_invariance(moments):
    ints, t1, t2 = moments
    kernel = _kernel()
    def scalar(x):
        fock = ints.fock + x * jnp.eye(ints.fock.shape[0])
        return _value(kernel, ints._replace(fock=fock), t1, t2)
    reference = scalar(0.)
    np.testing.assert_allclose(scalar(.17), reference, atol=2e-13, rtol=2e-12)
    first = jax.jit(jax.grad(scalar))(0.)
    second = jax.jit(jax.grad(jax.grad(scalar)))(0.)
    np.testing.assert_allclose(first, 0., atol=2e-12, rtol=0)
    np.testing.assert_allclose(second, 0., atol=2e-12, rtol=0)


def test_streamed_third_order_ad_is_rejected_explicitly(moments):
    ints, t1, t2 = moments
    direction = jnp.zeros_like(ints.fock)
    direction = direction.at[0, 1].set(.13).at[1, 0].set(.13)
    scalar = lambda x: _value(
        _kernel(), ints._replace(fock=ints.fock + x * direction), t1, t2)
    with pytest.raises(NotImplementedError, match="second.order"):
        jax.jit(jax.grad(jax.grad(jax.grad(scalar))))(0.)


@pytest.mark.parametrize("weights", [(1., 1.), (1., 0.), (0., 1.)])
def test_streamed_hvp_at_exact_degeneracy_matches_six_index_oracle(moments, weights):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    no = t1.shape[0]
    assert ints.fock[0, 0] == ints.fock[1, 1]
    assert ints.fock[no, no] == ints.fock[no + 1, no + 1]
    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    direction = _joint_response_direction(ints, t1, t2, 835)

    def hvp(kernel):
        gradient = jax.grad(_response_scalar(kernel, ints, weights), argnums=tuple(range(6)))
        return jax.jit(lambda *xs: jax.jvp(gradient, xs, direction)[1])(*args)

    expected = hvp(_general_correction)
    actual = hvp(_kernel())
    assert max(np.max(np.abs(value)) for value in expected) > 1e-9, (
        "The joint direction must expose a nonzero Hessian-vector product")
    for value, reference in zip(actual, expected):
        assert np.all(np.isfinite(value))
        np.testing.assert_allclose(value, reference, atol=2e-11, rtol=2e-9)


def test_streamed_gradgrad_direction_matches_six_index_oracle(moments):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    direction = _joint_response_direction(ints, t1, t2, 835)

    def curvature(kernel):
        scalar = _response_scalar(kernel, ints)
        curve = lambda x: scalar(*(a + x * d for a, d in zip(args, direction)))
        return jax.jit(jax.grad(jax.grad(curve)))(0.)

    expected = curvature(_general_correction)
    actual = curvature(_kernel())
    assert np.isfinite(actual)
    assert abs(expected) > 1e-9, "The direction must expose a nonzero curvature"
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-9)
    gradient = jax.grad(_response_scalar(_kernel(), ints), argnums=tuple(range(6)))
    hvp = jax.jit(lambda *xs: jax.jvp(gradient, xs, direction)[1])(*args)
    contraction = sum(jnp.vdot(d, h) for d, h in zip(direction, hvp))
    np.testing.assert_allclose(actual, contraction, atol=2e-11, rtol=2e-9)


def test_streamed_mixed_directions_have_symmetric_hessian(moments):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    left = _joint_response_direction(ints, t1, t2, 835)
    right = _joint_response_direction(ints, t1, t2, 836)

    def mixed(kernel, first, second):
        scalar = _response_scalar(kernel, ints)
        surface = lambda x, y: scalar(*(
            a + x * u + y * v for a, u, v in zip(args, left, right)))
        return jax.jit(jax.grad(jax.grad(surface, first), second))(0., 0.)

    expected = mixed(_general_correction, 0, 1)
    actual_xy = mixed(_kernel(), 0, 1)
    actual_yx = mixed(_kernel(), 1, 0)
    assert np.isfinite(actual_xy) and np.isfinite(actual_yx)
    np.testing.assert_allclose(actual_xy, expected, atol=2e-11, rtol=2e-9)
    np.testing.assert_allclose(actual_yx, expected, atol=2e-11, rtol=2e-9)
    np.testing.assert_allclose(actual_xy, actual_yx, atol=2e-12, rtol=2e-10)


def test_hvp_and_gradgrad_do_not_materialize_six_index_moments_or_panel_history(moments):
    ints, t1, t2 = moments
    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    direction = _joint_response_direction(ints, t1, t2, 835)
    scalar = _response_scalar(_kernel(), ints)
    gradient = jax.grad(scalar, argnums=tuple(range(6)))
    hvp = lambda *xs: jax.jvp(gradient, xs, direction)[1]
    curve = lambda x: scalar(*(a + x * d for a, d in zip(args, direction)))
    gradgrad = jax.grad(jax.grad(curve))
    largest_allowed = max(*(a.size for a in args), t1.shape[1] * t1.shape[0]**3)
    for fn, values in ((hvp, args), (gradgrad, (0.,))):
        shapes = list(_all_array_shapes(jax.make_jaxpr(fn)(*values)))
        assert shapes
        assert all(len(shape) < 6 for shape in shapes), shapes
        assert all(np.prod(shape) <= largest_allowed for shape in shapes), shapes


def test_streamed_denominator_guard_does_not_clip_physical_singularity(moments):
    ints, t1, t2 = moments
    kernel = _kernel()
    # Keep finite input tensors but make every occupied/virtual denominator zero.
    bad = ints._replace(fock=jnp.zeros_like(ints.fock))
    _, _, minimum, valid = kernel(bad, t1, t2, denominator_tol=1e-10)
    assert minimum == 0.
    assert not valid


def test_streamed_hvp_in_rotated_nondiagonal_basis_matches_six_index_oracle(moments):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    no, nv = t1.shape
    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    direction = _joint_response_direction(ints, t1, t2, 835)
    rng = np.random.default_rng(837)
    qo = np.linalg.qr(rng.normal(size=(no, no)))[0]
    qv = np.linalg.qr(rng.normal(size=(nv, nv)))[0]
    rotation = np.zeros((no + nv, no + nv))
    rotation[:no, :no], rotation[no:, no:] = qo, qv

    def rotate(values):
        # Independent NumPy transforms; do not call the production frame bridge.
        fock, ovvv, ooov, oovv, a1, a2 = map(np.asarray, values)
        return tuple(map(jnp.asarray, (
            rotation.T @ fock @ rotation,
            np.einsum("pi,qa,rb,sc,pqrs->iabc", qo, qv, qv, qv, ovvv),
            np.einsum("pi,qj,rk,sa,pqrs->ijka", qo, qo, qo, qv, ooov),
            np.einsum("pi,qj,ra,sb,pqrs->ijab", qo, qo, qv, qv, oovv),
            qo.T @ a1 @ qv,
            np.einsum("pi,qj,ra,sb,pqrs->ijab", qo, qo, qv, qv, a2),
        )))

    rotated, rotated_direction = rotate(args), rotate(direction)
    offdiagonal_norms = []
    for block in (slice(None, no), slice(no, None)):
        original_fock = np.asarray(args[0])[block, block]
        rotated_fock = np.asarray(rotated[0])[block, block]
        offdiagonal_norm = np.linalg.norm(rotated_fock - np.diag(np.diag(rotated_fock)))
        assert offdiagonal_norm > .05, "The rotated Fock block must be noncanonical"
        offdiagonal_norms.append(offdiagonal_norm)
        np.testing.assert_allclose(np.linalg.eigvalsh(rotated_fock),
                                   np.diag(original_fock), atol=2e-14, rtol=0)

    def evaluator(kernel):
        scalar = _response_scalar(kernel, ints)
        gradient = jax.grad(scalar, argnums=tuple(range(6)))
        return jax.jit(lambda values, tangent: (
            scalar(*values), jax.jvp(gradient, values, tangent)[1]))

    reference, actual = evaluator(_general_correction), evaluator(_kernel())
    ref_value, ref_hvp = reference(args, direction)
    rotated_ref_value, rotated_ref_hvp = reference(rotated, rotated_direction)
    value, hvp = actual(args, direction)
    rotated_value, rotated_hvp = actual(rotated, rotated_direction)
    assert max(np.max(np.abs(h)) for h in rotated_ref_hvp) > 1e-9
    for h, expected in zip(rotated_hvp, rotated_ref_hvp):
        assert np.all(np.isfinite(h))
        np.testing.assert_allclose(h, expected, atol=2e-11, rtol=2e-9)
    energies = (ref_value, rotated_ref_value, value, rotated_value)
    np.testing.assert_allclose(energies, ref_value, atol=2e-13, rtol=2e-12)
    curvatures = [sum(jnp.vdot(d, h) for d, h in zip(tangent, response))
                  for tangent, response in ((direction, ref_hvp),
                                            (rotated_direction, rotated_ref_hvp),
                                            (direction, hvp),
                                            (rotated_direction, rotated_hvp))]
    assert abs(curvatures[0]) > 1e-9
    np.testing.assert_allclose(curvatures, curvatures[0], atol=2e-11, rtol=2e-9)
    print("rotated triples HVP: Foo/Fvv offdiagonal norms = %.6f/%.6f; "
          "max HVP oracle error = %.3e; energy spread = %.3e Ha; "
          "curvature spread = %.3e Ha"
          % (*offdiagonal_norms,
             max(np.max(np.abs(h - r)) for h, r in zip(rotated_hvp, rotated_ref_hvp)),
             np.ptp(np.asarray(energies)), np.ptp(np.asarray(curvatures))))


def test_streamed_jit_hessian_api_matches_two_direction_six_index_oracle(moments):
    from gradscf.cc._spin_triples import _general_correction

    ints, t1, t2 = moments
    args = (ints.fock, ints.ovvv, ints.ooov, ints.oovv, t1, t2)
    left = _joint_response_direction(ints, t1, t2, 835)
    right = _joint_response_direction(ints, t1, t2, 836)

    def hessian(kernel):
        scalar = _response_scalar(kernel, ints)
        surface = lambda theta: scalar(*(
            a + theta[0] * u + theta[1] * v for a, u, v in zip(args, left, right)))
        return jax.jit(jax.hessian(surface))(jnp.zeros(2, dtype=jnp.float64))

    expected = hessian(_general_correction)
    actual = hessian(_kernel())
    assert actual.shape == expected.shape == (2, 2)
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(expected)) > 1e-9
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-9)
    np.testing.assert_allclose(actual, actual.T, atol=2e-12, rtol=2e-10)
    print("jit(jax.hessian) two-direction matrix = %s; max oracle error = %.3e; "
          "symmetry error = %.3e"
          % (np.asarray(actual).tolist(), np.max(np.abs(actual - expected)),
             np.max(np.abs(actual - actual.T))))
