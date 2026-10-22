"""Physical spin-block coordinates preserve the existing packed metric/dual."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.spin_amplitudes import SpinAmplitudeSpace


@pytest.mark.parametrize('occupied,virtual', [((2, 1), (2, 3)), ((0, 2), (3, 2)),
    ((1, 1), (0, 0)), ((0, 0), (2, 3))])
@pytest.mark.parametrize('method', ['ccsd', 'ccd'])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_direct_block_coordinates_match_old_projection_order_metric_and_dual(
        occupied, virtual, method, dtype):
    space = SpinAmplitudeSpace(occupied, virtual, method)
    pack = getattr(space, 'pack_blocks', None)
    unpack = getattr(space, 'unpack_blocks', None)
    assert callable(pack) and callable(unpack), 'Missing direct physical spin-block coordinates'
    oa, ob = occupied
    va, vb = virtual
    rng = np.random.default_rng(6537)
    t1 = tuple(jnp.asarray(rng.normal(size=s), dtype) for s in ((oa, va), (ob, vb)))
    t2 = tuple(jnp.asarray(rng.normal(size=s), dtype) for s in
               ((oa, oa, va, va), (oa, ob, va, vb), (ob, ob, vb, vb)))
    expected = space.pack(*space.from_blocks(t1, t2, dtype))
    actual = pack(t1, t2, dtype)
    assert actual.dtype == expected.dtype == dtype
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    blocks = unpack(actual)
    old = space.to_blocks(*space.unpack(expected))
    for a, b in zip((*blocks[0], *blocks[1]), (*old[0], *old[1])):
        assert a.dtype == b.dtype == dtype
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)
    norm = sum(jnp.vdot(a, a) for a in blocks[0])
    norm += sum(weight*jnp.vdot(a, a) for weight, a in zip((1., 4., 1.), blocks[1]))
    np.testing.assert_allclose(jnp.vdot(actual, actual), norm, atol=10*tolerance, rtol=10*tolerance)
    dual = jnp.asarray(rng.normal(size=actual.shape), dtype)
    l1, l2 = unpack(dual)
    pairing = sum(jnp.vdot(a, b) for a, b in zip(l1, blocks[0]))
    pairing += sum(weight*jnp.vdot(4*a, b) for weight, a, b in
                   zip((.25, 1., .25), l2, blocks[1]))
    np.testing.assert_allclose(pairing, jnp.vdot(dual, actual),
                               atol=10*tolerance, rtol=10*tolerance)


@pytest.mark.parametrize('method', ['ccsd', 'ccd'])
def test_ground_and_left_direct_boundaries_preserve_response(method, monkeypatch):
    import jax
    from gradscf import cc
    from gradscf.cc.uccsd import prepare_ucc_integrals
    from test_prepared_properties import _problem

    h, g = _problem()
    cfg = cc.CCConfig(method=method, residual_tol=1e-11, conv_tol=1e-12,
                      adjoint_tol=1e-12)
    old_from, old_to = SpinAmplitudeSpace.from_blocks, SpinAmplitudeSpace.to_blocks
    pack = getattr(SpinAmplitudeSpace, 'pack_blocks', None)
    unpack = getattr(SpinAmplitudeSpace, 'unpack_blocks', None)
    assert callable(pack) and callable(unpack), 'Missing direct physical spin-block coordinates'

    def value(x):
        ints, no, nv = prepare_ucc_integrals((h+x*jnp.eye(4)*.03, h),
            (g, g*(1+.07*x), g), nocc=(2, 1))
        state = cc.run_ucc_from_integrals(ints, no, nv, config=cfg)
        left = cc.solve_lambda_from_integrals(ints, state, config=cfg)
        return (state.total_energy+.13*jnp.sum(state.t2[1]**2)
                +.07*jnp.sum(left.l2[1]**2))

    def bundle(x):
        return (value(x), jax.jvp(value, (x,), (1.,))[1], jax.grad(value)(x),
                jax.jvp(jax.grad(value), (x,), (1.,))[1])

    def forbidden(*args, **kwargs):
        pytest.fail('Ground/Lambda boundary expanded a full spin amplitude')

    with monkeypatch.context() as trap:
        trap.setattr(SpinAmplitudeSpace, 'from_blocks', forbidden)
        trap.setattr(SpinAmplitudeSpace, 'to_blocks', forbidden)
        jax.make_jaxpr(value)(.01)

    with monkeypatch.context() as old:
        old.setattr(SpinAmplitudeSpace, 'pack_blocks',
                    lambda self, a, b, dtype: self.pack(*old_from(self, a, b, dtype)))
        old.setattr(SpinAmplitudeSpace, 'unpack_blocks',
                    lambda self, v: old_to(self, *self.unpack(v)))
        expected = jax.jit(bundle)(.01)

    monkeypatch.setattr(SpinAmplitudeSpace, 'from_blocks', forbidden)
    monkeypatch.setattr(SpinAmplitudeSpace, 'to_blocks', forbidden)
    actual = jax.jit(lambda x: bundle(x))(.01)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-9)
