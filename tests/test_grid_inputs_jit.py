"""A static molecular grid can be built under JIT without caching tracers."""
import jax
import jax.numpy as jnp
import numpy as np


def test_constant_geometry_grid_cache_inside_jit(monkeypatch):
    from gradscf import gto
    from gradscf.scf.inputs import build_rks_integral_inputs
    from gradscf.scf.inputs import grid

    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    spec = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True).to_spec()

    def integral(scale):
        data = build_rks_integral_inputs(atom=spec, basis="sto-3g", xc_spec="pbe",
                                         grids_level=0, integral_backend="native")
        return scale*jnp.sum(data.grid_weights*jnp.sum(data.ao**2, axis=1))

    expected = jax.jit(jax.grad(integral))(1.)
    np.testing.assert_allclose(integral(1.), expected, atol=1e-12, rtol=0)
    assert grid._GRID_AO_INPUT_CACHE
    for bundle in grid._GRID_AO_INPUT_CACHE.values():
        for array in (bundle.coords, bundle.grid_weights, bundle.ao, bundle.ao_deriv1):
            assert not isinstance(array, jax.core.Tracer)
            assert np.isfinite(np.asarray(array)).all()
    np.testing.assert_allclose(jax.jit(integral)(2.), 2*expected, atol=1e-12, rtol=0)
