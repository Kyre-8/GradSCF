"""One molecular memory budget reaches restricted and unrestricted response."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gradscf import tdscf
from gradscf.tddft.response_options import (
    ResponseKernelOptions,
    normalize_response_kernel_options,
    plan_response_memory,
)


def test_response_defaults_use_sixty_gb_and_larger_grid_blocks():
    options = normalize_response_kernel_options(None)
    assert options.max_memory == 60000.
    assert options.grid_block_size == 8192


@pytest.mark.parametrize("options, explicit, expected", [
    (None, None, 512.),
    ({"direct_workers": 3}, None, 512.),
    ({"max_memory": 256.}, None, 256.),
    ({"max_memory": 256.}, 128., 128.),
    (ResponseKernelOptions(max_memory=1024.), None, 1024.),
])
def test_td_inherits_mf_budget_with_explicit_override_precedence(options, explicit, expected):
    source = SimpleNamespace(xc="hf", max_memory=512.)
    for driver_type in (tdscf.TDA, tdscf.TDDFT):
        driver = driver_type(source, response_kernel_options=options, max_memory=explicit)
        resolved = normalize_response_kernel_options(driver._response_options())
        assert resolved.max_memory == expected


def test_td_array_reference_uses_common_default_budget():
    driver = tdscf.TDDFT(SimpleNamespace(xc="hf"))
    assert normalize_response_kernel_options(driver._response_options()).max_memory == 60000.


def test_resident_storage_reduces_joint_response_budget():
    arguments = (100, 6000, 5, np.float64, 128.)
    free = plan_response_memory(*arguments, integral_cache="none", grid_block_size=8192)
    occupied = plan_response_memory(*arguments, integral_cache="none", grid_block_size=8192,
                                    resident_bytes=80_000_000)
    assert occupied.batch_size <= free.batch_size
    assert occupied.grid_block_size < free.grid_block_size
    with pytest.raises(ValueError, match="memory|Memory"):
        plan_response_memory(*arguments, resident_bytes=128_000_000)


def test_global_neural_grid_fields_limit_vector_batches():
    arguments = (100, 6000, 5, np.float64, 128.)
    local = plan_response_memory(*arguments, integral_cache="none")
    global_model = plan_response_memory(*arguments, integral_cache="none",
                                        grid_vector_bytes=4_000_000)
    assert 1 <= global_model.batch_size < local.batch_size


def test_streamed_primitive_workspace_is_included_in_grid_budget():
    arguments = (100, 6000, 5, np.float64, 128.)
    ordinary = plan_response_memory(*arguments, integral_cache="none", grid_block_size=8192)
    primitive_heavy = plan_response_memory(*arguments, integral_cache="none", grid_block_size=8192,
                                           ao_point_bytes=500_000)
    assert primitive_heavy.grid_block_size < ordinary.grid_block_size


def test_response_resident_estimate_counts_grids_integrals_and_callback_fields():
    from gradscf.tddft.response_options import _response_memory_inputs

    shape = lambda dims: jax.ShapeDtypeStruct(dims, np.float64)
    density = shape((2, 4, 4))
    grid = SimpleNamespace(coords=shape((100, 3)), weights=shape((100,)))
    source = SimpleNamespace(grid=grid, mo_coeff=shape((2, 4, 4)), rdm1=density,
                             ao=shape((100, 4)), ao_deriv1=shape((4, 100, 4)))
    local = SimpleNamespace(response_feature_kind="GGA", grid_response_tensor=lambda _: None)
    opaque = SimpleNamespace(response_feature_kind="GGA", grid_response_hvp=lambda *_: None)
    base = _response_memory_inputs(source, None)
    fixed = _response_memory_inputs(source, local)
    global_model = _response_memory_inputs(source, opaque)
    assert base["resident_bytes"] >= 100 * 4 * 5 * 8
    assert fixed["resident_bytes"] > base["resident_bytes"]
    assert global_model["grid_vector_bytes"] >= 4 * 100 * 8
    source.eri_pair_matrix = shape((10, 10))
    assert _response_memory_inputs(source, None)["resident_bytes"] >= base["resident_bytes"] + 800


@pytest.mark.parametrize("platform, cached", [("cpu", True), ("gpu", False), (None, False)])
def test_integral_cache_uses_target_device_without_querying_default_backend(platform, cached, monkeypatch):
    from gradscf.integrals.backends.native.jk import NativeDirectBasis

    def unexpected_query():
        pytest.fail("Planning must not initialize or query unrelated devices")

    monkeypatch.setattr(jax, "default_backend", unexpected_query)
    provider = NativeDirectBasis(SimpleNamespace(topology=SimpleNamespace(nao=4)), None)
    device = None if platform is None else SimpleNamespace(platform=platform, memory_stats=lambda: {})
    plan = plan_response_memory(4, 3, 1, np.float64, 2., direct_basis=provider, device=device)
    assert bool(plan.eri_cache_bytes) is cached


def test_custom_hf_metadata_does_not_hide_callback_grid_workspace():
    from gradscf.tddft.response_options import _response_memory_inputs

    shape = lambda dims: jax.ShapeDtypeStruct(dims, np.float64)
    source = SimpleNamespace(grid=SimpleNamespace(coords=shape((100, 3)), weights=shape((100,))),
                             mo_coeff=shape((2, 4, 4)))
    custom = SimpleNamespace(xc_spec="hf", response_feature_kind="GGA",
                              grid_response_hvp=lambda *_: None)
    estimate = _response_memory_inputs(source, custom)
    assert estimate["grid_vector_bytes"] >= 4 * 100 * 8

