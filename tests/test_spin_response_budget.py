"""Spin channels share an integral cache, but retain separate AO work arrays."""
from types import SimpleNamespace

import numpy as np
import pytest

from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.tddft.response_options import plan_response_memory


@pytest.mark.parametrize("rpa", [False, True])
def test_two_spin_channels_share_budget_without_dropping_requested_roots(rpa):
    arguments = (100, 6000, 5, np.float64, 128.)
    restricted = plan_response_memory(*arguments, rpa=rpa, spin_channels=1)
    unrestricted = plan_response_memory(*arguments, rpa=rpa, spin_channels=2)
    assert 1 <= unrestricted.batch_size < restricted.batch_size <= 32
    assert 5 < unrestricted.max_subspace <= restricted.max_subspace
    assert unrestricted.eri_cache_bytes == 0


def test_spin_channels_pay_once_for_shared_s8_cache():
    provider = NativeDirectBasis(SimpleNamespace(topology=SimpleNamespace(nao=4)), None)
    arguments = (4, 3, 1, np.float64, .2)
    device = SimpleNamespace(platform="cpu")
    restricted = plan_response_memory(*arguments, direct_basis=provider, spin_channels=1, device=device)
    unrestricted = plan_response_memory(*arguments, direct_basis=provider, spin_channels=2, device=device)
    assert unrestricted.eri_cache_bytes == restricted.eri_cache_bytes == provider.eri_cache_nbytes
    assert 1 <= unrestricted.batch_size < restricted.batch_size
    per_vector = 8 * (2 * 60 * 4**2 + 4 * 3)
    assert 2 * unrestricted.eri_cache_bytes + unrestricted.batch_size * per_vector <= 80_000


@pytest.mark.parametrize("channels", [0, 3, True, 1.5])
def test_invalid_spin_channel_count_is_rejected(channels):
    with pytest.raises(ValueError, match="spin_channels"):
        plan_response_memory(4, 3, 1, np.float64, 1., spin_channels=channels)


def test_selected_native_jk_forwards_screening_and_worker_controls(monkeypatch):
    import jax.numpy as jnp
    from gradscf.integrals.backends.native.jk import _selected_jk

    calls = []

    def action(provider, density, **kwargs):
        calls.append(kwargs)
        return density, jnp.zeros_like(density)

    monkeypatch.setattr(NativeDirectBasis, "get_jk", action)
    density = jnp.eye(2)
    _selected_jk(NativeDirectBasis(None, None), density, with_k=False,
                 screening_threshold=1e-10, workers=3)
    assert calls == [dict(with_j=True, with_k=False, screening_threshold=1e-10, workers=3)]


def test_unrestricted_facade_forwards_the_shared_response_options():
    from gradscf import tdscf
    from gradscf.tddft.response_options import normalize_response_kernel_options

    source = SimpleNamespace(xc="hf")
    driver = tdscf.TDDFT(source, max_memory=128.,
                        response_kernel_options=dict(direct_workers=3, integral_cache="none"))
    options = normalize_response_kernel_options(
        driver._unrestricted_solver_kwargs(SimpleNamespace())["response_kernel_options"])
    assert options.max_memory == 128. and options.direct_workers == 3
    assert options.integral_cache == "none"
