"""Target-capacity planning must not confuse accelerator labels with host RAM."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from gradscf import _memory


def test_default_device_caps_consumers_without_an_explicit_device(monkeypatch):
    gpu = SimpleNamespace(platform="gpu", memory_stats=lambda: {
        "bytes_limit": 16_000_000, "bytes_in_use": 4_000_000})
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=gpu))
    assert _memory.workspace_bytes(60., resident_bytes=600_000) == 9_000_000


@pytest.mark.parametrize("label", ["gpu", "cuda", "tpu"])
def test_accelerator_label_never_falls_back_to_host_capacity(label):
    with pytest.raises(ValueError, match="concrete.*device"):
        _memory.available_memory_bytes(label)


def test_string_default_device_requires_a_concrete_capacity_target(monkeypatch):
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device="gpu"))
    with pytest.raises(ValueError, match="concrete.*device"):
        _memory.workspace_bytes(60.)


def test_unknown_device_object_is_not_assumed_to_be_cpu():
    with pytest.raises(TypeError, match="device"):
        _memory.available_memory_bytes(object())


def test_explicit_capacity_device_wins_over_default_context(monkeypatch):
    default = SimpleNamespace(platform="gpu", memory_stats=lambda: {"bytes_limit": 12_000_000})
    explicit = SimpleNamespace(platform="gpu", memory_stats=lambda: {"bytes_limit": 5_000_000})
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=default))
    assert _memory.workspace_bytes(60., device=explicit) == 4_000_000


def test_legacy_device_accessor_reaches_shared_capacity_planner():
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: {"bytes_limit": 5_000_000})
    assert _memory.workspace_bytes(60., device=lambda: device) == 4_000_000


def test_concrete_default_device_is_visible_inside_jit(monkeypatch):
    cpu = jax.devices("cpu")[0]
    observed = []

    def capacity(device=None):
        observed.append(device)
        return 10_000_000 if device is cpu else None

    monkeypatch.setattr(_memory, "available_memory_bytes", capacity)

    @jax.jit
    def planned(x):
        assert _memory.workspace_bytes(60.) == 8_000_000
        return x * x

    with jax.default_device(cpu):
        assert float(planned(jnp.asarray(2.))) == 4.
    assert observed == [cpu]


def test_abstract_array_inference_uses_context_without_device_initialization(monkeypatch):
    target = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=target))
    monkeypatch.setattr(jax, "devices", lambda *a, **k: pytest.fail("Unrequested device initialization"))
    infer = getattr(_memory, "array_device", None)
    assert callable(infer), "Missing shared array/default-device inference"
    shape = jax.ShapeDtypeStruct((2, 3), jnp.float64)
    assert infer(shape) is target


def test_array_inference_retains_known_eager_device_without_context(monkeypatch):
    device = SimpleNamespace(platform="cpu")
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=None))
    infer = getattr(_memory, "array_device", None)
    assert callable(infer), "Missing shared array/default-device inference"
    assert infer(SimpleNamespace(device=device)) is device
    assert infer(SimpleNamespace(device=lambda: device)) is device
    assert infer(jax.ShapeDtypeStruct((2, 3), jnp.float64)) is None


def test_known_eager_device_keeps_priority_over_another_default(monkeypatch):
    actual = SimpleNamespace(platform="gpu", memory_stats=lambda: {"bytes_limit": 5_000_000})
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device="cpu"))
    selected = _memory.array_device(SimpleNamespace(device=actual))
    assert selected is actual
    assert _memory.workspace_bytes(60., device=selected) == 4_000_000


@pytest.mark.parametrize("representation", ["dense", "df", "native"])
def test_ao2mo_preflight_uses_declared_target_before_contraction(monkeypatch, representation):
    from gradscf import ao2mo, integrals
    from gradscf.integrals.backends.native.jk import NativeDirectBasis

    coefficients = (jnp.ones((2, 1)),) * 4
    if representation == "native":
        topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 .8", basis="sto-3g")
        options = dict(direct_basis=NativeDirectBasis(
            integrals.make_plan(topology, backend="native"), parameters))
    else:
        options = ({"eri": jnp.ones((2,) * 4)} if representation == "dense"
                   else {"df_factors": jnp.ones((3, 2, 2))})
    # Traced arguments have no concrete device; their shared capacity plan
    # must use the declared target before any large contraction is compiled.
    target = SimpleNamespace(platform="gpu", memory_stats=lambda: {"bytes_limit": 100})
    monkeypatch.setattr(jax, "config", SimpleNamespace(
        jax_default_device=target, x64_enabled=jax.config.x64_enabled))
    with pytest.raises(ValueError, match="resident.*budget"):
        jax.jit(lambda cs, source: ao2mo.transform_eri_block(
            cs, max_memory=1., **source))(coefficients, options)
