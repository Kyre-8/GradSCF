from gradscf.tools import jax_runtime


def test_bounded_kernel_cache_evicts_cpu_groups_without_clearing_other_jits(monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np

    traces={}
    monkeypatch.setattr(jax_runtime,'_memory_mapping_usage',lambda:(1,100))
    def kernel(name):
        def f(x):
            traces[name]=traces.get(name,0)+1
            return x*x+1
        return jax.jit(f)

    a,b,c,gpu_like=[kernel(name) for name in ('a','b','c','other')]
    cache=jax_runtime.BoundedJitCache(max_entries=2)
    x=jnp.array([.2,.7])
    gpu_like(x).block_until_ready()
    for key,f in (('a',a),('b',b),('a',a),('c',c)):
        cache.activate(key,(f,));f(x).block_until_ready()
    assert len(cache)==2 and cache.evictions==1
    a(x).block_until_ready();gpu_like(x).block_until_ready()
    assert traces['a']==1 and traces['other']==1
    b(x).block_until_ready()
    assert traces['b']==2
    cache.activate('b',(b,))
    np.testing.assert_allclose(jax.grad(lambda x:jnp.sum(b(x)))(x),2*x)


def test_bounded_kernel_cache_rejects_invalid_capacity_and_releases_replaced_group(monkeypatch):
    import jax
    import jax.numpy as jnp
    import pytest
    monkeypatch.setattr(jax_runtime,'_memory_mapping_usage',lambda:(1,100))
    for capacity in (0,-1,True,1.5):
        with pytest.raises(ValueError):jax_runtime.BoundedJitCache(max_entries=capacity)
    a=jax.jit(lambda x:x+1);b=jax.jit(lambda x:x+2)
    cache=jax_runtime.BoundedJitCache(max_entries=1)
    cache.activate('same',(a,));a(jnp.array(0.)).block_until_ready()
    assert a._cache_size()==1
    cache.activate('same',(b,))
    assert a._cache_size()==0 and len(cache)==1


def test_mapping_pressure_preserves_unrelated_jits_when_local_eviction_suffices(monkeypatch):
    import jax
    import jax.numpy as jnp
    a=jax.jit(lambda x:x+1);other=jax.jit(lambda x:x+2)
    cache=jax_runtime.BoundedJitCache(max_entries=2)
    monkeypatch.setattr(jax_runtime,'_memory_mapping_usage',lambda:(1,100))
    cache.activate('a',(a,));a(jnp.array(0.)).block_until_ready();other(jnp.array(0.)).block_until_ready()
    readings=iter(((80,100),(10,100)))
    monkeypatch.setattr(jax_runtime,'_memory_mapping_usage',lambda:next(readings))
    cache.activate('a',(a,))
    assert a._cache_size()==0 and other._cache_size()==1 and cache.global_clears==0


def test_mapping_pressure_has_a_global_last_resort(monkeypatch):
    import jax
    cache=jax_runtime.BoundedJitCache(max_entries=2)
    readings=iter(((80,100),(80,100),(10,100)))
    monkeypatch.setattr(jax_runtime,'_memory_mapping_usage',lambda:next(readings))
    calls=[]
    monkeypatch.setattr(jax_runtime.jax,'clear_caches',lambda:calls.append(True))
    cache.activate('a',(jax.jit(lambda x:x),))
    assert calls==[True] and cache.global_clears==1


def test_unreleased_mapping_pressure_stops_before_dispatch(monkeypatch):
    import jax
    import pytest
    cache=jax_runtime.BoundedJitCache(max_entries=2)
    monkeypatch.setattr(jax_runtime,'_memory_mapping_usage',lambda:(80,100))
    monkeypatch.setattr(jax_runtime.jax,'clear_caches',lambda:None)
    with pytest.raises(RuntimeError,match='mapping'):
        cache.activate('a',(jax.jit(lambda x:x),))


def test_configure_jax_persistent_cache_ignores_missing_xla_cache_option(
    monkeypatch,
    tmp_path,
):
    updates = []

    def fake_update(name, value):
        if name == "jax_persistent_cache_enable_xla_caches":
            raise AttributeError(f"Unrecognized config option: {name}")
        updates.append((name, value))

    monkeypatch.setattr(jax_runtime.jax.config, "update", fake_update)

    result = jax_runtime.configure_jax_persistent_cache(
        cache_dir=str(tmp_path),
        min_compile_time_secs=0.0,
        min_entry_size_bytes=0,
    )

    assert result == str(tmp_path.resolve())
    assert ("jax_compilation_cache_dir", str(tmp_path.resolve())) in updates
    assert ("jax_enable_compilation_cache", True) in updates
    assert ("jax_persistent_cache_min_compile_time_secs", 0.0) in updates
    assert ("jax_persistent_cache_min_entry_size_bytes", 0) in updates


def test_configure_jax_persistent_cache_initializes_legacy_cache_api(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeCompilationCache:
        @staticmethod
        def set_cache_dir(path):
            calls.append(path)

    real_import = jax_runtime.importlib.import_module

    def fake_import_module(name):
        if name == "jax.experimental.compilation_cache.compilation_cache":
            return FakeCompilationCache
        return real_import(name)

    monkeypatch.setattr(jax_runtime.importlib, "import_module", fake_import_module)

    result = jax_runtime.configure_jax_persistent_cache(
        cache_dir=str(tmp_path),
        min_compile_time_secs=0.0,
        min_entry_size_bytes=0,
    )

    assert result == str(tmp_path.resolve())
    assert calls == [str(tmp_path.resolve())]
