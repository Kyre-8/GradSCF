from __future__ import annotations

import importlib
import gc
from collections import OrderedDict
from numbers import Integral
from pathlib import Path

import jax


DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS = 0.0
DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES = 0


def _memory_mapping_usage():
    """Linux executable mappings are limited independently of RSS/cgroup RAM."""
    try:
        with Path('/proc/self/maps').open() as stream:
            count=sum(1 for _ in stream)
        return count,int(Path('/proc/sys/vm/max_map_count').read_text())
    except (OSError,ValueError):
        return None


class BoundedJitCache:
    """Bound groups of owned JIT kernels; unrelated kernels normally stay hot.

    Activate before dispatch, after synchronizing the preceding computation.
    Linux map pressure first evicts owned kernels, then clears global staging
    caches only if needed. Persistent on-disk compilation entries are retained.
    """
    def __init__(self, *, max_entries=8):
        if isinstance(max_entries,bool) or not isinstance(max_entries,Integral) or max_entries<1:
            raise ValueError('max_entries must be a positive integer.')
        self.max_entries=int(max_entries)
        self._groups=OrderedDict()
        self.evictions=0
        self.global_clears=0
        self.mapping_count=None
        self.mapping_limit=None

    def __len__(self):
        return len(self._groups)

    def _evict(self, functions):
        for function in functions:function.clear_cache()
        self.evictions+=1

    def clear(self):
        while self._groups:
            self._evict(self._groups.popitem(last=False)[1])
        gc.collect()

    def activate(self, key, functions):
        functions=tuple(functions)
        if not functions or any(not callable(getattr(f,'clear_cache',None)) for f in functions):
            raise TypeError('Expected JIT kernels with clear_cache().')
        previous=self._groups.pop(key,None)
        if previous is not None and previous!=functions:self._evict(previous)
        while len(self._groups)>=self.max_entries:
            self._evict(self._groups.popitem(last=False)[1])
        usage=_memory_mapping_usage()
        if usage is not None and usage[0]>.5*usage[1]:
            # Include the just-touched group in a pressure-triggered eviction.
            self._groups[key]=functions
            self.clear()
            usage=_memory_mapping_usage()
            if usage is not None and usage[0]>.5*usage[1]:
                jax.clear_caches()
                gc.collect()
                self.global_clears+=1
                usage=_memory_mapping_usage()
                if usage is not None and usage[0]>.5*usage[1]:
                    raise RuntimeError(f'Insufficient compiler mapping headroom after cache eviction: {usage[0]}/{usage[1]}')
        if usage is not None:self.mapping_count,self.mapping_limit=usage
        self._groups[key]=functions


def configure_jax_persistent_cache(
    *,
    cache_dir: str | None,
    min_compile_time_secs: float = DEFAULT_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS,
    min_entry_size_bytes: int = DEFAULT_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    xla_caches: str | None = "xla_gpu_per_fusion_autotune_cache_dir",
) -> str | None:
    """Enable JAX persistent compilation cache and return resolved cache path.

    This is a best-effort helper. If ``cache_dir`` is empty, no changes are made.
    """

    if cache_dir is None:
        return None
    raw = str(cache_dir).strip()
    if not raw:
        return None

    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path)
    jax.config.update("jax_compilation_cache_dir", resolved)
    try:
        jax.config.update("jax_enable_compilation_cache", True)
    except AttributeError:
        pass
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(min_compile_time_secs),
    )
    jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        int(min_entry_size_bytes),
    )
    if xla_caches is not None:
        try:
            jax.config.update("jax_persistent_cache_enable_xla_caches", str(xla_caches))
        except AttributeError:
            pass
    try:
        compilation_cache = importlib.import_module(
            "jax.experimental.compilation_cache.compilation_cache"
        )
    except (ImportError, ModuleNotFoundError):
        compilation_cache = None
    if compilation_cache is not None and hasattr(compilation_cache, "set_cache_dir"):
        compilation_cache.set_cache_dir(resolved)
    return resolved
