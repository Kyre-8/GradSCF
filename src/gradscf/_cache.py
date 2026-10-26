"""Bounded caches for declared static host-array payloads."""
from collections import OrderedDict, namedtuple
from functools import wraps
from numbers import Integral
from threading import RLock

from ._memory import tree_storage_bytes


_CacheInfo = namedtuple('CacheInfo', 'hits misses maxsize currsize')


def array_cache(*, maxsize, max_bytes, payload_bytes=None):
    """LRU by entries and retained array bytes, with no numerical JAX state.

    Decorate static topology/host-array factories only. ``payload_bytes`` can
    explicitly describe a factory's known closure arrays; arbitrary closures
    are never inspected. Python keys/headers and JAX executable/allocator
    storage are outside this payload cap. Oversized values are returned but
    not retained. Existing cache_clear/cache_info callers remain supported.
    """
    if any(not isinstance(x, Integral) or isinstance(x, bool) or x < 1
           for x in (maxsize, max_bytes)):
        raise ValueError('Array cache entry and byte limits must be positive integers')

    def decorate(function):
        entries, lock = OrderedDict(), RLock()
        hits = misses = total = 0

        @wraps(function)
        def cached(*args, **kwargs):
            nonlocal hits, misses, total
            key = args, tuple(sorted(kwargs.items()))
            with lock:
                if key in entries:
                    hits += 1
                    value, size = entries.pop(key)
                    entries[key] = value, size
                    return value
                misses += 1
            value = function(*args, **kwargs)
            size = (tree_storage_bytes(value) if payload_bytes is None else
                    payload_bytes(value, *args, **kwargs))
            if not isinstance(size, Integral) or isinstance(size, bool) or size < 0:
                raise ValueError('Declared cache array payload must be nonnegative integer bytes')
            if size <= max_bytes:
                with lock:
                    if key not in entries:
                        while entries and (len(entries) >= maxsize or total+size > max_bytes):
                            _, (_, removed) = entries.popitem(last=False)
                            total -= removed
                        entries[key] = value, int(size)
                        total += int(size)
            return value

        def clear():
            nonlocal hits, misses, total
            with lock:
                entries.clear()
                hits = misses = total = 0

        def info():
            with lock:
                return _CacheInfo(hits, misses, maxsize, len(entries))

        def array_bytes():
            with lock:
                return total

        cached.cache_clear = clear
        cached.cache_info = info
        cached.cache_bytes = array_bytes
        cached.cache_parameters = lambda: dict(maxsize=maxsize, typed=False, max_bytes=max_bytes)
        return cached

    return decorate
