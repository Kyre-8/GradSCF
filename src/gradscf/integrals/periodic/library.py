"""Offline GTH basis/pseudopotential data, names, and valence metadata."""
import copy
from functools import lru_cache
from importlib.resources import files
import json
import re


@lru_cache(maxsize=1)
def _data():
    return json.loads(files('gradscf.integrals.periodic').joinpath('gth_data.json').read_text())


def _normalize(name):
    return re.sub(r'[-_\s]', '', str(name).lower())


def _symbol(symbol):
    return str(symbol).capitalize()


def basis_names(element=None):
    """Canonical available family names, optionally filtered by element."""
    return tuple(name for name, values in sorted(_data()['basis'].items())
                 if element is None or _symbol(element) in values)


def _candidates(name):
    data, key = _data(), _normalize(name)
    for family, values in data['basis'].items():
        if key == _normalize(family):
            return {symbol: family for symbol in values}
    return data['aliases'].get(key, {})


def basis_elements(name):
    """Element symbols with valid exported data for a family or CP2K alias."""
    candidates = _candidates(name)
    if not candidates:
        raise ValueError(f'Unknown periodic basis {name!r}; use basis_names() for available families.')
    return tuple(sorted(candidates))


def is_periodic_basis(name):
    return bool(_candidates(name)) or 'gth' in _normalize(name)


def _family(name, symbol):
    candidates = _candidates(name)
    symbol = _symbol(symbol)
    if symbol not in candidates:
        available = ', '.join(sorted(candidates)) or 'none; use basis_names()'
        raise ValueError(f'No valid {name!r} basis for {symbol}; available elements: {available}.')
    return candidates[symbol]


def load_basis(name, symbol):
    """Return independent shell data; accept PySCF and element-specific CP2K aliases."""
    return copy.deepcopy(_data()['basis'][_family(name, symbol)][_symbol(symbol)])


def basis_valence(name, symbol):
    """Declared GTH valence count, or None when the upstream basis has no label."""
    return _data()['basis_valence'][_family(name, symbol)][_symbol(symbol)]


def load_pseudo(name, symbol):
    """Return GTH parameters, including explicitly requested -qN core variants."""
    data, symbol = _data(), _symbol(symbol)
    match = re.fullmatch(r'(.+?)(?:q(\d+))?', _normalize(name))
    if match is None:
        raise ValueError('A GTH pseudopotential name is required.')
    key, q = match.groups()
    if key == 'gthlda':
        key = 'gthpade'
    family = next((n for n in data['pseudo'] if _normalize(n) == key), None)
    default = data['pseudo'].get(family, {}).get(symbol)
    if default is None:
        raise ValueError(f'No bundled GTH pseudopotential {name!r} for {symbol}.')
    if q is None or int(q) == sum(default[0]):
        return copy.deepcopy(default)
    variants = data['pseudo_variants'][family].get(symbol, {})
    if q not in variants:
        available = sorted({sum(default[0]), *map(int, variants)})
        raise ValueError(f'No {name!r} for {symbol}; available valence counts: {available}.')
    return copy.deepcopy(variants[q])
