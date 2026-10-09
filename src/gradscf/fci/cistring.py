"""Static alpha/beta occupation strings and shared fermionic phases."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np


def excite(det, holes, particles):
    """Apply ordered annihilators/creators with alpha orbitals before beta."""
    phase = 1
    for orbital in holes:
        if not det & (1 << orbital):
            return None, 0
        phase *= -1 if (det & ((1 << orbital) - 1)).bit_count() % 2 else 1
        det ^= 1 << orbital
    for orbital in reversed(particles):
        if det & (1 << orbital):
            return None, 0
        phase *= -1 if (det & ((1 << orbital) - 1)).bit_count() % 2 else 1
        det ^= 1 << orbital
    return det, phase


def unpack_nelec(norb, nelec, spin=None):
    if not isinstance(norb, Integral) or isinstance(norb, bool) or norb < 0:
        raise ValueError("norb must be a nonnegative integer")
    if spin is not None and (not isinstance(spin, Integral) or isinstance(spin,bool)):
        raise ValueError("spin must be an integer 2*Ms")
    if isinstance(nelec, Integral) and not isinstance(nelec, bool):
        spin = int(nelec) % 2 if spin is None else spin
        if (not isinstance(spin, Integral) or isinstance(spin, bool)
                or (nelec + spin) % 2):
            raise ValueError("spin must be an integer 2*Ms with the electron parity")
        nelec = ((nelec + spin)//2, (nelec - spin)//2)
    else:
        try:
            nelec = tuple(nelec)
        except TypeError as error:
            raise ValueError("nelec must be a total count or (nalpha,nbeta)") from error
        if spin is not None and (len(nelec) != 2 or nelec[0]-nelec[1] != spin):
            raise ValueError("spin disagrees with (nalpha,nbeta)")
    if len(nelec) != 2 or any(not isinstance(x, Integral) or isinstance(x, bool)
                              or not 0 <= x <= norb for x in nelec):
        raise ValueError("Electron counts must lie in [0,norb]")
    return tuple(map(int, nelec))


@dataclass(frozen=True)
class FCISpace:
    norb: int
    nelec: tuple[int, int]
    max_determinants: int = 1_000_000
    max_link_elements: int = 4_000_000

    def __post_init__(self):
        if not isinstance(self.nelec, tuple):
            raise TypeError("FCISpace.nelec must be a static tuple")
        unpack_nelec(self.norb, self.nelec)
        for name in ("max_determinants", "max_link_elements"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.size > self.max_determinants:
            raise ValueError(f"FCI requires {self.size} determinants; max_determinants={self.max_determinants}")
        if self.norb**2 * sum(self.shape) > self.max_link_elements:
            raise ValueError("FCI string maps exceed max_link_elements")

    @property
    def shape(self):
        return tuple(comb(self.norb, n) for n in self.nelec)

    @property
    def size(self):
        na, nb = self.shape
        return na * nb

    @property
    def alpha_strings(self):
        return make_strings(self.norb, self.nelec[0])

    @property
    def beta_strings(self):
        return make_strings(self.norb, self.nelec[1])


def make_fci_space(norb, nelec, *, spin=None, max_determinants=1_000_000,
                   max_link_elements=4_000_000):
    """Build static topology outside JIT; count capacities before enumeration."""
    electrons = unpack_nelec(norb, nelec, spin)
    return FCISpace(int(norb), electrons, max_determinants, max_link_elements)


@lru_cache(maxsize=16)
def make_strings(norb, nelec):
    """Increasing binary-string order, matching PySCF's alpha/beta CI layout."""
    unpack_nelec(norb, (nelec, 0))
    return tuple(sorted(sum(1 << p for p in occ) for occ in combinations(range(norb), nelec)))


@lru_cache(maxsize=16)
def _link_maps(norb, nelec):
    strings = make_strings(norb, nelec)
    lookup = {s: i for i, s in enumerate(strings)}
    source = np.zeros((norb*norb, len(strings)), dtype=np.int32)
    signs = np.zeros_like(source, dtype=np.int8)
    occupations = np.array([[(s >> p) & 1 for p in range(norb)] for s in strings], dtype=np.int8)
    for ket, string in enumerate(strings):
        for q in range(norb):
            if not string & (1 << q):
                continue
            for p in range(norb):
                target, sign = excite(string, (q,), (p,))
                if sign:
                    bra = lookup[target]
                    source[p*norb+q, bra], signs[p*norb+q, bra] = ket, sign
    for value in (source, signs, occupations):
        value.setflags(write=False)
    return source, signs, occupations
