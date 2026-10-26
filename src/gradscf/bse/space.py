"""Static occupied/virtual index spaces shared by optical and screening inputs."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class BSESpace:
    nmo: int
    nocc: int
    occupied: tuple[int, ...]
    virtual: tuple[int, ...]

    def __post_init__(self):
        if (
            not isinstance(self.nmo, Integral)
            or not isinstance(self.nocc, Integral)
            or self.nmo < 1 or not 0 <= self.nocc <= self.nmo
        ):
            raise ValueError(
                "BSE requires valid occupations in a finite orbital space"
            )
        if not isinstance(self.occupied, tuple) or not isinstance(self.virtual, tuple):
            raise TypeError("BSESpace indices must be static tuples")
        for indices, lo, hi in (
            (self.occupied, 0, self.nocc),
            (self.virtual, self.nocc, self.nmo),
        ):
            if any(
                not isinstance(p, Integral) or not lo <= p < hi for p in indices
            ) or len(set(indices)) != len(indices):
                raise ValueError(
                    "BSE occupied/virtual indices must be distinct and in their respective ranges"
                )

    @property
    def size(self):
        return len(self.occupied) * len(self.virtual)


@dataclass(frozen=True)
class SpinBSESpace:
    """Two collinear, spin-conserving transition blocks, alpha then beta."""

    channels: tuple[BSESpace, BSESpace]

    def __post_init__(self):
        if len(self.channels) != 2 or self.channels[0].nmo != self.channels[1].nmo:
            raise ValueError("Spin BSE requires two channels with the same MO dimension")

    @property
    def nmo(self):
        return self.channels[0].nmo

    @property
    def nocc(self):
        return tuple(c.nocc for c in self.channels)

    @property
    def occupied(self):
        return tuple(c.occupied for c in self.channels)

    @property
    def virtual(self):
        return tuple(c.virtual for c in self.channels)

    @property
    def size(self):
        return sum(c.size for c in self.channels)


def make_bse_space(nmo, nocc, *, occupied=None, virtual=None):
    if isinstance(nocc, tuple):
        if len(nocc) != 2 or any(x is not None and len(x) != 2 for x in (occupied, virtual)):
            raise ValueError("Spin occupations and windows require alpha/beta pairs")
        return SpinBSESpace(tuple(make_bse_space(
            nmo, count, occupied=None if occupied is None else occupied[s],
            virtual=None if virtual is None else virtual[s],
        ) for s, count in enumerate(nocc)))
    return BSESpace(
        nmo,
        nocc,
        tuple(range(nocc)) if occupied is None else tuple(occupied),
        tuple(range(nocc, nmo)) if virtual is None else tuple(virtual),
    )
