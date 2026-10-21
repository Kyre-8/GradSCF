"""Static determinant spaces; alpha orbitals precede beta orbitals.

For the rank-truncated CI hierarchy see Sherrill and Schaefer (1999),
doi:10.1016/S0065-3276(08)60532-8. See ci/REFERENCES.md.
"""
from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral
from gradscf.ao2mo import frozen_indices, unrestricted_frozen_indices


@dataclass(frozen=True)
class CISpace:
    nmo: int
    nocc: int
    max_excitation: int
    frozen: tuple[int, ...]
    determinants: tuple[int, ...]
    ranks: tuple[int, ...]

    @property
    def size(self):
        return len(self.determinants)


@dataclass(frozen=True)
class UCISpace:
    """Fixed (Nalpha,Nbeta) determinant space, without total-spin projection."""
    nmo: int
    nocc: tuple[int, int]
    max_excitation: int
    frozen: tuple[tuple[int, ...], tuple[int, ...]]
    determinants: tuple[int, ...]
    ranks: tuple[int, ...]

    @property
    def size(self):
        return len(self.determinants)


def _selected_ranks(max_excitation, excitation_ranks):
    if excitation_ranks is None:
        return None
    try:
        ranks = tuple(excitation_ranks)
    except TypeError as error:
        raise ValueError("excitation_ranks must be an integer sequence including 0") from error
    if (not ranks or any(not isinstance(r, Integral) or not 0 <= r <= max_excitation for r in ranks)
            or len(set(ranks)) != len(ranks) or 0 not in ranks):
        raise ValueError("excitation_ranks must contain unique ranks in [0,max_excitation], including 0")
    return tuple(sorted(map(int, ranks)))


def make_uci_space(nmo, nocc, *, max_excitation=2, frozen=None, max_determinants=1_000_000,
                   excitation_ranks=None):
    """Rank-truncated alpha/beta space; frozen electrons remain explicit."""
    if not isinstance(nmo, Integral) or nmo < 1:
        raise ValueError("nmo must be a positive integer")
    if not isinstance(max_excitation, Integral) or max_excitation < 0:
        raise ValueError("max_excitation must be a nonnegative integer")
    if not isinstance(max_determinants, Integral) or max_determinants < 1:
        raise ValueError("max_determinants must be a positive integer")
    selected = _selected_ranks(max_excitation, excitation_ranks)
    limit = max_excitation if selected is None else max(selected)
    frozen = unrestricted_frozen_indices(nmo, nocc, frozen)
    nocc = tuple(map(int, nocc))
    occupied = [tuple(i for i in range(no) if i not in f) for no, f in zip(nocc, frozen)]
    virtual = [tuple(i for i in range(no, nmo) if i not in f) for no, f in zip(nocc, frozen)]
    counts = [[comb(len(o), k)*comb(len(v), k)
               for k in range(min(limit, len(o), len(v))+1)]
              for o, v in zip(occupied, virtual)]
    rank_pairs = [(i, j) for i in range(len(counts[0])) for j in range(len(counts[1]))
                  if i+j <= max_excitation and (selected is None or i+j in selected)]
    count = sum(counts[0][i]*counts[1][j] for i, j in rank_pairs)
    if count > max_determinants:
        raise ValueError(f"CI space requires {count} determinants; max_determinants={max_determinants}")
    channels = []
    for spin, (no, occ, vir) in enumerate(zip(nocc, occupied, virtual)):
        ref = (1 << no)-1
        channels.append({rank: [ref ^ sum(1 << p for p in holes+particles)
                          for holes in combinations(occ, rank)
                          for particles in combinations(vir, rank)]
                         for rank in sorted({pair[spin] for pair in rank_pairs})})
    entries = sorted((i+j, a | (b << nmo))
                     for i, j in rank_pairs for a in channels[0][i] for b in channels[1][j])
    return UCISpace(int(nmo), nocc, int(max_excitation), frozen,
                    tuple(d for _, d in entries), tuple(r for r, _ in entries))


def make_ci_space(nmo, nocc, *, max_excitation=2, frozen=None, max_determinants=1_000_000,
                  excitation_ranks=None):
    """Generate only retained excitations, without enumerating the FCI space.

    Frozen occupied electrons remain explicitly occupied in all determinants;
    frozen virtual orbitals remain empty. Spin is conserved separately per channel.
    excitation_ranks=(0,2) selects CID; None retains every rank up to the limit.
    """
    if not isinstance(nmo, Integral) or nmo < 1:
        raise ValueError("nmo must be a positive integer")
    if not isinstance(nocc, Integral) or nocc < 0 or nocc > nmo:
        raise ValueError("nocc must be an integer in [0, nmo]")
    if not isinstance(max_excitation, Integral) or max_excitation < 0:
        raise ValueError("max_excitation must be a nonnegative integer")
    if not isinstance(max_determinants, Integral) or max_determinants < 1:
        raise ValueError("max_determinants must be a positive integer")
    selected = _selected_ranks(max_excitation, excitation_ranks)
    limit = max_excitation if selected is None else max(selected)
    frozen = frozen_indices(nmo, nocc, frozen)
    occ = tuple(i for i in range(nocc) if i not in frozen)
    vir = tuple(i for i in range(nocc, nmo) if i not in frozen)
    max_rank = min(limit, len(occ), len(vir))
    counts = [comb(len(occ), k) * comb(len(vir), k) for k in range(max_rank + 1)]
    rank_pairs = [(i, j) for i in range(len(counts)) for j in range(len(counts))
                  if i+j <= max_excitation and (selected is None or i+j in selected)]
    count = sum(counts[i]*counts[j] for i, j in rank_pairs)
    if count > max_determinants:
        raise ValueError(f"CI space requires {count} determinants; max_determinants={max_determinants}")
    reference = (1 << nocc) - 1
    channels = {}
    for rank in sorted({r for pair in rank_pairs for r in pair}):
        strings = []
        for holes in combinations(occ, rank):
            for particles in combinations(vir, rank):
                strings.append(reference ^ sum(1 << p for p in holes + particles))
        channels[rank] = strings
    entries = [(i + j, a | (b << nmo))
               for i, j in rank_pairs for a in channels[i] for b in channels[j]]
    entries.sort()
    return CISpace(int(nmo), int(nocc), int(max_excitation), frozen,
                   tuple(d for _, d in entries), tuple(r for r, _ in entries))
