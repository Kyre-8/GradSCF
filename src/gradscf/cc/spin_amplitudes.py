"""Independent spin-conserving coordinates for antisymmetric CC amplitudes."""
from dataclasses import dataclass
from itertools import combinations
from functools import cached_property
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class SpinAmplitudeSpace:
    nocc: tuple[int, int]
    nvir: tuple[int, int]
    method: str = "ccsd"

    @cached_property
    def indices(self):
        so = [0]*self.nocc[0] + [1]*self.nocc[1]
        sv = [0]*self.nvir[0] + [1]*self.nvir[1]
        singles = [(i, a) for i in range(len(so)) for a in range(len(sv))
                   if so[i] == sv[a] and self.method != "ccd"]
        doubles = [(i, j, a, b) for i, j in combinations(range(len(so)), 2)
                   for a, b in combinations(range(len(sv)), 2)
                   if so[i]+so[j] == sv[a]+sv[b]]
        return (np.asarray(singles, dtype=np.int32).reshape(-1, 2).T,
                np.asarray(doubles, dtype=np.int32).reshape(-1, 4).T)

    def pack(self, t1, t2):
        """Norm-preserving coordinates: each independent double occurs four times."""
        singles, doubles = self.indices
        return jnp.concatenate((t1[tuple(singles)], 2*t2[tuple(doubles)]))

    def unpack(self, vector):
        no, nv = sum(self.nocc), sum(self.nvir)
        singles, doubles = self.indices
        ns = singles.shape[1]
        t1 = jnp.zeros((no, nv), vector.dtype).at[tuple(singles)].set(vector[:ns])
        i, j, a, b = doubles
        v = vector[ns:]/2
        t2 = jnp.zeros((no, no, nv, nv), vector.dtype)
        t2 = t2.at[i, j, a, b].set(v).at[j, i, a, b].set(-v)
        return t1, t2.at[i, j, b, a].set(-v).at[j, i, b, a].set(v)

    def to_blocks(self, t1, t2):
        oa, va = self.nocc[0], self.nvir[0]
        return ((t1[:oa, :va], t1[oa:, va:]),
                (t2[:oa, :oa, :va, :va], t2[:oa, oa:, :va, va:],
                 t2[oa:, oa:, va:, va:]))

    @cached_property
    def _block_indices(self):
        """Physical-block gathers in the original, interleaved packed order."""
        oa, va = self.nocc[0], self.nvir[0]
        singles, doubles = self.indices
        si, sa = singles
        i, j, a, b = doubles
        singles_blocks = []
        for beta in (False, True):
            positions = np.flatnonzero((si >= oa) == beta).astype(np.int32)
            singles_blocks.append((positions, (si[positions]-oa*beta,
                                               sa[positions]-va*beta)))
        doubles_blocks = []
        for mask, offsets in (((j < oa), (0, 0, 0, 0)),
                              ((i < oa) & (j >= oa), (0, oa, 0, va)),
                              ((i >= oa), (oa, oa, va, va))):
            positions = np.flatnonzero(mask).astype(np.int32)
            local = tuple(index[positions]-offset for index, offset in zip(doubles, offsets))
            doubles_blocks.append((positions, local))
        return tuple(singles_blocks), tuple(doubles_blocks)

    def pack_blocks(self, t1, t2, dtype):
        """Project physical blocks directly into the existing packed metric."""
        blocks = self._validated_blocks(t1, t2, dtype)
        singles, doubles = self.indices
        ns = singles.shape[1]
        vector = jnp.zeros((ns+doubles.shape[1],), blocks[0].dtype)
        si, di = self._block_indices
        for block, (positions, local) in zip(blocks[:2], si):
            vector = vector.at[positions].set(block[local])
        for spin, (block, (positions, local)) in enumerate(zip(blocks[2:], di)):
            i, j, a, b = local
            values = block[local]
            if spin != 1:
                values = .25*(values-block[j, i, a, b]-block[i, j, b, a]+block[j, i, b, a])
            vector = vector.at[ns+positions].set(2*values)
        return vector

    def unpack_blocks(self, vector):
        """Unpack physical blocks without a spin-orbital zero-filled tensor."""
        oa, ob = self.nocc
        va, vb = self.nvir
        ns = self.indices[0].shape[1]
        si, di = self._block_indices
        t1 = tuple(jnp.zeros(shape, vector.dtype).at[local].set(vector[positions])
                   for shape, (positions, local) in zip(((oa, va), (ob, vb)), si))
        t2 = []
        shapes = ((oa, oa, va, va), (oa, ob, va, vb), (ob, ob, vb, vb))
        for spin, (shape, (positions, local)) in enumerate(zip(shapes, di)):
            i, j, a, b = local
            values = vector[ns+positions]/2
            block = jnp.zeros(shape, vector.dtype).at[local].set(values)
            if spin != 1:
                block = block.at[j, i, a, b].set(-values)
                block = block.at[i, j, b, a].set(-values).at[j, i, b, a].set(values)
            t2.append(block)
        return t1, tuple(t2)

    def _validated_blocks(self, t1, t2, dtype):
        oa, ob = self.nocc
        va, vb = self.nvir
        shapes1 = ((oa, va), (ob, vb))
        shapes2 = ((oa, oa, va, va), (oa, ob, va, vb), (ob, ob, vb, vb))
        if len(t1) != 2 or len(t2) != 3:
            raise ValueError("UCC amplitudes require two singles and three doubles blocks")
        blocks = tuple(jnp.asarray(a) for a in (*t1, *t2))
        if any(jnp.iscomplexobj(a) for a in blocks):
            raise NotImplementedError("UCC amplitudes must be real")
        if any(a.shape != s for a, s in zip(blocks, shapes1+shapes2)):
            raise ValueError("Initial UCC amplitude shapes do not match the active space")
        dtype = jnp.result_type(dtype, *blocks)
        return tuple(a.astype(dtype) for a in blocks)

    def from_blocks(self, t1, t2, dtype):
        oa, ob = self.nocc
        va, vb = self.nvir
        ta, tb, taa, tab, tbb = self._validated_blocks(t1, t2, dtype)
        dtype = ta.dtype
        # Project same-spin restart amplitudes onto their antisymmetric subspace.
        project = lambda t: .25*(t-t.swapaxes(0, 1)-t.swapaxes(2, 3)+t.transpose(1, 0, 3, 2))
        s = jnp.zeros((oa+ob, va+vb), dtype).at[:oa, :va].set(ta).at[oa:, va:].set(tb)
        d = jnp.zeros((oa+ob, oa+ob, va+vb, va+vb), dtype)
        d = d.at[:oa, :oa, :va, :va].set(project(taa)).at[oa:, oa:, va:, va:].set(project(tbb))
        d = d.at[:oa, oa:, :va, va:].set(tab).at[oa:, :oa, :va, va:].set(-tab.swapaxes(0, 1))
        d = d.at[:oa, oa:, va:, :va].set(-tab.swapaxes(2, 3))
        return s, d.at[oa:, :oa, va:, :va].set(tab.transpose(1, 0, 3, 2))
