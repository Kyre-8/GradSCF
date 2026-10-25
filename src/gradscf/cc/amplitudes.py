"""Orthonormal independent coordinates for t2[ijab] = t2[jiba]."""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class AmplitudeSpace:
    nocc: int
    nvir: int
    method: str = "ccsd"

    @property
    def singles(self):
        return self.method not in {"ccd", "lccd"}

    @property
    def doubles(self):
        return self.method != "ccs"

    @property
    def size(self):
        k = self.nocc * self.nvir
        return (k if self.singles else 0) + (k * (k + 1) // 2 if self.doubles else 0)

    def pack(self, t1, t2):
        if t1.shape != (self.nocc, self.nvir) or t2.shape != (
            self.nocc,
            self.nocc,
            self.nvir,
            self.nvir,
        ):
            raise ValueError("CC amplitude shape does not match the active space")
        if jnp.iscomplexobj(t1) or jnp.iscomplexobj(t2):
            raise NotImplementedError("Restricted CC amplitudes must be real")
        k = self.nocc * self.nvir
        parts = []
        if self.singles:
            parts.append(t1.reshape(-1))
        if self.doubles:
            i, j = np.tril_indices(k)
            matrix = t2.transpose(0, 2, 1, 3).reshape(k, k)
            weights = jnp.asarray(np.where(i == j, 1.0, np.sqrt(2.0)), dtype=t2.dtype)
            parts.append(matrix[i, j] * weights)
        return jnp.concatenate(parts)

    def unpack(self, vector):
        no, nv = self.nocc, self.nvir
        k = no * nv
        if self.singles:
            t1 = vector[:k].reshape(no, nv)
            vector = vector[k:]
        else:
            t1 = jnp.zeros((no, nv), dtype=vector.dtype)
        t2 = jnp.zeros((no, no, nv, nv), dtype=t1.dtype)
        if self.doubles:
            i, j = np.tril_indices(k)
            weights = jnp.asarray(np.where(i == j, 1.0, np.sqrt(2.0)), dtype=t1.dtype)
            matrix = jnp.zeros((k, k), dtype=t1.dtype)
            matrix = (
                matrix.at[i, j].set(vector / weights).at[j, i].set(vector / weights)
            )
            t2 = matrix.reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
        return t1, t2
