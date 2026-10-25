"""Bare spin-virtual action from the producer's existing assembly terms."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _SpinVirtualERI:
    blocks: tuple
    virtual: tuple
    terms: tuple

    def tree_flatten(self):
        return self.blocks, (self.virtual, self.terms)

    @classmethod
    def tree_unflatten(cls, metadata, blocks):
        return cls(tuple(blocks), *metadata)

    @classmethod
    def from_terms(cls, values, terms, virtual, dtype):
        requests = tuple(dict.fromkeys(term[1] for term in terms))
        indices = {request: index for index, request in enumerate(requests)}
        records = tuple((indices[request], permutation,
                         tuple((part.start, part.stop) for part in target), sign)
                        for _, request, permutation, target, sign in terms)
        return cls(tuple(values[request].astype(dtype) for request in requests), virtual, records)

    def contract(self, x, *, dtype):
        """Apply each original nonzero spin term without zero-padding vvvv."""
        nv = sum(self.virtual)
        if x.ndim < 2 or x.shape[-2:] != (nv, nv):
            raise ValueError('Spin virtual action requires (..., nvir, nvir) RHS blocks')
        result = jnp.zeros((*x.shape[:-2], nv, nv), jnp.result_type(dtype, x, *self.blocks))
        for index, permutation, bounds, sign in self.terms:
            a, b, e, f = (slice(start, stop) for start, stop in bounds)
            value = jnp.einsum('abef,...ef->...ab', self.blocks[index].transpose(permutation),
                               x[..., e, f])
            result = result.at[..., a, b].add(sign*value)
        return result
