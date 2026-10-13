"""KS response in Hermitian density coordinates, without eigenvector derivatives."""
from dataclasses import dataclass
from math import isfinite
import jax
import jax.numpy as jnp
import numpy as np
from .core import Problem


@dataclass(frozen=True)
class _Hermitian:
    n: int
    complex: bool

    @property
    def size(self):
        return self.n**2 if self.complex else self.n*(self.n+1)//2

    def pack(self, matrix):
        i, j = np.triu_indices(self.n, 1)
        diagonal = jnp.arange(self.n)
        pieces = (matrix[diagonal, diagonal].real, matrix[i, j].real)
        if self.complex:
            pieces += (matrix[i, j].imag,)
        return jnp.concatenate(pieces)

    def unpack(self, vector):
        i, j = np.triu_indices(self.n, 1)
        m = len(i)
        z = vector[self.n:self.n+m]
        if self.complex:
            z = z + 1j*vector[self.n+m:]
        out = (jnp.eye(self.n) * vector[:self.n]).astype(z.dtype)
        return out.at[i, j].add(z).at[j, i].add(z.conj())


def ks_problem(coefficients, nocc, inputs, fock, energy, *, overlap, occupation=1.,
               converged=True, tolerance=1e-7, gap_tolerance=1e-8):
    """Fixed-occupation KS problem in one or more AO blocks.

    coefficients: (nblocks, nao, nao), complete S-orthonormal reference MOs.
    fock(densities, inputs), overlap(inputs), energy(densities, inputs) reuse
    ground-state builders. nocc is a static count per block; occupation is 2
    for a restricted molecular block or 1 for explicit spin/k blocks.

    Q S Q=Q enforces the occupied projector. The vo commutator equations,
    together with oo/vv idempotency equations, form a local square chart.
    It remains regular at degeneracy within a fully occupied/empty block.
    """
    c = jnp.asarray(coefficients)
    if c.ndim != 3 or c.shape[1] != c.shape[2] or len(nocc) != len(c):
        raise ValueError('KS coefficients require complete square orbital blocks')
    n = c.shape[-1]
    if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k <= n for k in nocc):
        raise ValueError('nocc must contain integer occupation counts')
    if not isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError('gap_tolerance must be nonnegative and finite')
    adjoint = c.conj().swapaxes(-1, -2)
    q0 = jnp.stack([jnp.eye(n)*(jnp.arange(n)<no) for no in nocc])
    s0 = overlap(inputs)
    f0 = fock(occupation*(c @ q0 @ adjoint), inputs)
    chart = _Hermitian(n, any(jnp.iscomplexobj(a) for a in (c, s0, f0)))
    initial = jnp.concatenate([chart.pack(q) for q in q0])

    def projectors(x):
        return jnp.stack([chart.unpack(y) for y in x.reshape(len(c), chart.size)])

    def density(x):
        return occupation * (c @ projectors(x) @ adjoint)

    def residual(x, args):
        q = projectors(x)
        s = adjoint @ overlap(args) @ c
        f = adjoint @ fock(density(x), args) @ c
        idem = q @ s @ q-q
        comm = f @ q @ s-s @ q @ f
        result = []
        for block, no in enumerate(nocc):
            # Assemble the independent equations with additive, jet-safe ops.
            mask = (jnp.arange(n)[:, None] < no) == (jnp.arange(n)[None, :] < no)
            vo = (jnp.arange(n)[:, None] >= no) & (jnp.arange(n)[None, :] < no)
            lower = jnp.where(vo, comm[block], 0.)
            value = jnp.where(mask, idem[block], 0.) + lower + lower.conj().T
            result.append(chart.pack(value))
        return jnp.concatenate(result)

    def quantities(x, args):
        d = density(x)
        return {'density': d, 'energy': energy(d, args),
                'electron_number': jnp.einsum('bij,bji->b', overlap(args), d).real}

    metric = adjoint @ s0 @ c
    reference_fock = adjoint @ f0 @ c
    valid = jnp.asarray(converged) & jnp.all(jnp.isfinite(c))
    valid &= jnp.max(jnp.abs(metric-jnp.eye(n)), initial=0.) <= tolerance
    valid &= jnp.max(jnp.abs(reference_fock-reference_fock.conj().swapaxes(-1,-2)), initial=0.) <= tolerance
    # Spectral checks diagnose the base only; do not differentiate eigenvectors
    # inside an occupied degeneracy. Response itself uses projector equations.
    checked = jax.lax.stop_gradient(reference_fock)
    for block, no in enumerate(nocc):
        if 0 < no < n:
            gap = jnp.min(jnp.linalg.eigvalsh(checked[block, no:, no:])) - jnp.max(jnp.linalg.eigvalsh(checked[block, :no, :no]))
            valid &= gap > gap_tolerance
    return Problem(residual, initial, inputs, quantities, valid, tolerance)
