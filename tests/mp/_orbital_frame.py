"""Independent full-MO oracle for pure C/F/E/AO input-chart tests.

These models test the supplied-frame array API, not an SCF geometry. At zero
parameter the frames are orthogonal and the occupied/virtual Fock blocks have
internal degeneracy. Joint tangents deliberately remain live in every input.
"""
import jax.numpy as jnp
import numpy as np


def orbital_frame(parameter, *, unrestricted=False, canonical=False):
    rng = np.random.default_rng(2426)
    n = 4
    nocc = (2, 1) if unrestricted else (2,)
    base = rng.normal(size=(3, n, n)) * .04
    base = base + base.transpose(0, 2, 1)
    direction = rng.normal(size=base.shape) * .01
    direction = direction + direction.transpose(0, 2, 1)
    factors = jnp.asarray(base) + parameter * jnp.asarray(direction)
    eri = jnp.einsum('Lpq,Lrs->pqrs', factors, factors, precision='highest')
    frames, focks = [], []
    for spin, no in enumerate(nocc):
        frame = np.linalg.qr(rng.normal(size=(n, n)))[0]
        frames.append(jnp.asarray(frame) + parameter * jnp.asarray(rng.normal(size=(n, n)) * .02))
        energies = np.array([-1., -1., .6, .6]) if no == 2 else np.array([-.8, .4, .4, .9])
        tangent = rng.normal(size=(n, n)) * .025
        tangent = .5 * (tangent + tangent.T)
        same = (np.arange(n) < no)[:, None] == (np.arange(n) < no)[None, :]
        tangent = np.diag(np.diag(tangent)) if canonical else np.where(same, tangent, 0.)
        focks.append(jnp.diag(jnp.asarray(energies)) + parameter * jnp.asarray(tangent))
    energy = jnp.asarray(-2.4) + .07 * parameter
    return (tuple(frames) if unrestricted else frames[0]), tuple(focks), energy, factors, eri


def raw_oracle(coefficients, focks, energy, eri, *, unrestricted=False):
    """Rebuild h/G in full MO coordinates using explicit spin J/K algebra."""
    cs = coefficients if unrestricted else (coefficients,)

    def transform(left, right):
        return jnp.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, left, left, right, right,
                          optimize='optimal', precision='highest')

    if unrestricted:
        aa, ab, bb = transform(cs[0], cs[0]), transform(cs[0], cs[1]), transform(cs[1], cs[1])
        na, nb = 2, 1
        ma = (jnp.einsum('pqii->pq', aa[:, :, :na, :na])
              - jnp.einsum('piiq->pq', aa[:, :na, :na])
              + jnp.einsum('pqii->pq', ab[:, :, :nb, :nb]))
        mb = (jnp.einsum('pqii->pq', bb[:, :, :nb, :nb])
              - jnp.einsum('piiq->pq', bb[:, :nb, :nb])
              + jnp.einsum('iipq->pq', ab[:na, :na]))
        h = (focks[0] - ma, focks[1] - mb)
        e0 = .5 * (jnp.trace((h[0] + focks[0])[:na, :na])
                    + jnp.trace((h[1] + focks[1])[:nb, :nb]))
        return h, (aa, ab, bb), energy - e0
    g = transform(cs[0], cs[0])
    interaction = (2 * jnp.einsum('pqii->pq', g[:, :, :2, :2])
                   - jnp.einsum('piiq->pq', g[:, :2, :2]))
    h = focks[0] - interaction
    return h, g, energy - jnp.trace((h + focks[0])[:2, :2])


def ao_source(kind, factors, eri):
    if kind == 'df':
        return dict(df_factors=factors)
    if kind == 's4':
        i, j = np.tril_indices(eri.shape[0])
        return dict(eri_pair_matrix=eri[i[:, None], j[:, None], i[None, :], j[None, :]])
    return dict(eri=eri)
