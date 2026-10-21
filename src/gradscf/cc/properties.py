"""CC one- and two-particle properties including the stationary left state."""

import jax
import jax.numpy as jnp
from .amplitudes import AmplitudeSpace
from .integrals import prepare_integrals
from .rccsd import residual, correlation_energy
from .lambda_equations import solve_lambda
from .types import CCConfig
from .uccsd import prepare_ucc_integrals
from .spin_amplitudes import SpinAmplitudeSpace
from . import _spin_equations
from gradscf.ao2mo import frozen_indices, unrestricted_frozen_indices
from ._spin_density import active_density_parts


def _density_from_lambda(h1, eri, result, left, *, nocc, frozen, config):
    if isinstance(nocc, (tuple, list)):
        _, occupied, virtual = prepare_ucc_integrals(h1, eri, nocc=nocc, frozen=frozen)
        space = SpinAmplitudeSpace(occupied, virtual, config.method)
        t1, t2 = space.from_blocks(result.t1, result.t2, jnp.asarray(h1[0]).dtype)

        def lagrangian(raw_h):
            h = tuple(.5*(a+a.T) for a in raw_h)
            ints, _, _ = prepare_ucc_integrals(h, eri, nocc=nocc, frozen=frozen)
            r = space.pack(*_spin_equations.residual(t1, t2, ints))
            e = ints.reference_energy + _spin_equations.correlation_energy(t1, t2, ints)
            return e + jnp.vdot(left.adjoint, r).real

        densities = jax.grad(lagrangian)(tuple(map(jnp.asarray, h1)))
        return tuple(jnp.where(result.converged & left.converged, d, jnp.nan) for d in densities)
    no, nv = result.t1.shape
    space = AmplitudeSpace(no, nv, config.method)

    def lagrangian(raw_h):
        # Derivative for a real Hermitian perturbation; PySCF's public RDM1 is
        # also symmetrized. This is an orbital-unrelaxed, spin-summed MO density.
        h = 0.5 * (raw_h + raw_h.T)
        ints = prepare_integrals(h, eri, nocc=nocc, frozen=frozen)
        r = space.pack(*residual(result.t1, result.t2, ints, model=config.method))
        e = ints.reference_energy + correlation_energy(
            result.t1, result.t2, ints, model=config.method
        )
        return e + jnp.vdot(left.adjoint, r).real

    # T and lambda are held fixed for this *partial* derivative. Do not apply
    # stop_gradient: an outer derivative of the density must include both states'
    # implicit response, which is verified against reconverged finite differences.
    density = jax.grad(lagrangian)(jnp.asarray(h1))
    return jnp.where(result.converged & left.converged, density, jnp.nan)


def make_rdm1(h1, eri, result, *, nocc, frozen=None, config=None):
    """Real symmetric MO 1-RDM, including frozen occupied electrons.

    Restricted output is spin summed; unrestricted output is (alpha, beta).
    This includes T/Lambda response but not optimization of the reference
    orbitals. It is a CC-model density, not a CCSD(T)-corrected density.
    """
    cfg = CCConfig() if config is None else config
    left = solve_lambda(h1, eri, result, nocc=nocc, frozen=frozen, config=cfg)
    return _density_from_lambda(
        h1, eri, result, left, nocc=nocc, frozen=frozen, config=cfg
    )


def _rdm2_from_lambda(h1, eri, result, left, *, nocc, frozen, config):
    if config.method not in {"ccsd", "ccd", "ccs"}:
        raise NotImplementedError("The CC 2-RDM currently supports CCS, CCD and CCSD")
    unrestricted = isinstance(nocc, (tuple, list))
    if unrestricted:
        n = jnp.asarray(h1[0]).shape[0]
        counts = tuple(nocc)
        frozen_sets = unrestricted_frozen_indices(n, counts, frozen)
    else:
        n = jnp.asarray(h1).shape[0]
        counts = (nocc, nocc)
        frozen_sets = (frozen_indices(n, nocc, frozen),)*2
    occupied = [[s*n+i for i in range(no) if i not in fr]
                for s, (no, fr) in enumerate(zip(counts, frozen_sets))]
    virtual = [[s*n+i for i in range(no, n) if i not in fr]
               for s, (no, fr) in enumerate(zip(counts, frozen_sets))]
    space = SpinAmplitudeSpace(tuple(map(len, occupied)), tuple(map(len, virtual)))

    def to_spin(a, b):
        if unrestricted:
            return space.from_blocks(a, b, jnp.asarray(h1[0]).dtype)
        same_spin = b-b.swapaxes(2, 3)
        return space.from_blocks((a, a), (same_spin, b, same_spin), jnp.asarray(h1).dtype)

    t1, t2 = to_spin(result.t1, result.t2)
    l1, l2 = to_spin(left.l1, left.l2)
    active1, active2 = active_density_parts(t1, t2, l1, l2)
    active = jnp.asarray(occupied[0]+occupied[1]+virtual[0]+virtual[1], dtype=jnp.int32)
    dc = jnp.zeros((2*n, 2*n), t1.dtype).at[jnp.ix_(active, active)].set(active1)
    dm2 = jnp.zeros((2*n,)*4, t1.dtype).at[jnp.ix_(active, active, active, active)].set(active2)
    full_occ = jnp.asarray(list(range(counts[0]))+list(range(n, n+counts[1])), dtype=jnp.int32)
    p = jnp.zeros_like(dc).at[full_occ, full_occ].set(1.)
    # Restore reference and core-active terms in the full spin-orbital space.
    dm2 += (jnp.einsum("pq,rs->pqrs", p, dc)+jnp.einsum("pq,rs->pqrs", dc, p)
            -jnp.einsum("ps,rq->pqrs", p, dc)-jnp.einsum("ps,rq->pqrs", dc, p)
            +jnp.einsum("pq,rs->pqrs", p, p)-jnp.einsum("ps,rq->pqrs", p, p))
    dm2 = jnp.where(result.converged & left.converged, dm2, jnp.nan)
    aa, ab, bb = dm2[:n, :n, :n, :n], dm2[:n, :n, n:, n:], dm2[n:, n:, n:, n:]
    return (aa, ab, bb) if unrestricted else aa+ab+ab.transpose(2, 3, 0, 1)+bb


def make_rdm2(h1, eri, result, *, nocc, frozen=None, config=None):
    """Orbital-unrelaxed real MO 2-RDM, including Lambda and frozen electrons.

    dm2[p,q,r,s] = <a_p^+ a_r^+ a_s a_q>, Hermitian-symmetrized for real
    observables. Restricted: spin sum; unrestricted: (aa,ab,bb), whose energy
    contraction weights are (.5,1,.5). A dense spin-orbital intermediate is
    used: this is an in-core reference path, not a low-memory density algorithm.
    CCSD(T), CC2 and linearized-model 2-RDMs are not supplied by this function.
    """
    cfg = CCConfig() if config is None else config
    left = solve_lambda(h1, eri, result, nocc=nocc, frozen=frozen, config=cfg)
    return _rdm2_from_lambda(h1, eri, result, left, nocc=nocc, frozen=frozen, config=cfg)
