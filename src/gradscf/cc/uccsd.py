"""Real collinear UCCSD/UCCD using spin-orbital equations and shared solvers.

Only the required spin integral blocks are assembled from spatial integrals.
UHF and ROHF orbitals are accepted without canonicalization; no spin-adapted
ROCCSD claim is made. The integral blocks and amplitudes remain dense.
"""
from typing import NamedTuple
import jax.numpy as jnp
from gradscf.ao2mo import validate_unrestricted_integrals, unrestricted_frozen_indices
from gradscf.ao2mo.spin import _spin_eri_block
from ..solvers.nonlinear.iterate import NonlinearConfig, solve_nonlinear
from .types import CCConfig, CCResult
from .ground import linear_config, METHODS
from .spin_amplitudes import SpinAmplitudeSpace
from ._spin_equations import residual, correlation_energy


class SpinCCIntegrals(NamedTuple):
    fock: object
    oooo: object
    ooov: object
    oovv: object
    ovov: object
    ovvv: object
    vvvv: object
    reference_energy: object


def prepare_ucc_integrals(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0.):
    h1, eri = validate_unrestricted_integrals(h1, eri)
    dtype = jnp.result_type(*h1, *eri)
    ha, hb = (jnp.asarray(h, dtype=dtype) for h in h1)
    gaa, gab, gbb = eri = tuple(jnp.asarray(g, dtype=dtype) for g in eri)
    n = ha.shape[0]
    frozen = unrestricted_frozen_indices(n, nocc, frozen)
    # Core electrons contribute before the active projection.
    na, nb = nocc
    fa = (ha + jnp.einsum("pqii->pq", gaa[:, :, :na, :na])
          + jnp.einsum("pqii->pq", gab[:, :, :nb, :nb])
          - jnp.einsum("piiq->pq", gaa[:, :na, :na, :]))
    fb = (hb + jnp.einsum("pqii->pq", gbb[:, :, :nb, :nb])
          + jnp.einsum("iipq->pq", gab[:na, :na, :, :])
          - jnp.einsum("piiq->pq", gbb[:, :nb, :nb, :]))
    e = .5*(jnp.trace((ha+fa)[:na, :na]) + jnp.trace((hb+fb)[:nb, :nb]))
    e += nuclear_repulsion
    occ = tuple(tuple(i for i in range(no) if i not in fr)
                for no, fr in zip(nocc, frozen))
    vir = tuple(tuple(i for i in range(no, n) if i not in fr)
                for no, fr in zip(nocc, frozen))
    active = jnp.asarray(occ[0] + tuple(n+i for i in occ[1])
                         + vir[0] + tuple(n+i for i in vir[1]), dtype=jnp.int32)
    f = jnp.zeros((2*n, 2*n), dtype=dtype).at[:n, :n].set(fa).at[n:, n:].set(fb)
    f = f[jnp.ix_(active, active)]

    def antisymmetrized(p, q, r, s):
        first = _spin_eri_block(eri, (p, r, q, s))
        second = first if r == s else _spin_eri_block(eri, (p, s, q, r))
        return first.transpose(0, 2, 1, 3) - second.transpose(0, 2, 3, 1)

    o, v = occ, vir
    ints = SpinCCIntegrals(f, antisymmetrized(o, o, o, o),
                          antisymmetrized(o, o, o, v), antisymmetrized(o, o, v, v),
                          antisymmetrized(o, v, o, v), antisymmetrized(o, v, v, v),
                          antisymmetrized(v, v, v, v), e)
    return ints, tuple(map(len, occ)), tuple(map(len, vir))


def run_ucc(h1, eri, *, nocc, nuclear_repulsion=0., frozen=None, config=None, t1=None, t2=None):
    """Solve UCCSD or UCCD; return t1=(ta,tb), t2=(taa,tab,tbb).

    Differentiation includes amplitude response at a converged isolated root.
    Level shifting changes only the iteration preconditioner, not the residual.
    """
    cfg = CCConfig() if config is None else config
    if cfg.method not in {"ccsd", "ccd"}:
        raise NotImplementedError("Unrestricted models currently support CCSD and CCD")
    ints, occupied, virtual = prepare_ucc_integrals(h1, eri, nocc=nocc, frozen=frozen,
                                                   nuclear_repulsion=nuclear_repulsion)
    space = SpinAmplitudeSpace(occupied, virtual, cfg.method)
    no = sum(occupied)
    eps = jnp.diag(ints.fock)
    d1 = eps[:no, None]-eps[None, no:]-cfg.level_shift
    d2 = d1[:, None, :, None]+d1[None, :, None, :]
    guess1 = jnp.zeros_like(d1)
    guess2 = ints.oovv/jnp.where(jnp.abs(d2) > cfg.denominator_tol, d2, 1.)
    blocks1, blocks2 = space.to_blocks(guess1, guess2)
    guess1, guess2 = space.from_blocks(blocks1 if t1 is None else t1,
                                       blocks2 if t2 is None else t2, ints.fock.dtype)
    initial = space.pack(guess1, guess2)
    si, di = space.indices
    diagonal = jnp.concatenate((d1[tuple(si)], d2[tuple(di)]))
    minimum = jnp.min(jnp.abs(diagonal), initial=jnp.inf)
    valid = (minimum > cfg.denominator_tol) & jnp.isfinite(ints.reference_energy) & jnp.all(jnp.isfinite(ints.fock))
    safe = jnp.where(jnp.abs(diagonal) > cfg.denominator_tol, diagonal, 1.)
    r = lambda x: space.pack(*residual(*space.unpack(x), ints))
    e = lambda x: correlation_energy(*space.unpack(x), ints)
    solved = solve_nonlinear(r, initial, update=lambda x, rx: x+rx/safe, observable=e,
        config=NonlinearConfig(maxiter=cfg.max_cycle, residual_tol=cfg.residual_tol,
            energy_tol=cfg.conv_tol, diis_space=cfg.diis_space,
            diis_start=cfg.diis_start_cycle, damping=cfg.damping),
        linear_config=linear_config(cfg), valid_inputs=valid)
    ta, tb = space.to_blocks(*space.unpack(solved.solution))
    ecorr = e(solved.solution)
    return CCResult(ints.reference_energy+ecorr, ecorr, ints.reference_energy, ta, tb,
                    solved.residual_norm, solved.energy_change, solved.iterations,
                    solved.converged, minimum, jnp.asarray(METHODS.index(cfg.method), jnp.int32))
