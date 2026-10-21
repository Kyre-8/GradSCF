"""Real collinear UCCSD/UCCD using spin-orbital equations and shared solvers.

This in-core reference path materializes spin integrals. UHF and ROHF orbitals
are accepted without canonicalization; no spin-adapted ROCCSD claim is made.
"""
from typing import NamedTuple
import jax.numpy as jnp
from gradscf.ao2mo import spin_orbital_integrals, unrestricted_frozen_indices
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
    h, g = spin_orbital_integrals(h1, eri)
    n = h.shape[0]//2
    frozen = unrestricted_frozen_indices(n, nocc, frozen)
    full_occ = jnp.asarray(list(range(nocc[0])) + list(range(n, n+nocc[1])), dtype=jnp.int32)
    # Core electrons contribute before the active projection.
    f = h + jnp.sum(g[:, :, full_occ, full_occ], axis=-1) - jnp.sum(g[:, full_occ, full_occ, :], axis=1)
    e = .5*jnp.sum((h+f)[full_occ, full_occ]) + nuclear_repulsion
    occ = [[s*n+i for i in range(no) if i not in fr] for s, (no, fr) in enumerate(zip(nocc, frozen))]
    vir = [[s*n+i for i in range(no, n) if i not in fr] for s, (no, fr) in enumerate(zip(nocc, frozen))]
    active = jnp.asarray(occ[0]+occ[1]+vir[0]+vir[1], dtype=jnp.int32)
    no = sum(map(len, occ))
    f = f[jnp.ix_(active, active)]
    g = g[jnp.ix_(active, active, active, active)]
    g = g.transpose(0, 2, 1, 3)-g.transpose(0, 2, 3, 1)
    o, v = slice(None, no), slice(no, None)
    ints = SpinCCIntegrals(f, g[o, o, o, o], g[o, o, o, v], g[o, o, v, v],
                          g[o, v, o, v], g[o, v, v, v], g[v, v, v, v], e)
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
