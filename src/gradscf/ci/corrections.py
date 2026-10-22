"""Head-Gordon singlet CIS(D); distinct from variational CISD."""
from types import SimpleNamespace

import jax.numpy as jnp
import jax
from gradscf.ao2mo import transform_eri_blocks
from gradscf._memory import (DEFAULT_MAX_MEMORY_MB,array_storage_bytes,
                             resolve_memory_mb,validate_reserved_bytes,workspace_bytes)

from ..tddft.cisd import restricted_cisd_second_order_correction
from ..tddft.types import TDAResult
from .space import frozen_indices
from .types import CISDCorrectionResult
from ..solvers.diagnostics import require_converged_derivative


def cis_d_correction(eri, mo_energy, singles, *, nocc, frozen=None, denominator_tol=1e-10):
    """Return unscaled singlet CIS(D) corrections, in Hartree.

    Inputs must describe canonical RHF orbitals and singlet CIS roots. Spatial
    amplitudes use solve_cis's unit norm. No empirical scaling, damping or level
    shift is applied. Near-singular MP2/excited denominators are marked invalid.

    For derivatives, solve_cis must use gradient_mode='implicit_eigenvector';
    otherwise this function reports NaN correction derivatives, not partial AD.
    Formula: <CIS|V|U2 HF> + <CIS|V|T2 U1 HF> - E_MP2.
    Method: Head-Gordon et al. (1994), doi:10.1016/0009-2614(94)00070-0.
    Working-equation companion (not a numerical software comparison):
    https://manual.q-chem.com/5.1/sect-excorr.html (7.38–7.40).
    See ci/REFERENCES.md for attribution and validation scope.
    """
    if isinstance(singles.singlet, bool) and not singles.singlet:
        raise NotImplementedError("CIS(D) currently requires singlet CIS roots")
    eri, eps = jnp.asarray(eri), jnp.asarray(mo_energy)
    if jnp.iscomplexobj(eri) or jnp.iscomplexobj(eps):
        raise NotImplementedError("CIS(D) requires real integrals and orbital energies")
    if eps.ndim != 1 or eri.shape != (eps.size,) * 4:
        raise ValueError("Inconsistent MO energies and ERI dimensions")
    if not isinstance(nocc, int) or not 0 < nocc < eps.size:
        raise ValueError("CIS(D) requires occupied and virtual orbitals")
    if denominator_tol <= 0:
        raise ValueError("denominator_tol must be positive")
    frozen = frozen_indices(eps.size, nocc, frozen)
    occ = [i for i in range(nocc) if i not in frozen]
    vir = [i for i in range(nocc, eps.size) if i not in frozen]
    if not occ or not vir:
        raise ValueError("CIS(D) requires active occupied and virtual orbitals")
    if singles.amplitudes.shape != (singles.excitation_energies.size, len(occ), len(vir)):
        raise ValueError("CIS amplitudes do not match the active orbital space")
    active = jnp.asarray(occ + vir, dtype=jnp.int32)
    eps = eps[active]
    g = eri[jnp.ix_(active, active, active, active)]
    no, nm = len(occ), len(occ) + len(vir)
    reference = SimpleNamespace(mo_coeff=jnp.eye(nm, dtype=g.dtype), mo_energy=eps,
                                mo_occ=jnp.concatenate([jnp.full((no,), 2.), jnp.zeros((nm-no,))]),
                                mo_eri=g, nocc=no)
    return _evaluate(reference,singles,no,eps,denominator_tol)


def _evaluate(reference,singles,no,eps,denominator_tol,blocks=None):
    denominator = (eps[None, None, no:, None] + eps[None, None, None, no:]
                   - eps[:no, None, None, None] - eps[None, :no, None, None])
    omegas = singles.excitation_energies
    min_abs = jnp.minimum(jnp.min(jnp.abs(denominator)), jnp.min(
        jnp.abs(denominator[None, ...] - omegas[:, None, None, None, None]), axis=(1, 2, 3, 4)))
    # The legacy restricted-TDA convention has ||X||^2=1/2.
    tda = TDAResult(omegas, singles.amplitudes / jnp.sqrt(2.))
    raw = restricted_cisd_second_order_correction(reference, tda,_blocks=blocks)
    norm = jnp.sum(singles.amplitudes**2, axis=(1, 2))
    valid = (singles.converged & singles.singlet & (min_abs > denominator_tol) & jnp.isfinite(raw)
             & (jnp.abs(norm - 1) < 1e-7))
    raw = require_converged_derivative(raw, valid & singles.amplitude_response)
    correction = jnp.where(valid, raw, jnp.nan)
    return CISDCorrectionResult(omegas + correction, correction, omegas, min_abs, valid)


def cis_d_from_orbitals(coefficients,mo_energy,singles,*,nocc,frozen=None,
                        denominator_tol=1e-10,max_memory=DEFAULT_MAX_MEMORY_MB,
                        eri=None,eri_pair_matrix=None,df_factors=None,direct_basis=None,
                        reserved_bytes=0):
    """Prepare four selected spatial blocks for the unchanged CIS(D) formula.

    This retains its existing spin intermediates and root contractions; known
    spin buffers are checked before the shared AO2MO batch. No complete AO/MO
    ERI or full spin tensor is created. C/eps/amplitude response stays live.
    """
    maximum=resolve_memory_mb(max_memory)
    reserved_bytes=validate_reserved_bytes(reserved_bytes)
    c,eps=jnp.asarray(coefficients),jnp.asarray(mo_energy)
    if c.ndim != 2 or eps.shape != (c.shape[1],) or jnp.iscomplexobj(c) or jnp.iscomplexobj(eps):
        raise ValueError('Supply real orbital coefficients and matching MO energies')
    if not isinstance(nocc,int) or not 0 < nocc < c.shape[1] or denominator_tol <= 0:
        raise ValueError('CIS(D) requires occupied/virtual orbitals and positive denominator_tol')
    if isinstance(singles.singlet,bool) and not singles.singlet:
        raise NotImplementedError('CIS(D) currently requires singlet CIS roots')
    frozen=frozen_indices(c.shape[1],nocc,frozen)
    occupied=[i for i in range(nocc) if i not in frozen]
    virtual=[i for i in range(nocc,c.shape[1]) if i not in frozen]
    no,nv=len(occupied),len(virtual)
    if not no or not nv or singles.amplitudes.shape != (singles.excitation_energies.size,no,nv):
        raise ValueError('CIS amplitudes do not match the active orbital space')
    dtype=jnp.result_type(c,eps,singles.amplitudes,singles.excitation_energies,
                          *jax.tree.leaves((eri,eri_pair_matrix,df_factors)),
                          jnp.float64 if direct_basis is not None else c)
    spin_buffers=16*(4*no**2*nv**2+no*nv**3+no**3*nv)*dtype.itemsize
    # The unchanged shared helper vmaps roots; account for that root-dependent
    # moment/denominator workspace in addition to the root-independent blocks.
    root_buffers=singles.excitation_energies.size*80*no**2*nv**2*dtype.itemsize
    resident=(reserved_bytes+array_storage_bytes(c,eps,*jax.tree.leaves(singles))
              +3*spin_buffers+root_buffers)
    workspace_bytes(maximum,resident_bytes=resident,device=getattr(c,'device',None))
    co,cv=c[:,jnp.asarray(occupied)],c[:,jnp.asarray(virtual)]
    requests=((0,1,0,1),(1,1,1,0),(1,0,1,1),(0,0,1,0))
    spatial=transform_eri_blocks((co,cv),requests,eri=eri,eri_pair_matrix=eri_pair_matrix,
        df_factors=df_factors,direct_basis=direct_basis,max_memory=maximum,reserved_bytes=resident)
    values=dict(zip(requests,spatial))
    def chemist(types):
        tensor=values[types]
        indices=tuple(jnp.repeat(jnp.arange(n),2) for n in tensor.shape)
        spin=tuple(jnp.tile(jnp.arange(2),n).reshape(tuple(2*n if i == axis else 1 for i in range(4)))
                   for axis,n in enumerate(tensor.shape))
        return tensor[jnp.ix_(*indices)]*((spin[0] == spin[1]) & (spin[2] == spin[3]))
    def antisym(p,q,r,s):
        direct=chemist((p,r,q,s)).transpose(0,2,1,3)
        exchange=chemist((p,s,q,r)).transpose(0,2,3,1)
        return direct-exchange
    blocks=(antisym(0,0,1,1),antisym(1,1,1,0),antisym(0,1,0,0),chemist((0,1,0,1)))
    active=jnp.asarray(occupied+virtual,dtype=jnp.int32)
    eps=eps[active]
    reference=SimpleNamespace(mo_coeff=jnp.eye(no+nv,dtype=dtype),mo_energy=eps,
        mo_occ=jnp.concatenate((jnp.full((no,),2.),jnp.zeros(nv))),nocc=no)
    return _evaluate(reference,singles,no,eps,denominator_tol,blocks)
