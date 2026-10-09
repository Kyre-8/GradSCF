"""Active-space Hamiltonians and eager common-orbital SCF adaptation."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
import jax.numpy as jnp
from ..integrals.mo import validate_integrals, transform_integrals
from ..scf.reference import _array_signature, reference_state_signature
from .cistring import unpack_nelec


@dataclass(frozen=True)
class FCIReference:
    h1: object
    eri: object
    nelec: object
    ecore: object = 0.


def orbital_selection(norb, core=0, active=None):
    if isinstance(core, Integral) and not isinstance(core, bool):
        if not 0 <= core <= norb:
            raise ValueError("core count must be between 0 and norb")
        core = tuple(range(core))
    else:
        try:
            core = tuple(core)
        except TypeError as error:
            raise ValueError("core must be a count or orbital index sequence") from error
    try:
        active = None if active is None else tuple(active)
    except TypeError as error:
        raise ValueError("active must be an orbital index sequence") from error
    for name, indices in [('core',core),('active',() if active is None else active)]:
        if any(not isinstance(x,Integral) or isinstance(x,bool) or not 0 <= x < norb for x in indices) or len(set(indices)) != len(indices):
            raise ValueError(f"{name} indices must be distinct integers in [0,norb)")
    active = tuple(p for p in range(norb) if p not in core) if active is None else tuple(active)
    if set(core) & set(active):
        raise ValueError("Core and active orbital sets must be disjoint")
    return tuple(map(int,core)), tuple(map(int,active))


def fold_core(h1, eri, *, core=0, active=None, ecore=0.):
    """Return (h_active, eri_active, constant) for a doubly occupied core.

    Orbitals outside core+active are empty. Active orbital order is preserved.
    No electron count is inferred and no SCF or orbital optimization is run.
    """
    h,g = validate_integrals(h1,eri)
    core,active = orbital_selection(h.shape[0],core,active)
    c,a = jnp.asarray(core,dtype=jnp.int32),jnp.asarray(active,dtype=jnp.int32)
    hactive = h[jnp.ix_(a,a)]
    constant = jnp.asarray(ecore)
    if constant.shape != () or jnp.iscomplexobj(constant):
        raise ValueError("ecore must be a real scalar")
    if core:
        hactive = hactive + 2*jnp.einsum('pqii->pq',g[jnp.ix_(a,a,c,c)])
        hactive = hactive - jnp.einsum('piiq->pq',g[jnp.ix_(a,c,c,a)])
        gcc = g[jnp.ix_(c,c,c,c)]
        constant += 2*jnp.trace(h[jnp.ix_(c,c)]) + 2*jnp.einsum('iijj->',gcc) - jnp.einsum('ijji->',gcc)
    return hactive,g[jnp.ix_(a,a,a,a)],constant


def source_dimensions(source, spin=None):
    if isinstance(source,FCIReference):
        shape = np.shape(source.h1)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("FCIReference.h1 must be square")
        n = shape[0]
        return n,unpack_nelec(n,source.nelec,spin)
    from ..scf.facade import RKS
    from ..scf.roks import ROKS
    if not isinstance(source,(RKS,ROKS)):
        raise NotImplementedError("FCI accepts real common-orbital RKS/ROHF sources or FCIReference")
    if not source.converged or source.scf_result is None or source._scf_inputs is None:
        raise RuntimeError("Run and converge SCF before FCI")
    if np.ndim(source.mo_coeff) != 2:
        raise NotImplementedError("FCI requires a common spatial orbital frame")
    if source._cached_scf_key != source._scf_signature():
        raise RuntimeError("SCF inputs changed; run SCF again before FCI")
    n = source.mo_coeff.shape[1]
    return n,unpack_nelec(n,source.mol.nelectron,source.mol.spin if spin is None else spin)


def source_signature(source):
    if isinstance(source,FCIReference):
        return (unpack_nelec(np.shape(source.h1)[0],source.nelec),
                *(_array_signature(a) for a in (source.h1,source.eri,source.ecore)))
    return reference_state_signature(source)


def active_reference(source, core, active):
    if isinstance(source,FCIReference):
        return fold_core(source.h1,source.eri,core=core,active=active,ecore=source.ecore)
    inputs = source._scf_inputs
    coefficients = jnp.asarray(source.mo_coeff)
    overlap = np.asarray(inputs.overlap)
    c = np.asarray(coefficients)
    if (np.iscomplexobj(c) or not np.isfinite(c).all()
            or not np.allclose(c.T@overlap@c,np.eye(c.shape[1]),atol=1e-8,rtol=0)):
        raise ValueError("FCI requires finite real S-orthonormal orbitals")
    if getattr(inputs,'df_factors',None) is not None:
        representation = dict(df_factors=inputs.df_factors)
    elif getattr(inputs,'eri',None) is not None:
        key = 'eri' if np.ndim(inputs.eri)==4 else 'eri_pair_matrix'
        representation = {key:inputs.eri}
    else:
        pair = getattr(inputs,'eri_pair_matrix',None)
        if pair is None and hasattr(inputs,'response_eri_pair_matrix'):
            pair = inputs.response_eri_pair_matrix()
        if pair is None:
            raise ValueError("SCF source supplies no transformable two-electron integrals")
        representation = dict(eri_pair_matrix=pair)
    hcore = jnp.asarray(inputs.hcore)
    effective = hcore
    constant = jnp.asarray(inputs.nuclear_repulsion)
    if core:
        cc = coefficients[:,jnp.asarray(core,dtype=jnp.int32)]
        density = 2*cc@cc.T
        if 'df_factors' in representation:
            from ..df import build_jk_from_df
            coulomb,exchange = build_jk_from_df(representation['df_factors'],density)
        elif 'eri_pair_matrix' in representation:
            from ..integrals.layouts import build_jk_from_packed
            coulomb,exchange = build_jk_from_packed(representation['eri_pair_matrix'],density)
        else:
            from ..integrals.contraction import exchange_matrix
            eri = representation['eri']
            coulomb = jnp.einsum('pqrs,rs->pq',eri,density,precision='highest')
            exchange = exchange_matrix(eri,density)
        potential = coulomb-.5*exchange
        constant += jnp.sum(density*(hcore+.5*potential).T)
        effective = hcore+potential
    # Core folding stays in AO space. Only nactive^4 MO ERIs are produced,
    # even if many core orbitals are frozen.
    ca = coefficients[:,jnp.asarray(active,dtype=jnp.int32)]
    h,g = transform_integrals(effective,ca,**representation)
    return h,g,constant
