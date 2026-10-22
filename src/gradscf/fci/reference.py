"""Active-space Hamiltonians and eager common-orbital SCF adaptation."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
import jax
import jax.numpy as jnp
from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, array_storage_bytes,
    resolve_memory_mb, tree_storage_bytes, validate_reserved_bytes, workspace_bytes)
from gradscf.ao2mo import validate_integrals, transform_integrals
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


def active_reference(source, core, active, *, max_memory=None):
    """Adapt an eager common frame; max_memory budgets active preparation."""
    max_memory = resolve_memory_mb(max_memory,source)
    if isinstance(source,FCIReference):
        shape = np.shape(source.h1)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("FCIReference.h1 must be square")
        core,active = orbital_selection(shape[0],core,active)
        arrays = (source.h1,source.eri,source.ecore)
        itemsize = max(8,*(np.dtype(getattr(a,'dtype',np.float64)).itemsize for a in arrays))
        resident = array_storage_bytes(*arrays)
        # NumPy/sequences may require a backend copy. This is a conservative
        # forecast: CPU jnp.asarray can alias NumPy storage. Existing JAX and
        # abstract/traced inputs require no additional full-array copy here.
        numerical = (jax.Array,jax.core.Tracer,jax.ShapeDtypeStruct)
        if not isinstance(source.h1,numerical):
            resident += shape[0]**2*itemsize
        if not isinstance(source.eri,numerical):
            resident += shape[0]**4*itemsize
        device = getattr(source.h1,'device',None)
        device = device() if callable(device) else device
        available = workspace_bytes(max_memory,resident_bytes=resident,device=device)
        nc,na = len(core),len(active)
        # fold_core gathers the active ERI, both active/core blocks, and the
        # closed-core ERI, while retaining one-body gather/update versions.
        required = itemsize*(na**4 + 3*na**2 + 2*na**2*nc**2 + nc**4 + nc**2)
        if required > available:
            raise ValueError("Explicit FCI core-folding arrays exceed the working-memory budget")
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
    elif getattr(inputs,'eri',None) is not None and np.size(inputs.eri):
        key = 'eri' if np.ndim(inputs.eri)==4 else 'eri_pair_matrix'
        representation = {key:inputs.eri}
    else:
        pair = getattr(inputs,'eri_pair_matrix',None)
        from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
        direct = getattr(inputs,'direct_basis',None)
        if pair is not None and np.size(pair):
            representation = dict(eri_pair_matrix=pair)
        elif isinstance(direct,(NativeDirectBasis,ProjectedNativeDirectBasis)):
            representation = dict(direct_basis=direct)
        else:
            if hasattr(inputs,'response_eri_pair_matrix'):
                nao = coefficients.shape[0]
                npair = nao*(nao+1)//2
                itemsize = max(8,np.dtype(jnp.result_type(coefficients,inputs.hcore)).itemsize)
                resident = tree_storage_bytes(source,coefficients)
                device = getattr(coefficients,'device',None)
                device = device() if callable(device) else device
                available = workspace_bytes(max_memory,resident_bytes=resident,device=device)
                # Legacy packed construction retains an accumulator, a
                # chunk and their sum, including one sentinel AO-pair row.
                if 3*(npair+1)**2*itemsize > available:
                    raise ValueError("Legacy packed AO ERI construction buffers exceed the working-memory budget")
                pair = inputs.response_eri_pair_matrix()
            if pair is None:
                raise ValueError("SCF source supplies no transformable two-electron integrals")
            representation = dict(eri_pair_matrix=pair)
    numerical = (jax.Array,jax.core.Tracer,jax.ShapeDtypeStruct)
    child = (inputs.hcore,coefficients,inputs.nuclear_repulsion,*jax.tree.leaves(representation))
    reserved = tree_storage_bytes(source,exclude=tuple(x for x in child if isinstance(x,numerical)))
    return _active_from_orbitals(inputs.hcore,coefficients,core=core,active=active,
        ecore=inputs.nuclear_repulsion,max_memory=max_memory,reserved_bytes=reserved,**representation)


def _check_native_active_memory(hcore, coefficients, constant, basis, ncore, nactive,
                                max_memory, reserved_bytes=0):
    """Check known CAS/core arrays before native J/K or shell evaluation.

    The block transformer additionally checks its minimum shell tile. These
    forward-array estimates do not bound complete solver/AD process residency.
    """
    from gradscf.integrals.backends.native.jk import (
        NativeDirectBasis, ProjectedNativeDirectBasis, native_jk_workspace_bytes,
    )
    nao = coefficients.shape[0]
    required = 8*(3*nactive**4 + 3*nactive**2 + nao*(ncore+nactive))
    primitive = basis
    while isinstance(primitive,ProjectedNativeDirectBasis):
        transform = primitive.transform
        # Promotion/projection buffers, plus primitive density/J/K versions
        # retained by the closed-core action, use the native float64 ABI.
        required += 8*transform.shape[0]*(transform.shape[1]+ncore+nactive)
        if ncore:
            required += 8*4*transform.shape[0]**2
        primitive = primitive.primitive_basis
    if not isinstance(primitive,NativeDirectBasis):
        raise TypeError("direct_basis must be a native direct integral basis")
    if ncore:
        required += 8*8*nao**2 + native_jk_workspace_bytes(primitive,1,0.)
    arrays = (hcore,coefficients,constant,*jax.tree.leaves(basis))
    device = next((getattr(a,'device',None) for a in arrays
                   if not isinstance(a,jax.core.Tracer) and getattr(a,'device',None) is not None),None)
    if callable(device):
        device = device()
    if device is None:
        device = getattr(jax.config,'jax_default_device',None)
    available = workspace_bytes(max_memory,
        resident_bytes=reserved_bytes+array_storage_bytes(*arrays),device=device)
    if required > available:
        raise ValueError("Native FCI active/core arrays exceed the working-memory budget")


def _active_from_orbitals(hcore, coefficients, *, core=0, active=None, ecore=0.,
                          max_memory=DEFAULT_MAX_MEMORY_MB, eri=None,
                          eri_pair_matrix=None, df_factors=None, direct_basis=None,
                          reserved_bytes=0):
    """Prepare a fixed real spatial frame without eager SCF validation.

    Return (h_active, eri_active, Ecore), preserving supplied active order.
    Coefficients, one-electron/nuclear inputs and the native provider remain
    live JAX inputs; this does not differentiate an SCF source or choose MOs.
    Native coordinates retain the shared direct operators' order-two AD
    boundary. Only the necessary nactive**4 MO ERI output is produced.
    """
    max_memory = resolve_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    hcore,coefficients,constant = map(jnp.asarray,(hcore,coefficients,ecore))
    if coefficients.ndim != 2 or hcore.shape != (coefficients.shape[0],)*2:
        raise ValueError("Inconsistent AO Hamiltonian and MO coefficient dimensions")
    if jnp.iscomplexobj(hcore) or jnp.iscomplexobj(coefficients):
        raise NotImplementedError("FCI requires real orbitals and integrals")
    if constant.shape != () or jnp.iscomplexobj(constant):
        raise ValueError("ecore must be a real scalar")
    core,active = orbital_selection(coefficients.shape[1],core,active)
    representation = {name:value for name,value in (
        ('eri',eri),('eri_pair_matrix',eri_pair_matrix),
        ('df_factors',df_factors),('direct_basis',direct_basis)) if value is not None}
    if len(representation) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, df_factors, direct_basis")
    if direct_basis is not None:
        _check_native_active_memory(hcore,coefficients,constant,direct_basis,
                                    len(core),len(active),max_memory,reserved_bytes)
    ca = coefficients[:,jnp.asarray(active,dtype=jnp.int32)]

    def transform(effective, *retained):
        reserved = reserved_bytes+tree_storage_bytes(hcore, coefficients, ecore, constant, *retained,
                                                     exclude=(effective, ca))
        return transform_integrals(effective, ca, max_memory=max_memory,
                                    reserved_bytes=reserved, **representation)

    if direct_basis is not None:
        # The block planner's minimum shell-tile check precedes core J/K.
        h,g = transform(hcore)
    effective = hcore
    if core:
        cc = coefficients[:,jnp.asarray(core,dtype=jnp.int32)]
        density = 2*cc@cc.T
        if direct_basis is not None:
            coulomb,exchange = direct_basis.get_jk(density,screening_threshold=0.,workers=1)
        elif 'df_factors' in representation:
            from gradscf.integrals.molecular.jk import build_jk_from_df
            coulomb,exchange = build_jk_from_df(representation['df_factors'],density)
        elif 'eri_pair_matrix' in representation:
            from gradscf.integrals.molecular.jk import build_jk_from_packed
            coulomb,exchange = build_jk_from_packed(representation['eri_pair_matrix'],density)
        else:
            from gradscf.integrals.molecular.jk import exchange_matrix
            eri = representation['eri']
            coulomb = jnp.einsum('pqrs,rs->pq',eri,density,precision='highest')
            exchange = exchange_matrix(eri,density)
        potential = coulomb-.5*exchange
        constant += jnp.sum(density*(hcore+.5*potential).T)
        effective = hcore+potential
        if direct_basis is not None:
            h = h + ca.T@potential@ca
    # Core folding stays in AO space. Only nactive^4 MO ERIs are produced,
    # even if many core orbitals are frozen.
    if direct_basis is None:
        h,g = transform(effective, cc, density, potential) if core else transform(effective)
    return h,g,constant
