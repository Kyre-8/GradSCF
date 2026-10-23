"""Internal eager HF source adapter for selected post-HF transformations."""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gradscf._memory import (array_storage_bytes, resolve_memory_mb,
                             validate_reserved_bytes, workspace_bytes)
from gradscf.ao2mo.reference import transform_fock


class SCFSource(NamedTuple):
    coefficients: object
    focks: tuple
    nocc: object
    reference_energy: object
    hcore: object
    ao_kwargs: dict


def prepare_scf_source(source, *, max_memory=None, allow_rohf=False, reserved_bytes=0):
    """Read current RHF/UHF orbitals, solved Focks and one AO representation.

    No J/K or complete MO ERI is constructed. Native direct providers remain
    direct; legacy restricted JAX providers retain their packed AO fallback,
    with the construction budget checked before materializing that fallback.
    ``hcore`` is existing AO metadata, without a full MO one-body transform.
    """
    from gradscf.scf.facade import RKS, UKS
    from gradscf.scf.roks import ROKS
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis

    max_memory = resolve_memory_mb(max_memory, source)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    if not isinstance(allow_rohf, bool):
        raise TypeError('allow_rohf must be a static bool')
    rohf = allow_rohf and isinstance(source, ROKS)
    if (not isinstance(source, (RKS, UKS)) and not rohf) or str(source.xc).strip().lower() != "hf":
        raise NotImplementedError("Post-HF source preparation requires RHF/UHF; ROHF and DFT are not supported")
    source._check_reference_source()
    coeff = jnp.asarray(source.mo_coeff)
    if jnp.iscomplexobj(coeff):
        raise NotImplementedError("Post-HF source preparation requires real orbitals")
    unrestricted = isinstance(source, UKS) or rohf
    if rohf:
        inputs, result = source._scf_inputs, source.scf_result
        if inputs is None or result is None:
            raise RuntimeError('Solved ROHF spin Focks are missing; run SCF again before post-HF preparation')
        ns, cs = (inputs.nalpha, inputs.nbeta), (coeff, coeff)
        focks = tuple(transform_fock(f, coeff) for f in
                      (result.fock_matrix_alpha, result.fock_matrix_beta))
        reference_energy = result.total_energy
        hcore = inputs.hcore
        factors, direct = inputs.df_factors, getattr(inputs,'direct_basis',None)
        eri, pair = inputs.eri, getattr(inputs,'eri_pair_matrix',None)
    elif unrestricted:
        ref = source.reference
        if ref is None or ref.fock_matrix is None:
            raise RuntimeError("Solved UHF Fock data is missing; run SCF again before post-HF preparation")
        ns = (ref.nocc_alpha, ref.nocc_beta)
        cs = tuple(coeff)
        focks = tuple(transform_fock(f, c) for c, f in zip(cs, ref.fock_matrix))
        reference_energy = getattr(ref, "mf_energy", None)
        hcore = ref.h1e
        factors, direct, eri, pair = ref.df_factors, ref.direct_basis, ref.rep_tensor, None
    else:
        inputs = source._scf_inputs
        if inputs.nelectron % 2:
            raise ValueError("Restricted HF requires an even electron count")
        ns, cs = (inputs.nelectron // 2,), (coeff,)
        focks = (transform_fock(source.scf_result.fock_matrix, coeff),)
        reference_energy = getattr(source.scf_result, "total_energy", None)
        hcore = inputs.hcore
        factors, direct = inputs.df_factors, inputs.direct_basis
        eri, pair = inputs.eri, inputs.eri_pair_matrix
    if reference_energy is None:
        raise RuntimeError("Solved HF energy data is missing; run SCF again before post-HF preparation")
    expected = tuple((np.arange(c.shape[1]) < no).astype(int) for c, no in zip(cs, ns))
    expected = sum(expected) if rohf else np.stack(expected) if unrestricted else 2 * expected[0]
    if not np.array_equal(np.asarray(source.mo_occ), expected):
        raise ValueError("Post-HF source preparation requires integer occupations and occupied orbitals first")
    if factors is not None:
        kwargs = {"df_factors": factors}
    elif isinstance(direct, (NativeDirectBasis, ProjectedNativeDirectBasis)):
        kwargs = {"direct_basis": direct}
    elif eri is not None and eri.size:
        kwargs = {"eri" if eri.ndim == 4 else "eri_pair_matrix": eri}
    elif pair is not None:
        kwargs = {"eri_pair_matrix": pair}
    elif not unrestricted and direct is not None:
        npair = coeff.shape[0] * (coeff.shape[0] + 1) // 2
        itemsize = max(8, np.dtype(jnp.result_type(coeff, inputs.hcore)).itemsize)
        resident = array_storage_bytes(coeff, *focks, inputs.overlap, inputs.hcore,
            inputs.ao, inputs.ao_deriv1, inputs.ao_laplacian, inputs.coords, inputs.grid_weights)
        device = getattr(coeff, "device", None)
        device = device() if callable(device) else device
        available = workspace_bytes(max_memory, resident_bytes=resident+reserved_bytes, device=device)
        # Packed JAX construction can retain an accumulator, a chunk result
        # and their sum; scan accumulators also carry a sentinel AO-pair row.
        required = 3 * (npair + 1)**2 * itemsize
        if required > available:
            raise ValueError("Legacy packed AO ERI construction buffers exceed the max_memory budget")
        pair = inputs.response_eri_pair_matrix()
        if pair is None:
            raise ValueError("Post-HF source preparation requires two-electron integral data")
        kwargs = {"eri_pair_matrix": pair}
    else:
        raise ValueError("Post-HF source preparation requires two-electron integral data")
    return SCFSource(cs if unrestricted else coeff, focks,
                     ns if unrestricted else ns[0], reference_energy, hcore, kwargs)
