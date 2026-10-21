"""Internal eager HF source adapter for selected post-HF transformations."""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gradscf._memory import array_storage_bytes, resolve_memory_mb, workspace_bytes
from gradscf.ao2mo.reference import transform_fock


class SCFSource(NamedTuple):
    coefficients: object
    focks: tuple
    nocc: object
    reference_energy: object
    hcore: object
    ao_kwargs: dict


def prepare_scf_source(source, *, max_memory=None):
    """Read current RHF/UHF orbitals, solved Focks and one AO representation.

    No J/K or complete MO ERI is constructed. Native direct providers remain
    direct; legacy restricted JAX providers retain their packed AO fallback,
    with the construction budget checked before materializing that fallback.
    ``hcore`` is existing AO metadata, without a full MO one-body transform.
    """
    from gradscf.scf.facade import RKS, UKS
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis

    max_memory = resolve_memory_mb(max_memory, source)
    if not isinstance(source, (RKS, UKS)) or str(source.xc).lower() != "hf":
        raise NotImplementedError("Post-HF source preparation requires RHF/UHF; ROHF and DFT are not supported")
    source._check_reference_source()
    coeff = jnp.asarray(source.mo_coeff)
    if jnp.iscomplexobj(coeff):
        raise NotImplementedError("Post-HF source preparation requires real orbitals")
    unrestricted = isinstance(source, UKS)
    if unrestricted:
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
    expected = np.stack(expected) if unrestricted else 2 * expected[0]
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
        available = workspace_bytes(max_memory, resident_bytes=resident, device=device)
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
