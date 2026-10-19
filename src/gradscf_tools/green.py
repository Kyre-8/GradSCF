"""Input adapters for independent GREEN reference calculations.

This module changes auxiliary representations, not the electronic Hamiltonian.
It contains no mean-field or GW solver and has no GREEN runtime dependency.
"""
import json
from pathlib import Path

import numpy as np


def pair_adjoint_factors(factor, partners, *, tolerance=1e-11,
                         max_memory_bytes=256*1024**2):
    """Represent a self-inverse transfer in GREEN's pair-adjoint convention.

    ``factor[k,P,m,n]`` joins k to ``partners[k]``. GREEN reconstructs the
    reverse pair by an adjoint, while an arbitrary complex auxiliary rotation
    need not satisfy that relation componentwise. The Hermitian and
    anti-Hermitian parts provide a compatible auxiliary representation:

        R(B)[k] = B[partners[k]].conj().T
        C = concatenate((B + R(B))/2, (B - R(B))/(2j), axis=aux)

    The complete cross-k pair Gram must remain unchanged within ``tolerance``.
    Otherwise this routine rejects the input rather than changing its Coulomb
    operator. Only self-inverse momentum transfers are supported here.
    Zero padding to a shared auxiliary dimension is a separate file-writing
    operation. The memory bound covers the principal factor/Gram temporaries.
    """
    b, indices = np.asarray(factor), np.asarray(partners)
    if (b.ndim != 4 or min(b.shape) < 1 or b.shape[-1] != b.shape[-2]
            or b.dtype.kind not in 'fc' or not np.isfinite(b).all()):
        raise ValueError('Use finite (nk,naux,nmo,nmo) floating-point factors')
    if b.dtype.itemsize < (16 if b.dtype.kind == 'c' else 8):
        raise ValueError('Reference factors require float64 or complex128')
    nk, naux, nmo, _ = b.shape
    if (indices.shape != (nk,) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= nk)
            or not np.array_equal(indices[indices], np.arange(nk))):
        raise ValueError('partners must describe a self-inverse momentum transfer')
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('Use a positive relative Gram tolerance below one')
    columns = nk*nmo*nmo
    estimated = 16*(3*columns*columns+4*naux*columns)
    if estimated > max_memory_bytes:
        raise MemoryError(f'Pair Gram certificate needs approximately {estimated} bytes')
    reverse = b[indices].conj().swapaxes(-1, -2)
    converted = np.concatenate(((b+reverse)/2, (b-reverse)/(2j)), axis=1)
    original_columns = b.transpose(1, 0, 2, 3).reshape(naux, -1)
    converted_columns = converted.transpose(1, 0, 2, 3).reshape(2*naux, -1)
    original_gram = original_columns.conj().T@original_columns
    converted_gram = converted_columns.conj().T@converted_columns
    difference = converted_gram-original_gram
    norm = np.linalg.norm(original_gram)
    error = float(np.linalg.norm(difference)/max(norm, np.finfo(float).tiny))
    if error > tolerance:
        raise ValueError(f'Pair-adjoint representation changes the Coulomb Gram by {error:.3e}')
    return converted, dict(original_rank=naux, converted_rank=2*naux,
        relative_gram_error=error, maximum_gram_error=float(np.max(abs(difference))))


def export_source(model, directory, *, kmesh=None, tolerance=1e-11):
    """Write an orthonormal source model in GREEN 1.1 input/DF format.

    Only complete source meshes with self-inverse transfers are supported.
    Full k/q/pair lists disable spatial and time-reversal reduction. Bare
    Hartree/exchange and correlation factors are written separately, with
    exact zero padding to a common auxiliary dimension. ``q0_treatment``
    must be ``IGNORE_G0`` when running the external solver.

    This writes input, not a reference result or an executable validation.
    h5py is needed only when the exporter is called. Existing output
    directories are preserved.
    """
    import h5py
    from gradscf.gw.pbc.source import validate_model
    from gradscf.pbc.kpoints import kmesh_shape

    validate_model(model)
    actual_mesh = kmesh_shape(model.lattice, model.kpoints)
    if kmesh is not None and tuple(kmesh) != actual_mesh:
        raise ValueError('kmesh must match the complete source mesh')
    nk, nmo = np.shape(model.energy)
    if model.madelung != 0.:
        raise ValueError('The body-only reference exporter requires zero Madelung correction')
    blocks, certificates = {}, {}
    for name, original in (('df_hf', model.factors), ('df_corr', model.correlation_factors)):
        result = [pair_adjoint_factors(value, np.asarray(model.table)[:, q], tolerance=tolerance)
                  for q, value in enumerate(original)]
        blocks[name] = [value for value, _ in result]
        certificates[name] = [certificate for _, certificate in result]
    naux = max(block.shape[1] for group in blocks.values() for block in group)
    pairs = [(left, right) for left in range(nk) for right in range(left+1)]
    fractional = np.mod(np.round(np.asarray(model.kpoints)@np.asarray(model.lattice).T/(2*np.pi), 12), 1.)
    if np.max(abs(fractional[model.zero_q])) > 1e-10:
        raise ValueError('The reference source must be a Gamma-centered momentum group')
    indices = np.arange(nk, dtype=np.int64)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with h5py.File(directory/'input.h5', 'x') as data:
        data.attrs['__green_version__'] = '1.1.0'
        for name, value in dict(nao=nmo, nso=nmo, ns=1, nel_cell=2*model.nocc,
                                nk=nk, NQ=naux).items():
            data['params/'+name] = value
        data['HF/H-k'] = np.asarray(model.hcore, dtype=complex)[None]
        data['HF/Fock-k'] = np.asarray(model.reference_fock, dtype=complex)[None]
        data['HF/S-k'] = np.tile(np.eye(nmo, dtype=complex), (1, nk, 1, 1))
        data['HF/Energy_nuc'] = float(model.nuclear_repulsion)
        data['HF/madelung'] = 0.
        data['HF/mo_energy'] = np.asarray(model.energy)
        data['HF/mo_coeff'] = np.tile(np.eye(nmo, dtype=complex), (nk, 1, 1))
        data['native/ao_to_mo_coeff'] = np.asarray(model.coeff)
        for label, dimension in (('k', nmo), ('q', naux)):
            prefix = 'symmetry/'+label+'/'
            data[prefix+('nk' if label == 'k' else 'nq')] = nk
            data[prefix+('ink' if label == 'k' else 'inq')] = nk
            data[prefix+'mesh_scaled'] = fractional
            data[prefix+'mesh'] = np.asarray(model.kpoints)
            data[prefix+'weight_ibz'] = np.ones(nk)
            data[prefix+'bz2ibz'] = indices
            data[prefix+'ibz2bz'] = indices
            data[prefix+'tr_conj'] = np.zeros(nk, dtype=np.int64)
            data[prefix+'n_stars'] = nk
            identity = np.tile(np.eye(dimension, dtype=complex), (nk, 1, 1))
            for matrix in (('k_sym_transform_ao',) if label == 'k' else
                           ('k_sym_transform_j2c', 'k_sym_transform_p0')):
                data[prefix+matrix] = identity
            for k in range(nk):
                data[prefix+'stars/'+str(k)] = np.array([k], dtype=np.int64)
        data['symmetry/k/nk_list'] = np.asarray(actual_mesh, dtype=np.int64)
        for name in ('conj_pairs_list', 'trans_pairs_list', 'kpair_irre_list'):
            data['symmetry/pairs/'+name] = np.arange(len(pairs), dtype=np.int64)
        data['symmetry/pairs/kpair_idx'] = np.asarray(pairs, dtype=np.int64)
        data['symmetry/pairs/num_kpair_stored'] = len(pairs)
    for name, factors in blocks.items():
        folder = directory/name
        folder.mkdir()
        with h5py.File(folder/'meta.h5', 'x') as data:
            data.attrs['__green_version__'] = '1.1.0'
            data['chunk_size'] = 1
        for pair_index, (left, right) in enumerate(pairs):
            transfers = np.flatnonzero(np.asarray(model.table)[left] == right)
            if len(transfers) != 1:
                raise ValueError('Each stored pair must have exactly one momentum transfer')
            factor = factors[int(transfers[0])][left]
            padded = np.zeros((1, naux, nmo, nmo), dtype=complex)
            padded[0, :len(factor)] = factor
            with h5py.File(folder/f'VQ_{pair_index}.h5', 'x') as data:
                data[str(pair_index)] = padded.view(np.float64)
    report = dict(input_file=str(directory/'input.h5'), naux=naux, kmesh=list(actual_mesh),
        source_frame='fixed orthonormal reference MO', q0_treatment='IGNORE_G0',
        factor_certificates=certificates, external_reader_validated=False)
    (directory/'export.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
