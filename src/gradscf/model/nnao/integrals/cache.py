"""Read-only fixed-primitive integral cache storage."""
from pathlib import Path
import math
import hashlib
import json
import struct
import zipfile
import numpy as np


def fixed_signature(layout, symbols, coords, basis_family, auxbasis, topology, parameters):
    """Preserve the fixed-primitive cache identity used by accepted pilot runs."""
    payload = dict(symbols=symbols, coords=coords.tolist(), basis=basis_family,
                   auxbasis=auxbasis, cart=layout.topology.cart,
                   angular_momenta=topology.angular_momenta,
                   primitive_counts=topology.primitive_counts,
                   exponents=[np.asarray(x).tolist() for x in parameters.exponents])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_fixed_cache(path, signature, storage):
    import jax.numpy as jnp
    names = ('overlap', 'hcore', 'df_factors')
    with np.load(path, allow_pickle=False) as cache:
        if str(cache['signature'].item()) != signature:
            raise ValueError('Integral cache does not match this geometry and basis.')
        if storage == 'mmap':
            return tuple(map_npz_numeric(path, name) for name in names)
        array = np.asarray if storage == 'host' else jnp.asarray
        return tuple(array(cache[name]) for name in names)


def write_fixed_cache(path, signature, overlap, hcore, factors):
    path = Path(path)
    if not str(path).endswith('.npz'):
        path = Path(str(path)+'.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, signature=np.asarray(signature), overlap=np.asarray(overlap),
             hcore=np.asarray(hcore), df_factors=np.asarray(factors))
    return path

def map_npz_numeric(path,name):
    """Read-only map a numeric ZIP_STORED NPY member of the original NPZ file."""
    with zipfile.ZipFile(path) as archive,Path(path).open('rb') as stream:
        member=archive.getinfo(name+'.npy')
        if member.compress_type!=zipfile.ZIP_STORED or member.flag_bits&1:
            raise ValueError('NPZ mmap requires unencrypted ZIP_STORED members; compressed arrays cannot be mapped.')
        stream.seek(member.header_offset)
        header=struct.unpack('<4s5H3I2H',stream.read(30))
        if header[0]!=b'PK\x03\x04' or header[3]!=zipfile.ZIP_STORED or header[2]&1:
            raise ValueError('Invalid ZIP_STORED local header for NPZ mmap.')
        start=member.header_offset+30+header[-2]+header[-1]
        stream.seek(start)
        version=np.lib.format.read_magic(stream)
        readers={(1,0):np.lib.format.read_array_header_1_0,
                 (2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError(f'Unsupported NPY version for NPZ mmap: {version}.')
        shape,fortran,dtype=readers[version](stream)
        if dtype.hasobject or dtype.kind not in 'biufc':
            raise ValueError('NPZ mmap supports only numeric arrays without object dtype.')
        offset=stream.tell();size=math.prod(shape)*dtype.itemsize
        if offset+size!=start+member.file_size:
            raise ValueError('NPZ numeric payload size differs from its NPY header.')
    return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape,
                     order='F' if fortran else 'C')
