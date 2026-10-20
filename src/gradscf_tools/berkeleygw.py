"""Export fixed Gaussian Bloch orbitals to BerkeleyGW 4 binary files.

The WFN layout follows ``Common/wfn_rho_vxc_io_inc.f90`` in BerkeleyGW 4.0:
little-endian GNU Fortran records, complex128 coefficients and one spin.
Only scalar, restricted, integer-occupied 3D references are supported.
Orbital coefficients are preserved; an orthonormality failure is rejected.

The orbital FFT box is rectangular. Its coefficients are zero-padded to the
complete kinetic spheres expected by BGW; this never changes an orbital.
The larger density box prevents cyclic aliasing of two orbital Fourier
series in the exported representation. BGW's
spherical dielectric/bare cutoffs remain independent convergence parameters;
exporting a WFN does not establish a matched self-energy calculation.
"""
from pathlib import Path
import hashlib
import json
import re
import struct

import numpy as np


HARTREE_TO_EV = 27.211386245988


def _integers(values):
    return np.asarray(values, dtype='<i4').tobytes()


def _reals(values):
    return np.asarray(values, dtype='<f8').tobytes()


def _record(stream, payload):
    marker = struct.pack('<i', len(payload))
    stream.write(marker)
    stream.write(payload)
    stream.write(marker)


def _grid(mesh):
    return np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(n)*n).astype(int)
        for n in mesh], indexing='ij'), axis=-1).reshape(-1, 3)


def write_wfn(path, *, lattice_bohr, mesh, kpoints_fractional, kgrid,
              energies_hartree, occupations, periodic_values, atom_numbers,
              atom_positions_bohr, shift=(0., 0., 0.),
              wavefunction_cutoff_rydberg=None,
              orthonormality_tolerance=1e-8):
    """Write BGW complex binary WFN from periodic orbital values ``u_kn(r)``.

    ``periodic_values`` has shape ``(nk, prod(mesh), nmo)`` in bohr**(-3/2),
    with uniform real-space samples. ``occupations`` are per-spin (0 or 1).
    Energies are Hartree on input, Rydberg in WFN. Lattice vectors are rows.
    No orbitals are discarded, renormalized, or orthogonalized during export.
    ``wavefunction_cutoff_rydberg`` can give WFN and WFNq a common header
    cutoff; it must cover all actual G+k vectors and never truncates them.
    """
    lattice = np.asarray(lattice_bohr, dtype=float)
    mesh = tuple(int(n) for n in mesh)
    points = np.asarray(kpoints_fractional, dtype=float)
    energy = np.asarray(energies_hartree, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    values = np.asarray(periodic_values, dtype=complex)
    numbers = np.asarray(atom_numbers, dtype=int)
    positions = np.asarray(atom_positions_bohr, dtype=float)
    kgrid = np.asarray(kgrid, dtype=int)
    shift = np.asarray(shift, dtype=float)
    if lattice.shape != (3, 3) or np.linalg.det(lattice) <= 0:
        raise ValueError('Use a right-handed finite (3,3) lattice in bohr')
    if len(mesh) != 3 or min(mesh) < 1 or any(n % 2 == 0 for n in mesh):
        raise ValueError('Use three positive odd FFT mesh dimensions')
    if energy.ndim != 2 or not energy.size or occ.shape != energy.shape:
        raise ValueError('Energies and occupations must have shape (nk,nmo)')
    nk, nmo = energy.shape
    ngrid = int(np.prod(mesh))
    if points.shape != (nk, 3) or values.shape != (nk, ngrid, nmo):
        raise ValueError('K points or periodic orbital values have incompatible shapes')
    if kgrid.shape != (3,) or min(kgrid) < 1 or int(np.prod(kgrid)) != nk:
        raise ValueError('Export a complete uniform k mesh with an explicit kgrid')
    if shift.shape != (3,) or np.any(abs(shift) > 1.):
        raise ValueError('Shift must have three components between -1 and 1')
    if numbers.ndim != 1 or not len(numbers) or positions.shape != (len(numbers), 3):
        raise ValueError('Atom numbers and Cartesian positions must have matching shapes')
    if any(not np.isfinite(array).all() for array in (lattice, points, energy, occ, values, positions)):
        raise ValueError('Export arrays must be finite')
    nocc = np.sum(occ, axis=1).astype(int)
    if not np.array_equal(occ, (np.arange(nmo)[None] < nocc[:, None]).astype(float)) or np.any(nocc != nocc[0]):
        raise ValueError('Use identical integer Aufbau occupations at every k point')
    if not 0 < nocc[0] < nmo:
        raise ValueError('Export requires occupied and empty bands')
    volume = float(np.linalg.det(lattice))
    coefficients = np.fft.fftn(values.reshape((nk,)+mesh+(nmo,)), axes=(1, 2, 3)).reshape(values.shape)
    coefficients *= np.sqrt(volume)/ngrid
    gram = np.einsum('kgm,kgn->kmn', coefficients.conj(), coefficients)
    error = float(np.max(abs(gram-np.eye(nmo))))
    if error > orthonormality_tolerance:
        raise ValueError(f'Exported orbitals are not orthonormal: maximum error {error:.3e}')
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    gvecs = _grid(mesh)
    actual_cutoff = float(np.max(np.sum(((gvecs[None]+points[:, None])@reciprocal)**2, axis=-1)))
    ecutwfc = actual_cutoff if wavefunction_cutoff_rydberg is None else float(wavefunction_cutoff_rydberg)
    if not np.isfinite(ecutwfc) or ecutwfc < actual_cutoff-1e-10:
        raise ValueError('Declared wavefunction cutoff must cover every actual G+k vector')
    if np.max(abs(points)) > 1.+1e-10:
        raise ValueError('Use fractional reciprocal representatives between -1 and 1')
    # BGW genwf_mpi reconstructs the first ngk vectors of a kinetic-energy
    # sort. A rectangular subset is insufficient even when its norm is one.
    # Use a common bounding box determined only by the declared cutoff and
    # lattice, so WFN and WFNq retain the same full G-space/header FFT mesh.
    bounds = np.ceil(np.sqrt(ecutwfc)*np.linalg.norm(lattice, axis=1)/(2*np.pi)+1).astype(int)
    candidates = _grid(tuple(2*bounds+1))
    wave_vectors, wave_coefficients = [], []
    for k, point in enumerate(points):
        keep = np.sum(((candidates+point)@reciprocal)**2, axis=-1) <= ecutwfc+1e-10
        vectors = candidates[keep]
        in_original = np.all(abs(vectors) <= (np.asarray(mesh)-1)//2, axis=1)
        indices = np.ravel_multi_index(tuple((vectors % np.asarray(mesh)).T), mesh)
        padded = np.zeros((len(vectors), nmo), dtype=complex)
        padded[in_original] = coefficients[k, indices[in_original]]
        wave_vectors.append(vectors)
        wave_coefficients.append(padded)
    ngk = np.array([len(vectors) for vectors in wave_vectors])
    density_mesh = tuple(map(int, 4*bounds+1))
    full_gvecs = _grid(density_mesh)
    ng = len(full_gvecs)
    ecutrho = max(ecutwfc, float(np.max(np.sum((full_gvecs@reciprocal)**2, axis=-1))))
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve the existing exported file: {path}')
    with path.open('xb') as stream:
        write = lambda payload: _record(stream, payload)
        write(b'WFN-Complex'.ljust(32)+b'GradSCF fixed orbitals'.ljust(32)+b'00:00:00'.ljust(32))
        write(struct.pack('<5id3id', 1, ng, 1, 0, len(numbers), ecutrho, nk, nmo, int(ngk.max()), ecutwfc))
        write(_integers(density_mesh)+_integers(kgrid)+_reals(shift))
        # Fortran lattice vectors are columns; row-major Python flattening
        # of our row-vector lattice gives exactly that record ordering.
        write(_reals([volume, 1.])+_reals(lattice)+_reals(lattice@lattice.T))
        write(_reals([(2*np.pi)**3/volume, 1.])+_reals(reciprocal)+_reals(reciprocal@reciprocal.T))
        write(_integers(np.eye(3, dtype=int)))
        write(_reals(np.zeros(3)))
        write(b''.join(_reals(position)+_integers(number) for position, number in zip(positions, numbers)))
        write(_integers(ngk))
        write(_reals(np.full(nk, 1/nk)))
        write(_reals(points))
        write(_integers(np.ones(nk, dtype=int)))
        write(_integers(nocc))
        write(_reals(2*energy))
        write(_reals(occ))
        def block(payload, count):
            write(_integers(1))
            write(_integers(count))
            write(payload)
        block(_integers(full_gvecs), ng)
        for k in range(nk):
            block(_integers(wave_vectors[k]), int(ngk[k]))
            for band in range(nmo):
                block(np.asarray(wave_coefficients[k][:, band], dtype='<c16').tobytes(), int(ngk[k]))
    return dict(nmo=nmo, nocc=int(nocc[0]), kgrid=kgrid.tolist(),
        orbital_fft_mesh=list(mesh), density_fft_mesh=list(density_mesh),
        max_orthonormality_error=error, ecutwfc_rydberg=ecutwfc,
        actual_max_gk_squared_rydberg=actual_cutoff,
        wavefunction_g_counts=ngk.tolist(), zero_padding_only=True,
        ecutrho_rydberg=ecutrho, wavefunction=str(path),
        scope='Fixed scalar Gaussian Bloch orbitals; exact zero padding to kinetic spheres; BGW cutoffs require separate validation')


def write_vxc(path, kpoints_fractional, diagonal_hartree):
    """Write diagonal BGW ``vxc.dat`` in eV, with one-based band/spin IDs."""
    points = np.asarray(kpoints_fractional, dtype=float)
    diagonal = np.asarray(diagonal_hartree)
    if diagonal.ndim != 2 or points.shape != (len(diagonal), 3):
        raise ValueError('Use (nk,3) points and (nk,nmo) Vxc diagonals')
    if not np.isfinite(points).all() or not np.isfinite(diagonal).all():
        raise ValueError('Vxc data must be finite')
    with Path(path).open('x') as stream:
        for point, elements in zip(points, diagonal):
            stream.write(' '.join(f'{x:.12f}' for x in point)+f' {len(elements)} 0\n')
            for band, value in enumerate(elements, 1):
                value = complex(value)*HARTREE_TO_EV
                stream.write(f'1 {band} {value.real:.14e} {value.imag:.14e}\n')


def export_scf(mf, directory, *, kpoints=None, kgrid=None, shift=(0., 0., 0.),
               wfn_name='WFN', vxc_name='vxc.dat', wavefunction_cutoff_rydberg=None):
    """Export a converged restricted LDA/GGA reference or its shifted mesh.

    Optional ``kpoints`` are Cartesian inverse-bohr points. They are evaluated
    in the same converged SCF potential, which is required for a true WFNq.
    ``shift`` uses BGW's units of 1/kgrid. A shifted export can disable Vxc
    output with ``vxc_name=None``. It does not run BerkeleyGW.
    """
    from gradscf.pbc.scf import validate_reference
    from gradscf.pbc.kpoints import kmesh_shape
    from gradscf.pbc._kpoint import grid_xc_potential, project_xc_potential
    from gradscf.integrals.periodic.fft import build_band_inputs
    from gradscf.dft.xc import hybrid_coeff, parse_recipe, xc_type

    validate_reference(mf, integer=True, equal_weights=True)
    kind = xc_type(mf.xc)
    recipe = parse_recipe(mf.xc)
    recipe.require_complete_support()
    if recipe.exact_exchange.omega is not None:
        raise NotImplementedError('BGW export of screened starting potentials is not implemented.')
    if mf.cell.dimension != 3 or mf.unrestricted or kind not in ('LDA', 'GGA') or hybrid_coeff(mf.xc):
        raise NotImplementedError('BGW export currently requires a restricted pure LDA/GGA 3D reference')
    lattice = np.asarray(mf.cell.lattice)
    if kpoints is None:
        points = np.asarray(mf.kpts)
        target = mf.inputs
        energy = np.asarray(mf.result.mo_energy_spin[0])
        coefficients = np.asarray(mf.result.mo_coeff_spin[0])
    else:
        points = np.asarray(kpoints, dtype=float)
        energy, coefficients = map(np.asarray, mf.get_bands(points, chunk_size=1))
        target = build_band_inputs(mf.cell, points, mf.inputs)
    kgrid = kmesh_shape(lattice, points) if kgrid is None else kgrid
    fractions = points@lattice.T/(2*np.pi)
    nocc = int(mf.cell.nelectron//2)
    occupations = np.broadcast_to((np.arange(energy.shape[-1]) < nocc).astype(float), energy.shape)
    periodic_values = np.einsum('kga,kam->kgm', np.asarray(target.ao[:, 0]), coefficients)
    # Evaluate the same projected grid Vxc that the native query subtracts.
    _, potential = grid_xc_potential(mf.result.density_spin, mf.inputs, xc=mf.xc)
    vxc = np.asarray(project_xc_potential(target.ao, mf.inputs.weights, potential, kind)[0])
    vxc_diagonal = np.einsum('kam,kab,kbm->km', coefficients.conj(), vxc, coefficients)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = write_wfn(directory/wfn_name, lattice_bohr=lattice, mesh=mf.cell.mesh,
        kpoints_fractional=fractions, kgrid=kgrid, shift=shift,
        wavefunction_cutoff_rydberg=wavefunction_cutoff_rydberg,
        energies_hartree=energy, occupations=occupations, periodic_values=periodic_values,
        atom_numbers=mf.cell.topology.nuclear_charges, atom_positions_bohr=mf.cell.coords)
    report.update(xc=mf.xc, basis=str(mf.cell.basis), pseudo=str(mf.cell.pseudo),
        native_exxdiv=mf.exxdiv, native_madelung_hartree=float(mf.inputs.madelung),
        source_kpoints_fractional=(np.asarray(mf.kpts)@lattice.T/(2*np.pi)).tolist(),
        query_kpoints_fractional=fractions.tolist(), shifted_query=kpoints is not None,
        max_vxc_imaginary_hartree=float(np.max(abs(vxc_diagonal.imag))))
    if vxc_name is not None:
        write_vxc(directory/vxc_name, fractions, vxc_diagonal)
    with (directory/(wfn_name+'_export.json')).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def _key(point, spin, band):
    return tuple(np.round(point, 8)), int(spin), int(band)


def _text_blocks(path, *, matrix=False, hartree_to_ev=HARTREE_TO_EV):
    if not np.isfinite(hartree_to_ev) or hartree_to_ev <= 0:
        raise ValueError('The build energy conversion must be finite and positive')
    lines = [line.split() for line in Path(path).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    result, offset, seen = [], 0, set()
    while offset < len(lines):
        header = lines[offset]
        if len(header) != (5 if matrix else 4):
            raise ValueError(f'Invalid BGW block header in {path}')
        point, count = np.asarray(header[:3], dtype=float), int(header[3])
        if matrix and int(header[4]) != 0:
            raise NotImplementedError('Only diagonal BGW matrix-element tables are supported')
        if count < 1 or offset+count >= len(lines):
            raise ValueError(f'BGW block is truncated in {path}')
        for cells in lines[offset+1:offset+count+1]:
            if len(cells) != (4 if matrix else 5):
                raise ValueError(f'Invalid or truncated BGW data row in {path}')
            spin, band = int(cells[0]), int(cells[1])-1
            numbers = np.asarray(cells[2:], dtype=float)/hartree_to_ev
            key = _key(point, spin, band)
            if not np.isfinite(point).all() or not np.isfinite(numbers).all() or spin < 1 or band < 0 or key in seen:
                raise ValueError(f'Invalid or duplicate BGW state in {path}')
            seen.add(key)
            result.append((point, spin, band, numbers))
        offset += count+1
    return result


def read_eqp(path, *, hartree_to_ev=HARTREE_TO_EV):
    """Read raw BGW eqp0/eqp1 states into Hartree, preserving complex values.

    Bands become zero-based; spins retain BGW's one-based convention. An
    eqp1 file contains no root certificate: consult the Sigma flags and use
    the shared selector on its tabulated self-energy before claiming a QP.
    Set hartree_to_ev to twice the installed BGW RYD constant when its
    legacy code-unit convention must be recovered exactly.
    """
    return [dict(kpoint_fractional=point, spin=spin, band=band,
                 reference_hartree=values[0], qp_hartree=values[1]+1j*values[2])
            for point, spin, band, values in _text_blocks(path, hartree_to_ev=hartree_to_ev)]


def read_sigma(path, *, hartree_to_ev=HARTREE_TO_EV):
    """Read the 15-column CD spectrum into per-state Hartree arrays.

    A directory additionally reads actual diagonal ``x.dat`` and ``vxc.dat``.
    Raw imaginary signs are preserved. Each state must have a finite strictly
    increasing frequency table. No interpolation, root selection, linewidth
    interpretation or extrapolation occurs in this parser.
    BGW's fixed-width format3000 is accepted when full I4 frequency indices
    join the preceding band field (iw>=1000), as well as whitespace tables.
    """
    path = Path(path)
    if not np.isfinite(hartree_to_ev) or hartree_to_ev <= 0:
        raise ValueError('The build energy conversion must be finite and positive')
    directory = path if path.is_dir() else None
    rows = []
    for line in (path/'spectrum.dat' if directory else path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        cells = line.split()
        if len(cells) != 15:
            # shiftenergy_dyn.f90 format3000:
            # 3F12.5,2x,3I4,2x,9F12.5. Do not guess joined integer digits.
            if (len(line) < 160 or line[36:38].strip() or line[50:52].strip()
                    or line[160:].strip()):
                raise ValueError('Use a finite 15-column BerkeleyGW CD spectrum')
            cells = ([line[start:start+12] for start in (0, 12, 24)]
                     +[line[start:start+4] for start in (38, 42, 46)]
                     +[line[start:start+12] for start in range(52, 160, 12)])
        try:
            rows.append([float(cell) for cell in cells])
        except ValueError as error:
            raise ValueError('Use a finite 15-column BerkeleyGW CD spectrum') from error
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 15 or not np.isfinite(values).all():
        raise ValueError('Use a finite 15-column BerkeleyGW CD spectrum')
    if np.any(values[:, 3:6] != np.rint(values[:, 3:6])) or np.any(values[:, 3:6] < 1):
        raise ValueError('BGW spin, band and frequency indices must be positive integers')
    grouped = {}
    for row in values:
        grouped.setdefault(_key(row[:3], row[3], row[4]-1), []).append(row)
    matrices = {}
    flags = {}
    if directory:
        for name in ('x.dat', 'vxc.dat'):
            matrices[name] = {_key(point, spin, band):numbers[0]+1j*numbers[1]
                for point, spin, band, numbers in _text_blocks(directory/name, matrix=True, hartree_to_ev=hartree_to_ev)}
        if (directory/'sigma_hp.log').exists():
            point, spin = None, None
            for line in (directory/'sigma_hp.log').read_text().splitlines():
                match = re.search(r'\bk\s*=\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+).*spin\s*=\s*(\d+)', line)
                if match:
                    point, spin = tuple(map(float, match.groups()[:3])), int(match.group(4))
                cells = line.split()
                if point is not None and len(cells) == 12 and cells[0].isdigit():
                    flag = cells[-1]
                    if flag == 'unique' or flag.startswith(('MULT:', 'extrap', 'NO_SOLN')):
                        flags[_key(point, spin, int(cells[0])-1)] = flag
    result = []
    channels = ('total', 'residue', 'correlation', 'integral')
    for key, rows in grouped.items():
        rows = np.asarray(rows)
        rows = rows[np.argsort(rows[:, 6])]
        if len(rows) < 2 or np.any(np.diff(rows[:, 6]) <= 0):
            raise ValueError('Each BGW state needs strictly increasing frequency samples')
        point, spin, band = key
        record = dict(kpoint_fractional=np.asarray(point), spin=spin, band=band,
                      frequency_hartree=rows[:, 6]/hartree_to_ev)
        for index, channel in enumerate(channels):
            record[f'sigma_{channel}_hartree'] = (rows[:, 7+2*index]+1j*rows[:, 8+2*index])/hartree_to_ev
        if directory:
            if any(key not in matrix for matrix in matrices.values()):
                raise ValueError('The BGW spectrum state lacks exchange or Vxc data')
            record.update(exchange_hartree=matrices['x.dat'][key], vxc_hartree=matrices['vxc.dat'][key])
            record['reported_qp_flag'] = flags.get(key)
        result.append(record)
    return result


def _read_fortran_record(stream, *, maximum_bytes, digest=None):
    marker = stream.read(4)
    if len(marker) != 4:
        raise ValueError('Truncated GNU Fortran record marker')
    size, = struct.unpack('<i', marker)
    if not 0 <= size <= maximum_bytes:
        raise ValueError('Unsupported segmented, big-endian or oversized Fortran record')
    payload = stream.read(size)
    trailer = stream.read(4)
    if len(payload) != size or trailer != marker:
        raise ValueError('Truncated or inconsistent GNU Fortran record')
    if digest is not None:
        digest.update(marker+payload+trailer)
    return payload


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def copy_eps0_without_head_feedback(source, destination, *, head_floor=1e-12,
                                   maximum_memory_bytes=256*1024**2,
                                   identity_tolerance=1e-8):
    """Copy a verified BGW4 GNU/CPLX CD eps0mat with head-free body inverse.

    For each stored inverse S=[[A,B],[C,E]], replace its body by
    E-C*A**(-1)*B, head by one and wings by zero. This is the inverse of the
    original dielectric's body block, not a fitted self-energy correction.
    Retarded and advanced matrices are transformed independently. The
    original, every header byte, G ordering and frequency are preserved.

    Only the inspected non-subspace, one-q, complex full-frequency format
    is accepted. Two streamed passes retain only head vectors and two small
    certification frames. Singular heads or failed inverse identities raise;
    they are never clipped. The finite small q in the original remains.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError('Preserve original epsilon files; choose a fresh destination')
    if not np.isfinite(head_floor) or head_floor <= 0 or not np.isfinite(identity_tolerance) or identity_tolerance <= 0:
        raise ValueError('Use positive finite head and identity tolerances')
    original_digest = hashlib.sha256()
    with source.open('rb') as stream:
        read = lambda: _read_fortran_record(stream, maximum_bytes=maximum_memory_bytes, digest=original_digest)
        headers = [read() for _ in range(9)]
        if not headers[0].startswith(b'chiGG0') or len(headers[1]) != 20:
            raise ValueError('Use the inspected BGW4 extended full-frequency header')
        mode, nfrequency, subspace, matrix_subspace, _ = struct.unpack('<5i', headers[1])
        if mode != 2 or nfrequency < 1 or subspace or matrix_subspace:
            raise ValueError('Only non-subspace complex full-frequency eps0mat is supported')
        nq, = struct.unpack('<i', headers[7][:4])
        if nq != 1:
            raise ValueError('Use a single small-q eps0mat')
        ng, = struct.unpack('<i', headers[8][:4])
        if ng < 1 or len(headers[8]) != 4+12*ng:
            raise ValueError('Inconsistent full G-vector header')
        vectors = np.frombuffer(headers[8][4:], '<i4').reshape(ng, 3)
        matches = np.flatnonzero(np.all(vectors == 0, axis=1))
        if len(matches) != 1:
            raise ValueError('The global G-vector set needs one unique G=0 head')
        headers.extend([read() for _ in range(3)])
        ngq, nmatrix = struct.unpack('<2i', headers[9][:8])
        if ngq != ng or not 1 <= nmatrix <= ng or len(headers[9]) != 8+8*ng:
            raise ValueError('Inconsistent sorted matrix G-vector map')
        order = np.frombuffer(headers[9][8:], '<i4').reshape(ng, 2)[:nmatrix, 0]
        positions = np.flatnonzero(order == matches[0]+1)
        if len(positions) != 1:
            raise ValueError('The dielectric matrix needs one unique G=0 head')
        head = int(positions[0])
        if len(headers[10]) != 8*ng or len(headers[11]) != 24:
            raise ValueError('Unexpected kinetic/q record lengths')
        predicted = sum(map(len, headers))+4*nmatrix*nfrequency*16+8*nmatrix**2*16
        if predicted > maximum_memory_bytes:
            raise ValueError('Schur-copy working arrays exceed the declared memory bound')
        offset = stream.tell()
        head_row = np.empty((2, nmatrix, nfrequency), dtype=complex)
        head_column = np.empty_like(head_row)
        samples = sorted(set((0, nfrequency-1)))
        before = np.empty((2, len(samples), nmatrix, nmatrix), dtype=complex)
        for column in range(nmatrix):
            for channel in range(2):
                for row in range(nmatrix):
                    payload = read()
                    if len(payload) != nfrequency*16:
                        raise ValueError('Expected one complex frequency vector per matrix element')
                    values = np.frombuffer(payload, '<c16')
                    if not np.isfinite(values).all():
                        raise ValueError('The original inverse dielectric matrix is nonfinite')
                    if row == head:
                        head_row[channel, column] = values
                    if column == head:
                        head_column[channel, row] = values
                    before[channel, :, row, column] = values[samples]
        if stream.read(1):
            raise ValueError('Unexpected trailing q points or matrix records')
        denominator = head_row[:, head]
        minimum = float(np.min(abs(denominator)))
        if minimum < head_floor:
            raise ValueError(f'The inverse head is singular or below head_floor: {minimum:.3e}')
    source_hash = original_digest.hexdigest()
    after = np.empty_like(before)
    largest_change = 0.
    output_digest = hashlib.sha256()
    with source.open('rb') as original, destination.open('xb') as output:
        def write(payload):
            _record(output, payload)
            marker = struct.pack('<i', len(payload))
            output_digest.update(marker+payload+marker)
        for header in headers:
            write(header)
        original.seek(offset)
        for column in range(nmatrix):
            for channel in range(2):
                for row in range(nmatrix):
                    payload = _read_fortran_record(original, maximum_bytes=maximum_memory_bytes)
                    values = np.frombuffer(payload, '<c16')
                    if row == head or column == head:
                        transformed = np.full(nfrequency, 1. if row == column else 0., dtype=complex)
                    else:
                        transformed = values-head_column[channel, row]*head_row[channel, column]/denominator[channel]
                        largest_change = max(largest_change, float(np.max(abs(transformed-values))))
                    after[channel, :, row, column] = transformed[samples]
                    write(np.asarray(transformed, '<c16').tobytes())
    body = np.flatnonzero(np.arange(nmatrix) != head)
    errors = []
    for channel in range(2):
        for frame in range(len(samples)):
            epsilon = np.linalg.inv(before[channel, frame])
            direct = np.linalg.inv(epsilon[np.ix_(body, body)])
            actual = after[channel, frame][np.ix_(body, body)]
            errors.append(float(np.linalg.norm(actual-direct)/max(np.linalg.norm(direct), np.finfo(float).tiny)))
    maximum_error = max(errors)
    if maximum_error > identity_tolerance or not np.isfinite(maximum_error):
        raise ArithmeticError(f'Written Schur copy failed its independent block inverse identity: {maximum_error:.3e}')
    if _file_sha256(source) != source_hash:
        raise ArithmeticError('The original epsilon file changed during the copy')
    return dict(source=str(source), destination=str(destination), matrix_size=nmatrix,
        frequencies=nfrequency, head_index=head, min_abs_inverse_head=minimum,
        certified_frequency_indices=samples, max_identity_relative_error=maximum_error,
        max_absolute_body_change=largest_change, source_sha256=source_hash,
        destination_sha256=output_digest.hexdigest(), original_modified=False,
        predicted_working_bytes=predicted,
        transformation='S_body -> S_body - S_body,head S_head,body / S_head,head; head=1,wings=0',
        scope='Exact block inverse conversion of the original finite-small-q dielectric; not q0 extrapolation or fitting')


__all__ = ['write_wfn', 'write_vxc', 'export_scf', 'read_sigma', 'read_eqp',
           'copy_eps0_without_head_feedback']
