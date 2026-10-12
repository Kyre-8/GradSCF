"""NNAO data, source provenance and explicit workflow runtime setup."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import socket

PACKAGE = Path(__file__).resolve().parent

def repository_root():
    for parent in PACKAGE.parents:
        if (parent/'src/gradscf').is_dir() and (parent/'pyproject.toml').is_file():
            return parent
    raise RuntimeError('This provenance-checked workflow requires a GradSCF source checkout.')

def vendor_directory():
    return PACKAGE/'vendor/mace-jax'

def verify_mace_source():
    import mace_jax
    expected = vendor_directory().resolve()
    if not Path(mace_jax.__file__).resolve().is_relative_to(expected):
        raise RuntimeError(f'MACE source mismatch: {mace_jax.__file__}; expected {expected}. '
                           'Install the model-local vendored project or add it to PYTHONPATH.')

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def scientific_source_hash(root):
    digest = hashlib.sha256()
    package = Path(root) / 'src' / 'gradscf'
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.json', '.dat'}:
            digest.update(str(path.relative_to(package)).encode())
            digest.update(file_hash(path).encode())
    return digest.hexdigest()

def verify_source_imports(root, *, importer=None):
    """Reject installed or setup copies even if they report the same version."""
    if importer is None:
        from importlib import import_module
        importer = import_module
    package = Path(root).resolve() / 'src' / 'gradscf'
    sources = {'gradscf': '__init__.py', 'gradscf.scf.rks': 'scf/rks.py',
               'gradscf.solvers.nonlinear.fixed_point': 'solvers/nonlinear/fixed_point.py'}
    confirmed = {}
    for name, relative in sources.items():
        imported = importer(name)
        actual = Path(imported.__file__).resolve()
        expected = (package / relative).resolve()
        if actual != expected:
            raise RuntimeError(f'Source mismatch for {name}: imported {actual}; '
                               f'expected {expected}. Set PYTHONPATH after activating the environment.')
        confirmed[name] = dict(path=str(actual), sha256=file_hash(actual))
    return confirmed

def atomic_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)

def read_structures(path, *, ids=None, limit=None):
    """Keep source order and fail explicitly on malformed/out-of-scope input."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    names = [row['structure_id'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate structure IDs in input.')
    if ids is not None:
        unknown = sorted(set(ids) - set(names))
        if unknown:
            raise ValueError(f'Unknown structure IDs: {unknown}')
        rows = [row for row in rows if row['structure_id'] in set(ids)]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive.')
        rows = rows[:limit]
    if not rows:
        raise ValueError('No structures selected.')
    for row in rows:
        if row.get('charge') != 0 or row.get('spin') != 0:
            raise ValueError(f"{row['structure_id']}: pilot requires neutral spin=0 structures.")
        symbols = row['symbols']; coordinates = row['coords_angstrom']
        if not symbols or any(symbol not in {'H', 'C', 'N', 'O', 'F'} for symbol in symbols):
            raise ValueError(f"{row['structure_id']}: pilot supports CHNOF only.")
        if len(coordinates) != len(symbols) or any(
            len(point) != 3 or not all(math.isfinite(float(x)) for x in point)
            for point in coordinates
        ):
            raise ValueError(f"{row['structure_id']}: invalid coordinates.")
        row['name'] = row['structure_id']
    return rows

def record_directory(root, identifier):
    # Hash prevents traversal and collisions caused by replacing '/' with '_'.
    return Path(root) / hashlib.sha256(identifier.encode()).hexdigest()[:16]

def numerical_runtime(platform):
    # Set before importing the existing experiment, whose default is CPU.
    os.environ['JAX_PLATFORMS'] = platform
    os.environ['JAX_ENABLE_X64'] = '1'
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platforms', platform)
    verify_source_imports(repository_root())
    from .training import objective as module
    expected = 'gpu' if platform == 'cuda' else 'cpu'
    if any(device.platform != expected for device in jax.devices()):
        raise RuntimeError(f'Expected {expected}, got {jax.devices()}')
    return jax, module.BasisRHF

def experiment_metadata(experiment, jax):
    return dict(method='RHF', basis='szp663_direct', core_primitives=6,
        jk_backend='df', auxbasis=experiment.auxbasis, dtype=str(experiment.ps.dtype),
        jax_version=jax.__version__, devices=[str(device) for device in jax.devices()],
        backend=jax.default_backend(), device_kinds=[device.device_kind for device in jax.devices()],
        hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        imported_sources=verify_source_imports(repository_root()),
        nao=int(experiment.layout.topology.nao),
        primitive_nao=int(experiment.primitive_topology.nao),
        nelectron=int(experiment.nelectron), df_shape=list(experiment.rep.shape),
        cache_storage=experiment.cache_storage,
        df_bytes=int(experiment.rep.size * experiment.rep.dtype.itemsize),
        integral_signature=experiment.integral_signature,
        gradient='converged RHF density fixed point, implicit backward',
        scf_config=dict(max_cycle=150, conv_tol=1e-12,
                        conv_tol_density=1e-10, conv_tol_grad=1e-9),
        implicit_config=dict(tolerance=experiment.implicit_tolerance, max_iter=100, restart=40),
        scf_rescue_level_shift=experiment.scf_rescue_level_shift,
        implicit_diagnostics='Checked GMRES residual failures produce NaN; finite gradients required. '
            'The existing API does not expose the numerical adjoint residual.')

def cache_location(cache_dir, row):
    return record_directory(cache_dir, row['structure_id']) / 'integrals.npz'
