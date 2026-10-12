"""Accepted-state NNAO checkpoints and separate failed candidates."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np


def save_checkpoint(path, checkpoint, optimizer):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in checkpoint.items()
                if key not in {'parameters', 'gradient', 'optimizer'}}
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, parameters=np.asarray(checkpoint['parameters']),
            gradient=np.asarray(checkpoint['gradient']),
            optimizer=np.frombuffer(optimizer.encode(checkpoint['optimizer']), dtype=np.uint8),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    os.replace(temporary, path)

def load_checkpoint(path, identity, optimizer, initial):
    with np.load(path, allow_pickle=False) as saved:
        checkpoint = json.loads(str(saved['metadata'].item()))
        if checkpoint.get('identity') != identity:
            raise ValueError('Checkpoint identity differs: data, source, cache or configuration changed.')
        checkpoint.update(parameters=np.array(saved['parameters']), gradient=np.array(saved['gradient']),
                          optimizer=optimizer.decode(saved['optimizer'].tobytes(), optimizer.init(initial)))
    if (checkpoint['parameters'].shape != np.asarray(initial).shape
            or checkpoint['gradient'].shape != np.asarray(initial).shape
            or not np.isfinite(checkpoint['parameters']).all()
            or not np.isfinite(checkpoint['gradient']).all()):
        raise ValueError('Checkpoint parameter/gradient shape or values are invalid.')
    return checkpoint

def save_failure_snapshot(path, vector, error, summary):
    """Save the unaccepted trial, following explicit diagnostics through wrappers."""
    arrays = dict(parameters=np.asarray(vector)); metadata = dict(summary)
    cause = error
    while cause is not None:
        if hasattr(cause, 'structure_id'):
            metadata.setdefault('structure_id', cause.structure_id)
        if hasattr(cause, 'failure_arrays'):
            arrays.update(cause.failure_arrays)
            metadata['details'] = cause.failure_details
        cause = cause.__cause__
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays,
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    os.replace(temporary, path)
