"""Independent NumPy Fourier/full-chi/Sigma check; shared GradSCF orbitals.

The dense oracle evaluates actual Dyson residuals at the native QP energies.
It does not import a reduced screening cache or claim independent SCF roots.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from periodic_bands import OUTPUT, reference
from gradscf_tools.periodic_gw_reference import evaluate_query


def run(output=OUTPUT):
    output = Path(output)
    started = perf_counter()
    native = np.load(output/'bands.npz')
    indices = [0, 1, 3]
    mf = reference()
    checked = evaluate_query(mf, native['kpoints'][indices], [0, 1],
        native['qp_energy'][indices], nw=16, eta=1e-3)
    sigma_error = float(np.max(np.abs(checked['sigma_c']-native['sigma_c'][indices])))
    residual_error = float(np.max(np.abs(checked['residual']-native['qp_residual'][indices])))
    passed = sigma_error < 2e-10 and residual_error < 2e-10
    np.savez(output/'numpy_reference.npz', **{key: value for key, value in checked.items()
             if key != 'scope'}, checked_indices=indices)
    sources = [__file__, 'src/gradscf_tools/periodic_gw_reference.py']
    report = dict(complete=True, passed=passed, scope=checked['scope'], checked_indices=indices,
        sigma_max_error_hartree=sigma_error, residual_max_error_hartree=residual_error,
        max_independent_residual_hartree=float(np.max(np.abs(checked['residual']))),
        max_two_sided_difference_hartree=checked['max_two_sided_difference'],
        elapsed_seconds=perf_counter()-started, tolerance_hartree=2e-10,
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (output/'numpy_reference.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not passed:
        raise AssertionError('Independent Fourier/chi/W/Sigma check failed')
    return report


if __name__ == '__main__':
    run()
