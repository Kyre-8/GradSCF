"""Separate finite-basis KS comparisons; these do not validate OFDFT code AD.

The k-mesh sequence quantifies refinement increments, not a converged KS error
bar. Same local OEPP, PZ-LDA, cell and 17^3 quadrature as the Al WGC example.
Edit KMESH below, run on jy_gpu_1 CPU, then large_plot.py reads the saved records.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource

import numpy as np
from pyscf import lib

try:
    from . import atlas_ks_pyscf
except ImportError:
    import atlas_ks_pyscf


KMESH = 4
OUTPUT = Path('reproducibility/ad_showcase/P08_ofdft_parameter_response/large_system_scaling')


def calculate(kmesh=KMESH):
    lib.num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    info, density = atlas_ks_pyscf.calculate('Al', spacing=.18, kmesh=(kmesh,)*3,
        basis_level=3, sigma=.005, max_cycle=100)
    info.update(hostname=platform.node(),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=hashlib.sha256(Path(atlas_ks_pyscf.__file__).read_bytes()).hexdigest(),
        interpretation='Finite-basis, finite-k, finite-smearing KS; WGC difference is not pure KEDF error')
    np.savez_compressed(OUTPUT/f'al_ks_k{kmesh}.npz', density=density)
    (OUTPUT/f'al_ks_k{kmesh}.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2), flush=True)
    return info


if __name__ == '__main__':
    calculate()
