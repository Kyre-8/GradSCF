"""WGC99 Al/Mg: independent optimizations and a finer-grid check.

DFTpy uses wgc_dftpy.py, an independent NumPy energy/potential adapter; its
native dev KEDF collection does not contain WGC. See WGC_REPRODUCTION.md.
"""
from pathlib import Path
from atlas_compare import compare


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    compare(spacings={'Al': (.18, .14), 'Mg': (.18, .14)}, kinetic='wgc',
            output=root / 'atlas_wgc_results.json',
            export_volumes=root / 'atlas_wgc_density_volumes.npz')
