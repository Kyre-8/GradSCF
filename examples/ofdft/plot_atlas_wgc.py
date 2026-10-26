"""Plot WGC versus independent Gaussian KS, using the shared contour renderer.

Reads the recorded densities only; does not rerun an electronic calculation.
"""
from pathlib import Path
import numpy as np
from plot_atlas_contours import plot_contours


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    with np.load(root / 'atlas_wgc_density_volumes.npz') as saved:
        volumes = dict(saved)
    with np.load(root / 'atlas_ks_pyscf_densities.npz') as saved:
        for symbol in ('Al', 'Mg'):
            volumes[f'{symbol}_wgc_ks'] = saved[f'{symbol}_rho']
    plot_contours(volumes, None, 'wgc', reference='ks',
                  labels=('GradSCF WGC99', 'PySCF KS (Gaussian basis)'),
                  subtitle='WGC99 (second-order, γ = 2.7) versus KS  ·  OEPP  ·  PZ-LDA',
                  note='KS: Al 10³ / Mg 8³ k points, σ = 0.005 Ha. Finite-basis reference; not original CASTEP data.',
                  filename='atlas_density_contours_wgc_ks',
                  contour_levels={'Al': np.arange(.8, 3.61, .2),
                                  'Mg': np.arange(1., 1.51, .05)},
                  label_positions=[[(.52,.52),(.13,.46),(.71,.69),(.27,.83),(.9,.92)],
                                   [(.2,.55),(.6,.5),(.37,.78),(.76,.2),(.07,.92)],
                                   [(.18,.12),(.26,.4),(.55,.85),(.88,.6),(.9,.15)],
                                   [(.14,.16),(.33,.46),(.51,.76),(.86,.87),(.8,.33)]])
    print('Saved WGC–KS contours (PNG and vector PDF).')
