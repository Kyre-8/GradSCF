"""A neutral H2 chain or layer with genuine reduced-dimensional electrostatics.

The first lattice vector is periodic in a wire; the first two are periodic
in a slab. Vacuum directions have one unshifted k point. These finite models
illustrate SCF and band evaluation with the same Coulomb boundary throughout.
"""
from pathlib import Path
import json

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
from gradscf.pbc import dft, gto


def run(dimension=1):
    if dimension == 1:
        lattice = np.diag([5., 16., 16.])
        positions = [[.4, 8., 8.], [1.8, 8., 8.]]
        mesh = (25, 81, 81)
        query = [[.13, 0., 0.], [.31, 0., 0.]]
    elif dimension == 2:
        lattice = np.diag([5., 6., 18.])
        positions = [[.4, .5, 8.7], [1.8, .7, 9.2]]
        mesh = (25, 31, 91)
        query = [[.13, -.04, 0.], [.31, .07, 0.]]
    else:
        raise ValueError('Choose a one-dimensional wire or two-dimensional slab.')

    cell = gto.M(atom=list(zip(['H', 'H'], positions)), a=lattice,
        unit='Bohr', dimension=dimension, basis='gth-szv', pseudo='gth-pade',
        mesh=mesh, cart=True, precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='svwn',
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9).run()
    assert mf.converged

    # The AO overlap metric is required when counting electrons from a DM.
    count = np.einsum('k,kpq,kqp->', np.asarray(mf.inputs.kweights),
        np.asarray(mf.make_rdm1()), np.asarray(mf.inputs.overlap)).real
    np.testing.assert_allclose(count, cell.nelectron, atol=2e-10)
    bands, _ = mf.get_bands(np.asarray(query))  # absolute k vectors, bohr^-1
    assert np.isfinite(bands).all()
    report = dict(dimension=dimension, energy_hartree=float(mf.e_tot),
        electron_number=float(count), mesh=mesh, lattice_bohr=lattice.tolist(),
        query_kpoints_bohr_inverse=query, bands_hartree=np.asarray(bands).tolist(),
        scope='Neutral finite wire/slab model; no material convergence claim.')
    folder = Path('reproducibility/ad_showcase/PBC_reduced_dimensions')
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f'dimension_{dimension}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{dimension}D energy / Eh:', report['energy_hartree'])
    print('Electron number:', report['electron_number'])
    print('Off-mesh bands / Eh:\n', np.asarray(bands))
    return report


if __name__ == '__main__':
    run(1)
    run(2)

# Measured on jy_gpu_1, CPU float64, 2026-10-16:
# 1D energy / Eh: -1.103487497321063
# Electron number: 2.0
# Off-mesh bands / Eh:
# [[-0.44025968  0.38293740]
#  [-0.41646368  0.22125627]]
# 2D energy / Eh: -1.1304553395985697
# Electron number: 2.0
# Off-mesh bands / Eh:
# [[-0.44791400  0.28469041]
#  [-0.42173826  0.16423652]]
