"""Shared physical inputs for the graphite examples; edit settings here.

The lattice is a fixed, representative AB graphite geometry, not a relaxed
PBE prediction or a geometry extracted from the experimental specimen.
This inexpensive mesh is a pilot. Convergence is tested separately.
"""
from pathlib import Path
import numpy as np
from scipy.constants import physical_constants
from gradscf.pbc import gto, dft

OUTPUT = Path('graphite_results')
HARTREE_TO_MEV = physical_constants['Hartree energy in eV'][0]*1000.
KB_HARTREE = physical_constants['Boltzmann constant in eV/K'][0]/physical_constants['Hartree energy in eV'][0]
MASSES = np.full(4, 12.011)  # atomic mass units

# Hexagonal lattice with a 120-degree angle between the in-plane rows.
A_ANGSTROM = 2.46
C_ANGSTROM = 6.708
BASIS = 'gth-szv'
FFT_MESH = (15, 15, 31)
K_MESH = (3, 3, 1)
ELECTRON_TEMPERATURES = (300., 800., 1377.)  # kelvin, independently of Bose T
SCF_CONTROLS = dict(max_cycle=200, damping=.25, conv_tol=1e-12,
                    conv_tol_density=1e-11, conv_tol_grad=1e-10)


def calculation_settings():
    """JSON-compatible settings stored with every artifact to prevent mixing runs."""
    return dict(a_angstrom=A_ANGSTROM, c_angstrom=C_ANGSTROM, basis=BASIS,
                pseudo='gth-pbe', xc='pbe', fft_mesh=list(FFT_MESH),
                k_mesh=list(K_MESH), scf_controls=SCF_CONTROLS,
                geometry='fixed AB graphite, not relaxed', exxdiv=None)


def make_cell(*, basis=BASIS, mesh=FFT_MESH):
    lattice = np.array([[A_ANGSTROM, 0., 0.],
                        [-A_ANGSTROM/2, np.sqrt(3)*A_ANGSTROM/2, 0.],
                        [0., 0., C_ANGSTROM]])
    fractional_positions = np.array([[0., 0., 0.], [1/3, 2/3, 0.],
                                     [0., 0., .5], [2/3, 1/3, .5]])
    return gto.M(atom=[('C', r) for r in fractional_positions @ lattice],
                 a=lattice, unit='Angstrom', basis=basis, pseudo='gth-pbe',
                 mesh=mesh, precision=1e-10)


def mean_field(cell, temperature, *, kmesh=K_MESH, xc='pbe'):
    """Construct a finite-electron-temperature calculation with shared controls."""
    return dft.KRKS(cell, kpts=cell.make_kpts(kmesh), xc=xc,
                    temperature=temperature*KB_HARTREE, exxdiv=None,
                    **SCF_CONTROLS)


def optical_directions():
    """Four real in-plane, intralayer-opposite displacements at Gamma.

    Each row has unit Euclidean norm. Both relative layer signs and both
    Cartesian polarizations are retained, without assigning Raman/IR labels
    before checking eigenvectors and finite-grid symmetry.
    """
    rows = []
    for layer_sign in (1., -1.):
        for axis in (0, 1):
            row = np.zeros((4, 3))
            row[:, axis] = np.array([1., -1., layer_sign, -layer_sign])/2
            rows.append(row)
    return np.asarray(rows)


def inplane_directions():
    """The complete eight-dimensional in-plane space, including shear modes."""
    return np.eye(12).reshape(12, 4, 3)[[0, 1, 3, 4, 6, 7, 9, 10]]
