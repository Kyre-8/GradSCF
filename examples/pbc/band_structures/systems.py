"""Explicit structures and paths shared by native and independent examples.

The lattice parameters are fixed example inputs, not fitted or reoptimized
separately for each functional. ASE supplies geometry/path bookkeeping only.
"""
from ase.build import bulk, graphene


BASIS = 'gth-dzvp-molopt-sr'
PSEUDO = 'gth-pbe'


def structure(name, *, vacuum=20.):
    if name == 'silicon':
        atoms = bulk('Si', 'diamond', a=5.431)
        path, dimension = 'GXWKGL', 3
    elif name == 'gaas':
        atoms = bulk('GaAs', 'zincblende', a=5.6533)
        path, dimension = 'GXWKGL', 3
    elif name == 'gan':
        atoms = bulk('GaN', 'wurtzite', a=3.189, c=5.185, u=.377)
        path, dimension = 'GMKGALHA', 3
    elif name == 'graphene':
        # ASE's vacuum is per side. This pilot has total cell height `vacuum`.
        atoms = graphene(a=2.46, vacuum=vacuum/2)
        path, dimension = 'GMKG', 2
    else:
        raise ValueError('Choose silicon, gaas, gan or graphene.')
    return atoms, path, dimension


def settings(name):
    """Finite-model example settings; material convergence is a separate study."""
    mesh = {'silicon': (31, 31, 31), 'gaas': (41, 41, 41),
            'gan': (41, 41, 65), 'graphene': (25, 25, 161)}[name]
    return dict(basis=BASIS, pseudo=PSEUDO, mesh=mesh,
        kmesh=(3, 3, 1) if name == 'graphene' else (2, 2, 2),
        temperature=2e-4 if name == 'graphene' else None,
        vacuum=20., path_points=121, chunk_size=2)
