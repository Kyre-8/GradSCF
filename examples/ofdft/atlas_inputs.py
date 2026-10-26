"""Reproducible Mg/Al inputs for ATLAS-style TF+lambda*vW and WGC comparisons.

These inputs follow the paper's material classes, not its full unpublished
input decks. Volumes per atom are taken from Table 1 OF rows. Mg c/a
is assumed ideal sqrt(8/3), because that ratio is not specified in the table.
Pseudopotentials are author-supplied OEPP files bundled with DFTpy dev fbc47b4e;
exact identity with the files used in the 2015 paper cannot be established.

The small RECPOT reader is eager preparation at fixed cell/ions. It matches the
CASTEP RECPOT units and interpolating-cubic convention; it is not a general
pseudopotential library or a nuclear/lattice-AD interface.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from ase import Atoms, units


DATA = Path(__file__).with_name('data')
VALENCE = {'Mg':2, 'Al':3}


def crystal(symbol, spacing=.18):
    """Return ASE atoms and an odd mesh with vector spacings <= target Angstrom."""
    if symbol=='Al':
        a=(4*18.435)**(1/3)
        lattice=a/2*np.array([[0,1,1],[1,0,1],[1,1,0]])
        fractional=[[0,0,0]]
    elif symbol=='Mg':
        ratio=np.sqrt(8/3)
        a=(2*22.225/(np.sqrt(3)/2*ratio))**(1/3)
        lattice=a*np.array([[1,0,0],[-.5,np.sqrt(3)/2,0],[0,0,ratio]])
        fractional=[[0,0,0],[2/3,1/3,.5]]
    else:
        raise ValueError('This example defines only fcc Al and hcp Mg.')
    counts=np.ceil(np.linalg.norm(lattice,axis=1)/spacing).astype(int)
    mesh=tuple(int(n if n%2 else n+1) for n in counts)
    return Atoms([symbol]*len(fractional),cell=lattice,scaled_positions=fractional,pbc=True),mesh


def pseudo_path(symbol):
    return DATA/f'{symbol}_lda.oe01.recpot'


def pz_lda_energy_density(rho):
    """Unpolarized Dirac exchange + Perdew-Zunger (1981) correlation, Ha/Bohr^3.

    PZ81 Table I, DOI 10.1103/PhysRevB.23.5048. Its piecewise parameterization
    is used unchanged. The documented 1e-30 floor is inactive in these solids.
    """
    n=jnp.maximum(rho,1e-30)
    rs=(3/(4*jnp.pi*n))**(1/3)
    high=.0311*jnp.log(rs)-.048+.002*rs*jnp.log(rs)-.0116*rs
    low=-.1423/(1+1.0529*jnp.sqrt(rs)+.3334*rs)
    exchange=-.75*(3/jnp.pi)**(1/3)*n**(1/3)
    return n*(exchange+jnp.where(rs<1,high,low))


def pz_lda_energy(params, features):
    del params
    return jnp.sum(features.weights*pz_lda_energy_density(features.rho))


def read_recpot(path):
    lines=Path(path).read_text().splitlines()
    end=next(i for i,line in enumerate(lines) if 'END COMMENT' in line)
    if lines[end+1].split()!=['3','5']:
        raise ValueError('Expected scalar CASTEP RECPOT 3/5 format.')
    maximum=float(lines[end+2])*units.Bohr
    stop=next(i for i in range(end+3,len(lines)) if lines[i].strip()=='1000')
    values=np.fromstring(' '.join(lines[end+3:stop]),sep=' ')/(units.Hartree*units.Bohr**3)
    wavevectors=np.linspace(0,maximum,len(values))
    valence=round((values[0]-values[1])*wavevectors[1]**2/(4*np.pi))
    return wavevectors,values,valence


def gaussian_free_inputs(atoms, mesh):
    """Independent GradSCF local-potential and Ewald assembly from RECPOT."""
    from gradscf import ofdft
    from gradscf.integrals.periodic.ao import reciprocal_grid
    from gradscf.integrals.periodic.coulomb import ewald_energy
    symbol=atoms.get_chemical_symbols()[0]
    lattice=np.asarray(atoms.cell)/units.Bohr
    coordinates=atoms.positions/units.Bohr
    charge=VALENCE[symbol]
    data=ofdft.periodic_inputs(lattice,mesh,nelectron=len(atoms)*charge)
    vectors=np.asarray(reciprocal_grid(data.lattice,mesh))
    q=np.linalg.norm(vectors,axis=1)
    grid,values,read_charge=read_recpot(pseudo_path(symbol))
    if read_charge!=charge or q.max()>=grid[-1]:
        raise ValueError('RECPOT valence/range does not cover this calculation.')
    # An interpolating not-a-knot cubic on all nonzero tabulated wavevectors.
    radial=CubicSpline(grid[1:],values[1:])(q)
    low=q<grid[1]
    dx=q[low]/grid[1]
    # Even quartic through (-2,-1,0,1,2); q=0 retains the finite core correction.
    c2=(-2*values[2]+32*values[1]-30*values[0])/24
    c4=(2*values[2]-8*values[1]+6*values[0])/24
    radial[low]=values[0]+c2*dx**2+c4*dx**4
    structure=np.exp(-1j*vectors@coordinates.T).sum(axis=1)
    vg=radial*structure
    potential=np.fft.ifftn((vg*np.prod(mesh)/np.linalg.det(lattice)).reshape(mesh)).real.reshape(-1)
    ionic=ewald_energy(jnp.asarray(lattice),jnp.asarray(coordinates),jnp.full(len(atoms),charge),
                       precision=1e-12,reference_lattice=lattice)
    return replace(data,external_potential=jnp.asarray(potential),nuclear_repulsion=ionic)


def provenance(symbol):
    path=pseudo_path(symbol)
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_commit':'fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0',
            'source_path':f'examples/DATA/{path.name}'}
