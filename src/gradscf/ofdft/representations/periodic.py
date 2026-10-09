"""Neutral 3D periodic grid and Gaussian density-amplitude representations.

Explicit grid potentials use a G=0 Hartree background convention. Callers own
ionic energy and the finite local-potential zero mode. Cell adapters accept only
local GTH potentials; nonlocal projector channels are rejected, never discarded.
"""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from ...integrals.periodic.ao import reciprocal_grid, bloch_grid
from ...integrals.periodic.coulomb import coulomb_kernel, ewald_energy
from ...integrals.periodic.pseudo import local_potential
from ..types import OFDFTInputs


def periodic_inputs(lattice, mesh, *, nelectron, external_potential=None, nuclear_repulsion=0.):
    mesh = tuple(mesh)
    if len(mesh) != 3 or any(not isinstance(n, (int, np.integer)) or n < 3 or n % 2 == 0 for n in mesh):
        raise ValueError('Use three odd FFT mesh sizes >=3 (symmetric +/-G pairs).')
    lattice = jnp.asarray(lattice)
    if jnp.iscomplexobj(lattice):
        raise ValueError('lattice must be real.')
    lattice = lattice.astype(jnp.float64)
    if lattice.shape != (3,3):
        raise ValueError('lattice must have shape (3,3), in Bohr.')
    if not isinstance(lattice, jax.core.Tracer) and (not np.isfinite(lattice).all() or np.linalg.det(lattice) <= 0):
        raise ValueError('lattice must be finite and right handed.')
    volume = jnp.linalg.det(lattice)
    size = int(np.prod(mesh))
    fractions = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), axis=-1).reshape(-1,3)
    potential = jnp.zeros(size) if external_potential is None else jnp.asarray(external_potential)
    if potential.size != size:
        raise ValueError('Local potential must contain one real value per grid point.')
    return OFDFTInputs('periodic', jnp.full(size,volume/size), jnp.asarray(fractions)@lattice,
        jnp.asarray(nelectron,dtype=lattice.dtype), potential.reshape(-1),
        nuclear_repulsion=jnp.asarray(nuclear_repulsion), lattice=lattice, mesh=mesh)


def spectral_gradient(values, lattice, mesh):
    g = reciprocal_grid(lattice, mesh)
    transformed = jnp.fft.fftn(values.reshape(mesh)).reshape(-1)
    return jnp.stack([jnp.fft.ifftn((1j*g[:,i]*transformed).reshape(mesh)).real.reshape(-1)
                      for i in range(3)], axis=-1)


def hartree_potential(rho, data):
    kernel = coulomb_kernel(reciprocal_grid(data.lattice,data.mesh)).reshape(data.mesh)
    return jnp.fft.ifftn(jnp.fft.fftn(rho.reshape(data.mesh))*kernel).real.reshape(-1)


def periodic_gaussian_inputs(data, topology, parameters):
    """Project a periodic grid problem onto a Gamma Gaussian amplitude space.

    Uses the same quadrature/FFT operators as the grid problem, without an
    AO-pair potential cache. Increase mesh until product-density aliasing is small.
    """
    ao, _, _, _ = bloch_grid(topology, parameters, data.lattice, data.mesh, jnp.zeros((1,3)))
    values, gradients = ao[0,0].real, ao[0,1:4].real.transpose(1,2,0)
    overlap = values.T@(data.weights[:,None]*values)
    kinetic = .5*jnp.einsum('gpi,g,gqi->pq',gradients,data.weights,gradients)
    return replace(data, representation='periodic_gaussian', ao=values, ao_gradient=gradients,
                   overlap=overlap, kinetic_matrix=kinetic)


def inputs_from_cell(cell, *, representation='periodic', parameters=None, lattice=None):
    if representation not in ('periodic','periodic_gaussian'):
        raise ValueError('Cell representation must be periodic or periodic_gaussian.')
    if not hasattr(cell,'topology'):
        cell = cell.build()
    if cell.spin != 0 or cell.charge != 0 or cell.dimension != 3:
        raise ValueError('OFDFT cell adapter requires a neutral, unpolarized 3D cell.')
    for pp in cell.pseudopotentials:
        if any(channel[1] for channel in pp[5:]):
            raise ValueError('OFDFT requires a local pseudopotential; nonlocal GTH projectors are unsupported.')
    parameters = cell.parameters if parameters is None else parameters
    lattice = cell.lattice if lattice is None else lattice
    data = periodic_inputs(lattice,cell.mesh,nelectron=cell.nelectron)
    gv = reciprocal_grid(data.lattice,data.mesh)
    vg = local_potential(gv,parameters.nuclear_coords,cell.pseudopotentials)
    potential = jnp.fft.ifftn((vg*int(np.prod(cell.mesh))/jnp.linalg.det(data.lattice)).reshape(cell.mesh)).real.reshape(-1)
    enuc = ewald_energy(data.lattice,parameters.nuclear_coords,cell.charges,
                       precision=cell.precision,reference_lattice=cell.lattice)
    data = replace(data,external_potential=potential,nuclear_repulsion=enuc)
    return periodic_gaussian_inputs(data,cell.topology,parameters) if representation=='periodic_gaussian' else data
