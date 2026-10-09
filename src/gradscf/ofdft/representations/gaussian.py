"""Molecular Gaussian inputs, without invoking an HF/KS calculation."""
from dataclasses import replace
import jax.numpy as jnp
import numpy as np
from ... import integrals
from ...data.molecule import nuclear_repulsion_energy
from ..types import OFDFTInputs


def gaussian_inputs(mol, *, grids_level=1, integral_backend='native', coordinates=None):
    """Build all-electron molecular OFDFT inputs; optional coordinates are Bohr.

    The native full ERI geometry path supports first coordinate derivatives.
    Basis exponent/coefficient derivatives require the JAX reference backend.
    Eager native construction uses packed ERIs; traced geometry uses full ERIs
    because the packed native integral call has no geometry AD rule.
    """
    import jax
    if mol.spin != 0 or not mol.cart:
        raise ValueError('Molecular OFDFT currently requires unpolarized Cartesian AOs.')
    spec = mol.to_spec()
    if coordinates is not None:
        spec = replace(spec, coords_bohr=jnp.asarray(coordinates))
    topology, parameters = integrals.prepare_basis(spec,mol.basis)
    plan = integrals.make_plan(topology,backend=integral_backend)
    basis = integrals.basis_from_molecule_spec(spec,basis=mol.basis,precompute_eri_groups=False)
    coords, weights = integrals.build_molecular_grid_from_spec(spec,level=grids_level)
    ao, deriv = integrals.evaluate_cartesian_ao_with_derivatives(basis,coords)
    overlap = plan.evaluate('overlap',parameters)
    kinetic = plan.evaluate('kinetic',parameters)
    external = plan.evaluate('nuclear',parameters)
    traced = isinstance(parameters.nuclear_coords,jax.core.Tracer)
    eri = plan.evaluate('eri',parameters,aosym='s4' if integral_backend=='native' and not traced else 's1')
    nelectron = int(np.asarray(mol.to_spec().charges).sum())-mol.charge
    return OFDFTInputs('gaussian',weights,coords,jnp.asarray(nelectron,dtype=ao.dtype),external,
        nuclear_repulsion_energy(spec),overlap,kinetic,ao,deriv[1:4].transpose(1,2,0),eri)
