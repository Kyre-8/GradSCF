"""Molecular response for the existing external dft.Functional contract."""
from dataclasses import replace
import jax
import jax.numpy as jnp
from ..dft.functional import Functional
from ..dft.derivatives import xc_energy_and_potential_from_density
from ..scf.reference import as_reference
from ..scf.differentiable import (
    _coulomb_matrix, _df_factors_from_molecule, _unrestricted_channel,
    _spin_resolved_density_matrix, _repulsion_integrals_from_molecule, _scf_problem_arrays,
)
from ..scf.energy import XCContribution, unrestricted_energy, unrestricted_fock
from ..integrals.molecular.jk import build_j_from_df
from .ks import ks_problem


def functional_problem(source, functional, params, *, converged=True, tolerance=1e-7):
    """Use a supplied self-consistent state without detaching its parameter AD.

    Restricted references remain in the equal-spin subspace. Unrestricted
    references retain independent alpha/beta response. density is the spin sum;
    values['density_spin'] exposes both channels. No SCF is run here.
    """
    if not isinstance(functional, Functional):
        raise TypeError('External DFPT requires a dft.Functional energy model')
    molecule = as_reference(source)
    unrestricted = getattr(molecule, 'nocc_alpha', None) is not None
    nocc = ((molecule.nocc_alpha, molecule.nocc_beta) if unrestricted
            else (getattr(molecule, 'nocc', None),))
    if any(no is None for no in nocc):
        raise ValueError('External DFPT requires static integer occupation counts')
    c, occ, _ = _unrestricted_channel(molecule)
    if jnp.iscomplexobj(c):
        raise ValueError('dft.Functional requires real molecular orbitals')
    expected = jnp.stack([(jnp.arange(c.shape[-1]) < no).astype(occ.dtype)
                          for no in (nocc if unrestricted else nocc*2)])
    valid = converged & jnp.all(jnp.abs(occ-expected) <= tolerance)
    reference_density = jnp.einsum('spi,si,sqi->spq', c, expected, c)
    valid &= jnp.max(jnp.abs(reference_density-_spin_resolved_density_matrix(molecule))) <= tolerance
    if not unrestricted:
        valid &= jnp.max(jnp.abs(reference_density[0]-reference_density[1])) <= tolerance
    spin_density = lambda d: d if unrestricted else jnp.stack([d[0]/2]*2)

    def evaluate(d, args):
        p, mol = args
        ds = spin_density(d)
        xc = xc_energy_and_potential_from_density(
            p, molecule=mol, density=ds, xc_energy_fn=functional.energy_for_density)
        factors = _df_factors_from_molecule(mol)
        total = ds.sum(axis=0)
        coulomb = (_coulomb_matrix(_repulsion_integrals_from_molecule(mol), total) if factors is None
                   else build_j_from_df(factors, total))
        contribution = XCContribution(xc.xc_energy, xc.vxc_matrix, 0.)
        exchange = jnp.zeros_like(ds)
        energy = unrestricted_energy(ds, mol.h1e, coulomb, exchange, contribution,
                                      nuclear_repulsion=mol.nuclear_repulsion)
        fock = unrestricted_fock(mol.h1e, coulomb, exchange, contribution)
        return energy, fock if unrestricted else fock.mean(axis=0, keepdims=True)

    args = (params, molecule)
    problem = ks_problem(c if unrestricted else c[:1], nocc, args,
        lambda d,a: evaluate(d,a)[1], lambda d,a: evaluate(d,a)[0],
        overlap=lambda a: jnp.broadcast_to(_scf_problem_arrays(a[1])[-1],
                                           (len(nocc), c.shape[1], c.shape[1])),
        occupation=1. if unrestricted else 2., converged=valid, tolerance=tolerance)
    observe = problem.quantities
    def quantities(x, a):
        values = observe(x, a)
        ds = spin_density(values['density'])
        return dict(values, density=ds.sum(axis=0), density_spin=ds,
                    electron_number=values['electron_number'].sum())

    def potential_tangent(value):
        value = jnp.asarray(value)
        if value.shape != molecule.h1e.shape or jnp.iscomplexobj(value):
            raise ValueError('External molecular potential must be a real AO matrix')
        value = value.astype(molecule.h1e.dtype)
        value *= jnp.where(jnp.max(jnp.abs(value-value.T)) <= 1e-10, 1., jnp.nan)
        zero = jax.tree.map(jnp.zeros_like, args)
        return zero[0], replace(zero[1], h1e=value)

    return replace(problem, quantities=quantities, potential_tangent=potential_tangent,
        potential_path=lambda tangent,t: (params, replace(molecule, h1e=molecule.h1e+t*tangent[1].h1e)))
