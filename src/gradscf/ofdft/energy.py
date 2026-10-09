"""One differentiable energy for all OFDFT representations and KEDF callbacks."""
import jax.numpy as jnp
from ..integrals.periodic.ao import reciprocal_grid
from ..integrals import build_jk_from_packed, build_j_from_eri_pair_matrix
from ..df import build_j_from_df
from ..dft.libxc_jax.jax_libxc import (
    restricted_feature_bundle_from_rho_grad_tau, eval_xc_energy_density,
)
from .types import KineticFeatures, EnergyComponents
from .representations.periodic import spectral_gradient, hartree_potential


def amplitude(q, data):
    return q if data.representation=='periodic' else data.ao@q


def density_features(q, data):
    phi = amplitude(q,data)
    if data.representation=='periodic':
        grad_phi = spectral_gradient(phi,data.lattice,data.mesh)
        vw = .5*jnp.sum(data.weights*jnp.sum(grad_phi**2,axis=-1))
    else:
        grad_phi = jnp.einsum('gpi,p->gi',data.ao_gradient,q)
        vw = q@data.kinetic_matrix@q
    return KineticFeatures(phi**2,2*phi[:,None]*grad_phi,data.weights,data.coordinates,vw,phi,
        reciprocal_grid(data.lattice,data.mesh) if data.mesh else None,
        jnp.linalg.det(data.lattice) if data.mesh else None,data.mesh)


def energy_components(q, data, *, kinetic, kinetic_params, xc=None, xc_energy_fn=None, xc_params=None):
    features = density_features(q,data)
    rho = features.rho
    if data.representation=='gaussian':
        density = jnp.outer(q,q)  # density-amplitude product, not a KS occupation matrix
        external = jnp.sum(density*data.external_potential)
        if data.df_factors is not None:
            coulomb = build_j_from_df(data.df_factors,density)
        elif data.eri.ndim==4:
            coulomb = jnp.einsum('pqrs,rs->pq',data.eri,density)
        elif data.eri.ndim==2:
            coulomb = build_j_from_eri_pair_matrix(data.eri,density)
        else:
            coulomb, _ = build_jk_from_packed(data.eri,density)
        hartree = .5*jnp.sum(density*coulomb)
    else:
        external = jnp.sum(data.weights*rho*data.external_potential)
        hartree = .5*jnp.sum(data.weights*rho*hartree_potential(rho,data))
    if xc_energy_fn is not None:
        exc = xc_energy_fn(xc_params,features)
    elif xc is None:
        exc = jnp.asarray(0.,dtype=rho.dtype)
    else:
        bundle = restricted_feature_bundle_from_rho_grad_tau(rho,features.grad_rho)
        exc = jnp.sum(data.weights*eval_xc_energy_density(xc,bundle))
    return EnergyComponents(kinetic(kinetic_params,features),external,hartree,exc,data.nuclear_repulsion)
