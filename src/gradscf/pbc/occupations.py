"""Fermi-Dirac occupations at fixed spin populations; temperature is kBT in Eh."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from ..solvers.nonlinear import bisect, attach_root
from ..solvers.eigen.spectral import matrix_function


class Occupations(NamedTuple):
    occupations: object
    chemical_potential: object
    entropy: object  # S/kB per cell, dimensionless
    converged: object


@jax.custom_jvp
def _fermi(z):
    return jax.nn.sigmoid(z)


@_fermi.defjvp
def _fermi_jvp(primals,tangents):
    (z,),(dz,)=primals,tangents
    value=_fermi(z)
    return value,value*jax.nn.sigmoid(-z)*dz


def fermi_occupations(energies, nelec, temperature, kweights):
    """Per-spin occupations (spin,k,band), with implicit chemical-potential AD.

    Weights must be positive and normalized. Empty/full channels are exact and
    report -/+infinite mu; their density and entropy derivatives remain finite.
    Invalid inputs or unresolved number equations return NaNs and False.
    """
    e,number,t,w = map(jnp.asarray, (energies,nelec,temperature,kweights))
    if e.ndim != 3 or number.shape != (e.shape[0],) or w.shape != (e.shape[1],) or t.shape != ():
        raise ValueError('Expected energies (spin,k,band), spin counts, scalar temperature, and k weights')
    n = e.shape[-1]
    valid = (jnp.all(jnp.isfinite(e)) & jnp.isfinite(t) & (t>0)
             & jnp.all(jnp.isfinite(w)) & jnp.all(w>0) & (jnp.abs(w.sum()-1)<1e-12)
             & jnp.all(jnp.isfinite(number)) & jnp.all((number>=0)&(number<=n)))
    active = (number>0)&(number<n)
    target = jnp.where(active,number,n/2)
    safe_t = jnp.where(t>0,t,1.)
    def residual(mu):
        z=(mu[:,None,None]-e)/safe_t
        filled=z>0
        # Subtract the integer reference before adding exponentially small
        # electron/hole tails. Direct sum(f)-N loses the gap chemical potential.
        tails=jnp.where(filled,-_fermi(-z),_fermi(z))
        reference=jnp.einsum('k,skn->s',w,filled.astype(e.dtype))-target
        return reference+jnp.einsum('k,skn->s',w,tails)
    margin = jnp.maximum(1.,64*safe_t)
    mu,ok = bisect(residual, e.min(axis=(1,2))-margin, e.max(axis=(1,2))+margin)
    slope=jax.jvp(residual,(mu,),(jnp.ones_like(mu),))[1]
    valid &= jnp.all(ok & (~active | (slope>jnp.finfo(e.dtype).tiny)))
    mu = attach_root(residual,mu,converged=valid,
        tangent_solve=lambda op,rhs:rhs/jnp.where(active,op(jnp.ones_like(rhs)),1.))
    z = (mu[:,None,None]-e)/safe_t
    f = _fermi(z)
    # log-sigmoid avoids 0*log(0) for saturated tails.
    entropy = -f*jax.nn.log_sigmoid(z)-_fermi(-z)*jax.nn.log_sigmoid(-z)
    f = jnp.where(active[:,None,None],f,jnp.where(number[:,None,None]==0,0.,1.))
    entropy = jnp.where(active[:,None,None],entropy,0.)
    mu = jnp.where(active,mu,jnp.where(number==0,-jnp.inf,jnp.inf))
    mask = jnp.where(valid,1.,jnp.nan)
    return Occupations(f*mask,mu*mask,jnp.einsum('k,skn->',w,entropy)*mask,valid)


def fermi_density(fock, overlap, nelec, temperature, kweights):
    """Generalized Hermitian Fermi matrix; first density AD is safe at degeneracy."""
    x = jnp.linalg.inv(jnp.linalg.cholesky(overlap)).conj().swapaxes(-1,-2)
    orth = x.conj().swapaxes(-1,-2)@fock@x
    result = fermi_occupations(jnp.linalg.eigvalsh(orth),nelec,temperature,kweights)
    n = fock.shape[-1]
    def spin(a,mu,count):
        mu = jnp.where(jnp.isfinite(mu),mu,0.)
        p = jnp.stack([mu,jnp.asarray(temperature)])
        matrix = jax.vmap(lambda h:matrix_function(h,p,lambda e,p:_fermi((p[0]-e)/p[1]),
                                                   divided_difference=_fermi_divided_difference))(a)
        return jnp.where(count==0,0.,jnp.where(count==n,jnp.eye(n),matrix))
    matrix = jax.vmap(spin)(orth,result.chemical_potential,jnp.asarray(nelec))
    return (x@matrix@x.conj().swapaxes(-1,-2))*jnp.where(result.converged,1.,jnp.nan),result


def _fermi_divided_difference(values, params):
    """Exact stable Fermi secants, including the coincident-energy limit.

    Avoid subtracting nearly equal occupations and avoid positive exponentials
    in the tails. The natural small scale is temperature, not an orbital gap
    regularizer or a threshold in Hartree.
    """
    mu,t=params
    z=(mu-values)/t
    delta=jnp.abs(values[:,None]-values[None,:])
    prefactor=jax.nn.sigmoid(jnp.maximum(z[:,None],z[None,:]))*jax.nn.sigmoid(-jnp.minimum(z[:,None],z[None,:]))
    ratio=jnp.expm1(-delta/t)/jnp.where(delta==0,1.,delta)
    return prefactor*jnp.where(delta==0,-1/t,ratio)


def density_entropy(density, overlap, kweights):
    """Dimensionless von Neumann entropy of an ensemble density (for SCF free energy)."""
    l = jnp.linalg.cholesky(overlap)
    values = jnp.linalg.eigvalsh(l.conj().swapaxes(-1,-2)@density@l)
    p = jnp.clip(values,jnp.finfo(values.dtype).eps,1-jnp.finfo(values.dtype).eps)
    return -jnp.einsum('k,skn->',kweights,p*jnp.log(p)+(1-p)*jnp.log1p(-p))
