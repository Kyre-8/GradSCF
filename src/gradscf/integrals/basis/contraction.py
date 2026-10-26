"""JAX contraction of fixed primitive integrals, with basis normalization.

Native integrals can be evaluated outside AD when exponents/centers are fixed.
These transforms expose coefficient derivatives without a native basis JVP.
Full primitive ERIs are intended for small-system validation, not a scalable cache.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
from gradscf.integrals.basis.normalization import normalized_shell_coefficients, radial_primitive_norm


def primitive_basis(topology, parameters):
    """Expand each contraction into independent normalized primitives."""
    return replace(topology, contraction_counts=topology.primitive_counts), replace(
        parameters, coefficients=tuple(jnp.eye(n, dtype=c.dtype) for n,c in
            zip(topology.primitive_counts,parameters.coefficients)))


def contraction_matrix(topology, parameters):
    """Matrix T in chi_contracted = chi_primitive @ T; supports Cartesian/real spherical."""
    from .types import check_parameter_shapes
    check_parameter_shapes(topology,parameters)
    angular = tuple((l+1)*(l+2)//2 if topology.cart else 2*l+1 for l in topology.angular_momenta)
    nprim = sum(n*m for n,m in zip(topology.primitive_counts,angular))
    t = jnp.zeros((nprim,topology.nao),dtype=parameters.coefficients[0].dtype)
    pi=ci=0
    for l,m,a,c in zip(topology.angular_momenta,angular,parameters.exponents,parameters.coefficients):
        normalized=normalized_shell_coefficients(l,a,c)/radial_primitive_norm(l,a)[:,None]
        block=jnp.kron(normalized,jnp.eye(m,dtype=c.dtype))
        t=t.at[pi:pi+block.shape[0],ci:ci+block.shape[1]].set(block)
        pi+=block.shape[0];ci+=block.shape[1]
    return t


def contract_integrals(tensors, transform):
    """Contract overlap/kinetic/nuclear/dipole/ERI tensors with the same T."""
    t=jnp.asarray(transform)
    result={}
    for name,value in tensors.items():
        if name=='eri':
            if value.shape != (t.shape[0],)*4:
                raise ValueError('Primitive ERI dimensions do not match transform.')
            result[name]=jnp.einsum('pqrs,pi,qj,rk,sl->ijkl',value,t,t,t,t,optimize=True)
        elif name in {'overlap','kinetic','nuclear','dipole','hcore','ecp'}:
            if value.shape[-2:] != (t.shape[0],)*2:
                raise ValueError('Primitive matrix dimensions do not match transform.')
            result[name]=jnp.einsum('pi,...pq,qj->...ij',t,value,t)
        else:
            raise ValueError(f'Unknown integral operator {name!r}.')
    return result


__all__=['primitive_basis','contraction_matrix','contract_integrals']
