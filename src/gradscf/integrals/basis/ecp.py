"""Scalar semilocal Gaussian ECP values, with no external chemistry runtime.

Nuclear charges in the ordinary IntegralPlan must already be effective core
charges Z-ncore. Coefficient differentiation uses fixed primitive ECP matrices
and JAX contraction. Direct ECP geometry/exponent AD is not implemented.
"""
from dataclasses import dataclass
from collections import defaultdict
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class AtomicECP:
    ncore: int
    # (angular channel: -1 local, radial power, exponent, coefficient)
    terms: tuple[tuple[int,int,float,float],...]

    def __post_init__(self):
        if not isinstance(self.ncore,int) or self.ncore<0:raise ValueError('ncore must be a nonnegative integer.')
        for l,power,a,c in self.terms:
            if l not in range(-1,5) or power not in range(7) or not np.isfinite([a,c]).all() or a<=0:
                raise ValueError('Invalid scalar ECP term.')

    def as_raw(self):
        """Return the common [ncore, angular channels / radial powers] data format."""
        groups=defaultdict(lambda:[[] for _ in range(7)])
        for l,power,a,c in self.terms:groups[l][power].append([a,c])
        return [self.ncore,[[l,groups[l]] for l in sorted(groups)]]


def evaluate_ecp(plan,parameters,ecps):
    if plan.backend!='native':raise NotImplementedError('ECP integrals require the native backend.')
    if ecps is None or len(ecps)!=len(plan.topology.nuclear_charges):
        raise ValueError('Provide one AtomicECP or None per nucleus.')
    if max(plan.topology.angular_momenta)>3:raise NotImplementedError('Scalar ECP supports orbital l<=3.')
    atm,bas,env=plan.pack(parameters)
    rows=[];values=[];offset=env.size
    for atom,potential in enumerate(ecps):
        if potential is None:continue
        groups=defaultdict(list)
        for l,power,a,c in potential.terms:groups[l,power].append((a,c))
        for (l,power),terms in sorted(groups.items()):
            n=len(terms)
            rows.append([atom,l,n,power,0,offset,offset+n,0])
            values.extend(a for a,c in terms);values.extend(c for a,c in terms);offset+=2*n
    if not rows:return jnp.zeros((plan.topology.nao,)*2,dtype=env.dtype)
    env=jnp.concatenate((env,jnp.asarray(values,dtype=env.dtype)))
    from gradscf.integrals._native import register_integrals
    register_integrals()
    call=jax.ffi.ffi_call('gradscf_ecp_cpu_v1',jax.ShapeDtypeStruct((plan.topology.nao,)*2,jnp.float64),vmap_method='sequential')
    return call(jnp.asarray(atm),jnp.asarray(bas),jnp.asarray(rows,dtype=jnp.int32),env,cart=np.int32(plan.topology.cart))


__all__=['AtomicECP']
