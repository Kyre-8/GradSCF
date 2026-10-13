"""Array-level periodic response with explicit external-functional parameters."""
from dataclasses import replace
import jax
import jax.numpy as jnp
from .core import Problem
from .ks import ks_problem,_Hermitian
from ..pbc._kpoint import energy_and_fock
from ..pbc.occupations import fermi_density,density_entropy
from ..pbc.kpoints import weights_for


def periodic_problem(inputs,result,*,mesh,nelec,xc='hf',exxdiv='ewald',
                     functional=None,params=None,temperature=None,tolerance=1e-7):
    """Use a converged periodic SCF result without detaching its parameter AD.

    Inputs/model parameters are explicit leaves. Static q=0, collinear spin,
    fixed integer occupations or fixed-spin-population Fermi occupations.
    """
    inputs=inputs._replace(hcore=inputs.hcore.astype(jnp.complex128))
    shape=result.density_spin.shape
    _,nk,n,_=shape
    thermal=temperature is not None
    args=(inputs,params,jnp.asarray(temperature),jnp.asarray(nelec,dtype=float)) if thermal else (inputs,params)
    def evaluate(d,a):
        return energy_and_fock(d,a[0],mesh=mesh,xc=xc,exxdiv=exxdiv,functional=functional,params=a[1])

    if thermal:
        chart=_Hermitian(n,True)
        pack=lambda d:jnp.concatenate([chart.pack(m) for m in d.reshape(-1,n,n)])
        unpack=lambda x:jnp.stack([chart.unpack(v) for v in x.reshape(-1,chart.size)]).reshape(shape)
        def residual(x,a):
            d=unpack(x)
            target,_=fermi_density(evaluate(d,a)[1],a[0].overlap,a[3],a[2],weights_for(a[0]))
            return pack(d-target)
        def quantities(x,a):
            d=unpack(x)
            e=evaluate(d,a)[0]
            entropy=density_entropy(d,a[0].overlap,weights_for(a[0]))
            count=jnp.einsum('k,skij,kji->',weights_for(a[0]),d,a[0].overlap).real
            return dict(density=d,energy=e,free_energy=e-a[2]*entropy,entropy=entropy,electron_number=count)
        problem=Problem(residual,pack(result.density_spin),args,quantities,
                        result.converged,tolerance,series_supported=False)
    else:
        coeff=result.mo_coeff_spin.astype(jnp.complex128)
        problem=ks_problem(coeff.reshape(2*nk,n,n),tuple(no for no in nelec for _ in range(nk)),args,
            lambda d,a:evaluate(d.reshape(shape),a)[1].reshape(2*nk,n,n),
            lambda d,a:evaluate(d.reshape(shape),a)[0],
            overlap=lambda a:jnp.tile(a[0].overlap,(2,1,1)),converged=result.converged,tolerance=tolerance)
        observe=problem.quantities
        def quantities(x,a):
            values=observe(x,a)
            return dict(values,density=values['density'].reshape(shape),
                        electron_number=jnp.sum(values['electron_number'].reshape(2,nk)*weights_for(a[0])))
        problem=replace(problem,quantities=quantities)

    def potential_tangent(value):
        value=jnp.asarray(value,dtype=inputs.hcore.dtype)
        if value.shape!=inputs.hcore.shape:raise ValueError('Periodic potential must have shape (nk,nao,nao)')
        value*=jnp.where(jnp.max(jnp.abs(value-value.conj().swapaxes(-1,-2)))<1e-10,1.,jnp.nan)
        zero=jax.tree.map(jnp.zeros_like,args)
        return (zero[0]._replace(hcore=value),)+zero[1:]
    return replace(problem,potential_tangent=potential_tangent,
        potential_path=lambda tangent,t:(inputs._replace(hcore=inputs.hcore+t*tangent[0].hcore),)+args[1:])
