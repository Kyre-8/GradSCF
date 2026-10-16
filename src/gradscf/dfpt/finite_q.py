"""Static finite-q response with Fock, overlap and direct-density sources.

Solve (1-chi0_q K_q) dD_q = chi0_q v_q with the shared Problem/linear solver.
This is the finite-basis spectral form of the Sternheimer equation, with all
AO bands retained. QSource supports moving-metric/Pulay response; the primitive
displacement adapter supplies geometry sources. See Baroni et al., Rev. Mod. Phys.
73, 515 (2001), Sec. II.C.1.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from .core import Problem
from ..pbc._kpoint import energy_and_fock
from ..pbc.response import q_kernel,q_density,_validate_q_domain
from ..pbc.occupations import _fermi_divided_difference
from ..pbc.kpoints import weights_for,reciprocal_vectors


class QSource(NamedTuple):
    """Fixed-density Fock/overlap variations and direct spin-density envelope."""
    fock: object
    overlap: object
    rho_spin: object


def _independent_response(result,transfer,temperature):
    c=result.mo_coeff_spin;e=result.mo_energy_spin;f=result.mo_occ_spin
    partner=transfer.partners;n=e.shape[-1]
    delta=e[:,partner,:,None]-e[:,:,None,:]
    valid=jnp.all(jnp.isfinite(e))&jnp.all(jnp.isfinite(c))&jnp.all(jnp.isfinite(f))
    if temperature is None:
        difference=f[:,partner,:,None]-f[:,:,None,:]
        divided=difference/jnp.where(jnp.abs(delta)>1e-8,delta,1.)
        valid &= jnp.all(jnp.abs(f-jnp.rint(f))<1e-10)&jnp.all((f>=0)&(f<=1))
        occupied=jnp.max(jnp.where(f>.5,e,-jnp.inf),axis=(1,2))
        empty=jnp.min(jnp.where(f<.5,e,jnp.inf),axis=(1,2))
        valid &= jnp.all(empty-occupied>1e-8)
    else:
        t=jnp.asarray(temperature);mu=result.chemical_potential
        if mu is None or jnp.shape(mu)!=(e.shape[0],):
            raise ValueError('Finite-temperature reference requires one chemical potential per spin')
        valid &= jnp.isfinite(t)&(t>0)
        expected=jax.nn.sigmoid((mu[:,None,None]-e)/t)
        valid &= jnp.all(~jnp.isnan(mu))&jnp.all(jnp.abs(f-expected)<1e-10)
        def spin(values,chemical_potential):
            safe_mu=jnp.where(jnp.isfinite(chemical_potential),chemical_potential,0.)
            def pair(upper,lower):
                all_values=jnp.concatenate([upper,lower])
                return _fermi_divided_difference(all_values,jnp.array([safe_mu,t]))[:n,n:]
            return jax.vmap(pair)(values[partner],values)*jnp.isfinite(chemical_potential)
        divided=jax.vmap(spin)(e,mu)
    def apply(v,overlap=None):
        # JAX 0.8.1 CPU constant folding can miscompile gather+conjugate+
        # transpose of a closed-over complex frame. Keep the frame at runtime;
        # the barrier is an AD identity and does not detach its derivatives.
        coeff=jax.lax.optimization_barrier(c)
        mo=coeff[:,partner].conj().swapaxes(-1,-2)@v@coeff
        response=divided*mo
        if overlap is not None:
            metric=coeff[:,partner].conj().swapaxes(-1,-2)@overlap@coeff
            # Generalized spectral derivative: (f_i-f_j)/(e_i-e_j) dF
            # minus [ (f_i+f_j)/2 + (e_i+e_j)/2 * divided ] dS.
            weighted=.5*(f[:,partner,:,None]+f[:,:,None,:])
            weighted+=.5*(e[:,partner,:,None]+e[:,:,None,:])*divided
            response-=weighted*metric
        return coeff[:,partner]@response@coeff.conj().swapaxes(-1,-2)
    return apply,valid


def finite_q_problem(inputs,result,transfer,*,mesh,xc='hf',exxdiv='ewald',
                     temperature=None,tolerance=1e-7,functional=None,params=None):
    """A finite-q response snapshot, with explicit static momentum-transfer data.

    Potential and density blocks have shape (2,nk,nao,nao), row k+q/column k;
    spin-independent potentials may omit the spin axis. They are generally
    non-Hermitian. Physical real perturbations combine q and its adjoint -q.
    Equal k weights and a closed mesh are required. Zero T requires an insulator;
    finite T reuses stable Fermi secants, including coincident-energy limits.
    External Functional models require q_energy_fn; params remain differentiable
    along with the reference. Construct this problem inside the AD function.
    """
    _validate_q_domain(inputs, exxdiv)
    if np.allclose(np.asarray(transfer.q)-np.rint(transfer.q),0.,atol=1e-10,rtol=0):
        raise ValueError('Use periodic_problem for q=0, including chemical-potential response')
    shape=result.density_spin.shape;size=result.density_spin.size
    if shape!=(2,len(inputs.kpoints),inputs.hcore.shape[-1],inputs.hcore.shape[-1]):
        raise ValueError('Inconsistent spin/k/AO reference shapes')
    if len(transfer.partners)!=shape[1]:raise ValueError('Momentum transfer must cover the k mesh')
    chi,valid=_independent_response(result,transfer,temperature)
    interaction=q_kernel(inputs,result.density_spin,transfer,mesh=mesh,xc=xc,exxdiv=exxdiv,
                         functional=functional,params=params)
    c=result.mo_coeff_spin;s=inputs.overlap;eps=result.mo_energy_spin
    fock=energy_and_fock(result.density_spin,inputs,mesh=mesh,xc=xc,exxdiv=exxdiv,
                        functional=functional,params=params)[1]
    rebuilt=(c*result.mo_occ_spin[...,None,:])@c.conj().swapaxes(-1,-2)
    valid &= result.converged & jnp.all(jnp.abs(weights_for(inputs)-1/shape[1])<1e-12)
    valid &= jnp.max(jnp.abs(c.conj().swapaxes(-1,-2)@s@c-jnp.eye(shape[-1])))<=tolerance
    valid &= jnp.max(jnp.abs(fock@c-(s@c)*eps[...,None,:]))<=tolerance
    valid &= jnp.max(jnp.abs(rebuilt-result.density_spin))<=tolerance
    reciprocal=reciprocal_vectors(inputs.lattice)
    mismatch=inputs.kpoints[transfer.partners]-inputs.kpoints-(transfer.q+transfer.shifts)@reciprocal
    valid &= jnp.max(jnp.abs(mismatch))<=tolerance
    pack=lambda d:jnp.concatenate([d.real.ravel(),d.imag.ravel()])
    unpack=lambda x:(x[:size]+1j*x[size:]).reshape(shape)
    zero=QSource(jnp.zeros(shape,dtype=complex),jnp.zeros(inputs.overlap.shape,dtype=complex),
                 jnp.zeros((2,len(inputs.weights)),dtype=complex))
    def residual(x,source):
        d=unpack(x)
        return pack(d-chi(source.fock+interaction(d),source.overlap))
    def quantities(x,source):
        d=unpack(x)
        # Nonzero-q first-order energy/particle-number averages vanish over
        # the commensurate supercell. Second order involves other q sectors.
        return dict(density=d,rho_spin=q_density(inputs,transfer,d)+source.rho_spin,
                    energy=jnp.array(0.),electron_number=jnp.array(0.))
    def potential_tangent(value):
        value=jnp.asarray(value,dtype=jnp.complex128)
        if value.shape==shape[1:]:value=jnp.broadcast_to(value,shape)
        if value.shape!=shape:raise ValueError('Finite-q potential requires (nk,nao,nao) or (2,nk,nao,nao)')
        return zero._replace(fock=value)
    return Problem(residual,jnp.zeros(2*size),zero,quantities,
        valid,tolerance,potential_tangent,series_supported=False)
