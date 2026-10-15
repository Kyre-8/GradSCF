"""Current periodic density features for the shared dft.Functional interface."""
import jax
import jax.numpy as jnp
from ..dft.functional import DensityInputs
from .kpoints import weights_for


class PeriodicDensityInputs(DensityInputs):
    """Real physical fields from complex (spin,k,ao,ao) Hermitian densities.

    AO values are the periodic Bloch parts u_k; derivative channels represent
    (grad+i*k)u_k. Grid weights and normalized k weights are distinct. Energies
    returned by an external Functional are total XC energies per cell in Eh.
    """
    @property
    def ao(self):return self._molecule.ao[:,0]

    @property
    def ao_deriv1(self):return self._molecule.ao

    @property
    def weights(self):return self._molecule.weights

    @property
    def coordinates(self):return self._molecule.coords

    @property
    def overlap_matrix(self):return self._molecule.overlap

    @property
    def kweights(self):return weights_for(self._molecule)

    @property
    def kpoints(self):return self._molecule.kpoints

    @property
    def lattice(self):return self._molecule.lattice

    @property
    def ionic_charges(self):return self._molecule.ionic_charges

    @property
    def rho_spin(self):
        return jnp.einsum('k,kgp,skpq,kgq->sg',self.kweights,self.ao,self.density_matrix,self.ao.conj()).real

    @property
    def rho(self):return self.rho_spin.sum(axis=0)

    @property
    def grad_rho_spin(self):
        return 2*jnp.einsum('k,kgp,skpq,kxgq->sgx',self.kweights,self.ao,
                           self.density_matrix,self.ao_deriv1[:,1:4].conj()).real

    @property
    def grad_rho(self):return self.grad_rho_spin.sum(axis=0)

    @property
    def tau_spin(self):
        deriv=self.ao_deriv1[:,1:4]
        return .5*jnp.einsum('k,kxgp,skpq,kxgq->sg',self.kweights,deriv,self.density_matrix,deriv.conj()).real

    @property
    def tau(self):return self.tau_spin.sum(axis=0)

    @property
    def laplacian_rho_spin(self):
        laplacian=self._molecule.ao_laplacian
        if laplacian is None:raise AttributeError('Periodic inputs do not supply ao_laplacian')
        return 2*jnp.einsum('k,kgp,skpq,kgq->sg',self.kweights,self.ao,
                           self.density_matrix,laplacian.conj()).real+4*self.tau_spin

    @property
    def laplacian_rho(self):return self.laplacian_rho_spin.sum(axis=0)


def density_inputs(inputs, density):
    nk,_,_,n=inputs.ao.shape
    dm=jnp.asarray(density)
    if dm.ndim==2 and nk==1:dm=dm[None]
    if dm.shape==(nk,n,n):dm=jnp.stack([dm/2,dm/2])
    if dm.shape!=(2,nk,n,n):
        raise ValueError('Periodic density requires (nk,nao,nao) or (2,nk,nao,nao)')
    return PeriodicDensityInputs(.5*(dm+dm.conj().swapaxes(-1,-2)),inputs)


def finite_q_kernel(functional,params,inputs,density,transfer):
    """Differentiate an explicit physical q/-q energy extension, matrix free.

    For E(D+Z+Z†), the Euclidean gradient is conj(jax.grad(E)). Its Hessian
    H has complex-linear part (H(z)-i H(i z))/2 = 2*w_k*K_q(z).
    Removing the anti-linear part also handles overlapping q/-q blocks at
    self-inverse q, without doubling the physical response. No reference or
    parameter is detached, so higher AD includes reference relaxation.
    """
    if functional.q_energy_fn is None:
        raise NotImplementedError('External finite-q response requires an explicit q_energy_fn energy extension')
    if functional.exact_exchange_fraction != 0.:
        raise NotImplementedError('External finite-q hybrids require an explicit hybrid adapter')
    density=jnp.asarray(density)
    nk=len(inputs.kpoints);n=inputs.overlap.shape[-1]
    if density.shape!=(2,nk,n,n):
        raise ValueError('Finite-q reference density requires (2,nk,nao,nao)')
    zero=jnp.zeros_like(density,dtype=jnp.result_type(density,1j))
    def energy(z):
        value=jnp.asarray(functional.q_energy_fn(params,inputs,density,transfer,z))
        if value.shape!=() or jnp.iscomplexobj(value):
            raise ValueError('q_energy_fn must return a real scalar XC energy per cell')
        return value
    _,hessian=jax.linearize(lambda z:jax.grad(energy)(z).conj(),zero)
    weights=weights_for(inputs)
    valid=jnp.all(jnp.isfinite(weights)&(weights>0))
    scale=4*jnp.where(weights>0,weights,1.)[None,:,None,None]
    def apply(tangent):
        tangent=jnp.asarray(tangent,dtype=zero.dtype)
        if tangent.shape!=density.shape:
            raise ValueError('Finite-q tangent requires the spin/k/AO reference shape')
        return (hessian(tangent)-1j*hessian(1j*tangent))/scale*jnp.where(valid,1.,jnp.nan)
    return apply
