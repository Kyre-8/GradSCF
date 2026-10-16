"""q=0 periodic TDA/TDHF/TDDFT with spin-conserving particle-hole response."""
import jax
import jax.numpy as jnp
import numpy as np
from ._kpoint import energy_and_fock
from .scf import validate_reference as _validate_reference
from ..dft.libxc_jax.jax_libxc import hybrid_coeff
from ..solvers.eigen.davidson import _davidson_lowest_symmetric
from ..solvers.eigen.rpa import _davidson_lowest_tdhf, solve_dense_rpa


def response_operators(mf,*,singlet=True):
    _validate_reference(mf)
    # Removing the Ewald term shifts occupied energies only. Spherical
    # truncation changes the entire kernel and cannot use that construction.
    if mf.exxdiv == 'vcut_sph' and hybrid_coeff(mf.xc) != 0:
        raise NotImplementedError('Periodic TD response does not support vcut_sph HF/hybrid references.')
    inputs=mf.inputs;result=mf.result
    coeff=result.mo_coeff_spin.astype(complex);density=result.density_spin.astype(complex)
    nk=coeff.shape[1];n=coeff.shape[-1]
    # G=0 exchange corrections must be removed consistently from both the
    # response kernel and its orbital-energy differences (PySCF convention).
    fock=lambda d:energy_and_fock(d,inputs,mesh=mf.cell.mesh,xc=mf.xc,exxdiv=None)[1]
    f0,derivative=jax.linearize(fock,density)
    energies=jnp.einsum('skpi,skpq,skqi->ski',coeff.conj(),f0,coeff).real
    blocks=[];gaps=[];dimension=0
    spins=range(2) if mf.unrestricted else range(1)
    for spin in spins:
        no=mf.cell.nelec[spin];nv=n-no
        for k in range(nk):
            blocks.append((spin,k,no,nv,dimension))
            gaps.append((energies[spin,k,no:][None,:]-energies[spin,k,:no,None]).reshape(-1))
            dimension+=no*nv
    if not dimension:raise ValueError('No occupied-virtual transitions in this basis.')
    gap=jnp.concatenate(gaps)
    def transition(vector):
        q=jnp.zeros_like(density)
        for spin,k,no,nv,offset in blocks:
            values=vector[offset:offset+no*nv].reshape(no,nv)
            co=coeff[spin,k,:,:no];cv=coeff[spin,k,:,no:]
            q=q.at[spin,k].set(cv@values.T@co.conj().T)
        if not mf.unrestricted:q=q.at[1].set(q[0]*(1 if singlet else -1))
        return q
    def project(f):
        values=[]
        for spin,k,no,nv,offset in blocks:
            co=coeff[spin,k,:,:no];cv=coeff[spin,k,:,no:]
            values.append((cv.conj().T@f[spin,k]@co).T.reshape(-1))
        return jnp.concatenate(values)
    def response(q):
        adjoint=q.conj().swapaxes(-1,-2)
        # Complex-linear extension of the physical Hermitian-density JVP.
        return .5*(derivative(q+adjoint)-1j*derivative(1j*(q-adjoint)))
    def a(vector):return gap*vector+project(response(transition(vector)))
    def b(vector):return project(response(transition(vector.conj()).conj().swapaxes(-1,-2)))
    return jax.jit(a),jax.jit(b),gap


class TDA:
    full=False

    def __init__(self,mf,*,nstates=3,singlet=True,q=None,max_dense=256):
        if q is not None and np.any(np.asarray(q)!=0):
            raise NotImplementedError('Periodic response currently supports q=0 only.')
        if nstates<1:raise ValueError('nstates must be positive.')
        self._scf=mf;self.nstates=nstates;self.singlet=singlet
        self.conv_tol=1e-7;self.max_cycle=100;self.max_dense=max_dense
        self._operators=None

    def _get_operators(self):
        _validate_reference(self._scf)
        # Cache only for the current completed reference object.
        signature=(id(self._scf.result),self.singlet,self._scf.xc,id(self._scf.inputs))
        if self._operators is None or self._reference != signature:
            self._operators=response_operators(self._scf,singlet=self.singlet)
            self._reference=signature
        return self._operators

    def get_ab(self):
        a,b,gap=self._get_operators();n=len(gap)
        if n>self.max_dense:raise ValueError('Dense response exceeds max_dense; use TDA matvec.')
        identity=jnp.eye(n,dtype=complex)
        return jax.vmap(a,in_axes=1,out_axes=1)(identity),jax.vmap(b,in_axes=1,out_axes=1)(identity)

    def kernel(self):
        a,b,gap=self._get_operators();n=len(gap);roots=min(self.nstates,n)
        gamma=not np.any(np.asarray(self._scf.kpts))
        if not self.full:
            apply=jax.jit(jax.vmap(a,in_axes=1,out_axes=1))
            self.e,x,ok=_davidson_lowest_symmetric(apply,nroots=roots,size=n,diag=gap.astype(complex),
                tol=self.conv_tol,max_iter=self.max_cycle,max_subspace=min(n,max(40,8*roots)))
            self.e=self.e.real
            self.xy=(x,jnp.zeros_like(x));residual=jnp.linalg.norm(apply(x)-x*self.e,axis=0)
            self.converged=jnp.isfinite(self.e)&(residual<self.conv_tol*1.01)&ok
        elif gamma:
            def vind(rows):
                def apply(row):
                    x,y=row[:n],row[n:]
                    return jnp.concatenate([a(x)+b(y),-b(x)-a(y)]).real
                return jax.vmap(apply)(rows)
            self.e,x,y,ok=_davidson_lowest_tdhf(jax.jit(vind),nroots=roots,size=n,diag=gap,
                tol=self.conv_tol,max_iter=self.max_cycle)
            rows=jnp.concatenate([x.T,y.T],axis=1)
            residual=jnp.linalg.norm(vind(rows)-self.e[:,None]*rows,axis=1)
            self.xy=(x,y);self.converged=jnp.isfinite(self.e)&(residual<self.conv_tol*1.01)&ok
        else:
            # The shared TDHF solver is real-only. Use a bounded exact solve
            # for complex k meshes until a complex symplectic iteration exists.
            aa,bb=self.get_ab()
            self.e,x,y,self.converged=solve_dense_rpa(aa,bb,nroots=roots,
                tol=self.conv_tol,min_frequency=1e-8,max_dense=self.max_dense)
            self.xy=(x,y)
        self._solution_reference=self._reference
        return self.e,self.xy

    def transition_velocity_dipole(self):
        from .optics import transition_velocity
        return transition_velocity(self)

    def oscillator_strength(self,*,gauge='velocity'):
        from .optics import oscillator_strength
        return oscillator_strength(self,gauge=gauge)

    def run(self):self.kernel();return self


class TDHF(TDA):
    full=True


class TDDFT(TDHF):
    pass
