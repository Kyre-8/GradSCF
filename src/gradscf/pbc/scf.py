"""Gamma-point periodic SCF using shared iterations and explicit FFTDF inputs."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from ..integrals.periodic.fft import build_inputs,get_jk
from ..scf.core import _orthogonalizer,_diagonalize_fock,_build_density_from_occ
from ..scf.uks import run_unrestricted_scf_scan,_unrestricted_xc_energy_and_potential_on_grid
from ..scf.rks import _vxc_matrix_from_grid_potential
from ..scf.energy import XCContribution,unrestricted_energy,unrestricted_fock
from ..scf.autodiff import SCFDifferentiationConfig,attach_scf_backward
from ..dft.libxc_jax.jax_libxc import hybrid_coeff,xc_type
from .kpoints import weights_for,time_reversal_plan
from ._kpoint import run_kpoint_scf


def validate_reference(mf, *, integer=True, equal_weights=True, external=False):
    """Check freshness and the reference domain required by a consumer."""
    if mf.result is None or not mf.converged:
        raise ValueError('Periodic response requires a converged SCF reference.')
    if (mf.xc!=mf._computed_xc or mf.cell.mesh!=mf._computed_mesh or mf.cell._version!=mf._cell_version):
        raise ValueError('Run kernel() again after changing the periodic reference.')
    if getattr(mf,'functional',None) is not getattr(mf,'_computed_functional',None):
        raise ValueError('Run kernel() again after changing the external functional.')
    if getattr(mf,'functional',None) is not None:
        if _params_signature(mf.params)!=mf._computed_params:
            raise ValueError('Run kernel() again after changing functional parameters.')
        if not external:raise NotImplementedError('This consumer does not yet support external periodic functionals.')
    if hasattr(mf,'_computed_temperature'):
        weights=np.full(len(mf.kpts),1/len(mf.kpts)) if mf.kweights is None else np.asarray(mf.kweights)
        if (mf.temperature!=mf._computed_temperature or mf.exxdiv!=mf._computed_exxdiv
                or not np.array_equal(mf.kpts,mf._computed_kpoints)
                or not np.array_equal(weights,mf._computed_kweights)):
            raise ValueError('Run kernel() again after changing temperature, k sampling or exchange correction.')
    if getattr(mf,'time_reversal',False)!=getattr(mf,'_computed_time_reversal',False):
        raise ValueError('Run kernel() again after changing time-reversal reduction.')
    if getattr(mf,'symmetry',False)!=getattr(mf,'_computed_symmetry',False):
        raise ValueError('Run kernel() again after changing spatial symmetry reduction.')
    if integer and (getattr(mf,'temperature',None) is not None
            or not np.allclose(mf.result.mo_occ_spin,np.rint(mf.result.mo_occ_spin),atol=1e-10,rtol=0)):
        raise NotImplementedError('This method currently requires zero-temperature integer occupations.')
    if equal_weights and not np.allclose(weights_for(mf.inputs),1/len(mf.kpts),atol=1e-12,rtol=0):
        raise NotImplementedError('This method currently requires equal k weights.')


def _params_signature(params):
    from ..scf.reference import _array_signature
    return jax.tree.structure(params),tuple(_array_signature(x) for x in jax.tree.leaves(params))


class PeriodicSCFResult(NamedTuple):
    total_energy: object
    density_spin: object       # (2,nk,nao,nao)
    mo_coeff_spin: object
    mo_energy_spin: object
    mo_occ_spin: object
    fock_spin: object
    converged: object
    cycles: object
    free_energy: object = None
    entropy: object = 0.
    chemical_potential: object = None


def gamma_energy_and_fock(density,inputs,*,xc='hf',exxdiv='ewald'):
    alpha=hybrid_coeff(xc);kind=xc_type(xc)
    j,k=get_jk(inputs,density,exxdiv=exxdiv,with_k=alpha!=0)
    ao=inputs.ao[0,0];deriv=inputs.ao[0]
    exc,va,vb,ga,gb=_unrestricted_xc_energy_and_potential_on_grid(
        ao=ao,ao_deriv1=deriv,weights=inputs.weights,density_a=density[0],density_b=density[1],
        xc_spec=xc,density_floor=1e-12,potential_clip=None,xc_kind=kind)
    potentials=jnp.stack([_vxc_matrix_from_grid_potential(ao=ao,ao_deriv1=deriv,
        ao_laplacian=jnp.zeros_like(ao),weights=inputs.weights,vxc_rho=v,vxc_grad=g,
        vxc_tau=jnp.zeros_like(v),vxc_lapl=jnp.zeros_like(v),xc_kind=kind)
        for v,g in ((va,ga),(vb,gb))])
    contribution=XCContribution(exc,potentials,alpha)
    return (unrestricted_energy(density,inputs.hcore[0],j,k,contribution,
                nuclear_repulsion=inputs.nuclear_repulsion),
            unrestricted_fock(inputs.hcore[0],j,k,contribution))


def run_gamma_scf(inputs,*,nelec,xc='hf',exxdiv='ewald',max_cycle=100,
                  conv_tol=1e-10,conv_tol_density=1e-8,conv_tol_grad=1e-7,
                  damping=0.,level_shift=0.,init_density=None,
                  differentiation=None,temperature=None,mesh=None,functional=None,params=None,reduction=None):
    """Array-valued SCF with implicit/explicit derivatives on fixed FFT topology."""
    if inputs.overlap.shape[0]!=1:
        raise NotImplementedError('This solver currently accepts Gamma only.')
    if temperature is not None or functional is not None or reduction is not None:
        if mesh is None:raise ValueError('Thermal/external Gamma SCF requires the explicit FFT mesh')
        density=None if init_density is None else jnp.asarray(init_density)[:,None]
        return run_kpoint_scf(inputs,mesh=mesh,nelec=nelec,xc=xc,exxdiv=exxdiv,
            max_cycle=max_cycle,conv_tol=conv_tol,conv_tol_density=conv_tol_density,
            conv_tol_grad=conv_tol_grad,damping=damping,level_shift=level_shift,
            init_density=density,differentiation=differentiation,temperature=temperature,
            functional=functional,params=params,reduction=reduction)
    if xc_type(xc) not in ('HF','LDA','GGA'):
        raise NotImplementedError('Periodic XC currently supports LDA/GGA/global hybrids.')
    n=inputs.overlap.shape[-1]
    if any(v<0 or v>n or v!=int(v) for v in nelec) or sum(nelec)==0 or max_cycle<1:
        raise ValueError('Zero-temperature SCF requires integer occupations and positive max_cycle.')
    config=differentiation or SCFDifferentiationConfig(mode='explicit')
    s=inputs.overlap[0];x=_orthogonalizer(s,1e-10)
    eps,c=_diagonalize_fock(inputs.hcore[0],x)
    occ=jnp.stack([(jnp.arange(n)<count).astype(s.dtype) for count in nelec])
    coeff=jnp.stack([c,c]);energies=jnp.stack([eps,eps])
    density=jax.vmap(_build_density_from_occ)(coeff,occ) if init_density is None else jnp.asarray(init_density)
    if density.shape!=(2,n,n):raise ValueError('Initial spin density must have shape (2,nao,nao).')
    def evaluate(d,args):return gamma_energy_and_fock(d,args,xc=xc,exxdiv=exxdiv)
    def builder(d,*_):
        e,f=evaluate(d,inputs)
        return f,f,e
    out=run_unrestricted_scf_scan(fock_builder=builder,density_spin=density,
        mo_coeff_spin=coeff,mo_occ_spin=occ,mo_energy_spin=energies,overlap=s,
        max_cycle=max_cycle,damping=damping,conv_tol=conv_tol,conv_tol_density=conv_tol_density,
        conv_tol_grad=conv_tol_grad,orthogonalization_eps=1e-10,level_shift=level_shift)
    density,coeff,energies,_,converged,cycles,*_=out
    if config.mode=='implicit':
        def residual(d,args):
            _,f=evaluate(d,args)
            transform=_orthogonalizer(args.overlap[0],1e-10)
            _,cnew=jax.vmap(lambda f:_diagonalize_fock(f,transform))(f)
            return jax.vmap(_build_density_from_occ)(cnew,occ)-d
        density=attach_scf_backward(inputs,solution=density,residual=residual,
                                    converged=converged,config=config)
        _,f=evaluate(density,inputs)
        energies,coeff=jax.vmap(lambda f:_diagonalize_fock(f,x))(f)
    energy,fock=evaluate(density,inputs)
    return PeriodicSCFResult(energy,density[:,None],coeff[:,None],energies[:,None],
                             occ[:,None],fock[:,None],converged,cycles,energy)


class _SCF:
    unrestricted=False
    default_xc='hf'
    k_sampling=False

    def __init__(self,cell,*,kpts=None,kweights=None,xc=None,exxdiv='ewald',temperature=None,
                 functional=None,params=None,time_reversal=False,symmetry=False,**controls):
        self.cell=cell if hasattr(cell,'topology') else cell.build()
        self.kpts=jnp.zeros((1,3)) if kpts is None else jnp.asarray(kpts).reshape(-1,3)
        if not self.k_sampling and (self.kpts.shape!=(1,3) or np.any(np.asarray(self.kpts)!=0)):
            raise NotImplementedError('First periodic SCF release supports Gamma only.')
        if not self.unrestricted and self.cell.spin!=0:
            raise ValueError('Restricted periodic SCF requires spin=0.')
        if functional is not None and (self.default_xc=='hf' or xc is not None):
            raise ValueError('Use a periodic KS class with one external Functional and no named xc baseline.')
        if params is not None and functional is None:raise ValueError('params requires an external functional')
        self.functional=functional;self.params=params
        self.xc=(self.default_xc if xc is None else xc) if functional is None else None
        if self.default_xc=='hf' and self.xc!='hf':
            raise ValueError('Use a periodic KS class for a DFT functional.')
        self.kweights=kweights;self.temperature=temperature
        self.time_reversal=bool(time_reversal);self.symmetry=bool(symmetry);self.reduction=None
        self.exxdiv=exxdiv;self.controls=controls;self.inputs=None;self.result=None

    def kernel(self,dm0=None):
        if not jax.config.x64_enabled:raise ValueError('Periodic SCF requires JAX float64.')
        self.inputs=build_inputs(self.cell,kpts=self.kpts,kweights=self.kweights)
        if self.symmetry:
            from .reduction import SpatialReduction
            self.reduction=SpatialReduction.from_cell(self.cell,self.kpts,
                kweights=np.asarray(self.inputs.kweights),time_reversal=self.time_reversal)
        elif self.time_reversal:
            self.reduction=time_reversal_plan(np.asarray(self.kpts)@np.asarray(self.cell.lattice).T/(2*np.pi))
        else:
            self.reduction=None
        if self.reduction is not None:
            w=np.asarray(self.inputs.kweights)
            if (not bool(self.reduction.consistent(self.inputs.overlap))
                    or not bool(self.reduction.consistent(self.inputs.hcore))
                    or not bool(self.reduction.weights_consistent(w))):
                raise ValueError('Inputs violate the requested symmetry; use the full mesh or refine the FFT grid.')
        if (self.functional is None and hybrid_coeff(self.xc)!=0 and self.exxdiv=='ewald'
                and not np.allclose(self.inputs.kweights,1/len(self.kpts),atol=1e-12,rtol=0)):
            raise NotImplementedError('Ewald exchange correction requires equal k weights; use exxdiv=None for general weights')
        density=None
        if dm0 is not None:
            d=jnp.asarray(dm0)
            density=d if self.unrestricted else jnp.stack([d/2,d/2])
        self.result=self._solve(self.inputs,params=self.params,init_density=density,reduction=self.reduction)
        self._computed_xc=self.xc;self._computed_mesh=self.cell.mesh;self._cell_version=self.cell._version
        self._computed_functional=self.functional;self._computed_params=_params_signature(self.params)
        self._computed_time_reversal=self.time_reversal
        self._computed_symmetry=self.symmetry
        self.e_tot=self.result.total_energy
        self.e_free=self.result.free_energy;self.entropy=self.result.entropy
        self.chemical_potential=self.result.chemical_potential
        self._computed_temperature=self.temperature
        self._computed_exxdiv=self.exxdiv
        self._computed_kpoints=np.asarray(self.kpts).copy()
        self._computed_kweights=np.asarray(self.inputs.kweights).copy()
        self.converged=bool(self.result.converged);self.cycles=int(self.result.cycles)
        self.mo_coeff=self.result.mo_coeff_spin if self.unrestricted else self.result.mo_coeff_spin[0]
        self.mo_energy=self.result.mo_energy_spin if self.unrestricted else self.result.mo_energy_spin[0]
        self.mo_occ=self.result.mo_occ_spin if self.unrestricted else self.result.mo_occ_spin.sum(axis=0)
        return self.e_tot

    def run(self,**controls):
        for name in ('temperature','functional','params','time_reversal','symmetry'):
            if name in controls:setattr(self,name,controls.pop(name))
        self.controls.update(controls);self.kernel();return self

    def _solve(self,inputs,*,params,init_density=None,reduction=None):
        if self.functional is not None and (self.default_xc=='hf' or self.xc is not None):
            raise ValueError('Use a periodic KS class with one external Functional and no named xc baseline.')
        if params is not None and self.functional is None:
            raise ValueError('params requires an external functional')
        if self.default_xc=='hf' and self.xc!='hf':
            raise ValueError('Use a periodic KS class for a DFT functional.')
        if self.temperature is not None and (not np.isfinite(self.temperature) or self.temperature<=0):
            raise ValueError('temperature is kBT in Eh and must be positive; use None for fixed occupations')
        gamma=self.kpts.shape==(1,3) and not np.any(np.asarray(self.kpts))
        solver=run_gamma_scf if gamma else run_kpoint_scf
        density=init_density[:,0] if gamma and init_density is not None else init_density
        return solver(inputs,mesh=self.cell.mesh,nelec=self.cell.nelec,xc=self.xc,exxdiv=self.exxdiv,
            init_density=density,temperature=self.temperature,functional=self.functional,
            params=params,reduction=reduction,**self.controls)

    def nuc_grad_method(self):
        from .grad import Gradients
        return Gradients(self)

    def to_reference(self):
        """Capture a validated periodic array state for the shared Trainer."""
        from .training import Reference
        return Reference.from_scf(self)

    def get_bands(self,kpts,*,chunk_size=4):
        from .bands import get_bands
        return get_bands(self,kpts,chunk_size=chunk_size)

    def make_rdm1(self):
        if self.result is None:raise RuntimeError('Run kernel() before requesting density.')
        return self.result.density_spin if self.unrestricted else self.result.density_spin.sum(axis=0)


class RHF(_SCF):
    pass


class UHF(_SCF):
    unrestricted=True


class KRHF(RHF):
    k_sampling=True


class KUHF(UHF):
    k_sampling=True
