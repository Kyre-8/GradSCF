"""NNAO RHF energy objectives using the existing implicit SCF solver."""
from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import scf
from gradscf.integrals.backends.native.jk import ProjectedNativeDirectBasis
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.solvers.nonlinear import ImplicitFixedPointConfig, implicit_fixed_point_solution
from gradscf.scf.core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from gradscf.model.nnao import prepare_basis

from ..integrals.fixed_exponents import contracted_one_electron, physical_inputs

def _run_scf_with_rescue(solver,kwargs,config,*,rescue_level_shift):
    """Use shifted iterations only after failure, then solve the raw problem."""
    primary=solver(**kwargs,config=config)
    unchanged=(primary,jnp.asarray(False),primary.cycles)
    if rescue_level_shift==0.:return unchanged
    def rescue(previous):
        shifted=solver(**kwargs,config=replace(config,level_shift=rescue_level_shift))
        polished=solver(**dict(kwargs,init_density=shifted.density_matrix,
            init_mo_coeff=shifted.mo_coeff,init_mo_energy=shifted.mo_energy),config=config)
        return polished,jnp.asarray(True),previous.cycles+shifted.cycles+polished.cycles
    return jax.lax.cond(primary.converged,lambda _:unchanged,rescue,primary)

class NonfiniteGradientError(RuntimeError):
    """Keep the failed backward arrays alongside its already validated primal."""
    def __init__(self,outputs,gradient,energy,info):
        super().__init__('Nonfinite energy or gradient')
        self.failure_arrays=dict(basis_outputs=np.array(outputs),coefficient_gradient=np.array(gradient))
        self.failure_details=dict(energy_hartree=float(energy),scf=info)

@lru_cache(maxsize=32)
def _df_rhf_solver(nelectron,implicit_tolerance,rescue_level_shift):
    """Cache SCF/response by algorithm configuration; integrals are operands."""
    cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=150,
                  conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)

    def value(inputs,nuclear_energy):
        s,h,factors=inputs;n=s.shape[0]
        mo_occ=jnp.zeros(n,dtype=s.dtype).at[:nelectron//2].set(2.)
        kwargs=dict(overlap=s,hcore=h,eri=None,df_factors=factors,
            nelectron=nelectron,nuclear_repulsion=nuclear_energy,
            ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
        result,rescued,cycles=_run_scf_with_rescue(run_rks_from_integrals_traceable,
            kwargs,cfg,rescue_level_shift=rescue_level_shift)
        initial_scf_energy=result.total_energy
        from gradscf.scf.stability import stabilize_df_rhf
        result,stability=stabilize_df_rhf(result,**kwargs,config=cfg)
        cycles=cycles+stability['stability_scf_cycles']

        def fixed_point(density,physical):
            overlap,core,df=physical
            j,k=build_jk_from_df(df,density)
            _,coeff=_diagonalize_fock(core+j-.5*k,
                                     _orthogonalizer(overlap,cfg.orthogonalization_eps))
            return _build_density_from_occ(coeff,mo_occ)

        density=implicit_fixed_point_solution(inputs,solution=result.density_matrix,
            fixed_point=fixed_point,
            config=ImplicitFixedPointConfig(tolerance=implicit_tolerance,max_iter=100,restart=40),
            converged=result.converged & stability['stable'],require_converged=True)
        j,k=build_jk_from_df(factors,density);fock=h+j-.5*k
        energy=jnp.sum(density*h)+.5*jnp.sum(density*j)-.25*jnp.sum(density*k)+nuclear_energy
        _,coeff=_diagonalize_fock(fock,_orthogonalizer(s,cfg.orthogonalization_eps))
        info=dict(converged=result.converged,scf_cycles=result.cycles,
            rescue_used=rescued,total_scf_cycles=cycles,
            fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(coeff,mo_occ)-density),
            orbital_residual=jnp.linalg.norm(fock@density@s-s@density@fock),
            min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],
            reconstruction_error=jnp.abs(energy-result.total_energy))
        info.update(stability)
        info['initial_scf_energy']=initial_scf_energy
        return energy,info

    return jax.jit(value),jax.jit(jax.value_and_grad(value,argnums=0,has_aux=True))

class BasisRHF:
    """RHF experiment, defaulting to methane; geometry may specify another molecule."""
    def __init__(self,bond=1.09,basis_family="szp442_direct",core_primitives=None,
                 geometry=None,jk_backend='direct',auxbasis='def2-universal-jkfit',
                 integral_cache=None,implicit_tolerance=1e-9,cache_storage='device',
                 scf_rescue_level_shift=0.0,*,log_exponent_scales=None,df_metric_factor=None):
        if not np.isfinite(implicit_tolerance) or implicit_tolerance<=0:
            raise ValueError('implicit_tolerance must be finite and positive.')
        self.implicit_tolerance=float(implicit_tolerance)
        if not np.isfinite(scf_rescue_level_shift) or scf_rescue_level_shift<0:
            raise ValueError('scf_rescue_level_shift must be finite and nonnegative.')
        self.scf_rescue_level_shift=float(scf_rescue_level_shift)
        if cache_storage not in {'device','host','mmap'}:
            raise ValueError('cache_storage must be device, host or mmap.')
        if cache_storage in {'host','mmap'} and (jk_backend!='df' or integral_cache is None):
            raise ValueError('Host/mmap cache_storage requires a prepared DF integral_cache.')
        self.cache_storage=cache_storage
        if jk_backend not in {'direct','df'}:raise ValueError('jk_backend must be direct or df.')
        if jk_backend=='df' and basis_family=='qvszps':raise NotImplementedError('DF experiment currently covers all-electron families.')
        self.jk_backend=jk_backend;self.auxbasis=auxbasis
        if core_primitives is not None and basis_family not in {'szp3_direct','szp442_direct','szp663_direct'}:
            raise ValueError('core_primitives applies only to all-electron direct basis families.')
        if not np.isfinite(bond) or bond<=0:raise ValueError('bond must be positive.')
        self.bond=float(bond)
        self.coords=np.vstack((np.zeros((1,3)),bond/np.sqrt(3)*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])))
        self.symbols=('C','H','H','H','H')
        self.molecule='CH4'
        if geometry is not None:
            if geometry.get('charge',0)!=0 or geometry.get('spin',0)!=0:
                raise ValueError('This RHF experiment currently requires a neutral closed-shell molecule.')
            self.symbols=tuple(geometry['symbols']);self.coords=np.asarray(geometry['coords_angstrom'],dtype=float)
            if self.coords.shape!=(len(self.symbols),3) or not np.isfinite(self.coords).all():
                raise ValueError('Invalid molecular geometry.')
            self.molecule=geometry['name'];self.bond=None
        if basis_family=='qvszps':
            from gradscf.model.nnao import prepare_grimme_basis
            self.layout=prepare_grimme_basis(list(zip(self.symbols,self.coords)),unit='Angstrom')
        elif basis_family in {'szp3_direct','szp442_direct','szp663_direct'}:
            from gradscf.model.nnao import prepare_direct_basis
            self.layout=prepare_direct_basis(list(zip(self.symbols,self.coords)),unit='Angstrom',basis_family=basis_family,core_primitives=core_primitives)
        elif basis_family=='szp3':
            self.layout=prepare_basis(list(zip(self.symbols,self.coords)),unit='Angstrom')
        else:raise ValueError('Unknown basis family')
        if log_exponent_scales is not None:
            if not hasattr(self.layout,'with_log_exponent_scales'):
                raise ValueError('Log exponent scales require a direct-contraction NNAO basis.')
            self.layout=self.layout.with_log_exponent_scales(log_exponent_scales)
        self.basis_family=basis_family
        self.nelectron=sum(self.layout.topology.nuclear_charges)
        if self.nelectron%2:raise ValueError('RHF requires an even electron count.')
        from ..integrals.context import prepare_fixed_context
        prepared=prepare_fixed_context(self.layout,self.symbols,self.coords,
            basis_family=basis_family,jk_backend=jk_backend,auxbasis=auxbasis,
            integral_cache=integral_cache,cache_storage=cache_storage,metric_factor=df_metric_factor)
        self.primitive_topology=prepared.topology
        self.integral_signature=prepared.signature
        self.primitive_direct=prepared.direct_basis
        self.ps,self.ph,self.rep=prepared.overlap,prepared.hcore,prepared.factors
        self.enuc=scf.nuclear_repulsion_energy(prepared.parameters.nuclear_coords,
            jnp.asarray(prepared.topology.nuclear_charges))
        if self.jk_backend=='df':
            self._projection=jax.jit(self._df_inputs)
            self._df_value,self._df_value_grad=_df_rhf_solver(
                self.nelectron,self.implicit_tolerance,self.scf_rescue_level_shift)
            self._value=self._staged_df_value
            self._value_grad=self._staged_df_value_grad
        else:
            self._value=jax.jit(self._implicit_value)
            self._value_grad=jax.jit(jax.value_and_grad(self._implicit_value,argnums=0,has_aux=True))

    def write_integral_cache(self,path):
        if self.jk_backend!='df':raise ValueError('Only the DF backend has a reusable integral cache.')
        from ..integrals.cache import write_fixed_cache
        return write_fixed_cache(path,self.integral_signature,self.ps,self.ph,self.rep)

    def _contracted_one_electron(self,outputs,ps,ph):
        return contracted_one_electron(self.layout, outputs, ps, ph)

    @staticmethod
    def _project_df_factors(t,rep):
        from gradscf.integrals.molecular.density_fitting import project_factors
        return project_factors(rep,t)

    def _df_inputs(self,outputs,ps,ph,rep):
        return physical_inputs(self.layout, outputs, ps, ph, rep)

    def _staged_df_value(self,outputs,ps,ph,rep):
        return self._df_value(self._projection(outputs,ps,ph,rep),self.enuc)

    def _staged_df_value_grad(self,outputs,ps,ph,rep):
        physical,pullback=jax.vjp(lambda c:self._projection(c,ps,ph,rep),outputs)
        value,cotangent=self._df_value_grad(physical,self.enuc)
        return value,pullback(cotangent)[0]

    def _jk(self,t,density,rep,df_factors):
        if self.jk_backend=='df':
            return build_jk_from_df(df_factors,density)
        return ProjectedNativeDirectBasis(self.primitive_direct,t).get_jk(density)

    def _implicit_value(self,outputs,ps,ph,rep):
        if self.jk_backend=='df':
            return self._df_value(self._df_inputs(outputs,ps,ph,rep),self.enuc)
        t,s,h=self._contracted_one_electron(outputs,ps,ph)
        df_factors=self._project_df_factors(t,rep) if self.jk_backend=='df' else None
        direct_basis=(ProjectedNativeDirectBasis(self.primitive_direct,t)
                      if self.jk_backend=='direct' else None)
        n=s.shape[0]
        cfg=RKSConfig(xc_spec='hf',jk_backend=self.jk_backend,max_cycle=150,
                      conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)
        solve_kwargs=dict(overlap=s,hcore=h,eri=None,df_factors=df_factors,direct_basis=direct_basis,
            nelectron=self.nelectron,nuclear_repulsion=self.enuc,ao=jnp.zeros((0,n)),
            ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
        result,rescue_used,total_cycles=_run_scf_with_rescue(run_rks_from_integrals_traceable,
            solve_kwargs,cfg,rescue_level_shift=self.scf_rescue_level_shift)
        mo_occ=jnp.zeros(n,dtype=s.dtype).at[:self.nelectron//2].set(2.)

        def density_fixed_point(density,outputs_local):
            t_local,s_local,h_local=self._contracted_one_electron(outputs_local,ps,ph)
            df_local=self._project_df_factors(t_local,rep) if self.jk_backend=='df' else None
            j_local,k_local=self._jk(t_local,density,rep,df_local)
            x_local=_orthogonalizer(s_local,cfg.orthogonalization_eps)
            _,coeff_local=_diagonalize_fock(h_local+j_local-.5*k_local,x_local)
            return _build_density_from_occ(coeff_local,mo_occ)

        density=implicit_fixed_point_solution(
            outputs,
            solution=result.density_matrix,
            fixed_point=density_fixed_point,
            config=ImplicitFixedPointConfig(tolerance=self.implicit_tolerance,max_iter=100,restart=40),
            converged=result.converged,
            require_converged=True,
        )
        j,k=self._jk(t,density,rep,df_factors)
        fock=h+j-.5*k
        energy=jnp.sum(density*h)+.5*jnp.sum(density*j)-.25*jnp.sum(density*k)+self.enuc
        residual=fock@density@s-s@density@fock
        _,raw_coeff=_diagonalize_fock(fock,_orthogonalizer(s,cfg.orthogonalization_eps))
        fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(raw_coeff,mo_occ)-density)
        info=dict(converged=result.converged,scf_cycles=result.cycles,
                  rescue_used=rescue_used,total_scf_cycles=total_cycles,
                  fixed_point_residual=fixed_point_residual,
                  orbital_residual=jnp.linalg.norm(residual),
                  min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],
                  reconstruction_error=jnp.abs(energy-result.total_energy))
        return energy,info

    @staticmethod
    def _checked_info(energy,info):
        row={key:bool(value) if key in {'converged','rescue_used','stable','stability_resolved'} else
             int(value) if key in {'scf_cycles','total_scf_cycles','stability_restarts','stability_scf_cycles'} else float(value)
             for key,value in info.items()}
        if not np.isfinite(energy) or not np.isfinite([
            row['orbital_residual'],row['min_overlap_eigenvalue'],row['reconstruction_error'],
            row['fixed_point_residual']
        ]).all():
            raise RuntimeError('Nonfinite energy or SCF diagnostics')
        if not row['converged'] or row['orbital_residual']>1e-7:
            raise RuntimeError(f'SCF is not stationary: {row}')
        if 'stable' in row and not row['stable']:
            raise RuntimeError(f'SCF stability is unresolved or unstable: {row}')
        if row['fixed_point_residual']>1e-7:
            raise RuntimeError(f'SCF density is not a raw Aufbau fixed point: {row}')
        if row['min_overlap_eigenvalue']<1e-8 or row['reconstruction_error']>1e-9:
            raise RuntimeError(f'Invalid basis or primitive reconstruction: {row}')
        return row

    def evaluate_value(self,outputs):
        """Reconverge SCF and validate its energy without executing backward."""
        energy,info=self._value(jnp.asarray(outputs),self.ps,self.ph,self.rep)
        return float(energy),self._checked_info(energy,info)

    def evaluate(self,outputs):
        (energy,info),gradient=self._value_grad(jnp.asarray(outputs),self.ps,self.ph,self.rep)
        row=self._checked_info(energy,info)
        if not np.isfinite(gradient).all():
            raise NonfiniteGradientError(outputs,gradient,energy,row)
        return float(energy),gradient,row

def embedded_outputs(layout,atom_shells):
    """Embed saved raw contractions in a larger nested primitive pool."""
    if len(atom_shells)!=len(layout.symbols):raise ValueError('Warm-start atom count differs.')
    blocks=[s for shells in atom_shells for s in shells]
    if len(blocks)!=len(layout.roles):raise ValueError('Warm-start shell count differs.')
    out=layout.reference_outputs()
    for i,(block,atom,slot,l,a,c) in enumerate(zip(blocks,layout.shell_atoms,layout.slots,
            layout.topology.angular_momenta,layout.parameters.exponents,layout.parameters.coefficients)):
        rows=np.asarray(block[1:]);n=len(rows)
        if block[0]!=l or rows.shape[1]!=1+c.shape[1] or n>len(a):raise ValueError('Warm-start shell topology differs.')
        np.testing.assert_allclose(rows[:,0],np.asarray(a[:n]),atol=0,rtol=0)
        if slot<0:
            np.testing.assert_allclose(rows[:,1:],c,atol=1e-14,rtol=0)
        else:
            vector=jnp.pad(jnp.asarray(rows[:,1]),(0,layout.max_primitives-n))
            out=out.at[atom,slot].set(vector/jnp.linalg.norm(vector))
    return out
