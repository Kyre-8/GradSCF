"""Periodic reference adapter for the shared Sample/Trainer, without an optimizer."""
from dataclasses import dataclass, fields, replace
import jax
import jax.numpy as jnp
from ..integrals.periodic.fft import PeriodicInputs
from .functional import density_inputs


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Reference:
    """Immutable array snapshot; sampling, populations and temperature are static.

    Energy labels denote internal energy per cell in Eh, including at finite
    temperature. SCF minimizes free energy, but the training observable remains E.
    """
    inputs: object
    result: object
    mesh: tuple
    nelec: tuple
    temperature: float | None
    exxdiv: str | None

    def tree_flatten(self):
        return (self.inputs,self.result),(self.mesh,self.nelec,self.temperature,self.exxdiv)

    @classmethod
    def tree_unflatten(cls,metadata,children):
        return cls(*children,*metadata)

    @classmethod
    def from_scf(cls,mf):
        from .scf import validate_reference
        validate_reference(mf,integer=False,equal_weights=False,external=True)
        return cls(mf.inputs,mf.result,tuple(mf.cell.mesh),tuple(mf.cell.nelec),
                   mf.temperature,mf.exxdiv)

    def __getattr__(self,name):
        # Expose the existing integral/feature payload, without copying fields
        # into a second representation or inventing molecular ERI attributes.
        if name in PeriodicInputs._fields:
            return getattr(object.__getattribute__(self,'inputs'),name)
        raise AttributeError(name)

    @property
    def rdm1(self):return self.result.density_spin

    @property
    def grid(self):return self.inputs

    @property
    def electron_count(self):return sum(self.nelec)

    @property
    def natm(self):return self.inputs.atom_coords.shape[0]

    def density(self):
        return density_inputs(self.inputs,self.rdm1).rho_spin.T

    def energy(self,params,functional):
        from ._kpoint import energy_and_fock
        return energy_and_fock(self.rdm1,self.inputs,mesh=self.mesh,
            functional=functional,params=params,exxdiv=self.exxdiv)[0]

    def validate_loss(self,config):
        allowed={'e0_total_mse_weight','e0_total_mae_weight',
                 'grid_density_mse_weight','self_consistent_e0_weight'}
        unsupported=[f.name for f in fields(config) if f.name.endswith('_weight')
                     and getattr(config,f.name)!=0 and f.name not in allowed]
        if unsupported:
            raise NotImplementedError('Periodic training supports energy and grid density losses; '
                                      f'unsupported: {", ".join(unsupported)}')

    def resolve(self,params,functional,config):
        from ..dft import Functional
        from ..scf.differentiable import DifferentiableSCFInfo
        from ._kpoint import run_kpoint_scf
        if not isinstance(functional,Functional):
            raise TypeError('Periodic training requires an external dft.Functional.')
        state=self
        if config.mode=='self_consistent':
            defaults=type(config)()
            for name in ('convergence_metric','iterate_selection','orthogonalization_eps',
                         'eigenvalue_jitter','vxc_clip'):
                if getattr(config,name)!=getattr(defaults,name):
                    raise NotImplementedError(f'Periodic training does not support changing {name}.')
            result=run_kpoint_scf(self.inputs,mesh=self.mesh,nelec=self.nelec,
                functional=functional,params=params,temperature=self.temperature,
                exxdiv=self.exxdiv,init_density=self.rdm1,max_cycle=config.max_cycle,
                damping=config.damping,level_shift=config.level_shift,
                conv_tol=config.energy_convergence_tolerance(),
                conv_tol_density=config.conv_tol_density,conv_tol_grad=config.conv_tol_grad,
                differentiation=config.differentiation_config())
            state=replace(self,result=result)
        result=state.result
        mapped=jnp.einsum('skpi,ski,skqi->skpq',result.mo_coeff_spin,
                          result.mo_occ_spin,result.mo_coeff_spin.conj())
        rms=jnp.sqrt(jnp.mean(jnp.abs(mapped-state.rdm1)**2))
        info=DifferentiableSCFInfo(mode=config.mode,converged=result.converged,
            cycles=result.cycles,final_rms_density=rms,
            rms_density_history=jnp.empty((0,),dtype=rms.dtype),
            selected_cycle=jnp.maximum(result.cycles-1,0),selected_rms_density=rms,
            best_cycle=jnp.asarray(-1),best_rms_density=jnp.asarray(jnp.nan,dtype=rms.dtype))
        return state,info
