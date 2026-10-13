"""Bounded commensurate-supercell DFPT phonons using shared atomic response.

Geometry sources and force derivatives use AD, not finite displacement. The
supercell response space is retained: this is a small-system reference path,
not a scalable primitive-cell phonon implementation. Odd diagonal replication
preserves the current symmetric FFT grid. Frequencies are signed hbar*omega
in Eh (negative means an imaginary mode); masses are supplied in atomic mass
units. Mode diagonalization is an eager numerical postprocessing operation.
"""
from dataclasses import replace
from typing import NamedTuple
import itertools
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants
from . import gto
from .scf import _SCF,validate_reference
from .kpoints import kmesh_shape
from .momentum import momentum_transfer,wrap_to_kmesh
from ..dfpt.atomic import atomic_problem
from ..dfpt.finite_q import QSource
from ..solvers.operators import as_operator
from ..solvers.eigen.dense import dense_vectors


class DisplacementSource(NamedTuple):
    perturbation: QSource
    hcore: object


class PhononResult(NamedTuple):
    force_constants: object
    dynamical_matrix: object
    frequencies: object
    modes: object
    residual_norm: object
    hermiticity_error: object
    asr_residual: object
    converged: object


def _modes(force_constants,masses,asr):
    phi=jnp.asarray(force_constants)
    masses=np.asarray(masses,dtype=float)
    if masses.ndim!=1 or phi.shape!=(3*len(masses),)*2 or not np.isfinite(masses).all() or np.any(masses<=0):
        raise ValueError('Supply one finite positive mass in amu per primitive-cell atom')
    phi=.5*(phi+phi.conj().T)
    if asr:
        translations=jnp.tile(jnp.eye(3),(len(masses),1))/jnp.sqrt(len(masses))
        projector=jnp.eye(len(phi))-translations@translations.T
        phi=projector@phi@projector
    ratio=physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
    scale=jnp.repeat(jnp.sqrt(jnp.asarray(masses)*ratio),3)
    dynamical=phi/scale[:,None]/scale[None,:]
    if not np.isfinite(dynamical).all():raise ValueError('Mass-weighted dynamical matrix is not finite')
    vectors=dense_vectors(as_operator(dynamical),nroots=len(phi),max_dense=len(phi))
    values=jnp.sum(vectors.conj()*(dynamical@vectors),axis=0).real
    error=jnp.linalg.norm(dynamical@vectors-vectors*values)
    if not np.isfinite(values).all() or float(error)>1e-8*max(1.,float(jnp.linalg.norm(dynamical))):
        raise RuntimeError('Phonon eigensolve failed its residual check')
    return dynamical,jnp.sign(values)*jnp.sqrt(jnp.abs(values)),vectors


def _replica(cell,repetitions):
    indices=np.array(list(itertools.product(*[range(n) for n in repetitions])))
    translations=jnp.asarray(indices)@cell.lattice
    coords=(cell.coords[None]+translations[:,None]).reshape(-1,3)
    symbols=cell.spec.symbols*len(indices)
    replica=gto.M(atom=list(zip(symbols,np.asarray(coords))),a=np.diag(repetitions)@cell.lattice,
        basis=cell.basis_spec,pseudo=dict(zip(cell.spec.symbols,cell.pseudopotentials)),
        unit='Bohr',mesh=tuple(np.asarray(cell.mesh)*repetitions),spin=cell.spin*len(indices),precision=cell.precision)
    # Preserve the supplied raw exponents/contractions rather than reloading a
    # named basis; every copy retains the same atom-attached shell topology.
    p=cell.parameters
    top=cell.topology
    replica.topology=replace(top,angular_momenta=top.angular_momenta*len(indices),
        primitive_counts=top.primitive_counts*len(indices),contraction_counts=top.contraction_counts*len(indices),
        nuclear_charges=top.nuclear_charges*len(indices))
    replica.parameters=replace(p,exponents=p.exponents*len(indices),coefficients=p.coefficients*len(indices),
        centers=(p.centers[None]+translations[:,None]).reshape(-1,3),nuclear_coords=coords)
    replica._shell_atoms=np.concatenate([cell._shell_atoms+i*len(cell.coords) for i in range(len(indices))])
    return replica,indices,translations


class Phonons:
    """Small-cell reference phonons for periodic HF/KS and Cell-based OFDFT.

    KS replication must match a Gamma-centered full k mesh. OFDFT accepts an
    explicit odd supercell. Arbitrary external model replication is not inferred.
    Every kernel call builds a fresh snapshot; no stale response cache is reused.
    """
    def __init__(self,source,*,masses,supercell=None,max_atoms=24,max_pair_elements=2_000_000,max_grid=20_000):
        self.source=source;self.masses=np.asarray(masses,dtype=float)
        if self.masses.ndim!=1 or not np.isfinite(self.masses).all() or np.any(self.masses<=0):
            raise ValueError('masses must be a finite positive vector in amu')
        if any(not np.isfinite(v) or v<1 for v in (max_atoms,max_pair_elements,max_grid)):
            raise ValueError('Phonon workspace limits must be finite and positive')
        self.supercell=supercell;self.max_atoms=max_atoms;self.max_pair_elements=max_pair_elements;self.max_grid=max_grid
        self.result=None;self.super_reference=None

    def _limits(self,cell,repetitions,*,nk=1):
        count=int(np.prod(repetitions));grid=count*int(np.prod(cell.mesh));nao=count*cell.topology.nao
        if count*len(cell.coords)>self.max_atoms or grid>self.max_grid:
            raise ValueError('System exceeds the bounded phonon atom/grid limits')
        gaussian=isinstance(self.source,_SCF) or self.source.inputs.representation=='periodic_gaussian'
        if gaussian and grid*nao**2*nk**2>self.max_pair_elements:
            raise ValueError('System exceeds the bounded phonon AO-pair workspace limit')

    def _prepare(self,q):
        from ..ofdft.api import OFDFT
        from ..ofdft.kinetic import KineticFunctional
        self.super_reference=None
        source=self.source
        if isinstance(source,_SCF):
            validate_reference(source,integer=False,equal_weights=True)
            cell=source.cell;mesh=kmesh_shape(cell.lattice,source.kpts)
            repetitions=np.asarray(mesh if self.supercell is None else self.supercell)
            scaled=np.asarray(source.kpts)@np.asarray(cell.lattice).T/(2*np.pi)
            if tuple(repetitions)!=mesh or not np.allclose(scaled*repetitions,np.rint(scaled*repetitions),atol=1e-9):
                raise ValueError('KS supercell must match the Gamma-centered reference k mesh')
        elif isinstance(source,OFDFT) and hasattr(source.system,'a'):
            if source.result is None or not source.converged:raise ValueError('Converge OFDFT before phonons')
            if not isinstance(source.kinetic,KineticFunctional) or source.xc_energy_fn is not None:
                raise NotImplementedError('External models need an explicit supercell extension')
            cell=source.system;repetitions=np.asarray((1,1,1) if self.supercell is None else self.supercell)
        else:raise TypeError('Phonons requires periodic HF/KS or Cell-based OFDFT')
        if repetitions.shape!=(3,) or np.any(repetitions<1) or np.any(repetitions!=repetitions.astype(int)):
            raise ValueError('supercell must contain three positive integers')
        repetitions=repetitions.astype(int)
        if np.any(repetitions%2==0):raise NotImplementedError('Reference phonons currently require odd supercell repetitions')
        if not np.allclose(q*repetitions,np.rint(q*repetitions),atol=1e-9):
            raise ValueError('q must be commensurate with the supercell')
        count=int(np.prod(repetitions));self._limits(cell,repetitions)
        if isinstance(source,OFDFT) and not bool(atomic_problem(source)._base()[1]):
            raise ValueError('OFDFT reference is stale or nonstationary; run it again')
        sup,indices,translations=_replica(cell,repetitions)
        if isinstance(source,_SCF):
            from . import dft
            cls=dft.UKS if source.unrestricted else dft.RKS
            reference=cls(sup,xc=source.xc,temperature=source.temperature,exxdiv=source.exxdiv,**source.controls)
            b=jnp.kron(jnp.exp(1j*translations@source.kpts.T)/jnp.sqrt(count),jnp.eye(cell.topology.nao))
            folded=jnp.stack([b@jax.scipy.linalg.block_diag(*d)@b.conj().T for d in source.result.density_spin])
            if np.max(np.abs(np.asarray(folded.imag)))>1e-9:
                raise NotImplementedError('This Gamma supercell path requires a real reference density')
            dm=folded.real[:,None]
            reference.kernel(dm0=dm if source.unrestricted else dm.sum(axis=0))
            if not reference.converged or not np.allclose(reference.result.density_spin[:,0],folded,atol=1e-6,rtol=1e-6):
                raise RuntimeError('Supercell and primitive references differ; refine the FFT mesh or check the SCF branch')
        else:
            cfg=source.config
            reference=OFDFT(sup,kinetic=source.kinetic,xc=cfg.xc,kinetic_params=source.kinetic_params,
                representation=source.inputs.representation,maxiter=cfg.maxiter,tolerance=cfg.tolerance,differentiation=cfg.differentiation)
            initial=source.result.coefficients
            initial=(jnp.tile(initial.reshape(cell.mesh),tuple(repetitions)).ravel()
                     if source.inputs.representation=='periodic' else jnp.tile(initial,count))
            reference.run(initial=initial)
            expected=jnp.tile(source.result.density.reshape(cell.mesh),tuple(repetitions)).ravel()
            if not np.allclose(reference.result.density,expected,atol=1e-6,rtol=1e-6):
                raise RuntimeError('Supercell density differs from the translated OFDFT reference')
            b=None
        if not reference.converged:raise RuntimeError('Supercell reference did not converge')
        self.super_reference=reference
        return cell,repetitions,indices,b

    @staticmethod
    def _q(q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(3,) or not np.isfinite(q).all():raise ValueError('q must be a finite fractional three-vector')
        return wrap_to_kmesh(q)

    def displacement(self,q,displacements):
        """AD fixed-D sources for the primitive finite_q_problem (nonzero q)."""
        from ..integrals.periodic.fft import build_inputs
        from ._kpoint import energy_and_fock
        from .functional import density_inputs
        q=self._q(q)
        if not isinstance(self.source,_SCF) or np.allclose(q,0.):
            raise NotImplementedError('Use atomic_problem for Gamma or OFDFT displacement response')
        cell,repetitions,indices,b=self._prepare(q)
        direction=jnp.asarray(displacements)
        if direction.shape!=cell.coords.shape or jnp.iscomplexobj(direction):
            raise ValueError('displacements must be real (natom,3) amplitudes in Bohr')
        ref=self.super_reference;sup=ref.cell;d=ref.result.density_spin
        phase=jnp.exp(2j*jnp.pi*(jnp.asarray(indices)@q))
        pattern=(phase[:,None,None]*direction[None]).reshape(-1,3)
        def fields(r):
            data=build_inputs(sup,parameters=sup.parameters_at(r))
            f=energy_and_fock(d,data,mesh=sup.mesh,xc=ref.xc,exxdiv=ref.exxdiv)[1]
            return data.hcore[0],data.overlap[0],f[:,0],density_inputs(data,d).rho_spin
        cosine=jax.jvp(fields,(sup.coords,),(pattern.real,))[1]
        sine=jax.jvp(fields,(sup.coords,),(pattern.imag,))[1]
        h,s,f,rho=jax.tree.map(lambda a,b:a+1j*b,cosine,sine)
        fractional=np.asarray(self.source.kpts)@np.asarray(cell.lattice).T/(2*np.pi)
        transfer=momentum_transfer(fractional,q);nk=len(indices);n=cell.topology.nao
        def fold(matrix):
            blocks=(b.conj().T@matrix@b).reshape(matrix.shape[:-2]+(nk,n,nk,n))
            return jnp.stack([blocks[...,p,:,k,:] for k,p in enumerate(transfer.partners)],axis=-3)
        nx,ny,nz=repetitions;mx,my,mz=cell.mesh
        cubes=rho.reshape(2,nx,mx,ny,my,nz,mz).transpose(0,1,3,5,2,4,6).reshape(2,nk,-1)
        envelope=jnp.einsum('r,srg->sg',phase.conj()/nk,cubes)
        wave=q@(2*jnp.pi*jnp.linalg.inv(cell.lattice).T)
        envelope*=jnp.exp(-1j*self.source.inputs.coords@wave)
        return DisplacementSource(QSource(fold(f),fold(s),envelope),fold(h))

    def kernel(self,q=(0.,0.,0.),*,asr=False,config=None):
        """Force constants and numerical modes; ASR projection is opt-in at Gamma."""
        self.result=None;self.super_reference=None;self.reference_forces=None
        q=self._q(q);gamma=np.allclose(q,0.)
        if asr and not gamma:raise ValueError('ASR projection is only defined here at Gamma')
        source=self.source;cell=source.cell if isinstance(source,_SCF) else source.system
        if len(self.masses)!=len(cell.coords):raise ValueError('One mass per primitive-cell atom is required')
        if gamma:
            self._limits(cell,(1,1,1),nk=len(source.kpts) if isinstance(source,_SCF) else 1)
            reference=source;indices=np.zeros((1,3),dtype=int)
        else:
            _,_,indices,_=self._prepare(q);reference=self.super_reference
        problem=atomic_problem(reference)
        self.reference_forces=problem.quantities(problem.solution,problem.inputs)['forces']
        phase=jnp.exp(2j*jnp.pi*(jnp.asarray(indices)@q));count=len(indices)
        nat=len(cell.coords);columns=[];residuals=[]
        respond=jax.jit(lambda direction:problem.response(tangent=direction,config=config))
        for direction in jnp.eye(3*nat).reshape(3*nat,nat,3):
            pattern=(phase[:,None,None]*direction[None]).reshape(-1,3)
            real=respond(pattern.real)
            if not bool(real.converged):raise RuntimeError('Atomic response did not converge')
            response=real.values['forces'].astype(complex);residuals.append(real.residual_norm)
            if not gamma:
                imag=respond(pattern.imag)
                if not bool(imag.converged):raise RuntimeError('Atomic response did not converge')
                response+=1j*imag.values['forces'];residuals.append(imag.residual_norm)
            column=-jnp.einsum('r,rai->ai',phase.conj()/count,response.reshape(count,nat,3))
            columns.append(column.ravel())
        phi=jnp.stack(columns,axis=1)
        error=jnp.linalg.norm(phi-phi.conj().T)
        if not np.isfinite(phi).all() or float(error)>1e-6*max(1.,float(jnp.linalg.norm(phi))):
            raise RuntimeError('Force constants fail the Hermiticity check; tighten reference/response convergence')
        translations=jnp.tile(jnp.eye(3),(nat,1))
        asr_error=jnp.linalg.norm(phi@translations) if gamma else None
        dynamical,frequencies,vectors=_modes(phi,self.masses,asr)
        self.result=PhononResult(phi,dynamical,frequencies,vectors,jnp.max(jnp.stack(residuals)),error,asr_error,True)
        return self.result
