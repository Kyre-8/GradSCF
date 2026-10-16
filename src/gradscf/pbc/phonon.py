"""Periodic DFPT phonons with primitive and independent supercell paths.

Geometry sources and force derivatives use AD, not finite displacement. The
default retains the bounded supercell reference; method='primitive' uses
primitive HF/KS geometry and electronic response. The supercell reference
requires odd replication; primitive response requires a closed equal-weight
k mesh. Frequencies are signed hbar*omega
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
from ..dfpt.displacement import DisplacementSource,displacement_source
from ..solvers.operators import as_operator
from ..solvers.eigen.dense import dense_vectors


class PhononResult(NamedTuple):
    force_constants: object
    dynamical_matrix: object
    frequencies: object
    modes: object
    residual_norm: object
    hermiticity_error: object
    asr_residual: object
    converged: object


class ProjectedPhononResult(NamedTuple):
    force_constants: object
    mass_matrix: object
    frequencies: object
    modes: object
    residual_norm: object
    subspace_residual_norm: object
    hermiticity_error: object
    converged: object


def _project_translations(phi):
    """Orthogonal Gamma ASR projection in Cartesian, unweighted coordinates."""
    nat=len(phi)//3
    translations=jnp.tile(jnp.eye(3,dtype=phi.dtype),(nat,1))/jnp.sqrt(nat)
    projector=jnp.eye(len(phi),dtype=phi.dtype)-translations@translations.T
    return projector@phi@projector


def _modes(force_constants,masses,asr):
    phi=jnp.asarray(force_constants)
    masses=np.asarray(masses,dtype=float)
    if masses.ndim!=1 or phi.shape!=(3*len(masses),)*2 or not np.isfinite(masses).all() or np.any(masses<=0):
        raise ValueError('Supply one finite positive mass in amu per primitive-cell atom')
    phi=.5*(phi+phi.conj().T)
    if asr:
        phi=_project_translations(phi)
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

    Supercell KS replication must match a Gamma-centered full k mesh. OFDFT
    accepts an explicit odd supercell. Primitive HF/KS needs no replication.
    Arbitrary external functional finite-q extensions are not inferred.
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

    def displacement(self,q,displacements,*,method='primitive'):
        """Fixed-D sources; independent supercell AD remains an explicit oracle."""
        self.super_reference=None
        if method=='primitive':return displacement_source(self.source,q,displacements)
        if method!='supercell':raise ValueError('Use primitive or supercell displacement sources')
        return self._supercell_displacement(q,displacements)

    def mesh(self,qmesh,*,asr=False,config=None,method='primitive'):
        """Sample a complete q mesh and return Fourier-interpolable force constants.

        qmesh must be commensurate with the reference k mesh (or the explicit
        OFDFT supercell). Raw matrices are collected before optional Gamma ASR
        projection in ForceConstants, preserving its diagnostic residual.
        """
        from .ifc import ForceConstants
        mesh=np.asarray(qmesh)
        if mesh.shape!=(3,) or not np.isfinite(mesh).all() or np.any(mesh<1) or np.any(mesh!=mesh.astype(int)):
            raise ValueError('q mesh must contain three positive integers')
        mesh=mesh.astype(int)
        if method not in ('primitive','supercell'):raise ValueError('Use primitive or supercell phonon response')
        if isinstance(self.source,_SCF):
            validate_reference(self.source,integer=False,equal_weights=True)
            cell=self.source.cell;reference_mesh=np.asarray(kmesh_shape(cell.lattice,self.source.kpts))
        else:
            cell=self.source.system
            reference_mesh=np.asarray((1,1,1) if self.supercell is None else self.supercell)
            if method=='primitive' and np.any(mesh!=1):
                raise NotImplementedError('Primitive finite-q OFDFT phonons are not available')
        if np.any(reference_mesh%mesh):raise ValueError('q mesh must be commensurate with the reference sampling')
        qpoints=np.asarray(list(itertools.product(*[np.arange(n)/n for n in mesh])))
        matrices=jnp.stack([self.kernel(q,config=config,method=method).force_constants for q in qpoints])
        return ForceConstants.from_qmesh(cell,qpoints,matrices,masses=self.masses,asr=asr)

    def _supercell_displacement(self,q,displacements):
        """Independent AD oracle retaining the bounded full supercell."""
        from ..integrals.periodic.fft import build_inputs
        from ._kpoint import energy_and_fock
        from .functional import density_inputs
        canonical=self._q(q)
        q=np.asarray(q,dtype=float)
        if not isinstance(self.source,_SCF) or np.allclose(canonical,0.):
            raise NotImplementedError('Use atomic_problem for Gamma or OFDFT displacement response')
        cell,repetitions,indices,b=self._prepare(canonical)
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

    def _force_constant_action(self, q, config, method):
        """Build one reference snapshot, shared by full and projected modes."""
        self.result=None;self.super_reference=None;self.reference_forces=None
        if method not in ('primitive','supercell'):raise ValueError('Use primitive or supercell phonon response')
        q=self._q(q);gamma=np.allclose(q,0.)
        source=self.source;cell=source.cell if isinstance(source,_SCF) else source.system
        if len(self.masses)!=len(cell.coords):raise ValueError('One mass per primitive-cell atom is required')
        primitive=method=='primitive' and not gamma
        if primitive:
            if not isinstance(source,_SCF):
                raise NotImplementedError('Primitive phonons currently require HF/KS')
            from ..dfpt.force_constants import primitive_force_response
            self._limits(cell,(1,1,1),nk=len(source.kpts))
            self.reference_forces,respond=primitive_force_response(source,q,config)
            indices=np.zeros((1,3),dtype=int)
        elif gamma:
            self._limits(cell,(1,1,1),nk=len(source.kpts) if isinstance(source,_SCF) else 1)
            reference=source;indices=np.zeros((1,3),dtype=int)
        else:
            _,_,indices,_=self._prepare(q);reference=self.super_reference
        if not primitive:
            problem=atomic_problem(reference)
            self.reference_forces=problem.quantities(problem.solution,problem.inputs)['forces']
            respond=jax.jit(lambda direction:problem.response(tangent=direction,config=config))
        phase=jnp.exp(2j*jnp.pi*(jnp.asarray(indices)@q));count=len(indices)
        nat=len(cell.coords)
        def action(direction):
            if primitive:
                column,residual,converged=respond(direction)
                if not bool(converged):raise RuntimeError('Atomic q response did not converge')
                return column, residual
            else:
                pattern=(phase[:,None,None]*direction[None]).reshape(-1,3)
                real=respond(pattern.real)
                if not bool(real.converged):raise RuntimeError('Atomic response did not converge')
                response=real.values['forces'].astype(complex)
                residual=real.residual_norm
                if not gamma:
                    imag=respond(pattern.imag)
                    if not bool(imag.converged):raise RuntimeError('Atomic response did not converge')
                    response+=1j*imag.values['forces']
                    residual=jnp.maximum(residual,imag.residual_norm)
            column=-jnp.einsum('r,rai->ai',phase.conj()/count,response.reshape(count,nat,3))
            return column.ravel(), residual
        return action

    def kernel(self,q=(0.,0.,0.),*,asr=False,config=None,method='supercell'):
        """Force constants and numerical modes; ASR projection is opt-in at Gamma.

        'primitive' constructs HF/KS geometry and response in the primitive cell;
        'supercell' is the independent reference and also supports OFDFT.
        """
        self.result=None;self.super_reference=None;self.reference_forces=None
        q=self._q(q);gamma=np.allclose(q,0.)
        if asr and not gamma:raise ValueError('ASR projection is only defined here at Gamma')
        action=self._force_constant_action(q,config,method)
        nat=len(self.masses);columns=[];residuals=[]
        for direction in jnp.eye(3*nat).reshape(3*nat,nat,3):
            column,residual=action(direction)
            columns.append(column);residuals.append(residual)
        phi=jnp.stack(columns,axis=1)
        error=jnp.linalg.norm(phi-phi.conj().T)
        if not np.isfinite(phi).all() or float(error)>1e-6*max(1.,float(jnp.linalg.norm(phi))):
            raise RuntimeError('Force constants fail the Hermiticity check; refine the FFT mesh or tighten convergence')
        translations=jnp.tile(jnp.eye(3),(nat,1))
        asr_error=jnp.linalg.norm(phi@translations) if gamma else None
        dynamical,frequencies,vectors=_modes(phi,self.masses,asr)
        self.result=PhononResult(phi,dynamical,frequencies,vectors,jnp.max(jnp.stack(residuals)),error,asr_error,True)
        return self.result

    def project(self, directions, q=(0.,0.,0.), *, config=None, method='primitive'):
        """Solve only the requested real Cartesian displacement directions.

        directions has shape (nvector,natom,3); rows must be independent but
        need not be normalized. Solve B† Phi B v = omega² B† M B v with M in
        electron masses. Returned modes are full Cartesian mass-weighted unit
        vectors; signed frequencies are hbar*omega in Hartree as in kernel().
        Each response is solved for a unit direction and rescaled afterwards,
        so arbitrary input amplitudes cannot weaken the absolute solver gate.

        This is an eager Rayleigh--Ritz projection, not a new eigenderivative
        contract. subspace_residual_norm measures the full dynamical residual
        (Hartree²): a projected eigenmode is an exact physical mode only when
        the subspace is invariant. No acoustic-sum-rule projection is applied.
        """
        self.result=None;self.super_reference=None;self.reference_forces=None
        basis=np.asarray(directions)
        if (np.ndim(self.masses)!=1 or not np.isfinite(self.masses).all()
                or np.any(np.asarray(self.masses)<=0)):
            raise ValueError('Supply finite positive masses in amu')
        nat=len(self.masses)
        if (basis.ndim!=3 or basis.shape[1:]!=(nat,3) or not len(basis)
                or len(basis)>3*nat or np.iscomplexobj(basis)
                or not np.isfinite(basis).all()):
            raise ValueError('directions must be finite real (nvector,natom,3) arrays')
        basis=basis.reshape(len(basis),-1).astype(float)
        if np.linalg.matrix_rank(basis)!=len(basis):
            raise ValueError('Projected directions must be linearly independent')
        ratio=physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
        mass=jnp.repeat(jnp.asarray(self.masses)*ratio,3)
        metric=jnp.asarray(basis)@(mass[:,None]*jnp.asarray(basis.T))
        if not np.isfinite(metric).all() or np.linalg.eigvalsh(metric).min()<=0:
            raise ValueError('Projected directions have an unrepresentable mass metric')
        if np.linalg.cond(metric)>1e12:
            raise ValueError('Projected directions have an ill-conditioned mass metric')
        whiten=jnp.linalg.inv(jnp.linalg.cholesky(metric)).T
        if not np.isfinite(whiten).all():
            raise ValueError('Projected mass metric whitening is not finite')
        lengths=np.linalg.norm(basis,axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths<=0):
            raise ValueError('Projected directions have unrepresentable lengths')
        action=self._force_constant_action(q,config,method)
        columns=[];residuals=[]
        for direction,length in zip(basis.reshape(-1,nat,3),lengths):
            column,residual=action(jnp.asarray(direction/length))
            columns.append(column*length);residuals.append(residual*length)
        actions=jnp.stack(columns,axis=1)
        basis=jnp.asarray(basis.T)
        phi=basis.T@actions
        error=jnp.linalg.norm(phi-phi.conj().T)
        # Test Hermiticity before symmetrizing, in coordinates independent of
        # the user's arbitrary basis amplitudes. The returned error remains
        # the raw projected force-constant norm for diagnostic consistency.
        matrix=whiten.T@phi@whiten
        matrix_error=jnp.linalg.norm(matrix-matrix.conj().T)
        if (not all(np.isfinite(value).all() for value in (phi,error,matrix,matrix_error))
                or float(matrix_error)>1e-6*max(1e-12,float(jnp.linalg.norm(matrix)))):
            raise RuntimeError('Projected force constants fail the Hermiticity check')
        matrix=.5*(matrix+matrix.conj().T)
        vectors=dense_vectors(as_operator(matrix),nroots=len(phi),max_dense=len(phi))
        values=jnp.sum(vectors.conj()*(matrix@vectors),axis=0).real
        modes=jnp.sqrt(mass)[:,None]*(basis@whiten@vectors)
        full_action=(actions@whiten@vectors)/jnp.sqrt(mass)[:,None]
        leakage=jnp.linalg.norm(full_action-modes*values)
        eigen_error=jnp.linalg.norm(matrix@vectors-vectors*values)
        if (not all(np.isfinite(v).all() for v in (values,modes,leakage,eigen_error))
                or float(eigen_error)>1e-8*max(1.,float(jnp.linalg.norm(matrix)))):
            raise RuntimeError('Projected phonon eigensolve failed its residual check')
        self.result=ProjectedPhononResult(phi,metric,jnp.sign(values)*jnp.sqrt(jnp.abs(values)),
            modes,jnp.max(jnp.stack(residuals)),leakage,error,True)
        return self.result
