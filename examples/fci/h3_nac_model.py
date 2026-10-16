"""Small no-HF H3 model shared by the X13 native example and its diagnostics."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import fci, gto, integrals, scf
from gradscf.integrals.molecular.ao2mo import transform_integrals
from gradscf.solvers import EigenSolverConfig, EigenResponseConfig, LinearSolverConfig


class H3FCI:
    """Three spatial AOs, (2 alpha, 1 beta), all nine determinants active."""

    def __init__(self, side=1.8):
        self.center = side*jnp.array([[0.,1/np.sqrt(3),0.],[-.5,-.5/np.sqrt(3),0.],
                                     [.5,-.5/np.sqrt(3),0.]])
        self.modes = jnp.stack([self.center@jnp.diag(jnp.array([1.,-1.,0.]))/side,
            self.center@jnp.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])/side])
        mol = gto.M(atom=list(zip(['H']*3,np.asarray(self.center))),basis='sto-3g',
                    unit='Bohr',spin=1,cart=True)
        self.topology,self.parameters = integrals.prepare_basis(mol.to_spec(),mol.basis)
        self.owners = np.argmin(np.linalg.norm(np.asarray(self.parameters.centers)[:,None]
                                               - np.asarray(self.center)[None,:],axis=-1),axis=1)
        self.plan = integrals.make_plan(self.topology,backend='native')
        self.space = fci.make_fci_space(self.topology.nao,(2,1))
        assert self.space.norb == 3 and self.space.size == 9
        self.config = EigenSolverConfig(method='dense',nroots=2,max_dense=16,atol=1e-12)
        linear = LinearSolverConfig(method='direct',rtol=1e-10,atol=1e-12,max_dense=32)
        self.eigenpairs = EigenResponseConfig(target='eigenpairs',linear_config=linear)
        self.subspace = EigenResponseConfig(target='subspace',linear_config=linear)
        # The existing doubled-basis native overlap construction is independent
        # of electron count; no artificial closed-shell HF object is needed.
        doubled = integrals.BasisTopology(self.topology.angular_momenta*2,
            self.topology.primitive_counts*2,self.topology.contraction_counts*2,
            self.topology.nuclear_charges,self.topology.cart)
        self.cross_plan = integrals.make_plan(doubled,backend='native')
        self.determinants = jnp.eye(self.space.size).reshape((self.space.size,)+self.space.shape)

    def coordinates(self,q):
        return self.center+jnp.einsum('q,qax->ax',q,self.modes)

    def data(self,coordinates,rotation):
        moving = replace(self.parameters,centers=coordinates[self.owners],nuclear_coords=coordinates)
        overlap = self.plan.evaluate('overlap',moving)
        frame = jax.scipy.linalg.solve_triangular(jnp.linalg.cholesky(overlap).T,
                                                 jnp.eye(3),lower=False)@rotation
        hcore = self.plan.evaluate('kinetic',moving)+self.plan.evaluate('nuclear',moving)
        h,g = transform_integrals(hcore,frame,eri=self.plan.evaluate('eri',moving))
        nuclear = scf.nuclear_repulsion_energy(coordinates,jnp.asarray(self.topology.nuclear_charges))
        return h,g,nuclear,frame

    def states(self,coordinates,rotation):
        h,g,nuclear,frame = self.data(coordinates,rotation)
        result = fci.solve_fci(h,g,self.space,ecore=nuclear,config=self.config,response=self.eigenpairs)
        return dict(energies=result.total_energies,coefficients=result.coefficients,frame=frame,
            coordinates=coordinates,converged=result.converged,response_valid=result.response_valid,
            residual_norms=result.residual_norms)

    def bundle(self,coordinates,rotation):
        h,g,nuclear,frame = self.data(coordinates,rotation)
        result = fci.solve_fci(h,g,self.space,ecore=nuclear,config=self.config,response=self.subspace,
                               probes=jnp.eye(self.space.size))
        return dict(energy_sum=result.energy_sum,projection=result.projection,frame=frame,
            coordinates=coordinates,converged=result.converged,response_valid=result.response_valid)

    def cross_overlap(self,left,right):
        parameters = integrals.BasisParameters(self.parameters.exponents*2,
            self.parameters.coefficients*2,jnp.concatenate([left[self.owners],right[self.owners]]),left)
        return self.cross_plan.evaluate('overlap',parameters)[:3,3:]

    def orbital_map(self,left,right):
        return left['frame'].T@self.cross_overlap(left['coordinates'],right['coordinates'])@right['frame']

    def state_overlaps(self,left,right):
        matrix = self.orbital_map(left,right)
        return jax.vmap(lambda bra:jax.vmap(lambda ket:fci.overlap(bra,ket,self.space,
            orbital_overlap=matrix))(right['coefficients']))(left['coefficients'])

    def properties(self,coordinates,rotation):
        anchor = jax.tree.map(jax.lax.stop_gradient,self.states(coordinates,rotation))
        nac = jax.jacfwd(lambda right:self.state_overlaps(anchor,self.states(right,rotation)))(coordinates)
        # Diagnostic partial: its missing orbital contribution is not called NAC.
        ci_only = jax.jacfwd(lambda right:jnp.einsum('iab,jab->ij',anchor['coefficients'],
                               self.states(right,rotation)['coefficients']))(coordinates)
        return dict(anchor,nac=nac,ci_only_partial=ci_only)

    def determinant_map(self,matrix):
        return jax.vmap(lambda bra:jax.vmap(lambda ket:fci.overlap(bra,ket,self.space,
            orbital_overlap=matrix))(self.determinants))(self.determinants)

    def transported_bundle(self,anchor,coordinates,rotation):
        current = self.bundle(coordinates,rotation)
        transport = self.determinant_map(self.orbital_map(anchor,current))
        # Projected into the fixed nine-determinant anchor space. This is not
        # a claim about the full infinite-electronic-Hilbert-space projector.
        # Away from the anchor, transport need not be orthogonal and this
        # compression need not be idempotent.
        return current['energy_sum'],transport@current['projection']@transport.T
