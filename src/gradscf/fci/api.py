"""Eager FCI facades and PySCF-shaped active-space solver methods."""
from dataclasses import fields, is_dataclass
import numpy as np
import jax
import jax.numpy as jnp
from gradscf._memory import DEFAULT_MAX_MEMORY_MB, array_storage_bytes, resolve_memory_mb
from ..solvers import EigenSolverConfig
from ..scf.reference import _array_signature
from .cistring import make_fci_space
from .solver import _plan_fci_memory, solve_fci
from .reference import FCIReference, orbital_selection, source_dimensions, source_signature, active_reference
from . import hamiltonian, rdm


class FCISolver:
    """Integral-in/CI-out engine for active-space methods.

    kernel(h1,eri,norb,nelec,ecore=...) returns total energies and alpha-by-beta
    CI matrices. One root is scalar/matrix; multiple roots retain a leading root
    axis. spin selects 2*Ms, not a total-S projection.
    max_memory is a decimal-MB budget shared by reference preparation, primal
    eigensolver and operator arrays. Response/AD and allocator residency are not
    a complete process-memory hard limit.
    """
    def __init__(self, mol=None, *, nroots=1, solver='davidson', conv_tol=1e-10,
                 max_cycle=100, max_space=40, max_dense=2048, spin=None,
                 gradient_mode='eigenvalue_only', adjoint_tol=1e-10,
                 adjoint_max_cycle=100, response=None, seed=0,
                 max_determinants=1_000_000, max_link_elements=4_000_000,
                 max_workspace_elements=hamiltonian.DEFAULT_WORKSPACE,
                 max_memory=DEFAULT_MAX_MEMORY_MB):
        self.mol = mol
        self.nroots,self.solver,self.conv_tol = nroots,solver,conv_tol
        self.max_cycle,self.max_space,self.max_dense = max_cycle,max_space,max_dense
        self.spin,self.gradient_mode,self.response = spin,gradient_mode,response
        self.adjoint_tol,self.adjoint_max_cycle,self.seed = adjoint_tol,adjoint_max_cycle,seed
        self.max_determinants,self.max_link_elements = max_determinants,max_link_elements
        self.max_workspace_elements = max_workspace_elements
        self.max_memory = resolve_memory_mb(max_memory)
        self._reset()

    def _reset(self):
        self.result = self.space = self.e_tot = self.ci = self.converged = None
        self._inputs = self._state_key = None

    def _space(self,norb,nelec):
        return make_fci_space(norb,nelec,spin=self.spin,max_determinants=self.max_determinants,
                              max_link_elements=self.max_link_elements)

    def _config(self):
        return EigenSolverConfig(method=self.solver,nroots=self.nroots,atol=self.conv_tol,
            maxiter=self.max_cycle,max_subspace=self.max_space if self.solver=='davidson' else None,max_dense=self.max_dense,
            gradient_mode=self.gradient_mode,adjoint_tol=self.adjoint_tol,
            adjoint_maxiter=self.adjoint_max_cycle,seed=self.seed)

    def _key(self):
        h,g,norb,nelec,ecore = self._inputs
        return (norb,tuple(nelec),self._config(),self.response,self.spin,
                self.max_determinants,self.max_link_elements,self.max_workspace_elements,self.max_memory,
                *(_array_signature(a) for a in (h,g,ecore)))

    def kernel(self,h1,eri,norb,nelec,ci0=None,ecore=0.,*,reserved_bytes=0):
        self._reset()
        self.space = self._space(norb,nelec)
        self._inputs = h1,eri,norb,self.space.nelec,ecore
        self.result = solve_fci(h1,eri,self.space,ecore=ecore,config=self._config(),
                                response=self.response,ci0=ci0,
                                max_workspace_elements=self.max_workspace_elements,
                                max_memory=self.max_memory,reserved_bytes=reserved_bytes)
        self.e_tot,self.ci = self.result.total_energies,self.result.coefficients
        self.converged = np.asarray(self.result.converged)
        if self.e_tot is not None and self.nroots == 1:
            self.e_tot,self.ci = self.e_tot[0],self.ci[0]
            self.converged = bool(self.converged[0])
        self._state_key = self._key()
        return self.e_tot,self.ci

    def run(self,*args,**kwargs):
        self.kernel(*args,**kwargs)
        return self

    def _property_inputs(self,ci,norb,nelec,root):
        if ci is not None:
            if norb is None or nelec is None:
                raise ValueError("Explicit CI vectors require norb and nelec")
            return ci,self._space(norb,nelec)
        if norb is not None or nelec is not None:
            raise ValueError("Supply both the explicit CI vector and its orbital/electron space")
        if self.result is None or self._state_key is None:
            raise RuntimeError("Run FCI before requesting state properties")
        if self._key() != self._state_key:
            raise RuntimeError("FCI inputs or configuration changed; run kernel again")
        if self.result.coefficients is None:
            raise RuntimeError("Subspace responses do not expose individual CI coefficients")
        if not isinstance(root,(int,np.integer)) or isinstance(root,bool) or not 0<=root<self.nroots:
            raise ValueError("root must select an available FCI state")
        if not bool(self.result.converged[root]):
            raise RuntimeError("Converge the FCI root before requesting state properties")
        return self.result.coefficients[root],self.space

    def make_rdm1(self,ci=None,norb=None,nelec=None,*,root=0):
        c,space = self._property_inputs(ci,norb,nelec,root)
        return rdm.make_rdm1(c,space,max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def make_rdm1s(self,ci=None,norb=None,nelec=None,*,root=0):
        c,space = self._property_inputs(ci,norb,nelec,root)
        return rdm.make_rdm1s(c,space,max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def make_rdm12(self,ci=None,norb=None,nelec=None,*,root=0):
        c,space = self._property_inputs(ci,norb,nelec,root)
        return rdm.make_rdm12(c,space,max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def make_rdm12s(self,ci=None,norb=None,nelec=None,*,root=0):
        c,space = self._property_inputs(ci,norb,nelec,root)
        return rdm.make_rdm12s(c,space,max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def make_rdm2(self,ci=None,norb=None,nelec=None,*,root=0):
        return self.make_rdm12(ci,norb,nelec,root=root)[1]

    def make_rdm2_block(self,indices,ci=None,norb=None,nelec=None,*,root=0,spin_resolved=False):
        """Requested spatial output only, inheriting the existing state budget."""
        c,space = self._property_inputs(ci,norb,nelec,root)
        return rdm.make_rdm2_block(c,space,indices,spin_resolved=spin_resolved,
            max_workspace_elements=self.max_workspace_elements,max_memory=self.max_memory)

    def trans_rdm1(self,bra,ket,norb,nelec):
        return rdm.trans_rdm1(bra,ket,self._space(norb,nelec),max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def trans_rdm12(self,bra,ket,norb,nelec):
        return rdm.trans_rdm12(bra,ket,self._space(norb,nelec),max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def trans_rdm1s(self,bra,ket,norb,nelec):
        return rdm.trans_rdm1s(bra,ket,self._space(norb,nelec),max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def trans_rdm12s(self,bra,ket,norb,nelec):
        return rdm.trans_rdm12s(bra,ket,self._space(norb,nelec),max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def spin_square(self,ci=None,norb=None,nelec=None,*,root=0):
        c,space = self._property_inputs(ci,norb,nelec,root)
        return rdm.spin_square(c,space,max_workspace_elements=self.max_workspace_elements,
            max_memory=self.max_memory)

    def absorb_h1e(self,h1,eri,norb,nelec,fac=1.):
        return hamiltonian.absorb_h1e(h1,eri,self._space(norb,nelec),fac)

    def contract_1e(self,h1,ci,norb,nelec):
        return hamiltonian.contract_1e(h1,ci,self._space(norb,nelec),
            max_workspace_elements=self.max_workspace_elements,max_memory=self.max_memory)

    def contract_2e(self,eri,ci,norb,nelec):
        return hamiltonian.contract_2e(eri,ci,self._space(norb,nelec),
            max_workspace_elements=self.max_workspace_elements,max_memory=self.max_memory)

    def make_hdiag(self,h1,eri,norb,nelec):
        return hamiltonian.make_hdiag(h1,eri,self._space(norb,nelec))

    def energy(self,h1,eri,ci,norb,nelec,*,ecore=0.):
        return hamiltonian.energy(h1,eri,ci,self._space(norb,nelec),ecore=ecore,
            max_workspace_elements=self.max_workspace_elements,max_memory=self.max_memory)


def _bound_source_arrays(source):
    """Known arrays already held by a bound source; never build a reference."""
    if isinstance(source, FCIReference):
        return source.h1, source.eri, source.ecore
    arrays, seen = [], set()

    def collect(value):
        if isinstance(value, type) or id(value) in seen:
            return
        seen.add(id(value))
        if is_dataclass(value):
            for item in fields(value):
                collect(getattr(value, item.name))
        else:
            leaves = jax.tree.leaves(value)
            if len(leaves) == 1 and leaves[0] is value:
                arrays.append(value)
            else:
                for leaf in leaves:
                    collect(leaf)

    for name in ("mo_coeff", "mo_occ", "mo_energy", "params"):
        collect(getattr(source, name, None))
    for name in ("_scf_inputs", "scf_result", "reference"):
        payload = getattr(source, name, None)
        # Start at known payload fields. Recurse only through declared data,
        # containers and registered PyTrees, never arbitrary model __dict__s.
        for value in getattr(payload, "__dict__", {}).values():
            collect(value)
    return tuple(arrays)


class _MolecularFCI(FCISolver):
    """Bind a source and share its resolved budget with preparation/actions."""

    def __init__(self,source,*,core=0,active=None,max_memory=None,**kwargs):
        super().__init__(max_memory=resolve_memory_mb(max_memory,source),**kwargs)
        self.source,self.core = source,core
        self.active = None if active is None else tuple(active)

    def _key(self):
        norb,_ = source_dimensions(self.source,self.spin)
        return (super()._key(),orbital_selection(norb,self.core,self.active),
                source_signature(self.source))

    def kernel(self,ci0=None):
        self._reset()
        norb,nelec = source_dimensions(self.source,self.spin)
        core,active = orbital_selection(norb,self.core,self.active)
        active_nelec = tuple(n-len(core) for n in nelec)
        # Forecast active integrals, absorption scratch, the diagonal and
        # requested CI states before any source transformation. Only a minimum
        # action tile must fit; a complete P*determinant stack is not required.
        space = self._space(len(active),active_nelec)
        config = self._config()
        if isinstance(self.source,FCIReference):
            dtype = jnp.result_type(getattr(self.source.h1,'dtype',np.float64),
                                    getattr(self.source.eri,'dtype',np.float64))
        else:
            # Native direct integral outputs use the float64 ABI even when
            # the supplied orbital frame has lower precision.
            dtype = jnp.result_type(self.source._scf_inputs.hcore,self.source.mo_coeff,jnp.float64)
        hmeta = jax.ShapeDtypeStruct((space.norb,)*2,dtype)
        gmeta = jax.ShapeDtypeStruct((space.norb,)*4,dtype)
        source_arrays = _bound_source_arrays(self.source)
        source_bytes = array_storage_bytes(*source_arrays)
        config,solver_bytes,_ = _plan_fci_memory(space,dtype,config,self.max_memory,
            arrays=(hmeta,gmeta),ci0=ci0,reserved_bytes=source_bytes)
        hamiltonian._action_tiles(space,dtype,1,self.max_workspace_elements,self.max_memory,
            arrays=(hmeta,gmeta,ci0),
            extra_bytes=source_bytes+solver_bytes+(3*space.norb**4+2*space.size)*jnp.dtype(dtype).itemsize)
        h,g,ecore = active_reference(self.source,core,active,max_memory=self.max_memory)
        # Keep the source's full AO/MO arrays charged after producing the CAS.
        # Shared objects are counted once if an adapter returns an input array.
        source_bytes = (array_storage_bytes(*_bound_source_arrays(self.source),h,g,ecore)
                        - array_storage_bytes(h,g,ecore))
        return super().kernel(h,g,len(active),active_nelec,ci0=ci0,ecore=ecore,
                               reserved_bytes=source_bytes)


def FCI(source=None,**kwargs):
    """Bind an SCF/FCIReference, or create an integral-driven FCISolver."""
    return FCISolver(**kwargs) if source is None else _MolecularFCI(source,**kwargs)


def kernel(h1,eri,norb,nelec,ci0=None,ecore=0.,**kwargs):
    return FCISolver(**kwargs).kernel(h1,eri,norb,nelec,ci0,ecore)
