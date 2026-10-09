"""Physical FCI outputs assembled with the common Hermitian solver/response."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from ..integrals.mo import validate_integrals
from ..solvers import EigenSolverConfig, EigenResponseConfig, LinearOperator, solve_hermitian
from ..solvers.diagnostics import require_converged_derivative
from .hamiltonian import DEFAULT_WORKSPACE, build_hamiltonian, coefficient_array


class FCIResult(NamedTuple):
    total_energies: object
    coefficients: object
    energy_sum: object
    projection: object
    ecore: object
    solver_result: object

    @property
    def converged(self):
        return self.solver_result.converged

    @property
    def response_valid(self):
        return self.solver_result.response_valid

    @property
    def residual_norms(self):
        return self.solver_result.residual_norms

    @property
    def electronic_energies(self):
        return None if self.total_energies is None else self.total_energies-self.ecore


def _physical_integrals_valid(h, g):
    scale = jnp.maximum(1., jnp.maximum(jnp.max(jnp.abs(h), initial=0.), jnp.max(jnp.abs(g), initial=0.)))
    tol = jnp.maximum(1e-10, 64*jnp.finfo(jnp.result_type(h,g)).eps*max(1,h.shape[0])) * scale
    return (jnp.all(jnp.isfinite(h)) & jnp.all(jnp.isfinite(g))
            & (jnp.max(jnp.abs(h-h.T), initial=0.) <= tol)
            & (jnp.max(jnp.abs(g-g.transpose(1,0,2,3)), initial=0.) <= tol)
            & (jnp.max(jnp.abs(g-g.transpose(0,1,3,2)), initial=0.) <= tol)
            & (jnp.max(jnp.abs(g-g.transpose(2,3,0,1)), initial=0.) <= tol))


def solve_fci(h1, eri, space, *, ecore=0., config=None, response=None,
              ci0=None, probes=None, max_workspace_elements=DEFAULT_WORKSPACE):
    """Solve a fixed electron/spatial-orbital space; numerical configs are shared.

    Coefficients have (nroots,nalpha_strings,nbeta_strings) shape. A subspace
    response instead returns only energy_sum and projector probes, including
    internal degeneracy. The constant ecore is kept out of spectral gap tests.
    """
    cfg = config or EigenSolverConfig(atol=1e-10, max_subspace=40)
    policy = response
    h,g = validate_integrals(h1,eri,space.norb)
    core = jnp.asarray(ecore)
    if core.shape != () or jnp.iscomplexobj(core):
        raise ValueError("ecore must be a real scalar")
    physical = _physical_integrals_valid(h,g) & jnp.isfinite(core)
    original = build_hamiltonian(h,g,space,max_workspace_elements=max_workspace_elements)
    op = LinearOperator(original.shape, original.dtype,
        lambda v: jnp.where(physical, original.matvec(v), jnp.nan),
        diagonal=jnp.where(physical, original.diagonal, jnp.nan),
        matmat=lambda v: jnp.where(physical, original.apply(v), jnp.nan))
    initial = None
    if ci0 is None and cfg.method == 'davidson':
        count = min(cfg.nroots+1,space.size)
        ci0 = jax.nn.one_hot(jnp.argsort(original.diagonal)[:count],space.size,
                             dtype=op.dtype).reshape((count,)+space.shape)
    if ci0 is not None:
        c = jnp.asarray(ci0)
        if c.shape in (space.shape, (space.size,)):
            c = coefficient_array(c,space)[None,:,:]
        if c.ndim != 3 or c.shape[1:] != space.shape or not c.shape[0] or jnp.iscomplexobj(c):
            raise ValueError("ci0 must be one CI matrix or a stack of CI matrices")
        if cfg.method == 'davidson':
            # Add the caller's guesses to full-support deterministic defaults, so
            # a warm start cannot restrict the search to one invariant spin sector.
            count = min(cfg.nroots+1,space.size)
            requested = 2*count if cfg.initial_guess_count is None else cfg.initial_guess_count
            width = min(space.size, max(count,requested,c.shape[0]))
            if cfg.max_subspace is not None:
                width = min(width,cfg.max_subspace)
            if c.shape[0] > width:
                raise ValueError("ci0 exceeds the Davidson subspace capacity")
            initial = jax.random.normal(jax.random.PRNGKey(cfg.seed),(space.size,width),dtype=op.dtype) / jnp.sqrt(float(space.size))
            initial = initial.at[:,:c.shape[0]].set(
                c.reshape(c.shape[0],space.size).T + 1e-3*initial[:,:c.shape[0]])
    solved = solve_hermitian(op, config=cfg, response=policy, probes=probes,
                             initial_vectors=initial)
    valid = solved.response_valid & physical
    solved = solved._replace(converged=solved.converged & physical, response_valid=valid)
    energies = coefficients = total_sum = None
    if solved.values is not None:
        energies = require_converged_derivative(solved.values+core,valid)
        coefficients = solved.vectors.T.reshape((cfg.nroots,)+space.shape)
        vector_response = (policy.target == 'eigenpairs' if policy is not None
                           else cfg.gradient_mode == 'implicit_eigenvector')
        if not vector_response:
            # Keep a live dependency so a stopped vector cannot give a silently
            # incomplete integral derivative of an RDM or transition property.
            sentinel = require_converged_derivative(solved.values, False)
            coefficients += (sentinel-jax.lax.stop_gradient(sentinel))[:,None,None]
        coefficients = require_converged_derivative(coefficients, valid[:,None,None])
    else:
        total_sum = require_converged_derivative(solved.eigenvalue_sum + cfg.nroots*core, valid)
    return FCIResult(energies, coefficients, total_sum, solved.projection, core, solved)
