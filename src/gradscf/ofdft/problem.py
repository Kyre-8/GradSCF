"""OFDFT constraint assembly; optimization and backward live in shared modules.

Use whitened amplitude x, ||x||=1, q=sqrt(N) M^(-1/2) x. This keeps the
KKT response well scaled on fine grids and differentiates the moving metric.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from ..solvers.nonlinear import minimize_sphere, SphereConfig
from ..scf.autodiff import attach_scf_backward
from ..dft.libxc_jax.jax_libxc import xc_type, hybrid_coeff
from .types import OFDFTConfig, OFDFTResult
from .kinetic import KineticFunctional
from .energy import energy_components, amplitude


def _to_coefficients(x, data):
    if data.representation=='periodic':
        return x*jnp.sqrt(data.nelectron/data.weights)
    lower = jnp.linalg.cholesky(data.overlap)
    return jnp.sqrt(data.nelectron)*solve_triangular(lower.T,x,lower=False)


def _to_sphere(q, data):
    if data.representation=='periodic':
        x = q*jnp.sqrt(data.weights/data.nelectron)
    else:
        x = jnp.linalg.cholesky(data.overlap).T@q/jnp.sqrt(data.nelectron)
    return x/jnp.linalg.norm(x)


@lru_cache(maxsize=64)
def _equations(kinetic, xc, xc_energy_fn):
    def components(x, args):
        data, params, xc_params = args
        return energy_components(_to_coefficients(x,data),data,kinetic=kinetic,
            kinetic_params=params,xc=xc,xc_energy_fn=xc_energy_fn,xc_params=xc_params)
    def energy(x, args):
        return components(x,args).total
    gradient = jax.grad(energy)
    def residual(state, args):
        x, eta = state[:-1],state[-1]
        return jnp.concatenate([gradient(x,args)-2*eta*x,jnp.asarray([jnp.vdot(x,x)-1])])
    return energy, gradient, residual, components


def _validate(data):
    if not jax.config.x64_enabled:
        raise ValueError('OFDFT requires float64.')
    if data.representation not in ('gaussian','periodic','periodic_gaussian'):
        raise ValueError('Unknown OFDFT representation.')
    for leaf in jax.tree.leaves(data):
        if hasattr(leaf,'dtype') and jnp.iscomplexobj(leaf):
            raise ValueError('OFDFT inputs must be real.')
    if data.weights.ndim!=1 or data.coordinates.shape!=(data.weights.size,3):
        raise ValueError('Inconsistent grid shapes.')
    if jnp.shape(data.nelectron)!=() or jnp.shape(data.nuclear_repulsion)!=():
        raise ValueError('nelectron and nuclear_repulsion must be scalars.')
    if not isinstance(data.weights,jax.core.Tracer):
        weights = np.asarray(data.weights)
        # Lebedev quadratures can have signed weights and Becke partitions zero
        # weights. The Gaussian metric is S, not the quadrature weights.
        bad = np.any(weights <= 0) if data.representation=='periodic' else False
        if not np.isfinite(weights).all() or bad or weights.sum() <= 0:
            raise ValueError('Invalid quadrature weights for the chosen representation.')
    if not isinstance(data.nelectron,jax.core.Tracer):
        number = np.asarray(data.nelectron)
        if not np.isfinite(number) or number <= 0:
            raise ValueError('nelectron must be finite and positive.')
    if data.representation!='periodic':
        n = data.overlap.shape[0]
        if data.overlap.shape!=(n,n) or data.ao.shape!=(data.weights.size,n):
            raise ValueError('Inconsistent Gaussian overlap/AO shapes.')
        if data.kinetic_matrix.shape!=(n,n) or data.ao_gradient.shape!=(data.weights.size,n,3):
            raise ValueError('Inconsistent Gaussian kinetic/AO-gradient shapes.')
        if not isinstance(data.overlap,jax.core.Tracer):
            eigenvalues=np.linalg.eigvalsh(np.asarray(data.overlap))
            if not np.isfinite(eigenvalues).all() or eigenvalues.min()<=1e-12*eigenvalues.max():
                raise ValueError('Gaussian overlap is singular/ill-conditioned; change the basis or grid.')
    if data.representation=='gaussian' and data.eri is None and data.df_factors is None:
        raise ValueError('Supply molecular ERIs or DF factors.')
    expected_potential = (data.overlap.shape[0],)*2 if data.representation=='gaussian' else data.weights.shape
    if data.external_potential.shape != expected_potential:
        raise ValueError('external_potential shape does not match the representation.')
    if data.representation=='gaussian':
        n = data.overlap.shape[0]
        npair = n*(n+1)//2
        if data.eri is not None and data.eri.shape not in ((n,)*4,(npair,npair),(npair*(npair+1)//2,)):
            raise ValueError('Invalid molecular ERI shape.')
        if data.df_factors is not None and (data.df_factors.ndim!=3 or data.df_factors.shape[1:]!=(n,n)):
            raise ValueError('DF factors must have shape (naux,nao,nao).')
    else:
        if len(data.mesh)!=3 or np.prod(data.mesh)!=data.weights.size or data.lattice.shape!=(3,3):
            raise ValueError('Inconsistent periodic mesh/lattice.')


def run_ofdft(data, *, kinetic=None, kinetic_params=None, xc_energy_fn=None, xc_params=None,
               initial=None, config=None):
    """Converge density and attach implicit response (or differentiate iterates).

    For training, pass network weights via kinetic_params/xc_params, never close
    over traced parameters in the callback. Eager facades are for input building;
    this array/PyTree interface supports JIT, grad, JVP and density-loss training.
    A stationary state is not a certificate of the global minimum.
    """
    _validate(data)
    cfg = config or OFDFTConfig()
    kinetic = KineticFunctional() if kinetic is None else kinetic
    if cfg.xc is not None and (xc_type(cfg.xc) not in ('LDA','GGA') or hybrid_coeff(cfg.xc)!=0):
        raise ValueError('OFDFT supports density-only LDA/GGA XC; hybrids/meta-GGA need orbital information.')
    if cfg.xc is not None and xc_energy_fn is not None:
        raise ValueError('Set config.xc=None when supplying a custom XC energy callback.')
    args = (data, {} if kinetic_params is None else kinetic_params, {} if xc_params is None else xc_params)
    if initial is None:
        initial = jnp.ones(data.weights.size if data.representation=='periodic' else data.overlap.shape[0])
    initial = jnp.asarray(initial,dtype=jnp.float64)
    expected = data.weights.size if data.representation=='periodic' else data.overlap.shape[0]
    if initial.shape!=(expected,):
        raise ValueError('Initial amplitude/coefficient vector has the wrong shape.')
    if not isinstance(initial,jax.core.Tracer):
        concrete = np.asarray(initial)
        if not np.isfinite(concrete).all() or not np.any(concrete):
            raise ValueError('Initial amplitude must be finite and nonzero.')
    energy, gradient, residual, components = _equations(kinetic,cfg.xc,xc_energy_fn)
    x0 = _to_sphere(initial,data)
    forward_args, seed = args,x0
    if cfg.differentiation.mode=='implicit':
        forward_args, seed = jax.tree.map(jax.lax.stop_gradient,(args,x0))
    forward = minimize_sphere(energy,seed,forward_args,
        config=SphereConfig(maxiter=cfg.maxiter,tolerance=cfg.tolerance,
            response_refinements=3,refine_stationary=cfg.differentiation.mode=='explicit'))
    x = forward.solution
    eta = .5*jnp.vdot(x,gradient(x,forward_args))
    state = jnp.concatenate([x,jnp.asarray([eta])])
    norm = jnp.linalg.norm(residual(state,forward_args))
    valid = forward.converged & jnp.isfinite(norm) & (norm<=cfg.tolerance*1.01)
    state = attach_scf_backward(args,solution=state,residual=residual,
        config=cfg.differentiation,converged=valid)
    q = _to_coefficients(state[:-1],data)
    phi = amplitude(q,data)
    density = phi**2
    number = jnp.sum(data.weights*q*q) if data.representation=='periodic' else q@data.overlap@q
    pieces = components(state[:-1],args)
    return OFDFTResult(pieces.total,density,phi,q,state[-1]/data.nelectron,
        number,jnp.sum(data.weights*density),norm,valid,forward.iterations,pieces)
