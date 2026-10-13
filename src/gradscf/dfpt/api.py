"""Eager adapters take snapshots; all response work uses array-valued problems."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from .ks import ks_problem
from .ofdft import ofdft_problem


def _molecular(mf, tolerance):
    from ..scf.facade import RKS
    from ..scf.rks import _make_jk_builder, _energy_and_raw_fock_for_density
    from ..dft.libxc_jax.jax_libxc import hybrid_coeff, xc_type
    if not isinstance(mf, RKS):
        raise NotImplementedError('Molecular facade currently supports RHF/RKS; use ks_problem for custom spin builders')
    mf.to_reference()  # validates a converged, current source and completes arrays
    data, cfg = mf._scf_inputs, mf._config()
    if xc_type(cfg.xc_spec) not in ('HF', 'LDA', 'GGA'):
        raise NotImplementedError('Molecular adapter currently supports HF/LDA/GGA/global hybrids')
    names = ('overlap', 'hcore', 'eri', 'eri_pair_matrix', 'df_factors', 'ao', 'ao_deriv1',
             'grid_weights', 'nuclear_repulsion')
    args = {key: getattr(data, key) for key in names}
    alpha, kind = hybrid_coeff(cfg.xc_spec), xc_type(cfg.xc_spec)
    def evaluate(d, a):
        jk = _make_jk_builder(a['eri'], cfg, eri_pair_matrix=a['eri_pair_matrix'],
                              df_factors=a['df_factors'], direct_basis=data.direct_basis,
                              with_k=bool(alpha))
        e, _, f, _, _ = _energy_and_raw_fock_for_density(
            density=d[0], mo_coeff=None, mo_occ=None, mo_energy=None, jk_builder=jk,
            ao=a['ao'], ao_deriv1=a['ao_deriv1'], weights=a['grid_weights'], h=a['hcore'],
            enuc=a['nuclear_repulsion'], alpha=alpha, cfg=cfg, xc_kind=kind)
        return e, f[None]
    problem = ks_problem(jnp.asarray(mf.mo_coeff)[None], (data.nelectron//2,), args,
                          lambda d,a: evaluate(d,a)[1], lambda d,a: evaluate(d,a)[0],
                          overlap=lambda a:a['overlap'][None], occupation=2.,
                          converged=mf.converged, tolerance=tolerance)
    observe = problem.quantities
    def quantities(x, a):
        values = observe(x, a)
        return dict(values, density=values['density'][0], electron_number=values['electron_number'][0])
    def potential_tangent(value):
        value = jnp.asarray(value)
        if value.shape != args['hcore'].shape or jnp.iscomplexobj(value):
            raise ValueError('Molecular potential perturbation must be a real AO matrix')
        value = value.astype(args['hcore'].dtype)
        value *= jnp.where(jnp.max(jnp.abs(value-value.T)) <= 1e-10, 1., jnp.nan)
        return dict(jax.tree.map(jnp.zeros_like, args), hcore=value)
    return replace(problem, quantities=quantities, potential_tangent=potential_tangent,
                    potential_path=lambda tangent,t:dict(args, hcore=args['hcore']+t*tangent['hcore']))


def _periodic(mf, tolerance):
    from ..pbc.scf import validate_reference
    from .periodic import periodic_problem
    validate_reference(mf,integer=False,equal_weights=False,external=True)
    return periodic_problem(mf.inputs,mf.result,mesh=tuple(mf.cell.mesh),nelec=mf.cell.nelec,
        xc=mf.xc,exxdiv=mf.exxdiv,functional=getattr(mf,"functional",None),
        params=getattr(mf,"params",None),temperature=getattr(mf,"temperature",None),tolerance=tolerance)


def DFPT(source, *, functional=None, params=None, converged=True, q=None, tolerance=1e-7):
    """Build a response snapshot from converged molecular/periodic KS or OFDFT.

    Static q=0 response. Periodic sources may use fixed or Fermi occupations.
    Rebuild after changing the source.
    Array users can construct ks_problem, periodic_problem or ofdft_problem.
    Periodic facades retain their own functional and params; use DFPT(mf).
    For a molecular external dft.Functional, supply functional and params with the state
    returned by DifferentiableSCF; pass its convergence flag as converged.
    Construct inside the differentiated function to retain reference relaxation.
    """
    from ..ofdft.api import OFDFT
    if q is not None and (np.shape(q)!=(3,) or np.any(np.asarray(q)!=0)):
        raise NotImplementedError('DFPT currently supports q=0 only')
    if functional is not None:
        if hasattr(source,'cell'):
            raise ValueError('Set the periodic Functional and params in SCF, then use DFPT(mf); '
                             'use periodic_problem for array-valued parameter differentiation.')
        from .functional import functional_problem
        return functional_problem(source, functional, params, converged=converged, tolerance=tolerance)
    if params is not None:
        raise ValueError('params requires an external functional')
    if isinstance(source, OFDFT):
        if source.result is None:
            raise ValueError('Run OFDFT before constructing a response problem')
        problem = ofdft_problem(source.inputs, source.result, kinetic=source.kinetic,
            kinetic_params=source.kinetic_params, xc_energy_fn=source.xc_energy_fn,
            xc_params=source.xc_params, config=source.config, tolerance=tolerance)
    else:
        problem = _periodic(source, tolerance) if hasattr(source, 'cell') else _molecular(source, tolerance)
    return replace(problem, converged=problem.converged & converged)
