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
    from ..pbc.tdscf import _validate_reference
    from ..pbc._kpoint import energy_and_fock
    _validate_reference(mf)
    result, inputs = mf.result, mf.inputs
    inputs = inputs._replace(hcore=inputs.hcore.astype(jnp.complex128))
    coeff = result.mo_coeff_spin.astype(jnp.complex128)
    mesh, xc, exxdiv = tuple(mf.cell.mesh), mf.xc, mf.exxdiv
    nk, n = coeff.shape[1], coeff.shape[-1]
    shape = (2, nk, n, n)
    def evaluate(d, a):
        e, f = energy_and_fock(d.reshape(shape), a, mesh=mesh, xc=xc, exxdiv=exxdiv)
        return e, f.reshape(2*nk, n, n)
    problem = ks_problem(coeff.reshape(2*nk, n, n),
                          tuple(no for no in mf.cell.nelec for _ in range(nk)), inputs,
                          lambda d,a:evaluate(d,a)[1], lambda d,a:evaluate(d,a)[0],
                          overlap=lambda a:jnp.tile(a.overlap, (2,1,1)),
                          converged=result.converged, tolerance=tolerance)
    observe = problem.quantities
    def quantities(x, a):
        values = observe(x, a)
        return dict(values, density=values['density'].reshape(shape),
                    electron_number=jnp.sum(values['electron_number'])/nk)
    def potential_tangent(value):
        value = jnp.asarray(value, dtype=inputs.hcore.dtype)
        if value.shape != inputs.hcore.shape:
            raise ValueError('Periodic potential perturbation must have shape (nk,nao,nao)')
        value *= jnp.where(jnp.max(jnp.abs(value-value.conj().swapaxes(-1,-2))) <= 1e-10, 1., jnp.nan)
        return jax.tree.map(jnp.zeros_like, inputs)._replace(hcore=value)
    return replace(problem, quantities=quantities, potential_tangent=potential_tangent,
                    potential_path=lambda tangent,t:inputs._replace(hcore=inputs.hcore+t*tangent.hcore))


def DFPT(source, *, functional=None, params=None, converged=True, q=None, tolerance=1e-7):
    """Build a response snapshot from converged molecular/periodic KS or OFDFT.

    q=0 static, fixed-occupation response only. Rebuild after changing the source.
    Array/custom-functional users can construct ks_problem/ofdft_problem directly.
    For an external dft.Functional, supply functional and params with the state
    returned by DifferentiableSCF; pass its convergence flag as converged.
    Construct inside the differentiated function to retain reference relaxation.
    """
    from ..ofdft.api import OFDFT
    if q is not None and (np.shape(q)!=(3,) or np.any(np.asarray(q)!=0)):
        raise NotImplementedError('DFPT currently supports q=0 only')
    if functional is not None:
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
