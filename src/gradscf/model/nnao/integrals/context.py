"""Prepare the NNAO orbital and fixed-auxiliary native integral plans."""
from dataclasses import dataclass


from gradscf import integrals
from gradscf.integrals.basis.contraction import primitive_basis
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan
from .cache import fixed_signature, read_fixed_cache


@dataclass(frozen=True)
class FixedIntegralData:
    topology: object
    parameters: object
    signature: str
    overlap: object
    hcore: object
    factors: object
    direct_basis: object


def prepare_fixed_context(layout, symbols, coords, *, basis_family, jk_backend,
                          auxbasis, integral_cache=None, cache_storage='device',
                          metric_factor=None):
    """Prepare fixed primitive data; never construct a four-index ERI."""
    topology, parameters = primitive_basis(layout.topology, layout.parameters)
    signature = fixed_signature(layout, symbols, coords, basis_family, auxbasis,
                                topology, parameters)
    direct_basis = None
    factors = None
    if integral_cache is not None:
        if jk_backend != 'df':
            raise ValueError('Integral caches currently support jk_backend="df" only.')
        overlap, hcore, factors = read_fixed_cache(integral_cache, signature, cache_storage)
    else:
        plan = integrals.make_plan(topology, backend='native')
        if jk_backend == 'direct':
            direct_basis = NativeDirectBasis(plan, parameters)
        overlap = plan.evaluate('overlap', parameters)
        hcore = plan.evaluate('kinetic', parameters) + plan.evaluate('nuclear', parameters)
        if basis_family == 'qvszps':
            hcore = hcore + plan.evaluate('ecp', parameters, ecps=layout.ecps)
        if jk_backend == 'df':
            at, ap = integrals.prepare_basis(list(zip(symbols, coords)), auxbasis,
                                             cart=layout.topology.cart)
            kwargs = {} if metric_factor is None else {'metric_factor': metric_factor}
            factors = make_auxiliary_plan(topology, at).factors(parameters, ap, **kwargs)
    for array in (overlap, hcore, factors):
        if hasattr(array, 'block_until_ready'):
            array.block_until_ready()
    return FixedIntegralData(topology, parameters, signature, overlap, hcore,
                             factors, direct_basis)

def prepare_native_context(layout, atom, auxbasis):
    auxiliary_topology, auxiliary_parameters = integrals.prepare_basis(
        atom, auxbasis, unit='Angstrom', cart=layout.topology.cart)
    auxiliary = make_auxiliary_plan(layout.topology, auxiliary_topology)
    return dict(layout=layout, plan=integrals.make_plan(layout.topology, backend='native'),
                aux=auxiliary, ap=auxiliary_parameters,
                metric=auxiliary.metric_factor(layout.parameters, auxiliary_parameters))
