"""Native contracted physical inputs and their coefficient/exponent pullback."""
from dataclasses import replace

def native_projection(context, exponent_keys):
    """Return contracted physical inputs and their native CPU pullback."""
    import jax
    from gradscf.integrals.molecular.density_fitting import unpack_factors

    layout = context['layout']
    indices = {key: exponent_keys.index(key) for key in layout.exponent_scale_keys}

    def physical(outputs, beta):
        scaled = layout.with_log_exponent_scales({key: beta[index] for key, index in indices.items()})
        bound = scaled.bind(outputs)
        # This training graph varies basis parameters at fixed geometry. bind()
        # gathers centers under JIT; retain the concrete centers for the metric signature.
        bound = replace(bound, centers=layout.parameters.centers,
                        nuclear_coords=layout.parameters.nuclear_coords)
        s = context['plan'].evaluate('overlap', bound)
        h = context['plan'].evaluate('kinetic', bound) + context['plan'].evaluate('nuclear', bound)
        packed = context['aux'].factors(bound, context['ap'], metric_factor=context['metric'])
        return s, h, unpack_factors(packed, layout.topology.nao)

    def pullback(outputs, beta, cotangent):
        return jax.vjp(physical, outputs, beta)[1](cotangent)

    return jax.jit(physical), jax.jit(pullback)
