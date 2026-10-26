"""Reproducible shared-pilot model and exponent channel configuration."""
EXPONENT_KEYS = (('H', 0), ('H', 1), *((symbol, l) for symbol in ('C', 'N', 'O', 'F') for l in range(3)))
MODEL_CONFIG = dict(elements=(1, 6, 7, 8, 9), channels=8, num_interactions=2,
                    max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
PARENT_SOURCE = '7df11aecdbf295818de02c0db67da3de8d3db5b1ccef29798114308e28d08ebb'
PARENT_CHECKPOINT_SHA = 'ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de'
