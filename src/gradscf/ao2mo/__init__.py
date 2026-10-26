"""AO-to-MO transformations and shared post-HF integral-space utilities."""

from gradscf.ao2mo.actions import VirtualERI
from gradscf.ao2mo.df import transform_df_factors

from gradscf.ao2mo.blocks import (
    transform_eri_block,
    transform_eri_blocks,
)
from gradscf.ao2mo.reference import (
    validate_integrals,
    transform_integrals,
    validate_unrestricted_integrals,
    transform_unrestricted_integrals,
)
from gradscf.ao2mo.spin import (
    frozen_indices,
    unrestricted_frozen_indices,
    spin_orbital_integrals,
)
from gradscf.ao2mo.response import (
    df_factors_to_mo_eri_slices,
    eri_pair_matrix_to_mo_eri_slices,
)

__all__ = [
    "VirtualERI",
    "transform_df_factors",
    "transform_eri_block",
    "transform_eri_blocks",
    "validate_integrals",
    "transform_integrals",
    "validate_unrestricted_integrals",
    "transform_unrestricted_integrals",
    "frozen_indices",
    "unrestricted_frozen_indices",
    "spin_orbital_integrals",
    "df_factors_to_mo_eri_slices",
    "eri_pair_matrix_to_mo_eri_slices",
]
