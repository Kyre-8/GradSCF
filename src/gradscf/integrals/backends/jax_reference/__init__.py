"""Lazy public exports for the integral layer."""
from importlib import import_module

_EXPORTS = {'DirectJKResult': 'gradscf.integrals.backends.jax_reference.direct_jk',
 'build_direct_jk_from_basis': 'gradscf.integrals.backends.jax_reference.direct_jk',
 'build_direct_jk_incremental': 'gradscf.integrals.backends.jax_reference.direct_jk',
 'build_hcore': 'gradscf.integrals.backends.jax_reference.one_electron',
 'dipole_element': 'gradscf.integrals.backends.jax_reference.one_electron',
 'dipole_matrix': 'gradscf.integrals.backends.jax_reference.one_electron',
 'kinetic_element': 'gradscf.integrals.backends.jax_reference.one_electron',
 'kinetic_matrix': 'gradscf.integrals.backends.jax_reference.one_electron',
 'nuclear_attraction_element': 'gradscf.integrals.backends.jax_reference.one_electron',
 'nuclear_attraction_matrix': 'gradscf.integrals.backends.jax_reference.one_electron',
 'overlap_hcore_matrices': 'gradscf.integrals.backends.jax_reference.one_electron',
 'overlap_element': 'gradscf.integrals.backends.jax_reference.one_electron',
 'overlap_matrix': 'gradscf.integrals.backends.jax_reference.one_electron',
 'rinv_element': 'gradscf.integrals.backends.jax_reference.one_electron',
 'rinv_matrices': 'gradscf.integrals.backends.jax_reference.one_electron',
 'rinv_matrix': 'gradscf.integrals.backends.jax_reference.one_electron',
 'build_j_from_eri_pair_matrix': 'gradscf.integrals.molecular.jk',
 'build_jk_from_eri_pair_matrix': 'gradscf.integrals.molecular.jk',
 'eri_pair_matrix_to_mo_eri_slices': 'gradscf.ao2mo',
 'schwarz_bounds': 'gradscf.integrals.backends.jax_reference.screening',
 'shell_pair_schwarz_bounds': 'gradscf.integrals.backends.jax_reference.screening',
 'eri_element': 'gradscf.integrals.backends.jax_reference.two_electron',
 'eri_pair_matrix_packed': 'gradscf.integrals.backends.jax_reference.two_electron',
 'eri_tensor': 'gradscf.integrals.backends.jax_reference.two_electron',
 'eri_tensor_screened': 'gradscf.integrals.backends.jax_reference.two_electron',
 'precompile_eri_kernels': 'gradscf.integrals.backends.jax_reference.two_electron'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
