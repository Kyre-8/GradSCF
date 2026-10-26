"""Lazy public exports for the integral layer."""
from importlib import import_module

_EXPORTS = {'DirectJKResult': 'gradscf.integrals.molecular.jk',
 'build_jk_from_packed': 'gradscf.integrals.molecular.jk',
 'make_auxiliary_plan': 'gradscf.integrals.molecular.density_fitting',
 'build_direct_jk_from_basis': 'gradscf.integrals.molecular.jk',
 'build_direct_jk_incremental': 'gradscf.integrals.molecular.jk',
 'build_hcore': 'gradscf.integrals.molecular.one_electron',
 'dipole_element': 'gradscf.integrals.molecular.one_electron',
 'dipole_matrix': 'gradscf.integrals.molecular.one_electron',
 'kinetic_element': 'gradscf.integrals.molecular.one_electron',
 'kinetic_matrix': 'gradscf.integrals.molecular.one_electron',
 'nuclear_attraction_element': 'gradscf.integrals.molecular.one_electron',
 'nuclear_attraction_matrix': 'gradscf.integrals.molecular.one_electron',
 'overlap_hcore_matrices': 'gradscf.integrals.molecular.one_electron',
 'overlap_element': 'gradscf.integrals.molecular.one_electron',
 'overlap_matrix': 'gradscf.integrals.molecular.one_electron',
 'rinv_element': 'gradscf.integrals.molecular.one_electron',
 'rinv_matrices': 'gradscf.integrals.molecular.one_electron',
 'rinv_matrix': 'gradscf.integrals.molecular.one_electron',
 'build_j_from_eri_pair_matrix': 'gradscf.integrals.molecular.jk',
 'build_jk_from_eri_pair_matrix': 'gradscf.integrals.molecular.jk',
 'eri_pair_matrix_to_mo_eri_slices': 'gradscf.ao2mo',
 'schwarz_bounds': 'gradscf.integrals.molecular.screening',
 'shell_pair_schwarz_bounds': 'gradscf.integrals.molecular.screening',
 'eri_element': 'gradscf.integrals.molecular.eri',
 'eri_pair_matrix_packed': 'gradscf.integrals.molecular.eri',
 'eri_tensor': 'gradscf.integrals.molecular.eri',
 'eri_tensor_screened': 'gradscf.integrals.molecular.eri',
 'precompile_eri_kernels': 'gradscf.integrals.molecular.eri',
 'BasisTopology': 'gradscf.integrals.basis',
 'BasisParameters': 'gradscf.integrals.basis',
 'prepare_basis': 'gradscf.integrals.basis',
 'make_plan': 'gradscf.integrals.plan',
 'IntegralPlan': 'gradscf.integrals.plan',
 'backend_capabilities': 'gradscf.integrals.capabilities',
 'build_rks_integral_inputs': 'gradscf.scf.inputs.assembly',
 'build_uks_integral_inputs': 'gradscf.scf.inputs.assembly',
 'RKSIntegralInputs': 'gradscf.scf.inputs.assembly',
 'UKSIntegralInputs': 'gradscf.scf.inputs.assembly'}
_EXPORTS.update({'CartesianAO': 'gradscf.integrals.basis', 'CartesianBasis': 'gradscf.integrals.basis', 'basis_from_molecule_spec': 'gradscf.integrals.basis', 'basis_from_spec': 'gradscf.integrals.basis', 'basis_from_pyscf_spec': 'gradscf.integrals.basis', 'cartesian_angular_tuples': 'gradscf.integrals.basis', 'load_basis_from_snapshot': 'gradscf.integrals.basis.data', 'build_molecular_grid': 'gradscf.integrals.grids', 'build_molecular_grid_from_spec': 'gradscf.integrals.grids', 'evaluate_cartesian_ao': 'gradscf.integrals.grids.ao', 'evaluate_cartesian_ao_with_derivatives': 'gradscf.integrals.grids.ao'})
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
