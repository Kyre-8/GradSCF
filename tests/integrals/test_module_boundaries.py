"""Canonical ownership and import direction of the integral package."""
from pathlib import Path
import subprocess
import sys


def test_basis_and_native_plan_do_not_import_reference_kernels():
    code = """
import sys
from gradscf.integrals import prepare_basis, make_plan
top, parameters = prepare_basis('H 0 0 0; H 0 0 .74', 'sto-3g')
plan = make_plan(top)
assert plan.topology is top
assert not any('.backends.jax_reference' in name for name in sys.modules)
from gradscf.integrals.basis.types import BasisParameters
assert isinstance(parameters, BasisParameters)
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_scf_assembly_has_one_owner():
    from gradscf.scf.inputs import RKSIntegralInputs, UKSIntegralInputs, build_rks_integral_inputs
    assert RKSIntegralInputs.__module__ == 'gradscf.scf.inputs.types'
    assert UKSIntegralInputs.__module__ == 'gradscf.scf.inputs.types'
    assert build_rks_integral_inputs.__module__ == 'gradscf.scf.inputs.assembly'


def test_public_builder_and_scf_import_orders():
    for statement in ('from gradscf.integrals import build_rks_integral_inputs',
                      'import gradscf.scf.inputs.assembly'):
        subprocess.run([sys.executable, '-c', statement], check=True, capture_output=True, text=True)


def test_df_kernels_are_owned_by_the_integral_layer():
    from gradscf.integrals.molecular.jk import build_jk_from_df
    from gradscf.integrals.molecular.factorization import eri_to_df_factors
    assert build_jk_from_df.__module__ == 'gradscf.integrals.molecular.jk'
    assert eri_to_df_factors.__module__ == 'gradscf.integrals.molecular.factorization'


def test_basis_contraction_is_independent_of_execution_backends():
    code = """
import sys
from gradscf.integrals.basis import prepare_basis
from gradscf.integrals.basis.contraction import contraction_matrix
top, parameters = prepare_basis('H 0 0 0', 'sto-3g')
matrix = contraction_matrix(top, parameters)
assert matrix.shape == (3, 1)
assert not any(name.startswith('gradscf.integrals.backends') for name in sys.modules)
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_replaced_implementation_paths_are_absent():
    root = Path('src/gradscf')
    for relative in ('integrals/basis.py', 'integrals/basis_data', 'integrals/normalization.py',
                     'integrals/assembly.py', 'integrals/input_types.py', 'integrals/input_grid.py',
                     'integrals/input_spin.py', 'integrals/input_cache.py', 'df',
                     'integrals/backends/jax_reference/basis.py'):
        assert not (root/relative).exists(), relative
