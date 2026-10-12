import cuequivariance as cue
from cuequivariance.segmented_polynomials import operation as sp_operation

from mace_jax.adapters import cuequivariance as cue_adapters


def test_operand_letters_support_large_indices():
    # Importing the local adapter package must not regress cuequivariance>=0.9's
    # unbounded operand naming behaviour.
    assert cue_adapters is not None
    large_input = sp_operation.IVARS[1000]
    large_output = sp_operation.OVARS[1000]

    assert isinstance(large_input, str)
    assert isinstance(large_output, str)
    assert large_input
    assert large_output


def test_segmented_polynomial_comparisons_handle_foreign_types():
    descriptor = cue.descriptors.fully_connected_tensor_product(
        cue.Irreps(cue.O3, '1x0e'),
        cue.Irreps(cue.O3, '1x0e'),
        cue.Irreps(cue.O3, '1x0e'),
    )

    polynomial = descriptor.polynomial
    operation = polynomial.operations[0]

    assert not (descriptor == object())
    assert not (polynomial == object())
    assert not (operation == object())
