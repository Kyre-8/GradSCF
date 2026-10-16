import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscf_adapters import basis_from_pyscf_mol_cart
from gradscf.integrals import evaluate_cartesian_ao


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for AO-grid comparison tests.")


def _make_water_cart():
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.cart = True
    mol.build()
    return mol


def test_evaluate_cartesian_ao_matches_pyscf_values_and_first_derivatives():
    _pyscf_or_skip()
    from pyscf.dft import numint

    mol = _make_water_cart()
    basis = basis_from_pyscf_mol_cart(mol, max_l=3)
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.3],
            [-0.4, 0.5, -0.2],
            [0.8, -0.7, 0.1],
        ],
        dtype=float,
    )

    ao_ref = np.asarray(numint.eval_ao(mol, coords, deriv=0), dtype=float)
    ao1_ref = np.asarray(numint.eval_ao(mol, coords, deriv=1), dtype=float)

    ao_jax = np.asarray(evaluate_cartesian_ao(basis, coords, deriv=0), dtype=float)
    ao1_jax = np.asarray(evaluate_cartesian_ao(basis, coords, deriv=1), dtype=float)

    assert ao_jax.shape == ao_ref.shape
    assert ao1_jax.shape == ao1_ref.shape
    assert np.allclose(ao_jax, ao_ref, atol=2e-7, rtol=2e-6)
    assert np.allclose(ao1_jax, ao1_ref, atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g*"])
def test_ao_gradient_second_directional_derivative_on_coordinate_planes(basis_name):
    """GGA Hessians need third spatial AO derivatives, including at x/y/z=0."""
    _pyscf_or_skip()
    from pyscf.dft import numint

    mol = _make_water_cart()
    mol.basis = basis_name
    mol.build()
    basis = basis_from_pyscf_mol_cart(mol, max_l=3)
    points = jnp.asarray([[0., 0., 0.], [0., .2, .1], [.3, 0., -.2], [.2, .1, 0.]])
    direction = jnp.asarray([.7, -.2, .4])
    ao_gradient = lambda shift: evaluate_cartesian_ao(basis, points+shift, deriv=1)
    second = jax.jit(lambda shift: jax.jvp(
        lambda x: jax.jvp(ao_gradient, (x,), (direction,))[1],
        (shift,), (direction,))[1])(jnp.zeros(3))

    # PySCF's Cartesian derivative ordering through order 3.
    powers = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
              (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2),
              (3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2),
              (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)]
    reference = np.asarray(numint.eval_ao(mol, np.asarray(points), deriv=3))
    expected = np.zeros((4, len(points), mol.nao))
    for component in range(4):
        for i in range(3):
            for j in range(3):
                power = list(powers[component])
                power[i] += 1
                power[j] += 1
                expected[component] += float(direction[i]*direction[j]) * reference[powers.index(tuple(power))]
    assert np.isfinite(np.asarray(second)).all()
    np.testing.assert_allclose(second, expected, atol=2e-8, rtol=2e-8)
