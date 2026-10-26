"""Independent radial CHG gradient/Hessian used by the hybrid water oracle."""
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np


def test_radial_chg_reference_matches_displacements_and_native_ad():
    from gradscf.dft.dispersion import wb97xd_dispersion_energy

    path = Path(__file__).resolve().parents[1]/"examples/vibrations/compare_water_hybrid_pyscf.py"
    calculate = runpy.run_path(str(path))["chg_correction"]
    symbols = ["O", "H", "H"]
    coordinates = np.array([[.2, -.1, .3], [.1, 1.43, 1.11], [-.4, -1.37, 1.07]])
    value, gradient, hessian = calculate(symbols, coordinates)
    fd_gradient, fd_hessian = [], []
    step = 1e-4
    for direction in np.eye(9).reshape(9, 3, 3):
        plus = calculate(symbols, coordinates+step*direction)
        minus = calculate(symbols, coordinates-step*direction)
        fd_gradient.append((plus[0]-minus[0])/(2*step))
        fd_hessian.append(((plus[1]-minus[1])/(2*step)).ravel())
    np.testing.assert_allclose(gradient.ravel(), fd_gradient, atol=2e-12, rtol=2e-7)
    np.testing.assert_allclose(hessian, np.column_stack(fd_hessian), atol=2e-11, rtol=2e-7)
    energy = lambda r: wb97xd_dispersion_energy(symbols, r)
    np.testing.assert_allclose(value, energy(jnp.asarray(coordinates)), atol=1e-13, rtol=1e-8)
    np.testing.assert_allclose(gradient, jax.grad(energy)(coordinates), atol=1e-12, rtol=1e-8)
    np.testing.assert_allclose(hessian, jax.hessian(energy)(coordinates).reshape(9, 9), atol=1e-12, rtol=1e-8)
