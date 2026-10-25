"""An identically zero HF XC kernel must not allocate grid-by-OV features."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
from gradscf.tddft import response


@pytest.fixture(scope="module")
def hf_grid_reference():
    mol = gto.M(atom="H 0 0 0; H 0 0 .8", basis="6-31g", cart=True)
    mf = scf.RHF(mol, integral_backend="native", jk_backend="direct",
                 conv_tol=1e-12).run()
    assert mf.converged
    reference = scf.as_reference(mf)
    # Use the ordinary molecular grid, not the one-point synthetic HF fixture.
    assert len(reference.grid.weights) > 100
    assert np.any(np.asarray(reference.ao) != 0)
    return reference


@pytest.mark.parametrize("method,singlet", [("tda", True), ("tda", False), ("tdhf", True)])
def test_hf_response_does_not_expand_zero_xc_grid_kernel(
        hf_grid_reference, monkeypatch, method, singlet):
    def forbidden(*args, **kwargs):
        raise AssertionError("HF zero XC kernel must not construct grid-by-OV features")

    monkeypatch.setattr(response, "_restricted_response_features", forbidden)
    monkeypatch.setattr(response, "_restricted_grid_xc_response", forbidden)
    options = {"two_electron_mode": "direct"}
    if method == "tda":
        action, _, gaps = response.build_restricted_tda_operator(
            hf_grid_reference, SemilocalResponseFunctional("hf"), singlet=singlet,
            response_kernel_options=options)
        dim = gaps.size
    else:
        action = response.build_restricted_tdhf_operator(
            hf_grid_reference, SemilocalResponseFunctional("hf"),
            response_kernel_options=options)
        nmo = hf_grid_reference.mo_coeff.shape[-1]
        dim = 2*hf_grid_reference.nocc*(nmo-hf_grid_reference.nocc)
    values = jnp.arange(2*dim, dtype=jnp.float64).reshape(2, dim) / 7
    applied = action(values)
    assert applied.shape == values.shape
    assert np.all(np.isfinite(np.asarray(applied)))
    assert np.linalg.norm(np.asarray(applied)) > 0
