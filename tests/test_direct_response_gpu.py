"""Keep CPU shell-direct integrals usable after response arrays move to CUDA."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def references():
    from gradscf import gto, scf
    from gradscf.tools.device import put_molecule_on_device

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CUDA device not enabled for this targeted test")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="6-31g", unit="Bohr")
        mf = scf.RHF(mol, jk_backend="direct", grids_level=0, conv_tol=1e-12).run()
        assert mf.converged
        reference = scf.as_reference(mf)
    return reference, put_molecule_on_device(reference, device=gpu), cpu, gpu


@pytest.mark.parametrize("full_td", [False, True])
def test_transferred_direct_response_matches_cpu_value_and_density_ad(references, full_td):
    from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
    from gradscf.tddft.response import build_restricted_tda_operator, build_restricted_tdhf_operator

    reference, moved, cpu, gpu = references
    build = build_restricted_tdhf_operator if full_td else build_restricted_tda_operator
    dim = reference.nocc * (reference.mo_coeff.shape[1] - reference.nocc)
    values = np.random.default_rng(909).normal(size=(3, (2 if full_td else 1) * dim))

    def evaluate(molecule, device):
        with jax.default_device(device):
            result = build(molecule, SemilocalResponseFunctional("hf"))
            action = result if full_td else result[0]
            x = jax.device_put(values, device)
            value = jax.jit(action)(x)
            derivative = jax.jit(jax.grad(lambda z: jnp.sum(action(z)**2)))(x)
            return value, derivative

    expected = evaluate(reference, cpu)
    actual = evaluate(moved, gpu)
    assert next(iter(actual[0].devices())).platform == "gpu"
    for observed, wanted in zip(actual, expected):
        np.testing.assert_allclose(observed, wanted, atol=2e-10, rtol=2e-11)
