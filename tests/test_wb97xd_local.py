"""Local omegaB97X-D channel against LibXC; does not include exact exchange."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft.libxc_jax.jax_xc_adapter import (
    eval_jax_xc_energy_density_from_unrestricted_density_gradients,
)


@pytest.mark.parametrize("point", [
    [[0.35, 0.04, -0.08, 0.03], [0.21, -0.05, 0.02, 0.07]],
    [[0.025, 0.03, 0.02, -0.01], [0.08, -0.02, 0.01, 0.05]],
    [[1.3, 0.4, -0.2, 0.1], [0.9, 0.3, 0.15, -0.2]],
])
def test_wb97xd_local_energy_potential_kernel_against_libxc(point):
    pytest.importorskip("jax_xc")
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.asarray(point, dtype=np.float64)
    reference = [libxc.eval_xc_eff("WB97X_D", rho[:, :, None], deriv=i, spin=1)
                 for i in range(3)]

    def energy(x):
        return eval_jax_xc_energy_density_from_unrestricted_density_gradients(
            "hyb_gga_xc_wb97x_d", x[0, 0], x[1, 0], x[0, 1:], x[1, 1:],
            allow_experimental_jax_xc=True,
        )

    value, potential = jax.value_and_grad(energy)(jnp.asarray(rho))
    kernel = jax.hessian(energy)(jnp.asarray(rho))
    errors = [float(abs(value - rho[:, 0].sum() * reference[0][0])),
              float(np.max(np.abs(potential - reference[1][..., 0]))),
              float(np.max(np.abs(kernel - reference[2][..., 0])))]
    print("wb97xd local energy/potential/kernel maxabs errors:", errors)
    np.testing.assert_allclose(value, rho[:, 0].sum() * reference[0][0], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(potential, reference[1][..., 0], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(kernel, reference[2][..., 0], rtol=2e-9, atol=2e-10)


def test_wb97xd_preset_exposes_only_validated_local_channel():
    from gradscf.dft.libxc_jax.jax_libxc import get_rsh_functional_preset, parse_xc
    from gradscf.dft.libxc_jax.jax_xc_adapter import jax_xc_functional_info

    preset = get_rsh_functional_preset("wb97xd")
    assert preset.jax_local_xc_spec == "hyb_gga_xc_wb97x_d"
    assert len(preset.local_term_specs) == 1
    assert preset.local_term_specs[0].fixed_omega == 0.2
    assert not preset.strict_jax_supported
    assert jax_xc_functional_info(preset.jax_local_xc_spec).status == "wrapped"
    assert parse_xc(preset.jax_local_xc_spec)[0].kind == "semilocal"
    with pytest.raises(KeyError):
        parse_xc("wb97xd")


def test_wb97xd_local_rejects_unvalidated_omega():
    with pytest.raises(ValueError, match="fixed.*0.2"):
        eval_jax_xc_energy_density_from_unrestricted_density_gradients(
            "hyb_gga_xc_wb97x_d", 0.3, 0.2, jnp.ones(3)*0.01,
            jnp.ones(3)*0.02, omega=0.3, allow_experimental_jax_xc=True,
        )


@pytest.mark.parametrize("point", [[0.8, 0.08, -0.03, 0.07], [0.03, 0.01, 0.02, 0.03]])
def test_wb97xd_restricted_local_energy_potential_kernel(point):
    from gradscf.dft.libxc_jax.jax_libxc import restricted_feature_bundle_from_rho_grad_tau
    from gradscf.dft.libxc_jax.jax_xc_adapter import eval_jax_xc_from_restricted_features
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.asarray(point, dtype=np.float64)
    reference = [libxc.eval_xc_eff("WB97X_D", rho[:, None], deriv=i, spin=0)
                 for i in range(3)]

    def energy(x):
        features = restricted_feature_bundle_from_rho_grad_tau(x[0], x[1:])
        return x[0] * eval_jax_xc_from_restricted_features("hyb_gga_xc_wb97x_d", features)

    value, potential = jax.value_and_grad(energy)(jnp.asarray(rho))
    kernel = jax.hessian(energy)(jnp.asarray(rho))
    np.testing.assert_allclose(value, rho[0]*reference[0][0], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(potential, reference[1][..., 0], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(kernel, reference[2][..., 0], rtol=2e-9, atol=2e-10)
