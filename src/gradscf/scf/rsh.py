"""Internal fixed-parameter original omegaB97X-D assembly contract."""
import jax.numpy as jnp

from ..dft.libxc_jax.jax_libxc import get_rsh_functional_preset


def rsh_preset(xc_spec):
    """Recognize the supported complete method, not a local XC channel."""
    try:
        preset = get_rsh_functional_preset(xc_spec)
    except KeyError:
        return None
    return preset if preset.name == "wb97x-d" else None


def local_xc_spec(xc_spec):
    preset = rsh_preset(xc_spec)
    return preset.jax_local_xc_spec if preset is not None else xc_spec


def validate_rsh_inputs(xc_spec, eri, eri_lr, dispersion_energy, *, jk_backend="full"):
    """Require both differentiable leaves before exposing the complete name."""
    preset = rsh_preset(xc_spec)
    if preset is None:
        if eri_lr is not None or dispersion_energy is not None:
            raise ValueError("eri_lr/dispersion_energy require the complete wb97xd method.")
        return None
    if jk_backend != "full":
        raise NotImplementedError("Range-separated wb97xd requires jk_backend='full'.")
    if eri_lr is None:
        raise ValueError("Complete wb97xd requires explicit eri_lr.")
    if dispersion_energy is None:
        raise ValueError("Complete wb97xd requires explicit dispersion_energy (zero is allowed).")
    if eri is None or jnp.ndim(eri) != 4 or len(set(jnp.shape(eri))) != 1:
        raise ValueError("Complete wb97xd requires a full four-index Coulomb eri.")
    if jnp.shape(eri_lr) != jnp.shape(eri):
        raise ValueError("eri_lr must have the same full AO shape as eri.")
    if jnp.ndim(dispersion_energy) != 0:
        raise ValueError("dispersion_energy must be a scalar in Hartree.")
    return preset


def reject_rsh_response(source):
    """Keep single-exchange response consumers outside the new SCF boundary."""
    if (rsh_preset(getattr(source, "xc", "")) is not None
            or getattr(source, "range_separated_coefficients", None) is not None
            or getattr(source, "eri_lr", None) is not None):
        raise NotImplementedError(
            "General range-separated wb97xd reference response (GW/DFPT/unrestricted TD) is not supported; "
            "use the dedicated restricted singlet TD adapter or explicit integral inputs "
            "with the orbital SCF derivative solver.")
