"""Reproduce the tail-Hessian fixture from the upstream expanded expression.

Run on the numerical-validation host with mpmath, jax_xc, and PySCF installed.
This is an independent 80-decimal evaluation of jax_xc 0.0.9's generated
``pol`` code, not of GradSCF's factored local kernel. It changes only the
private function's arithmetic namespace. The installed package is untouched.
"""
import hashlib
import inspect
from importlib.metadata import version
from types import FunctionType, SimpleNamespace

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import jax_xc.impl.hyb_gga_xc_wb97 as expanded
from pyscf.dft import libxc

from gradscf.dft.libxc_jax.jax_xc_adapter import (
    eval_jax_xc_energy_density_from_unrestricted_density_gradients as evaluate,
    load_jax_xc,
)


SOURCE_SHA256 = "f4ecc4b751ac699c6e3fc3bdc871da49151c965ebad7616e33cfd85cddd028ba"
COEFFICIENTS = {
    "c_x": [.777964, .66116, .574541, -5.25671, 11.6386],
    "c_ss": [1., -6.90539, 31.3343, -51.0533, 26.4423],
    "c_ab": [1., 1.79413, -12.0477, 14.0847, -8.50809],
}


def main():
    mp.mp.dps = 80
    source = inspect.getsource(expanded)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    assert source_hash == SOURCE_SHA256, "Re-audit a changed upstream expression"
    # .033774 / 1.709921, the rounded PW92 f_zeta_zeta(0) parameter.
    assert "0.19751789702565206229e-1" in source
    module, _ = load_jax_xc()
    functional = module.hyb_gga_xc_wb97x_d(polarized=True)
    parameters = inspect.getclosurevars(functional).nonlocals["p"]
    for key, expected in COEFFICIENTS.items():
        np.testing.assert_array_equal(getattr(parameters.params, key), expected)
    p = SimpleNamespace(
        params=SimpleNamespace(**{key: [mp.mpf(str(v)) for v in values]
                                  for key, values in COEFFICIENTS.items()}),
        cam_omega=mp.mpf(str(parameters.cam_omega)),
        zeta_threshold=mp.mpf(str(parameters.zeta_threshold)),
        dens_threshold=mp.mpf(str(parameters.dens_threshold)),
    )
    arithmetic = SimpleNamespace(pi=mp.pi, cbrt=lambda x: mp.root(x, 3),
        sqrt=mp.sqrt, log=mp.log, exp=mp.exp, logical_or=lambda a, b: bool(a) or bool(b))
    namespace = dict(expanded.pol.__globals__, jnp=arithmetic,
        jax=SimpleNamespace(lax=SimpleNamespace(erf=mp.erf)),
        lax_cond=lambda condition, a, b: a if condition else b)
    original = FunctionType(expanded.pol.__code__, namespace,
                            argdefs=expanded.pol.__defaults__)
    point = np.array([[.6, .9, -.2, .4], [.4, -.3, .7, .2]])*1e-11
    fields = [[mp.mpf(str(t)) for t in row] for row in point]
    for row in fields:
        row[0] += mp.mpf("1e-15")
        row[1] += mp.mpf("1e-15")

    def energy_mp(gradient_alpha_x):
        gradients = [list(fields[0][1:]), list(fields[1][1:])]
        gradients[0][0] = gradient_alpha_x
        sigma = (sum(t*t for t in gradients[0]),
                 sum(a*b for a, b in zip(*gradients)),
                 sum(t*t for t in gradients[1]))
        density = fields[0][0], fields[1][0]
        return sum(density)*original(p, density, sigma)

    high_precision = mp.diff(energy_mp, fields[0][1], 2)
    energy = lambda x: evaluate("hyb_gga_xc_wb97x_d", x[0, 0], x[1, 0], x[0, 1:], x[1, 1:])
    actual = float(jax.hessian(energy)(jnp.asarray(point))[0, 1, 0, 1])
    effective = point.copy()
    effective[:, :2] += 1e-15
    reference = float(libxc.eval_xc_eff("WB97X_D", effective[..., None], deriv=2,
                                       spin=1)[0, 1, 0, 1, 0])
    print("jax_xc", version("jax_xc"), "source SHA256", source_hash)
    print("coefficients", COEFFICIENTS, "f_zeta_zeta(0)", 1.709921)
    print("original expanded 80dps H[ga_x,ga_x]", mp.nstr(high_precision, 50))
    print("factored JAX", actual, "error vs 80dps", float(mp.mpf(actual)-high_precision))
    print("LibXC", reference, "error vs 80dps", float(mp.mpf(reference)-high_precision))


if __name__ == "__main__":
    main()
