"""PySCF-style DFT namespace for GradSCF.

Lazy ``__getattr__`` dispatch (same pattern as ``gradscf/__init__.py``) so
that importing the leaf subpackage ``gradscf.dft.xc`` does not pull
in the SCF facade chain (which would create a circular import via
``scf -> dft``).
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Functional": "dft.functional",
    "custom_xc": "dft.functional",
    "DensityInputs": "dft.functional",
    "XCEnergyPotentialResult": "dft.derivatives",
    "xc_energy_and_potential_from_density": "dft.derivatives",
    "xc_kernel_action": "dft.derivatives",
    "RKS": "scf",
    "UKS": "scf",
    "ROKS": "scf.roks",
    "ROKSConfig": "scf.roks",
    "ROKSResult": "scf.roks",
    "run_roks_from_integrals": "scf.roks",
    "GKS": "scf.gks",
    "GKSConfig": "scf.gks",
    "GKSResult": "scf.gks",
    "run_gks_from_integrals": "scf.gks",
    "RKSConfig": "scf.rks",
    "RKSResult": "scf.rks",
    "UKSConfig": "scf.uks",
    "UKSResult": "scf.uks",
    "run_rks_from_integrals": "scf.rks",
    "run_uks_from_integrals": "scf.uks",
    "CLASSIC_XC_SPECS": "dft.xc",
    "TraditionalXCFunctional": "dft.xc",
    "eval_xc_energy_density": "dft.xc",
    "eval_xc_response_tensor": "dft.xc",
    "hybrid_coeff": "dft.xc",
    "make_b3lyp_functional": "dft.xc",
    "make_classic_xc_functional": "dft.xc",
    "make_lda_functional": "dft.xc",
    "make_pbe0_functional": "dft.xc",
    "make_pbe_functional": "dft.xc",
    "parse_xc": "dft.xc",
    "semilocal_terms": "dft.xc",
    "xc_type": "dft.xc",
}

_SUBMODULES = ("xc", "functional", "derivatives", "hfx", "pt2", "dispersion")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    if name in _EXPORTS:
        module_name = _EXPORTS[name]
        if module_name.startswith("dft."):
            module = import_module(f".{module_name[4:]}", __name__)
        else:
            module = import_module(f"..{module_name}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_EXPORTS) + list(_SUBMODULES)
