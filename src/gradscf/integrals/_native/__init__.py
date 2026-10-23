"""Private native library loading; public access is through integrals backends."""

from functools import lru_cache
import ctypes
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def register_integrals():
    """Load only GradSCF's own CPU library and keep its handle alive."""
    import jax

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = Path(__file__).parent / "lib" / ("libgradscf_integrals" + suffix)
    if not library.is_file():
        raise RuntimeError(
            "GradSCF native CPU integrals are not built. Run "
            "`python -m gradscf.integrals._native.build` (requires CMake and a C/C++ compiler)."
        )
    handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
    targets = {"gradscf_ecp_cpu_v1": "GradSCFECP", "gradscf_integrals_cpu_v1": "GradSCFIntegrals",
               "gradscf_eri_slice_cpu_v1": "GradSCFERISlice",
               "gradscf_eri_slice_geometry_jvp_cpu_v1": "GradSCFERISliceGeometryJVP",
               "gradscf_eri_slice_geometry_vjp_cpu_v1": "GradSCFERISliceGeometryVJP",
               "gradscf_eri_slice_geometry_hessian_jvp_cpu_v1": "GradSCFERISliceGeometryHessianJVP",
               "gradscf_eri_slice_geometry_hessian_vjp_cpu_v1": "GradSCFERISliceGeometryHessianVJP",
               "gradscf_eri_slab_cpu_v1": "GradSCFERISlab",
               "gradscf_eri_slab_geometry_jvp_cpu_v1": "GradSCFERISlabGeometryJVP",
               "gradscf_eri_slab_geometry_vjp_cpu_v1": "GradSCFERISlabGeometryVJP",
               "gradscf_eri_slab_geometry_hessian_jvp_cpu_v1": "GradSCFERISlabGeometryHessianJVP",
               "gradscf_eri_slab_geometry_hessian_vjp_cpu_v1": "GradSCFERISlabGeometryHessianVJP",
               "gradscf_geometry_jvp_cpu_v1": "GradSCFGeometryJVP",
               "gradscf_geometry_vjp_cpu_v1": "GradSCFGeometryVJP",
               "gradscf_geometry_hessian_jvp_cpu_v1": "GradSCFGeometryHessianJVP",
               "gradscf_geometry_hessian_vjp_cpu_v1": "GradSCFGeometryHessianVJP",
               "gradscf_geometry_higher_jvp_cpu_v1": "GradSCFGeometryHigherJVP",
               "gradscf_geometry_higher_vjp_cpu_v1": "GradSCFGeometryHigherVJP",
               "gradscf_coefficient_jvp_cpu_v1": "GradSCFCoefficientJVP",
               "gradscf_coefficient_vjp_cpu_v1": "GradSCFCoefficientVJP",
               "gradscf_coefficient_hessian_jvp_cpu_v1": "GradSCFCoefficientHessianJVP",
               "gradscf_coefficient_hessian_vjp_cpu_v1": "GradSCFCoefficientHessianVJP",
               "gradscf_compact_cpu_v1": "GradSCFCompact",
               "gradscf_direct_jk_cpu_v1": "GradSCFDirectJK",
               "gradscf_direct_jk_geometry_jvp_cpu_v1": "GradSCFDirectJKGeometryJVP",
               "gradscf_direct_jk_geometry_vjp_cpu_v1": "GradSCFDirectJKGeometryVJP",
               "gradscf_direct_jk_geometry_vjp_cpu_v2": "GradSCFDirectJKGeometryVJPParallel",
               "gradscf_packed_jk_cpu_v1": "GradSCFPackedJK"}
    targets.update({"gradscf_exponent_jvp_cpu_v1": "GradSCFExponentJVP",
                    "gradscf_exponent_vjp_cpu_v1": "GradSCFExponentVJP"})
    for target, symbol in targets.items():
        if not hasattr(handle, symbol):
            raise RuntimeError("Native integral library is stale; rebuild with "
                               "python -m gradscf.integrals._native.build")
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(handle, symbol)),
                                    platform="cpu", api_version=1)
    return handle
