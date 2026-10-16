from __future__ import annotations

from dataclasses import dataclass
import weakref
import jax.numpy as jnp

from gradscf.tools.features import _contains_tracer, restricted_grid_response_variables
from gradscf.dft.libxc_jax.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type
from gradscf.scf.rsh import local_xc_spec, rsh_preset


_GRID_RESPONSE_TENSOR_CACHE: dict[
    tuple[int, str],
    tuple[weakref.ReferenceType[object], object],
] = {}


@dataclass(frozen=True)
class SemilocalResponseFunctional:
    xc_spec: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "xc_spec", str(self.xc_spec).lower())
        preset = rsh_preset(self.xc_spec)
        fraction = preset.default_sr_hf_fraction if preset is not None else hybrid_coeff(self.xc_spec)
        object.__setattr__(self, "exact_exchange_fraction", float(fraction))
        object.__setattr__(self, "range_separated_coefficients",
                           None if preset is None else preset.to_range_separated_hybrid_coefficients())
        kind = str(xc_type(local_xc_spec(self.xc_spec)))
        object.__setattr__(self, "response_feature_kind", "LDA" if kind == "HF" else kind)

    def grid_response_tensor(self, molecule):
        if xc_type(local_xc_spec(self.xc_spec)) == "HF":
            # Exact exchange is assembled separately. HF has no semilocal
            # kernel; use a single density-feature channel containing zeros.
            return jnp.zeros((1,1,len(molecule.grid.weights)),dtype=jnp.asarray(molecule.grid.weights).dtype)
        cache_key = (id(molecule), self.xc_spec)
        cached = _GRID_RESPONSE_TENSOR_CACHE.get(cache_key)
        if cached is not None:
            cached_ref, cached_tensor = cached
            if cached_ref() is molecule and not _contains_tracer(cached_tensor):
                return cached_tensor
            _GRID_RESPONSE_TENSOR_CACHE.pop(cache_key, None)
        rho, grad_rho, tau, _ = restricted_grid_response_variables(
            molecule,
            feature_kind=self.response_feature_kind,
        )
        if self.range_separated_coefficients is not None:
            # The closed-shell singlet perturbation has equal alpha/beta
            # density variations. Reuse the stable SCF spin Hessian, including
            # its one-sided offsets, then project to (rho, grad rho).
            from gradscf.scf.uks import _point_unrestricted_xc_hessian_kernel

            active = rho > molecule.xc_density_floor
            fields = jnp.concatenate([rho[:, None]/2, rho[:, None]/2,
                                       grad_rho/2, grad_rho/2], axis=-1)
            fields = jnp.where(active[:, None], fields, jnp.ones_like(fields))
            spin_hessian = _point_unrestricted_xc_hessian_kernel(
                local_xc_spec(self.xc_spec), "GGA")(fields)
            projection = jnp.asarray([[.5, 0, 0, 0], [.5, 0, 0, 0],
                [0, .5, 0, 0], [0, 0, .5, 0], [0, 0, 0, .5],
                [0, .5, 0, 0], [0, 0, .5, 0], [0, 0, 0, .5]], dtype=fields.dtype)
            tensor = jnp.einsum("ai,gab,bj->ijg", projection, spin_hessian, projection)
            tensor = jnp.where(active[None, None, :], tensor, 0.)
        else:
            _, tensor = eval_xc_response_tensor(
                self.xc_spec, rho, grad=grad_rho, tau=tau)
        if not _contains_tracer(tensor):
            _GRID_RESPONSE_TENSOR_CACHE[cache_key] = (weakref.ref(molecule), tensor)
        return tensor
