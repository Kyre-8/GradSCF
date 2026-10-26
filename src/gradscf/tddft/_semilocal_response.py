from __future__ import annotations

from dataclasses import dataclass
import weakref
import jax.numpy as jnp

from gradscf.tools.features import _contains_tracer, restricted_grid_response_variables
from gradscf.dft.xc import parse_recipe
from gradscf.dft.xc.evaluate import grid_hessian


_GRID_RESPONSE_TENSOR_CACHE: dict[
    tuple[int, str],
    tuple[weakref.ReferenceType[object], object],
] = {}


@dataclass(frozen=True)
class SemilocalResponseFunctional:
    xc_spec: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "xc_spec", str(self.xc_spec).lower())
        recipe = parse_recipe(self.xc_spec)
        if recipe.needs_laplacian:
            raise NotImplementedError("Restricted response supports tau-dependent MGGA; density Laplacians are unsupported.")
        coefficients = recipe.to_range_separated_hybrid_coefficients()
        fraction = recipe.exact_exchange.full_range if coefficients is None else coefficients[2]
        object.__setattr__(self, "exact_exchange_fraction", float(fraction))
        object.__setattr__(self, "range_separated_coefficients",
                           coefficients)
        kind = recipe.kind
        object.__setattr__(self, "response_feature_kind", "LDA" if kind == "HF" else kind)

    def grid_response_tensor(self, molecule):
        recipe = parse_recipe(self.xc_spec)
        recipe.require_complete_support(provided=() if getattr(molecule, "dispersion_energy", None) is None
                                        else ("dispersion",))
        if recipe.kind == "HF":
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
        # A restricted singlet perturbs equal alpha/beta fields. Project the
        # same spin Hessian used by UKS, including both kinetic densities.
        kind = self.response_feature_kind
        width = {"LDA": 1, "GGA": 4, "MGGA": 5}[kind]
        spin_width = {"LDA": 2, "GGA": 8, "MGGA": 10}[kind]
        projection = jnp.zeros((spin_width, width), dtype=rho.dtype)
        projection = projection.at[:2, 0].set(.5)
        blocks = [rho[:, None]]
        if kind in {"GGA", "MGGA"}:
            blocks.append(grad_rho)
            projection = projection.at[2:5, 1:4].set(.5*jnp.eye(3, dtype=rho.dtype))
            projection = projection.at[5:8, 1:4].set(.5*jnp.eye(3, dtype=rho.dtype))
        if kind == "MGGA":
            blocks.append(tau[:, None])
            projection = projection.at[8:10, 4].set(.5)
        fields = jnp.concatenate(blocks, axis=-1)@projection.T
        active = rho > getattr(molecule, "xc_density_floor", 1e-12)
        fields = jnp.where(active[:, None], fields, jnp.ones_like(fields))
        spin_hessian = grid_hessian(self.xc_spec, kind)(fields)
        tensor = jnp.einsum("ai,gab,bj->ijg", projection, spin_hessian, projection)
        tensor = jnp.where(active[None, None, :], tensor, 0.)
        if not _contains_tracer(tensor):
            reference = weakref.ref(molecule, lambda _: _GRID_RESPONSE_TENSOR_CACHE.pop(cache_key, None))
            _GRID_RESPONSE_TENSOR_CACHE[cache_key] = (reference, tensor)
        return tensor
