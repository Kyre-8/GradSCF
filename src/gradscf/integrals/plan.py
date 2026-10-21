"""Reusable static integral plans with freshly bound basis parameters."""
from dataclasses import dataclass, replace
from functools import lru_cache
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np

from .basis import (BasisTopology, BasisParameters, CartesianAO, CartesianBasis,
                    ContractedShell, cartesian_angular_tuples)
from .basis.normalization import normalized_shell_coefficients, radial_primitive_norm
from .capabilities import backend_capabilities
from .backends.native import packing as native_packing
from .basis.types import check_parameter_shapes


def _validate_omega(omega):
    """Static nonnegative erf-Coulomb range parameter, in inverse Bohr."""
    if isinstance(omega, jax.core.Tracer):
        raise NotImplementedError("omega must be static; omega differentiation is not supported.")
    if isinstance(omega, (bool, np.bool_)) or not isinstance(omega, Real):
        raise TypeError("omega must be a static real scalar.")
    omega = float(omega)
    if not np.isfinite(omega) or omega < 0 or omega > np.sqrt(np.finfo(np.float64).max) / 2:
        raise ValueError("omega must be finite, nonnegative and at most sqrt(float64.max)/2.")
    return omega


def _reference_basis(topology, parameters, *, precompute_eri_groups=True):
    if not topology.cart:
        raise NotImplementedError("The JAX reference backend supports Cartesian basis functions only.")
    aos, shells = [], []
    for l, a, raw, center in zip(topology.angular_momenta, parameters.exponents,
                                 parameters.coefficients, parameters.centers):
        coefficients = normalized_shell_coefficients(l, a, raw) / radial_primitive_norm(l, a)[:, None]
        angulars = tuple(cartesian_angular_tuples(l))
        for column in range(raw.shape[1]):
            start = len(aos)
            coeff = coefficients[:, column]
            aos.extend(CartesianAO(center, angular, a, coeff) for angular in angulars)
            shells.append(ContractedShell(center, angulars, a, coeff,
                                           np.arange(start, len(aos), dtype=np.int32)))
    return CartesianBasis(tuple(aos), shells=tuple(shells),
                          precompute_eri_groups=precompute_eri_groups,
                          atom_coords=parameters.nuclear_coords,
                          atom_charges=jnp.asarray(topology.nuclear_charges))


@dataclass(frozen=True)
class IntegralPlan:
    topology: BasisTopology
    backend: str
    atm: object
    bas: object
    env_size: int

    def pack(self, parameters, *, origin=None):
        """Pack native tables with current differentiable basis parameters."""
        return native_packing.pack(self, parameters, origin=origin)

    def _pack(self, parameters, coefficients, *, origin=None):
        return native_packing.pack_environment(self, parameters, coefficients, origin=origin)

    def coefficient_data(self, parameters, *, active_shells=None, separate_exponents=False):
        """Separate normalized coefficients and optional exponents from ENV."""
        return native_packing.coefficient_data(
            self, parameters, active_shells=active_shells,
            separate_exponents=separate_exponents)

    def get_jk(self, parameters, density, *, screening_threshold=0., workers=1,
               with_j=True, with_k=True, omega=0.):
        """Native shell-direct J/K with density and coordinate response.

        Coordinate AD requires zero screening cutoff. Numerical exponents
        and coefficients remain separate unsupported differentiation inputs.
        A static positive omega selects erf(omega*r12)/r12 in inverse Bohr;
        omega=0 retains ordinary Coulomb, including its exact density response.
        """
        omega = _validate_omega(omega)
        if self.backend!='native':raise NotImplementedError('get_jk requires the native backend.')
        from .backends.native.jk import direct_jk
        fixed=replace(parameters,centers=jnp.zeros_like(parameters.centers),
                      nuclear_coords=jnp.zeros_like(parameters.nuclear_coords))
        atm,bas,env=self.pack(fixed)
        if omega:
            env=env.at[8].set(omega)  # libcint PTR_RANGE_OMEGA
        coordinates=jnp.concatenate((parameters.nuclear_coords,parameters.centers))
        if density.shape[-2:]!=(self.topology.nao,)*2:raise ValueError('Density AO dimensions do not match the plan.')
        return direct_jk(atm,bas,env,density,cart=self.topology.cart,
                         cutoff=screening_threshold,workers=workers,
                         with_j=with_j,with_k=with_k,coordinates=coordinates)

    def evaluate(self, operator, parameters, *, origin=None, ecps=None, aosym='s1', omega=0.0,
                 shell_slice=None):
        """Bind current values and evaluate an operator.

        Reference one-electron calls skip ERI layouts. Reference eager ERI
        calls still rebuild their layouts when binding the current parameters.
        For native ERIs, fixed ``omega > 0`` selects ``erf(omega*r12)/r12``;
        zero retains ordinary Coulomb. Omega is a static nonnegative scalar
        in inverse Bohr; its derivatives and short-range ERIs are unsupported.
        Long-range s4/s8 values and geometry AD currently gather from the full
        ERI, so their temporary storage remains O(nao**4).
        A static eight-endpoint ``shell_slice`` selects four half-open native
        s1 ERI shell ranges, with local AO axes and geometry AD through order
        two. On CUDA, bounded windows/products use the CPU shell engine.
        """
        omega = _validate_omega(omega)
        if shell_slice is not None:
            if operator != "eri":
                raise ValueError("shell_slice applies only to ERI.")
            if aosym != "s1":
                raise ValueError("shell_slice requires the s1 ERI layout.")
            if self.backend != "native":
                raise NotImplementedError("shell_slice requires the native backend.")
            from .backends.native.ffi import _eri_shell_slice
            shell_slice, _ = _eri_shell_slice(self.bas, self.topology.cart, shell_slice)
        if omega and operator != "eri":
            raise ValueError("Nonzero omega applies only to the eri operator.")
        if omega and self.backend != "native":
            raise NotImplementedError("Long-range omega requires the native backend.")
        if aosym not in {'s1','s4','s8'}:raise ValueError('aosym must be s1, s4 or s8.')
        if aosym!='s1':
            if operator!='eri' or origin is not None or ecps is not None:
                raise ValueError('Packed layouts apply only to ordinary ERI.')
            if self.backend!='native':raise NotImplementedError('Packed plans require the native backend.')
            if omega:
                full = self.evaluate("eri", parameters, omega=omega)
                rows, cols = np.tril_indices(self.topology.nao)
                pair = full[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
                return pair if aosym == 's4' else pair[np.tril_indices(len(rows))]
            from .backends.native.eri import evaluate
            atm,bas,env=self.pack(parameters)
            npair=self.topology.nao*(self.topology.nao+1)//2
            shape=(npair,npair) if aosym=='s4' else (npair*(npair+1)//2,)
            return evaluate(atm,bas,env,shape,layout=0 if aosym=='s4' else 1,cart=self.topology.cart)
        if operator == "ecp":
            if origin is not None: raise ValueError("origin is not used for ECP integrals.")
            from .basis.ecp import evaluate_ecp
            return evaluate_ecp(self, parameters, ecps)
        if ecps is not None: raise ValueError("ecps is only used for the ecp operator.")
        caps = backend_capabilities(self.backend)
        if not caps.supports(operator):
            raise NotImplementedError(f"{operator!r} is not supported by {self.backend}.")
        check_parameter_shapes(self.topology, parameters)
        if origin is not None and operator != "dipole":
            raise ValueError("origin is only meaningful for the dipole operator.")
        if origin is not None:
            origin = jnp.asarray(origin)
            if origin.shape != (3,):
                raise ValueError("Integral origin must have shape (3,).")
        if operator == "dipole" and origin is None:
            charges = jnp.asarray(self.topology.nuclear_charges)
            total = sum(self.topology.nuclear_charges)
            origin = (jnp.einsum("a,ar->r", charges, parameters.nuclear_coords) / total
                      if total else jnp.zeros(3))
        if self.backend == "native":
            fixed = replace(parameters, centers=jnp.zeros_like(parameters.centers),
                            nuclear_coords=jnp.zeros_like(parameters.nuclear_coords))
            coords = jnp.concatenate([parameters.nuclear_coords, parameters.centers], axis=0)
            origin = jnp.zeros(3, dtype=jnp.float64) if origin is None else origin
            if operator != "eri":
                from .backends.native.autodiff.evaluation import evaluate_differentiable
                atm, bas, env, coefficients, exponents = self.coefficient_data(fixed,separate_exponents=True)
                return evaluate_differentiable(operator, atm, bas, env, coefficients,
                                             coords, origin, self.topology.nao,
                                             cart=self.topology.cart,exponents=exponents)
            from .backends.native.autodiff.geometry import evaluate_geometry
            atm, bas, env = self.pack(fixed)
            if omega:
                env = env.at[8].set(omega)  # libcint PTR_RANGE_OMEGA
            return evaluate_geometry(operator, atm, bas, env, coords, origin,
                                     self.topology.nao, cart=self.topology.cart,
                                     shell_slice=shell_slice)
        from .backends import jax_reference
        names = {"overlap": "overlap_matrix", "kinetic": "kinetic_matrix",
                 "nuclear": "nuclear_attraction_matrix", "dipole": "dipole_matrix", "eri": "eri_tensor"}
        kwargs = {"origin": origin} if operator == "dipole" else {}
        basis = _reference_basis(self.topology, parameters, precompute_eri_groups=operator == "eri")
        return getattr(jax_reference, names[operator])(basis, **kwargs)


@lru_cache(maxsize=32)
def make_plan(topology, *, backend="native"):
    """Cache only topology, index/pointer arrays and backend identity."""
    if backend not in {"native", "jax_reference"}:
        raise ValueError("Integral plans support native or jax_reference backends.")
    atm, bas, offset = native_packing.build_tables(topology)
    return IntegralPlan(topology, backend, atm, bas, offset)
