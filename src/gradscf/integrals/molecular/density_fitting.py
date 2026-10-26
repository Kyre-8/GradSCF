"""Auxiliary-basis RI from native 2c/3c integrals, never from full ERIs.

With a fixed auxiliary metric and geometry, native contracted three-center
integrals support orbital coefficient and first-order exponent AD. Primitive
RI factors may also be cached at fixed exponents and projected in JAX.
"""

from dataclasses import dataclass
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.integrals.basis import BasisTopology, BasisParameters
from gradscf.integrals.plan import make_plan


def _check_lindep(lindep):
    if not np.isfinite(lindep) or lindep <= 0:
        raise ValueError("lindep must be finite and positive.")


@dataclass(frozen=True)
class AuxiliaryMetricFactor:
    """Host-only whitening cache bound to its auxiliary basis and geometry."""

    signature: str
    lindep: float
    method: str
    matrix: object

    def __post_init__(self):
        _check_lindep(self.lindep)
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if (matrix.ndim != 2 or not all(matrix.shape)
                or not np.all(np.isfinite(matrix))):
            raise ValueError("Auxiliary metric factor must be a finite nonempty matrix.")
        if self.method == "cholesky":
            if (matrix.shape[0] != matrix.shape[1] or np.any(np.diag(matrix) <= 0)
                    or np.any(np.triu(matrix, 1) != 0)):
                raise ValueError("Auxiliary Cholesky factor must be lower triangular.")
        elif self.method != "eigh" or matrix.shape[0] > matrix.shape[1]:
            raise ValueError("Unknown auxiliary metric factorization or rank.")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    def whiten(self, three):
        from scipy.linalg import solve_triangular

        three = np.asarray(three)
        if (three.ndim != 2 or three.shape[0] != self.matrix.shape[1]
                or not np.all(np.isfinite(three))):
            raise ValueError("Three-center integrals must be finite and match the auxiliary metric.")
        factors = (solve_triangular(self.matrix, three, lower=True, check_finite=False)
                   if self.method == "cholesky" else self.matrix @ three)
        if not np.all(np.isfinite(factors)):
            raise ValueError("Whitened auxiliary factors must be finite.")
        return factors

    def whiten_differentiable(self, three):
        """Whiten orbital-parameter-dependent 3c data with a fixed metric."""
        from jax.scipy.linalg import solve_triangular

        three = jnp.asarray(three)
        if three.ndim != 2 or three.shape[0] != self.matrix.shape[1]:
            raise ValueError("Three-center shape differs from the auxiliary metric.")
        matrix = jnp.asarray(self.matrix)
        factors = (solve_triangular(matrix, three, lower=True)
                   if self.method == "cholesky" else matrix @ three)
        finite = jnp.all(jnp.isfinite(three)) & jnp.all(jnp.isfinite(factors))
        if not isinstance(finite, jax.core.Tracer) and not bool(finite):
            raise ValueError("Whitened auxiliary factors must be finite.")
        return jnp.where(finite, factors, jnp.nan)


@dataclass(frozen=True)
class AuxiliaryPlan:
    orbital: BasisTopology
    auxiliary: BasisTopology
    combined: object

    def _pack(self, parameters, auxiliary_parameters):
        p, a = parameters, auxiliary_parameters
        combined = BasisParameters(
            p.exponents + a.exponents,
            p.coefficients + a.coefficients,
            jnp.concatenate((p.centers, a.centers)),
            p.nuclear_coords,
        )
        return self.combined.pack(combined)

    def _three_center(self, parameters, auxiliary_parameters):
        from gradscf.integrals.backends.native.autodiff.evaluation import evaluate_differentiable

        p, a = parameters, auxiliary_parameters
        combined = BasisParameters(p.exponents + a.exponents,
            p.coefficients + a.coefficients, jnp.concatenate((p.centers, a.centers)),
            p.nuclear_coords)
        coords = jnp.concatenate((combined.nuclear_coords, combined.centers))
        # Coordinates and fixed auxiliaries remain explicit nondifferentiable
        # operands; orbital normalized coefficients and exponents enter native AD.
        fixed = BasisParameters(combined.exponents, combined.coefficients,
                                jnp.zeros_like(combined.centers),
                                jnp.zeros_like(combined.nuclear_coords))
        split = len(self.orbital.angular_momenta)
        atm, bas, env, coefficients, exponents = self.combined.coefficient_data(fixed,
            active_shells=split,separate_exponents=True)
        return evaluate_differentiable("three_center", atm, bas, env, coefficients,
            coords, jnp.zeros(3, dtype=jnp.float64), self.combined.topology.nao,
            cart=self.orbital.cart, split=split,exponents=exponents)

    def _signature(self, parameters, auxiliary_parameters):
        """Exclude orbital coefficients/exponents and repeated shell centers."""
        a = auxiliary_parameters
        fixed = (*a.exponents, *a.coefficients, a.centers, a.nuclear_coords,
                 parameters.centers, parameters.nuclear_coords)
        if any(isinstance(value, jax.core.Tracer) for value in fixed):
            raise NotImplementedError(
                "DF basis AD requires fixed auxiliary basis and geometry; "
                "auxiliary-parameter or geometry differentiation is not implemented.")
        digest = hashlib.sha256(repr(self.auxiliary).encode())
        digest.update(repr(self.orbital.nuclear_charges).encode())

        def add(value):
            array = np.ascontiguousarray(value, dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise ValueError("Auxiliary basis and geometry parameters must be finite.")
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
            return array

        for alpha in a.exponents:
            if np.any(add(alpha) <= 0):
                raise ValueError("Auxiliary exponents must be positive.")
        for coefficients in a.coefficients:
            add(coefficients)
        for coords in (a.centers, a.nuclear_coords, parameters.nuclear_coords):
            add(coords)
        centers = np.asarray(parameters.centers)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("Orbital geometry centers must have shape (nshell,3).")
        add(np.unique(centers, axis=0))
        return digest.hexdigest()

    def evaluate(self, parameters, auxiliary_parameters):
        from gradscf.integrals.backends.native.density_fitting import evaluate

        atm, bas, env = self._pack(parameters, auxiliary_parameters)
        n, m = self.orbital.nao, self.auxiliary.nao
        kwargs = dict(cart=self.orbital.cart, split=len(self.orbital.angular_momenta))
        metric = evaluate(atm, bas, env, (m, m), layout=2, **kwargs)
        three = evaluate(atm, bas, env, (m, n * (n + 1) // 2), layout=3, **kwargs)
        return metric, three

    def metric_factor(self, parameters, auxiliary_parameters, *, lindep=1e-12):
        """Cache the 2c metric only, for repeated 3c evaluations at fixed geometry.

        Orbital exponent/coefficient or shell-count changes preserve this cache
        when their physical centers and the complete auxiliary basis are fixed.
        This host factorization provides no exponent or geometry AD rule.
        """
        from gradscf.integrals.backends.native.density_fitting import evaluate

        _check_lindep(lindep)
        signature = self._signature(parameters, auxiliary_parameters)
        atm, bas, env = self._pack(parameters, auxiliary_parameters)
        m = self.auxiliary.nao
        metric = evaluate(atm, bas, env, (m, m), layout=2,
                          cart=self.orbital.cart, split=len(self.orbital.angular_momenta))
        return self._factor_metric(metric, signature, lindep)

    @staticmethod
    def _factor_metric(metric, signature, lindep):
        from scipy.linalg import cholesky, eigh

        m = np.asarray(metric)
        if not np.all(np.isfinite(m)):
            raise ValueError("Auxiliary Coulomb metric must be finite.")
        try:
            matrix = cholesky(m, lower=True)
            method = "cholesky"
        except np.linalg.LinAlgError:
            values, vectors = eigh(m)
            keep = values > lindep
            if not np.any(keep):
                raise ValueError("Auxiliary Coulomb metric has no positive independent modes.")
            matrix = (vectors[:, keep] / np.sqrt(values[keep])).T
            method = "eigh"
        return AuxiliaryMetricFactor(signature, lindep, method, matrix)

    def factors(self, parameters, auxiliary_parameters, *, lindep=1e-12, metric_factor=None):
        """Whiten packed three-center integrals by the auxiliary Coulomb metric.

        A host Cholesky factorization is used for this fixed-basis cache. If
        the auxiliary metric is numerically dependent, discard eigenmodes
        below ``lindep``. This rank choice is not a basis/geometry AD rule.
        Orbital coefficient AD requires a precomputed ``metric_factor``; its
        whitening is then performed in JAX. Auxiliary parameters and geometry
        must remain fixed, concrete values for cache signature validation.
        """
        _check_lindep(lindep)
        signature = self._signature(parameters, auxiliary_parameters)
        if metric_factor is None:
            metric, three = self.evaluate(parameters, auxiliary_parameters)
            metric_factor = self._factor_metric(metric, signature, lindep)
        else:
            if (not isinstance(metric_factor, AuxiliaryMetricFactor)
                    or metric_factor.signature != signature or metric_factor.lindep != lindep
                    or metric_factor.matrix.shape[1] != self.auxiliary.nao):
                raise ValueError("Cached auxiliary metric signature, geometry or lindep differs.")
            three = self._three_center(parameters, auxiliary_parameters)
            return metric_factor.whiten_differentiable(three)
        return jnp.asarray(metric_factor.whiten(three))


def make_auxiliary_plan(orbital, auxiliary):
    if orbital.cart != auxiliary.cart:
        raise ValueError("Orbital and auxiliary angular representations must match.")
    top = BasisTopology(
        orbital.angular_momenta + auxiliary.angular_momenta,
        orbital.primitive_counts + auxiliary.primitive_counts,
        orbital.contraction_counts + auxiliary.contraction_counts,
        orbital.nuclear_charges,
        orbital.cart,
    )
    return AuxiliaryPlan(orbital, auxiliary, make_plan(top))


def unpack_factors(packed, nao):
    """Unpack only a three-index tensor, never a four-index ERI."""
    packed = jnp.asarray(packed)
    if packed.ndim != 2 or packed.shape[1] != nao * (nao + 1) // 2:
        raise ValueError("Packed RI factor shape mismatch.")
    i, j = np.tril_indices(nao)
    out = jnp.zeros((packed.shape[0], nao, nao), dtype=packed.dtype)
    out = out.at[:, i, j].set(packed)
    return out.at[:, j, i].set(packed)


def project_factors(packed_primitive, transform, *, block_size=64):
    """B_c[Q] = T.T B_p[Q] T, unpacking at most block_size auxiliary rows."""
    t = jnp.asarray(transform)
    factors = jnp.asarray(packed_primitive)
    rank = factors.shape[0]
    width = min(int(block_size), rank)
    if rank == 0:
        return jnp.zeros((0, t.shape[1], t.shape[1]), dtype=t.dtype)
    if width < 1:
        raise ValueError("block_size must be positive.")

    def block(f):
        return jnp.einsum(
            "pi,Qpq,qj->Qij",
            t,
            unpack_factors(f, t.shape[0]),
            t,
            precision=jax.lax.Precision.HIGHEST,
        )

    out = jnp.zeros((rank, t.shape[1], t.shape[1]), dtype=jnp.result_type(t, factors))

    def step(i, out):
        b = jax.lax.dynamic_slice_in_dim(factors, i * width, width, axis=0)
        return jax.lax.dynamic_update_slice_in_dim(out, block(b), i * width, axis=0)

    # Recompute each block during reverse AD instead of saving every unpacked
    # primitive matrix; the loop boundary already prevents cross-step CSE.
    out = jax.lax.fori_loop(
        0, rank // width, jax.checkpoint(step, prevent_cse=False), out
    )
    start = rank // width * width
    if start < rank:
        out = out.at[start:].set(block(factors[start:]))
    return out


__all__ = ["AuxiliaryPlan", "AuxiliaryMetricFactor", "make_auxiliary_plan", "unpack_factors", "project_factors"]
