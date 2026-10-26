"""Shared differentiation policy for SCF fixed points and orbital stationary states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from jaxtyping import Array, PyTree

from ..solvers.nonlinear import ImplicitFixedPointConfig, implicit_fixed_point_solution


def normalize_scf_gradient_mode(mode: str) -> Literal['implicit', 'explicit']:
    """Normalize canonical names and the historical DFT ``impl``/``expl`` aliases."""
    modes = {'implicit': 'implicit', 'impl': 'implicit', 'explicit': 'explicit',
             'expl': 'explicit', 'unrolled': 'explicit'}
    if mode not in modes:
        raise ValueError(f"gradient_mode must be 'implicit' or 'explicit' (legacy aliases: 'impl', 'expl', 'unrolled'); got {mode!r}.")
    return modes[mode]


@dataclass(frozen=True)
class SCFDifferentiationConfig:
    """Backward policy, independent of the algorithm producing the SCF solution.

    ``explicit`` differentiates the supplied JAX computation (SCF uses lax.scan). ``implicit`` uses
    the local residual Jacobian; unconverged forward states (when required) and
    unsuccessful adjoint solves produce NaN cotangents, also under JIT.
    Regularization changes the adjoint system and therefore biases the response.
    """

    mode: Literal['implicit', 'explicit', 'impl', 'expl', 'unrolled'] = 'implicit'
    tolerance: float = 1e-9
    max_iter: int = 20
    restart: int | None = 40
    regularization: float = 0.0
    require_converged: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, 'mode', normalize_scf_gradient_mode(self.mode))
        if self.tolerance <= 0:
            raise ValueError('SCF adjoint tolerance must be positive.')
        if self.max_iter < 1 or (self.restart is not None and self.restart < 1):
            raise ValueError('SCF adjoint max_iter and restart must be positive.')
        if self.regularization < 0:
            raise ValueError('SCF adjoint regularization must be non-negative.')

    def as_implicit_config(self) -> ImplicitFixedPointConfig:
        return ImplicitFixedPointConfig(
            tolerance=self.tolerance, max_iter=self.max_iter,
            restart=self.restart, regularization=self.regularization,
        )


def attach_scf_backward(
    params: PyTree,
    *,
    solution: Array,
    residual: Callable[[Array, PyTree], Array],
    config: SCFDifferentiationConfig | None = None,
    converged: Array | bool = True,
    max_memory=None,
    reserved_bytes=0,
    operator_min_bytes=0,
) -> Array:
    """Attach a backward rule for ``residual(solution, params) = 0``.

    All differentiated physical inputs must be explicit leaves of ``params``;
    closing over traced parameters in ``residual`` is unsupported. The implicit
    mode discards derivatives of the forward trajectory and initial guess.
    It solves a matrix-free transposed residual system using the DFT GMRES core.
    The supplied convergence flag refers to the forward stationarity criterion.
    """
    cfg = SCFDifferentiationConfig() if config is None else config
    if cfg.mode == 'explicit':
        return solution
    return implicit_fixed_point_solution(
        params, solution=solution,
        fixed_point=lambda state, inputs: state - residual(state, inputs),
        config=cfg.as_implicit_config(), converged=converged,
        require_converged=cfg.require_converged,
        max_memory=max_memory, reserved_bytes=reserved_bytes,
        operator_min_bytes=operator_min_bytes,
    )
