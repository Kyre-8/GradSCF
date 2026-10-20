"""Shared real-orbital RKS/UKS stability analysis and bounded saddle escape.

Host diagnostics and bounded JAX primal checks select a solution branch;
attach implicit differentiation separately to the selected stationary root.
Internal stability does not establish a global minimum or UHF->GHF stability.
"""

from dataclasses import dataclass, replace, fields
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .orbital_optimization import _problem_functions, _rotation_functions
from ..solvers import LinearOperator, EigenSolverConfig, solve_hermitian
from .uhf import UHFResult, run_uhf_from_integrals
from .uks import UKSConfig, UKSResult, run_uks_from_integrals


@dataclass(frozen=True)
class OrbitalStabilityResult:
    stable: bool | None
    minimum_curvature: float
    eigenvalues: jax.Array
    residual_norms: jax.Array
    eigensolver_converged: bool
    mo_coeff: jax.Array
    gradient_norm: float = float("nan")
    stationary: bool = False
    spectrum_certified: bool = False
    channel: str = "internal"
    direction: jax.Array | None = None


UnrestrictedStabilityResult = OrbitalStabilityResult


@dataclass(frozen=True)
class SCFStabilizationResult:
    result: object
    stability: UnrestrictedStabilityResult
    restarts: int
    energy_history: tuple[float, ...]
    attempts: tuple = ()
    status: str = "unresolved"

    @property
    def stable(self):
        return self.stability.stable

    @property
    def minimum_curvature(self):
        return self.stability.minimum_curvature


UnrestrictedStabilizationResult = SCFStabilizationResult


@dataclass(frozen=True)
class StabilityAttempt:
    restart: int
    step: float
    energy: float
    converged: bool
    accepted: bool = False


def stabilize_df_rhf(result, *, max_restarts=4, stability_tol=1e-5,
                     eigensolver_tol=1e-7, step_sizes=(.5, 1.), **scf_inputs):
    """JIT-compatible real internal RHF check and bounded negative-mode escape.

    This selects a primal root only: attach implicit differentiation afterwards
    to the original physical inputs. Unresolved curvature or exhausted retries
    return stable=False, never a silent successful fallback. Small orbital
    spaces use a dense Hessian; larger spaces use HVP/Davidson estimates.
    Neither path materializes four-index ERIs.
    """
    from .rks import run_rks_from_integrals_traceable
    from ..integrals.molecular.jk import build_jk_from_df

    steps = _follow_controls(max_restarts, step_sizes, 1e-10)
    cfg = scf_inputs['config']
    if cfg.xc_spec.lower() != 'hf' or cfg.jk_backend != 'df':
        raise ValueError('stabilize_df_rhf requires DF-HF inputs.')
    if (not jax.config.x64_enabled or
            any(not np.isfinite(x) or x <= 0 for x in (stability_tol,eigensolver_tol))):
        raise ValueError('DF stability requires float64 and positive tolerances.')
    result = jax.tree.map(jax.lax.stop_gradient, result)
    keys = ('overlap', 'hcore', 'df_factors', 'nuclear_repulsion')
    data = {**scf_inputs, **{k: jax.lax.stop_gradient(scf_inputs[k]) for k in keys}}
    for key in ('init_density','init_mo_coeff','init_mo_energy','init_mo_occ'):
        data.pop(key,None)
    s,h,b,enuc = (data[k] for k in keys)
    if jnp.iscomplexobj(s) or jnp.iscomplexobj(result.mo_coeff) or data['nelectron'] % 2:
        raise ValueError('DF-RHF stability requires real closed-shell inputs.')
    n = s.shape[0]; no = int(data['nelectron'])//2
    if not 0 < no <= n:
        raise ValueError('Invalid DF-RHF closed-shell electron count.')
    occ = (1.,)*no + (0.,)*(n-no)
    expected_occ = 2*jnp.asarray(occ)
    dimension, rotate, density_from = _rotation_functions('roks', (occ,occ))
    if not dimension:
        valid = result.converged & jnp.all(result.mo_occ == expected_occ)
        return result, dict(stable=valid, minimum_curvature=jnp.asarray(0.,s.dtype),
                            stability_restarts=jnp.asarray(0), stability_resolved=valid,
                            stability_scf_cycles=jnp.asarray(0))
    origin = jnp.zeros(dimension, s.dtype)

    def energy(angles, coeff):
        d = jnp.sum(density_from(rotate(angles, coeff[None])), axis=0)
        j,k = build_jk_from_df(b,d)
        return jnp.sum(d*h)+.5*jnp.sum(d*j)-.25*jnp.sum(d*k)+enuc

    def analyze(current):
        gradient,apply = jax.linearize(jax.grad(lambda x:energy(x,current.mo_coeff)),origin)
        i,a = np.where(np.triu(np.asarray(occ)[:,None] != np.asarray(occ)[None,:],1))
        eps = jnp.diag(current.mo_coeff.T @ current.fock_matrix @ current.mo_coeff)
        diagonal = 4*(eps[a]-eps[i])
        operator = LinearOperator((dimension,dimension),s.dtype,apply,diagonal=diagonal,
                                  matmat=jax.vmap(apply,in_axes=1,out_axes=1))
        solved = solve_hermitian(operator,config=EigenSolverConfig(
            method='dense' if dimension<=128 else 'davidson',nroots=min(3,dimension),
            atol=eigensolver_tol,maxiter=100,max_subspace=min(40,dimension),
            initial_guess_count=min(8,dimension)))
        resolved = (current.converged & jnp.all(current.mo_occ == expected_occ) & jnp.all(solved.converged)
            & jnp.all(jnp.isfinite(solved.values))
            & (jnp.linalg.norm(gradient) <= 1e-7))
        minimum = solved.values[0]
        return resolved & (minimum >= -stability_tol),minimum,solved.vectors[:,0],resolved

    stable,minimum,direction,resolved = analyze(result)
    signed_steps = jnp.asarray([sign*step for step in steps for sign in (1.,-1.)],s.dtype)
    def condition(state):
        _,stable,_,_,resolved,count,improved,_ = state
        return (~stable) & resolved & improved & (count < max_restarts)

    def escape(state):
        current,_,_,direction,_,count,_,cycles = state
        def trial(carry, step):
            best,cycles = carry
            c = rotate(step*direction,current.mo_coeff[None])[0]
            candidate = run_rks_from_integrals_traceable(**data,
                init_mo_coeff=c,init_mo_energy=current.mo_energy)
            accept = (candidate.converged & jnp.isfinite(candidate.total_energy)
                      & (candidate.total_energy < best.total_energy-1e-10))
            return (jax.lax.cond(accept,lambda _:candidate,lambda _:best,None),
                    cycles+candidate.cycles),None
        (best,cycles),_ = jax.lax.scan(trial,(current,cycles),signed_steps)
        stable,minimum,direction,resolved = analyze(best)
        return best,stable,minimum,direction,resolved,count+1,best.total_energy<current.total_energy-1e-10,cycles

    result,stable,minimum,_,resolved,count,_,cycles = jax.lax.while_loop(condition,escape,
        (result,stable,minimum,direction,resolved,jnp.asarray(0),jnp.asarray(True),jnp.asarray(0)))
    return result,dict(stable=stable,minimum_curvature=minimum,
                       stability_restarts=count,stability_resolved=resolved,stability_scf_cycles=cycles)


def _curvature_check(
    *,
    coeff,
    dimension,
    rotate,
    energy,
    args,
    diagonal,
    converged,
    stability_tol,
    eigensolver_tol,
    max_cycle,
    gradient_tol=1e-6,
    solver="davidson",
    max_space=40,
    reference_gradient_norm=None,
):
    if not jax.config.x64_enabled:
        raise ValueError("Stability requires JAX float64")
    if (
        any(
            not np.isfinite(t) or t <= 0
            for t in (stability_tol, eigensolver_tol, gradient_tol)
        )
        or max_cycle < 1
        or max_space < 1
    ):
        raise ValueError("Stability tolerances and limits must be positive")
    if solver not in {"dense", "davidson"}:
        raise ValueError("Stability solver must be dense or davidson")
    empty = jnp.zeros(0, dtype=coeff.dtype)
    if not converged:
        return OrbitalStabilityResult(None, float("nan"), empty, empty, False, coeff)
    if dimension == 0:
        return OrbitalStabilityResult(
            True, float("inf"), empty, empty, True, coeff, 0.0, True, True
        )
    zero = jnp.zeros(dimension, dtype=coeff.dtype)
    gradient, hvp = jax.linearize(jax.grad(lambda x: energy(x, coeff, args)), zero)
    norm = float(0.5 * jnp.linalg.norm(gradient))
    if reference_gradient_norm is not None:
        norm = float(np.maximum(norm, reference_gradient_norm))
    if not np.isfinite(norm) or norm > gradient_tol:
        return OrbitalStabilityResult(
            None, float("nan"), empty, empty, False, coeff, norm
        )
    apply = jax.jit(jax.vmap(hvp, in_axes=1, out_axes=1))
    op = LinearOperator(
        (dimension, dimension), coeff.dtype, hvp, diagonal=diagonal, matmat=apply
    )
    solved = solve_hermitian(
        op,
        config=EigenSolverConfig(
            method=solver,
            nroots=min(3, dimension),
            atol=eigensolver_tol,
            maxiter=max_cycle,
            max_subspace=min(max_space, dimension),
            initial_guess_count=min(8, dimension),
        ),
    )
    values, vectors = solved.values, solved.vectors
    residuals = jnp.linalg.norm(apply(vectors) - vectors * values, axis=0)
    valid = bool(
        jnp.all(solved.converged)
        & jnp.all(jnp.isfinite(values))
        & jnp.all(residuals <= eigensolver_tol * 1.01)
    )
    minimum = float(values[0])
    stable = bool(minimum >= -stability_tol) if valid else None
    proposed = rotate(vectors[:, 0], coeff) if stable is False else coeff
    return OrbitalStabilityResult(
        stable,
        minimum,
        values,
        residuals,
        valid,
        proposed,
        norm,
        True,
        valid and solver == "dense",
        direction=vectors[:, 0] if stable is False else None,
    )


def _unrestricted_problem(
    coeff,
    occ,
    *,
    hcore,
    eri,
    nuclear_repulsion,
    ao,
    ao_deriv1,
    grid_weights,
    config,
    fock=None,
):
    if config.jk_backend != "full" or config.potential_clip is not None:
        raise NotImplementedError(
            "Stability requires full/packed ERIs and an unclipped XC potential."
        )
    if jnp.iscomplexobj(coeff):
        raise ValueError("Internal unrestricted stability requires real orbitals.")
    dimension, rotate, density_from, evaluate, energy = _problem_functions(
        "uks", config.xc_spec, "col", tuple(map(tuple, occ)), uks_config=config
    )
    args = dict(
        hcore=hcore,
        eri=jnp.asarray(eri),
        nuclear_repulsion=jnp.asarray(nuclear_repulsion),
        ao=jnp.asarray(ao),
        ao_deriv1=jnp.asarray(ao_deriv1),
        grid_weights=jnp.asarray(grid_weights),
    )
    if fock is None:
        _, fock = evaluate(density_from(coeff), args)
    eps = jnp.einsum("spi,spq,sqi->si", coeff, fock, coeff)
    gaps = []
    for spin in range(2):
        i, a = np.where(np.triu(occ[spin, :, None] != occ[spin, None, :], 1))
        gaps.append(2 * (occ[spin, i] - occ[spin, a]) * (eps[spin, a] - eps[spin, i]))
    return dimension, rotate, energy, args, jnp.concatenate(gaps)


def uks_stability(
    result: UKSResult | UHFResult,
    *,
    eri,
    ao,
    ao_deriv1,
    grid_weights,
    config: UKSConfig,
    stability_tol=1e-5,
    eigensolver_tol=1e-7,
    max_cycle=100,
    gradient_tol=1e-6,
    solver="davidson",
    max_space=40,
) -> UnrestrictedStabilityResult:
    """Real internal UHF/UKS curvature from the original spin energy evaluator.

    The shared public eigensolver sees HVP actions. Positive partial Ritz values
    are estimates, not global stability certificates. This host diagnostic does
    not require isolated-eigenvalue AD validity at a degenerate curvature.
    """
    coeff = jnp.stack([result.mo_coeff_alpha, result.mo_coeff_beta])
    occ = np.stack([result.mo_occ_alpha, result.mo_occ_beta])
    dimension, rotate, energy, args, diagonal = _unrestricted_problem(
        coeff,
        occ,
        hcore=result.hcore_matrix,
        eri=eri,
        nuclear_repulsion=result.nuclear_repulsion,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=grid_weights,
        config=config,
        fock=jnp.stack([result.fock_matrix_alpha, result.fock_matrix_beta]),
    )
    return _curvature_check(
        coeff=coeff,
        dimension=dimension,
        rotate=rotate,
        energy=energy,
        args=args,
        diagonal=diagonal,
        converged=result.converged,
        stability_tol=stability_tol,
        eigensolver_tol=eigensolver_tol,
        max_cycle=max_cycle,
        gradient_tol=gradient_tol,
        solver=solver,
        max_space=max_space,
    )


def _validate_stability_source(source):
    from .rsh import reject_rsh_response

    reject_rsh_response(source)
    if callable(getattr(source, '_is_external_xc', None)) and source._is_external_xc():
        raise NotImplementedError('Orbital stability for external XC energies is not implemented.')


def restricted_stability(
    source,
    *,
    stability_tol=1e-5,
    eigensolver_tol=1e-7,
    max_cycle=100,
    gradient_tol=None,
    solver="davidson",
    max_space=40,
    channel="internal",
):
    """Real internal or alpha/beta-opposite spin-channel restricted stability."""
    if channel not in {"internal", "spin"}:
        raise ValueError("Restricted stability channel must be internal or spin")
    from .diagnostics import _restricted_scf_state
    from .rks import _make_jk_builder, _energy_and_raw_fock_for_density
    from ..dft.xc import hybrid_coeff, xc_type

    _validate_stability_source(source)
    result, inputs = _restricted_scf_state(source)
    cfg = source._config()
    if cfg.jk_backend not in {"full","df"} or cfg.potential_clip is not None:
        raise NotImplementedError(
            "Restricted stability requires full/packed or DF ERIs and unclipped XC"
        )
    if cfg.jk_backend == 'df' and channel != 'internal':
        raise NotImplementedError('DF restricted stability currently supports the internal channel.')
    occ = np.asarray(result.mo_occ)
    if not np.all((occ == 0) | (occ == 2)):
        raise ValueError("Restricted stability requires integer 0/2 occupations")
    topology = tuple(map(tuple, np.stack([occ / 2, occ / 2])))
    dimension, rotate, density_from = _rotation_functions("roks", topology)
    pair = None if inputs.eri is not None or cfg.jk_backend == 'df' else inputs.response_eri_pair_matrix()
    if cfg.jk_backend == 'full' and inputs.eri is None and pair is None:
        raise ValueError("Restricted stability requires available AO ERIs")
    if channel == "spin":
        spin_cfg = UKSConfig(
            **{
                f.name: getattr(cfg, f.name)
                for f in fields(UKSConfig)
                if hasattr(cfg, f.name)
            }
        )
        c = jnp.stack([result.mo_coeff, result.mo_coeff])
        full_dim, rotate_full, energy_full, args, full_diagonal = _unrestricted_problem(
            c,
            np.stack([occ / 2, occ / 2]),
            hcore=inputs.hcore,
            eri=inputs.eri if inputs.eri is not None else pair,
            nuclear_repulsion=inputs.nuclear_repulsion,
            ao=inputs.ao,
            ao_deriv1=inputs.ao_deriv1,
            grid_weights=inputs.grid_weights,
            config=spin_cfg,
        )

        def spin_angles(x):
            return jnp.concatenate([x, -x])

        def rotate_spin(x, base):
            return rotate_full(spin_angles(x), base)

        def energy_spin(x, base, parameters):
            return energy_full(spin_angles(x), base, parameters)

        origin = jnp.zeros(full_dim, dtype=c.dtype)
        # A zero spin-projected gradient alone does not establish stationarity.
        full_gradient = jax.grad(energy_full)(origin, c, args)
        reference_norm = float(jnp.linalg.norm(full_gradient) / jnp.sqrt(2.0))
        if abs(float(energy_full(origin, c, args)) - float(result.total_energy)) > 1e-8:
            raise ValueError(
                "Restricted and unrestricted reference energies are inconsistent"
            )
        check = _curvature_check(
            coeff=c,
            dimension=full_dim // 2,
            rotate=rotate_spin,
            energy=energy_spin,
            args=args,
            diagonal=2 * full_diagonal[: full_dim // 2],
            converged=bool(source.converged and result.converged),
            stability_tol=stability_tol,
            eigensolver_tol=eigensolver_tol,
            max_cycle=max_cycle,
            gradient_tol=source.conv_tol_grad if gradient_tol is None else gradient_tol,
            solver=solver,
            max_space=max_space,
            reference_gradient_norm=reference_norm,
        )
        return replace(check, channel="spin")
    alpha = jnp.asarray(hybrid_coeff(cfg.xc_spec), dtype=result.mo_coeff.dtype)
    kind = xc_type(cfg.xc_spec)
    if kind not in {"HF", "LDA", "GGA"}:
        raise NotImplementedError(
            "Restricted stability supports HF/LDA/GGA/global hybrids"
        )
    jk = _make_jk_builder(
        inputs.eri,
        cfg,
        eri_pair_matrix=pair,
        df_factors=inputs.df_factors,
        with_k=abs(float(alpha)) > 1e-14,
        nocc=inputs.nelectron // 2,
    )

    def energy(angles, base, args):
        c = rotate(angles, base)
        density = jnp.sum(density_from(c), axis=0)
        return _energy_and_raw_fock_for_density(
            density=density,
            mo_coeff=c[0],
            mo_occ=result.mo_occ,
            mo_energy=result.mo_energy,
            jk_builder=jk,
            ao=inputs.ao,
            ao_deriv1=inputs.ao_deriv1,
            weights=inputs.grid_weights,
            h=inputs.hcore,
            enuc=inputs.nuclear_repulsion,
            alpha=alpha,
            cfg=cfg,
            xc_kind=kind,
        )[0]

    c = result.mo_coeff
    eps = jnp.diag(c.T @ result.fock_matrix @ c)
    i, a = np.where(np.triu(occ[:, None] != occ[None, :], 1))
    diagonal = 2 * (occ[i] - occ[a]) * (eps[a] - eps[i])
    check = _curvature_check(
        coeff=c[None],
        dimension=dimension,
        rotate=rotate,
        energy=energy,
        args={},
        diagonal=diagonal,
        converged=bool(source.converged and result.converged),
        stability_tol=stability_tol,
        eigensolver_tol=eigensolver_tol,
        max_cycle=max_cycle,
        gradient_tol=source.conv_tol_grad if gradient_tol is None else gradient_tol,
        solver=solver,
        max_space=max_space,
    )
    return replace(check, mo_coeff=check.mo_coeff[0])


# Keep the original UHF result names as compatibility aliases.
UHFStabilityResult = UnrestrictedStabilityResult
UHFStabilizationResult = UnrestrictedStabilizationResult


def uhf_stability(result: UHFResult, *, eri, **controls) -> UnrestrictedStabilityResult:
    """HF specialization of the shared unrestricted curvature check."""
    n = result.hcore_matrix.shape[0]
    return uks_stability(
        result,
        eri=eri,
        ao=jnp.zeros((0, n)),
        ao_deriv1=jnp.zeros((4, 0, n)),
        grid_weights=jnp.zeros(0),
        config=UKSConfig(xc_spec="hf"),
        **controls,
    )


def stabilize_uhf_from_integrals(
    *,
    max_restarts=5,
    stability_tol=1e-5,
    eigensolver_tol=1e-7,
    step_sizes=(0.5, 1.0),
    energy_tol=1e-10,
    **scf_inputs,
):
    """UHF specialization of the shared bounded stability/restart workflow."""
    return _stabilize(
        run_uhf_from_integrals,
        lambda result: uhf_stability(
            result,
            eri=scf_inputs["eri"],
            stability_tol=stability_tol,
            eigensolver_tol=eigensolver_tol,
        ),
        scf_inputs,
        max_restarts,
        step_sizes,
        energy_tol,
    )


def stabilize_uks_from_integrals(
    *,
    max_restarts=5,
    stability_tol=1e-5,
    eigensolver_tol=1e-7,
    step_sizes=(0.5, 1.0),
    energy_tol=1e-10,
    **scf_inputs,
):
    """SCF plus internal stability and directed restarts for LDA/GGA/hybrids.

    Takes the arguments of run_uks_from_integrals. Supply the identical grid
    and XC config for analysis and SCF. Only full-ERI, ordinary XC is supported.
    max_restarts=0 checks without restarting; unresolved checks return None.
    This discrete host-side branch selection is outside SCF differentiation.
    """
    if scf_inputs.get("bound_xc") is not None:
        raise NotImplementedError("Bound/Neural XC stability is not yet supported.")
    config = scf_inputs.get("config") or UKSConfig()
    if config.jk_backend != "full" or config.potential_clip is not None:
        raise NotImplementedError(
            "Stability requires full ERIs and an unclipped XC potential."
        )
    options = {
        key: scf_inputs[key] for key in ("eri", "ao", "ao_deriv1", "grid_weights")
    }
    return _stabilize(
        run_uks_from_integrals,
        lambda result: uks_stability(
            result,
            **options,
            config=config,
            stability_tol=stability_tol,
            eigensolver_tol=eigensolver_tol,
        ),
        scf_inputs,
        max_restarts,
        step_sizes,
        energy_tol,
    )


def _follow_controls(max_restarts, step_sizes, energy_tol):
    if (
        not isinstance(max_restarts, Integral)
        or isinstance(max_restarts, bool)
        or max_restarts < 0
    ):
        raise ValueError("max_restarts must be a nonnegative integer")
    steps = tuple(float(step) for step in step_sizes)
    if (
        not steps
        or any(not np.isfinite(s) or s <= 0 for s in steps)
        or len(set(steps)) != len(steps)
    ):
        raise ValueError("step_sizes must be distinct finite positive values")
    if not np.isfinite(energy_tol) or energy_tol < 0:
        raise ValueError("energy_tol must be finite and nonnegative")
    return steps


def _mode_density(coeff, occupations, check, step):
    """Existing occupation rotations convert one retained mode into a density."""
    if check.direction is None:
        raise ValueError("No resolved negative direction to follow")
    restricted = coeff.ndim == 2
    spin = restricted and check.channel == "spin"
    occ = np.asarray(occupations)
    if restricted:
        occ = np.stack([occ / 2, occ / 2])
        base = jnp.stack([coeff, coeff]) if spin else coeff[None]
    else:
        base = coeff
    _, rotate, density_from = _rotation_functions(
        "roks" if restricted and not spin else "uks", tuple(map(tuple, occ))
    )
    angles = step * check.direction
    if spin:
        angles = jnp.concatenate([angles, -angles])
    density = density_from(rotate(angles, base))
    return jnp.sum(density, axis=0) if restricted and not spin else density


def _follow_negative_modes(
    initial,
    *,
    analyze,
    restart,
    energy_of,
    max_restarts,
    step_sizes=(0.5, 1.0),
    energy_tol=1e-10,
):
    """One bounded acceptance/history loop for facades and integral APIs."""
    steps = _follow_controls(max_restarts, step_sizes, energy_tol)
    result, attempts = initial, []
    history = [float(energy_of(result))]
    if not np.isfinite(history[0]):
        raise ValueError("Initial SCF energy must be finite")
    for cycle in range(max_restarts + 1):
        check = analyze(result)
        if check.stable is not False:
            status = "stable" if check.stable is True else "unresolved"
            break
        if cycle == max_restarts:
            status = "max_restarts"
            break
        best, best_index, best_energy = None, None, history[-1]
        for magnitude in steps:
            for sign in (1.0, -1.0):
                step = sign * magnitude
                candidate = restart(result, check, step)
                energy = float(energy_of(candidate))
                valid = bool(candidate.converged) and bool(np.isfinite(energy))
                attempts.append(StabilityAttempt(cycle + 1, step, energy, valid))
                if valid and energy < history[-1] - energy_tol and energy < best_energy:
                    best, best_index, best_energy = candidate, len(attempts) - 1, energy
        if best is None:
            return SCFStabilizationResult(
                result,
                check,
                cycle + 1,
                tuple(history),
                tuple(attempts),
                "no_lower_converged_candidate",
            )
        attempts[best_index] = replace(attempts[best_index], accepted=True)
        result = best
        history.append(best_energy)
    return SCFStabilizationResult(
        result, check, cycle, tuple(history), tuple(attempts), status
    )


def _stabilize(
    solve, analyze, scf_inputs, max_restarts, step_sizes=(0.5, 1.0), energy_tol=1e-10
):
    step_sizes = _follow_controls(max_restarts, step_sizes, energy_tol)
    inputs = dict(scf_inputs)

    def restart(result, check, step):
        coeff = jnp.stack([result.mo_coeff_alpha, result.mo_coeff_beta])
        occ = jnp.stack([result.mo_occ_alpha, result.mo_occ_beta])
        density = _mode_density(coeff, occ, check, step)
        return solve(
            **dict(inputs, init_density_alpha=density[0], init_density_beta=density[1])
        )

    return _follow_negative_modes(
        solve(**inputs),
        analyze=analyze,
        restart=restart,
        energy_of=lambda state: state.total_energy,
        max_restarts=max_restarts,
        step_sizes=step_sizes,
        energy_tol=energy_tol,
    )


def unrestricted_stability(
    source,
    *,
    stability_tol=1e-5,
    eigensolver_tol=1e-7,
    max_cycle=100,
    gradient_tol=None,
    solver="davidson",
    max_space=40,
):
    """Facade adapter to the same real unrestricted curvature problem."""
    from .diagnostics import _unrestricted_scf_state

    _validate_stability_source(source)
    ref = _unrestricted_scf_state(source)
    coeff, occ = ref.mo_coeff, np.asarray(ref.mo_occ)
    dimension, rotate, energy, args, diagonal = _unrestricted_problem(
        coeff,
        occ,
        hcore=ref.h1e,
        eri=ref.rep_tensor,
        nuclear_repulsion=ref.nuclear_repulsion,
        ao=ref.ao,
        ao_deriv1=ref.ao_deriv1,
        grid_weights=ref.grid.weights,
        config=source._config(),
    )
    return _curvature_check(
        coeff=coeff,
        dimension=dimension,
        rotate=rotate,
        energy=energy,
        args=args,
        diagonal=diagonal,
        converged=bool(source.converged and ref.scf_converged),
        stability_tol=stability_tol,
        eigensolver_tol=eigensolver_tol,
        max_cycle=max_cycle,
        gradient_tol=source.conv_tol_grad if gradient_tol is None else gradient_tol,
        solver=solver,
        max_space=max_space,
    )


def stabilize_scf(
    source,
    *,
    channel="internal",
    max_restarts=5,
    step_sizes=(0.5, 1.0),
    energy_tol=1e-10,
    **stability_controls,
):
    """Clone and follow resolved negative modes; spin promotion requires an explicit channel."""
    from .facade import RKS, UKS, _fresh_scf_like
    from .roks import ROKS

    if not isinstance(source, (RKS, UKS)) or isinstance(source, ROKS):
        raise TypeError("Stabilization requires an RKS or UKS/UHF facade")
    if channel not in {"internal", "spin"} or (
        isinstance(source, UKS) and channel != "internal"
    ):
        raise ValueError(
            "Use internal, or explicit spin channel for a restricted source"
        )
    step_sizes = _follow_controls(max_restarts, step_sizes, energy_tol)

    def analyze(state):
        if isinstance(state, RKS):
            return restricted_stability(state, channel=channel, **stability_controls)
        return unrestricted_stability(state, **stability_controls)

    def restart(state, check, step):
        density = _mode_density(state.mo_coeff, state.mo_occ, check, step)
        target = (
            UKS if isinstance(state, RKS) and check.channel == "spin" else type(state)
        )
        return _fresh_scf_like(state, target, init_guess=density).run()

    if source.e_tot is None:
        initial = _fresh_scf_like(source, type(source)).run()
    else:
        from copy import copy
        from .diagnostics import _restricted_scf_state, _unrestricted_scf_state

        (_restricted_scf_state if isinstance(source, RKS) else _unrestricted_scf_state)(
            source
        )
        initial = copy(source)
    return _follow_negative_modes(
        initial,
        analyze=analyze,
        restart=restart,
        energy_of=lambda state: state.e_tot,
        max_restarts=max_restarts,
        step_sizes=step_sizes,
        energy_tol=energy_tol,
    )
