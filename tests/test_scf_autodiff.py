"""CPU float64 regressions for the shared SCF backward attachment."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import implicit_fixed_point_solution


def test_fixed_point_argument_gradients_are_preserved():
    def solve(scale, offset):
        return implicit_fixed_point_solution(
            scale, solution=offset / (1.0 - scale),
            fixed_point=lambda x, p, args: p * x + args['offset'],
            fixed_point_args={'offset': offset},
        )
    derivatives = jax.grad(solve, argnums=(0, 1))(jnp.array(0.2), jnp.array(1.4))
    np.testing.assert_allclose(derivatives, [1.4 / 0.8**2, 1.0 / 0.8], rtol=1e-9)


def test_shared_implicit_nonlinear_root_and_density_loss_pytree_jit():
    from gradscf.scf.autodiff import attach_scf_backward

    def solve(params, initial):
        root = jax.lax.stop_gradient(jnp.sqrt(params['scale'] / params['nested'][0]))
        return attach_scf_backward(
            params, solution=root + 0.0 * initial,
            residual=lambda x, p: p['nested'][0] * x**2 - p['scale'],
        )
    params = {'scale': jnp.array(8.0), 'nested': (jnp.array(2.0),)}
    grads, initial_grad = jax.jit(jax.grad(lambda p, x: solve(p, x)**3, argnums=(0, 1)))(params, jnp.array(1.0))
    np.testing.assert_allclose(grads['scale'], 1.5, rtol=1e-10)
    np.testing.assert_allclose(grads['nested'][0], -6.0, rtol=1e-10)
    np.testing.assert_array_equal(initial_grad, 0.0)


def test_unrolled_preserves_finite_iterates_and_initial_guess_derivative():
    from gradscf.scf.autodiff import SCFDifferentiationConfig, attach_scf_backward

    def solve(p, initial):
        x = initial
        for _ in range(3):
            x = p * x + 1.0
        return attach_scf_backward(p, solution=x, residual=lambda x, p: x - p*x - 1.0,
                                   config=SCFDifferentiationConfig(mode='unrolled'), converged=False)
    grads = jax.grad(solve, argnums=(0, 1))(jnp.array(0.2), jnp.array(2.0))
    np.testing.assert_allclose(grads, [1.64, 0.008], rtol=1e-10)


def test_nonconverged_implicit_retains_primal_but_rejects_backward():
    from gradscf.scf.autodiff import SCFDifferentiationConfig, attach_scf_backward

    def solve(p, converged, require=True):
        return attach_scf_backward(p, solution=jnp.array(2.0), residual=lambda x, p: x-p,
                                   converged=converged, config=SCFDifferentiationConfig(require_converged=require))
    assert solve(jnp.array(3.0), False) == 2.0
    assert jnp.isnan(jax.jit(jax.grad(solve))(jnp.array(3.0), jnp.array(False)))
    assert jax.grad(lambda p: solve(p, False, False))(jnp.array(3.0)) == 1.0


def test_failed_adjoint_rejects_approximate_gradient():
    from gradscf.scf.autodiff import SCFDifferentiationConfig, attach_scf_backward

    matrix = jnp.diag(jnp.array([1.0, 3.0, 10.0]))
    def loss(params):
        x = attach_scf_backward(params, solution=jnp.zeros(3),
                                residual=lambda x, p: matrix @ x - p,
                                config=SCFDifferentiationConfig(max_iter=1, restart=1, tolerance=1e-12))
        return jnp.sum(x)
    assert jnp.all(jnp.isnan(jax.grad(loss)(jnp.zeros(3))))


@pytest.mark.parametrize('mode, expected', [('implicit', 'implicit'), ('impl', 'implicit'), ('explicit', 'explicit'), ('unrolled', 'explicit'), ('expl', 'explicit')])
def test_mode_normalization_and_legacy_configuration(mode, expected):
    from gradscf.scf.autodiff import SCFDifferentiationConfig, normalize_scf_gradient_mode
    from gradscf.scf.differentiable import DifferentiableSCFConfig
    from gradscf.model.neural_xc.training.config import MolecularTrainingConfig
    assert normalize_scf_gradient_mode(mode) == expected
    assert SCFDifferentiationConfig(mode=mode).mode == expected
    assert DifferentiableSCFConfig(gradient_mode=mode).differentiation_config().mode == expected
    assert MolecularTrainingConfig(scf_gradient_mode=mode).scf_gradient_mode == mode


def test_unknown_mode_rejected():
    from gradscf.scf.autodiff import SCFDifferentiationConfig
    with pytest.raises(ValueError, match='mode'):
        SCFDifferentiationConfig(mode='typo')


@pytest.mark.parametrize('mode, alias', [('implicit', 'impl'), ('unrolled', 'expl')])
def test_dft_canonical_modes_match_legacy_density_and_derivative(mode, alias):
    from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig
    from test_differentiable_scf import _make_toy_restricted_reference, _ToyRestrictedFunctional

    molecule = _make_toy_restricted_reference()
    functional = _ToyRestrictedFunctional()
    def loss(p, name):
        config = DifferentiableSCFConfig(mode='self_consistent', gradient_mode=name, max_cycle=3)
        out, info = DifferentiableSCF(config).run(molecule, functional, {'strength': p})
        assert info.mode == ('self_consistent_implicit' if mode == 'implicit' else 'self_consistent')
        return jnp.sum(out.rdm1[0])
    p = jnp.array(0.1, dtype=jnp.float32)
    canonical = jax.value_and_grad(lambda p: loss(p, mode))(p)
    legacy = jax.value_and_grad(lambda p: loss(p, alias))(p)
    np.testing.assert_allclose(canonical, legacy, rtol=1e-6)


def test_dft_required_convergence_rejects_unconverged_backward():
    from gradscf.scf import DifferentiableSCF, DifferentiableSCFConfig
    from test_differentiable_scf import _make_toy_restricted_reference, _ToyRestrictedFunctional

    molecule = _make_toy_restricted_reference()
    functional = _ToyRestrictedFunctional()
    config = DifferentiableSCFConfig(mode='self_consistent', gradient_mode='implicit',
                                    max_cycle=1, require_converged_iterates=True)
    def loss(p):
        out, info = DifferentiableSCF(config).run(molecule, functional, {'strength': p})
        assert not bool(info.converged)
        return jnp.sum(out.rdm1[0])
    value, grad = jax.value_and_grad(loss)(jnp.array(0.1, dtype=jnp.float32))
    assert jnp.isfinite(value)
    assert jnp.isnan(grad)


def test_implicit_disconnected_forward_while_loop():
    from gradscf.scf.autodiff import attach_scf_backward

    def solve(p):
        x = jax.lax.while_loop(lambda x: x < 2.0, lambda x: x + p, jnp.array(0.0))
        return attach_scf_backward(p, solution=x, residual=lambda x, p: x-2*p)
    assert jax.grad(solve)(jnp.array(1.0)) == 2.0


def test_empty_stationarity_space_has_zero_implicit_response():
    from gradscf.scf.autodiff import attach_scf_backward

    def loss(p):
        state = attach_scf_backward(p, solution=jnp.zeros(0), residual=lambda x, p: p*x)
        return jnp.sum(state)
    assert jax.grad(loss)(jnp.array(2.0)) == 0.0


def test_dft_accepts_the_same_backward_policy_as_orbital_scf():
    from gradscf.scf import SCFDifferentiationConfig, DifferentiableSCFConfig, DifferentiableSCF
    from test_differentiable_scf import _make_toy_restricted_reference, _ToyRestrictedFunctional
    policy = SCFDifferentiationConfig(mode='implicit', require_converged=True)
    config = DifferentiableSCFConfig(mode='self_consistent', differentiation=policy, max_cycle=1)
    assert config.differentiation_config() is policy
    def loss(p):
        result, info = DifferentiableSCF(config).run(_make_toy_restricted_reference(),
            _ToyRestrictedFunctional(), {'strength': p})
        assert info.mode == 'self_consistent_implicit'
        return jnp.sum(result.rdm1[0])
    value, derivative = jax.value_and_grad(loss)(jnp.array(.1,dtype=jnp.float32))
    assert jnp.isfinite(value)
    assert jnp.isnan(derivative)
