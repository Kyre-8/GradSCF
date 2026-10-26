"""Known-array budgets preserve the existing orbital SCF algorithms and AD."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import orbital_optimization as orbital
from gradscf.scf.autodiff import SCFDifferentiationConfig, attach_scf_backward
from gradscf.solvers.linear import implicit as linear
from gradscf.solvers.nonlinear import fixed_point, minimize
from gradscf_tools.molecular_response import MolecularEnergy


def _inputs():
    return dict(overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-1., .5])),
        eri=jnp.zeros((2,)*4), nuclear_repulsion=0., ao=jnp.zeros((0, 2)),
        ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        mo_coeff=jnp.stack((jnp.eye(2),)*2), mo_occ=np.array([[1., 0.], [0., 0.]]),
        gradient_tolerance=1e-9, max_iterations=50)


def test_moving_hf_preflight_precedes_integrals():
    @dataclass
    class Parameters:
        centers: object
        nuclear_coords: object

    def forbidden(*args, **kwargs):
        pytest.fail('An impossible moving HF budget reached native integrals')

    energy = MolecularEnergy.__new__(MolecularEnergy)
    energy.coordinates = jnp.zeros((1, 3))
    energy.coefficients = jax.ShapeDtypeStruct((64, 64), jnp.float64)
    energy.occupations = np.zeros((2, 64))
    energy.parameters = Parameters(energy.coordinates, energy.coordinates)
    energy.owners = np.zeros(1, dtype=int)
    energy.plan = SimpleNamespace(evaluate=forbidden)
    energy.spec = SimpleNamespace(charges=(1,))
    energy.kind, energy.jk_backend = 'HF', 'full'
    energy.max_memory, energy.reserved_bytes = .001, 0
    with pytest.raises(ValueError, match='memory|budget|resident'):
        energy.inputs(energy.coordinates)


def test_molecular_solver_relays_original_mb_and_seed_owners(monkeypatch):
    energy = MolecularEnergy.__new__(MolecularEnergy)
    energy.unrestricted, energy.jk_backend = True, 'full'
    energy.coefficients = jnp.stack((jnp.eye(2),)*2)
    energy.occupations = np.array([[1., 0.], [0., 0.]])
    energy.coordinates = jnp.zeros((1, 3))
    energy.parameters = {'exponents': jnp.ones(2)}
    energy.plan = SimpleNamespace(topology=None)
    energy.owners = np.zeros(1, dtype=int)
    energy.gradient_tolerance, energy.xc = 1e-9, 'hf'
    energy.differentiation = SCFDifferentiationConfig()
    energy.max_memory, energy.reserved_bytes = 8., 4096
    supplied = _inputs()
    supplied.pop('mo_coeff'); supplied.pop('mo_occ')
    supplied.pop('gradient_tolerance'); supplied.pop('max_iterations')
    monkeypatch.setattr(energy, 'inputs', lambda *a, **k: supplied)

    class Captured(Exception):
        pass
    def checked(**kwargs):
        assert kwargs.get('max_memory') == 8.
        assert kwargs.get('reserved_bytes', 0) >= 4096+2*8+3*8
        raise Captured
    monkeypatch.setattr('gradscf.scf.minimize_uks_from_integrals', checked)
    with pytest.raises(Captured):
        energy.solve_with_inputs(energy.coordinates)


@pytest.mark.parametrize('method', ['uks', 'roks'])
def test_minimizer_budget_precedes_metric_transport(method, monkeypatch):
    monkeypatch.setattr(orbital, '_orthonormalize', lambda *a: pytest.fail(
        'Orbital transport preceded a known exhausted caller budget'))
    inputs = _inputs()
    if method == 'roks':
        inputs['mo_coeff'] = inputs['mo_coeff'][0]
    with pytest.raises(ValueError, match='memory|budget|resident'):
        getattr(orbital, 'minimize_'+method+'_from_integrals')(
            **inputs, max_memory=.001, reserved_bytes=4096)


def test_forward_fixed_buffers_precede_the_objective():
    def forbidden(*a):
        pytest.fail('The forward objective preceded its fixed-buffer preflight')
    with pytest.raises(ValueError, match='memory|budget|resident'):
        minimize.solve_orbitals({}, jnp.eye(2)[None], energy=forbidden,
            rotate=lambda angles, base: base, dimension=33, max_iterations=50,
            gradient_tolerance=1e-9, max_memory=.001, reserved_bytes=4096)


def test_forward_does_not_shrink_its_original_polishing_restart(monkeypatch):
    def shrink(size, dtype, config, *a, **k):
        return replace(config, restart=1), 1
    monkeypatch.setattr(minimize, 'plan_linear', shrink, raising=False)
    with pytest.raises(ValueError, match='restart|fixed|forward'):
        minimize.solve_orbitals({}, jnp.eye(2)[None],
            energy=lambda angles, base, args: jnp.sum(angles**2),
            rotate=lambda angles, base: base, dimension=33, max_iterations=1,
            gradient_tolerance=1e-9, max_memory=8.)


def test_compat_linear_entry_relays_budget_without_changing_controls(monkeypatch):
    class Captured(Exception):
        pass
    def checked(action, rhs, **kwargs):
        assert kwargs['max_memory'] == 8. and kwargs['reserved_bytes'] == 4096
        assert kwargs['operator_min_bytes'] == 512
        cfg = kwargs['config']
        assert (cfg.rtol, cfg.maxiter, cfg.restart) == (1e-10, 17, 7)
        raise Captured
    monkeypatch.setattr(linear, 'checked_linear_solve', checked)
    with pytest.raises(Captured):
        linear.solve_implicit_linear_system(lambda x: x, jnp.ones(3),
            tol=1e-10, max_iter=17, restart=7, max_memory=8.,
            reserved_bytes=4096, operator_min_bytes=512)


def test_scf_root_relays_response_owners_and_actual_scratch(monkeypatch):
    class Captured(Exception):
        pass
    def checked(action, rhs, **kwargs):
        assert kwargs.get('max_memory') == 8.
        assert kwargs.get('reserved_bytes', 0) >= 4096+3*8
        assert kwargs.get('operator_min_bytes', 0) > 512
        assert kwargs['tol'] == 1e-10 and kwargs['max_iter'] == 17
        raise Captured
    monkeypatch.setattr(fixed_point, 'solve_implicit_linear_system', checked)
    def state(p):
        return attach_scf_backward({'p': p}, solution=jnp.zeros(3),
            residual=lambda x, a: 2*x-a['p'],
            config=SCFDifferentiationConfig(tolerance=1e-10, max_iter=17, restart=7),
            max_memory=8., reserved_bytes=4096, operator_min_bytes=512)
    with pytest.raises(Captured):
        jax.jvp(state, (jnp.ones(3),), (jnp.ones(3),))


def test_response_exhaustion_precedes_linear_work(monkeypatch):
    monkeypatch.setattr(fixed_point, 'solve_implicit_linear_system', lambda *a, **k:
        pytest.fail('Known response exhaustion reached a linear solve'))
    def state(p):
        return attach_scf_backward(p, solution=jnp.zeros(3),
            residual=lambda x, a: 2*x-a, max_memory=.001, reserved_bytes=4096)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        jax.jvp(state, (jnp.ones(3),), (jnp.ones(3),))
