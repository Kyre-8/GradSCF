"""Native joint training uses both derivatives and transactional checkpoints."""
from importlib import import_module
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def driver():
    return import_module("gradscf.model.nnao.training.native")


def test_explicit_compilation_cache_root_is_namespaced_by_native_binary(tmp_path):
    from types import SimpleNamespace
    m=driver();args=SimpleNamespace(compilation_cache_dir=tmp_path/'shared',output_dir=tmp_path/'run')
    first=m.compilation_cache_path(args,'a'*64)
    second=m.compilation_cache_path(args,'b'*64)
    assert first.parent==args.compilation_cache_dir and first!=second
    args.compilation_cache_dir=None
    assert m.compilation_cache_path(args,'a'*64)==args.output_dir/'compilation-cache'/('a'*16)


def test_joint_checkpoint_initialization_preserves_learned_exponents(tmp_path):
    m=driver();path=tmp_path/'parent.npz'
    vector=np.concatenate([np.linspace(-.1,.1,7171),np.linspace(-.25,.1,14)])
    metadata=dict(epoch=1000,identity=dict(model_config=m.MODEL_CONFIG,
        exponent_keys=[list(x) for x in m.base.EXPONENT_KEYS]))
    np.savez(path,parameters=vector,metadata=np.asarray(json.dumps(metadata)))
    theta,beta,parent=m.read_initial_parameters(path,log_bound=.25)
    np.testing.assert_array_equal(theta,vector[:-14])
    np.testing.assert_array_equal(beta,vector[-14:])
    assert parent['epoch']==1000
    metadata['identity']['exponent_keys'].reverse()
    np.savez(path,parameters=vector,metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError,match='exponent'):
        m.read_initial_parameters(path,log_bound=.25)


def test_parent_parity_preserves_raw_reference_while_training_uses_stable_energy(tmp_path):
    from types import SimpleNamespace
    m=driver(); engine=object.__new__(m.NativeJointEngine)
    engine.batch=0;engine.args=SimpleNamespace(log_bound=.25,output_dir=tmp_path)
    engine.rows=[dict(structure_id='sample')];engine.parent_energies={'sample':-1.}
    def molecule(vector,key):
        return -1.2,np.ones_like(vector),dict(converged=True,stable=True,
            orbital_residual=0.,fixed_point_residual=0.,min_overlap_eigenvalue=1.,
            reconstruction_error=0.,initial_scf_energy=-1.)
    engine.molecule=molecule
    value,_,_=engine.evaluate(np.zeros(15))
    assert value==-1.2
    gate=json.loads((tmp_path/'parent-parity.json').read_text())
    assert gate['max_absolute_error_hartree']==0.
    assert gate['stable_energies']['sample']==-1.2


def test_joint_adam_rates_bounds_and_serialization():
    m = driver()
    optimizer = m.JointAdam(2, learning_rate=1e-5, exponent_learning_rate=1e-3, log_bound=.25)
    x = jnp.array([.4, .2, .2499, -.2499])
    g = jnp.array([1., -1., -1., 1.])
    state = optimizer.init(x)
    trial, state = optimizer.update(x, g, state)
    np.testing.assert_allclose(trial[:2], [.39999, .20001], atol=1e-10)
    np.testing.assert_allclose(trial[2:], [.25, -.25], atol=1e-12)
    restored = optimizer.decode(optimizer.encode(state), optimizer.init(x))
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)


def test_failed_joint_candidate_keeps_parameters_and_adam_moments(tmp_path):
    m = driver()
    optimizer = m.JointAdam(2, learning_rate=1e-5, exponent_learning_rate=1e-3, log_bound=.25)
    initial = jnp.zeros(4)
    calls = []

    def evaluate(x):
        calls.append(np.asarray(x))
        if len(calls) > 1:
            raise RuntimeError('invalid implicit gradient')
        return -1., np.ones(4), []

    result = m.base.trainer.run_training(initial, evaluate, optimizer, tmp_path, {}, epochs=2)
    assert result['status'] == 'failed' and result['accepted_epoch'] == 0
    saved = m.base.trainer.load_checkpoint(tmp_path/'last.npz', {}, optimizer, initial)
    np.testing.assert_array_equal(saved['parameters'], initial)
    for a, b in zip(jax.tree.leaves(saved['optimizer']), jax.tree.leaves(optimizer.init(initial))):
        np.testing.assert_array_equal(a, b)


def test_native_joint_projection_pullback_matches_finite_difference():
    from gradscf import integrals
    from gradscf.model.nnao import prepare_direct_basis

    m = driver()
    atom = 'H 0 0 0; H 0 0 .74'
    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct', core_primitives=6)
    at, ap = integrals.prepare_basis(atom, 'def2-universal-jkfit', cart=False)
    aux = integrals.make_auxiliary_plan(layout.topology, at)
    context = dict(layout=layout, plan=integrals.make_plan(layout.topology), aux=aux, ap=ap,
        metric=aux.metric_factor(layout.parameters, ap))
    forward, backward = m.native_projection(context, m.EXPONENT_KEYS)
    outputs = layout.reference_outputs()
    beta = jnp.zeros(len(m.base.EXPONENT_KEYS))
    physical = forward(outputs, beta)
    cotangent = jax.tree.map(lambda a: jnp.sin(jnp.arange(a.size).reshape(a.shape))*.01, physical)
    co, cb = backward(outputs, beta, cotangent)
    do = .02*jnp.cos(jnp.arange(outputs.size).reshape(outputs.shape))
    db = jnp.linspace(-.05, .05, beta.size)
    def scalar(t):
        return sum(jnp.vdot(a, b) for a, b in zip(forward(outputs+t*do, beta+t*db), cotangent))
    h = 1e-4
    fd = (8*(scalar(h)-scalar(-h))-scalar(2*h)+scalar(-2*h))/(12*h)
    np.testing.assert_allclose(jnp.vdot(co, do)+jnp.vdot(cb, db), fd, atol=2e-8, rtol=2e-6)
    assert np.linalg.norm(cb[:2]) > 1e-8
    np.testing.assert_array_equal(cb[2:], 0.)


def test_frozen_mace_constants_are_excluded_from_directions_and_updates():
    from flax import nnx
    from jax.flatten_util import ravel_pytree

    m = driver()
    model = nnx.Dict(weight=nnx.Param(jnp.array([.4, .6])),
                     act_scale=nnx.Param(jnp.array(1.7), is_mutable=False))
    _, parameters, _ = nnx.split(model, nnx.Param, ...)
    vector, _ = ravel_pytree(parameters)
    mask = m.base.trainable_parameter_mask(parameters)
    np.testing.assert_array_equal(mask, [False, True, True])
    direction = np.ones(3)*mask

    def forward(x):
        return jax.lax.stop_gradient(x[0])*jnp.sum(x[1:]**2)

    analytic = jax.jvp(forward, (vector,), (jnp.asarray(direction),))[1]
    h = 1e-4
    fd = (forward(vector+h*direction)-forward(vector-h*direction))/(2*h)
    np.testing.assert_allclose(analytic, fd, atol=1e-10)
    optimizer = m.JointAdam(3, learning_rate=1e-5, exponent_learning_rate=1e-3,
                           log_bound=.25, trainable_mask=mask)
    x = jnp.concatenate((vector, jnp.zeros(14)))
    candidate, _ = optimizer.update(x, jnp.ones_like(x), optimizer.init(x))
    assert candidate[0] == x[0]
    assert candidate[1] != x[1] and candidate[-1] != x[-1]


def test_joint_model_metadata_roundtrips_through_training_checkpoint(tmp_path):
    m = driver()
    identity = dict(model_config=m.MODEL_CONFIG)
    assert json.loads(json.dumps(identity)) == identity
    optimizer = m.JointAdam(2, learning_rate=1e-5, exponent_learning_rate=1e-3, log_bound=.25)
    initial = jnp.array([.4, .6, 0., 0.])
    def evaluate(x):
        return float(jnp.vdot(x, x)), 2*np.asarray(x), []
    result = m.base.trainer.run_training(initial, evaluate, optimizer, tmp_path, identity, epochs=1)
    assert result['status'] == 'complete' and result['accepted_epoch'] == 1
