"""DF warm starts and stable-root selection without a four-index ERI."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.rks import RKSConfig, _make_jk_builder, run_rks_from_integrals_traceable
from gradscf.integrals.molecular.jk import build_jk_from_df


def problem():
    # Both determinants are Aufbau fixed points; orbital 0 is a saddle.
    return dict(overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-2., -2.5])),
        df_factors=jnp.eye(2)[None], eri=None, nelectron=2, nuclear_repulsion=0.,
        ao=jnp.zeros((0,2)), ao_deriv1=jnp.zeros((4,0,2)), grid_weights=jnp.zeros(0),
        config=RKSConfig(xc_spec='hf',jk_backend='df',conv_tol=1e-12,
                         conv_tol_density=1e-10,conv_tol_grad=1e-9))


def test_df_jk_respects_density_when_orbitals_do_not_represent_it():
    data=problem(); b=data['df_factors']; occ=jnp.array([2.,0.])
    builder=_make_jk_builder(None,data['config'],df_factors=b,with_k=True,nocc=1)
    for d in (jnp.diag(jnp.array([0.,2.])),jnp.diag(jnp.array([.3,1.7]))):
        actual=jax.jit(builder)(d,jnp.eye(2),occ)
        expected=build_jk_from_df(b,d)
        for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,atol=1e-14)


def test_orbital_hint_does_not_change_density_response_at_matching_primal():
    data=problem();b=data['df_factors'];d=jnp.diag(jnp.array([2.,0.]))
    builder=_make_jk_builder(None,data['config'],df_factors=b,with_k=True,nocc=1)
    direction=jnp.diag(jnp.array([.1,-.1]))
    response=jax.jvp(lambda x:builder(x,jnp.eye(2),jnp.array([2.,0.])),(d,),(direction,))[1]
    reference=jax.jvp(lambda x:build_jk_from_df(b,x),(d,),(direction,))[1]
    for x,y in zip(response,reference):np.testing.assert_allclose(x,y,atol=1e-14)
    def objective(t,fast):
        factors=b+t*jnp.array([[[.04,.03],[.03,-.02]]])
        density=d+t*direction
        if fast:
            kernel=_make_jk_builder(None,data['config'],df_factors=factors,with_k=True,nocc=1)
            matrices=kernel(density,jnp.eye(2),jnp.array([2.,0.]))
        else:matrices=build_jk_from_df(factors,density)
        return sum(jnp.sum(x*x) for x in matrices)
    np.testing.assert_allclose(jax.grad(jax.grad(lambda t:objective(t,True)))(0.),
                               jax.grad(jax.grad(lambda t:objective(t,False)))(0.),atol=1e-12)
    hint_tangent=jax.jvp(lambda c:builder(d,c,jnp.array([2.,0.])),
                         (jnp.eye(2),),(jnp.ones((2,2)),))[1]
    for x in hint_tangent:np.testing.assert_array_equal(x,0.)


def test_density_only_warm_start_matches_consistent_orbitals():
    data=problem();d=jnp.diag(jnp.array([2.,0.]))
    # One iteration exposes the initial J/K mismatch without relying on convergence basins.
    data['config']=replace(data['config'],max_cycle=1)
    only=run_rks_from_integrals_traceable(**data,init_density=d)
    both=run_rks_from_integrals_traceable(**data,init_density=d,init_mo_coeff=jnp.eye(2))
    np.testing.assert_allclose(only.density_matrix,both.density_matrix,atol=1e-12)


def test_stable_df_rhf_escapes_a_converged_aufbau_saddle_and_reports_budget():
    from gradscf.scf.stability import stabilize_df_rhf
    data=problem()
    saddle=run_rks_from_integrals_traceable(**data,init_mo_coeff=jnp.eye(2))
    assert bool(saddle.converged)
    np.testing.assert_allclose(saddle.total_energy,-3.,atol=1e-12)
    stopped,info=stabilize_df_rhf(saddle,**data,max_restarts=0)
    assert not bool(info['stable']) and float(info['minimum_curvature']) < -1.
    fixed,info=jax.jit(lambda:stabilize_df_rhf(saddle,**data))()
    assert bool(info['stable']) and int(info['stability_restarts'])>0
    np.testing.assert_allclose(fixed.total_energy,-4.,atol=1e-10)
    assert float(info['minimum_curvature'])>1.


def test_training_rejects_unresolved_stability():
    from gradscf.model.nnao.training import objective as m
    info=dict(converged=True,rescue_used=False,scf_cycles=3,total_scf_cycles=3,
        orbital_residual=0.,fixed_point_residual=0.,min_overlap_eigenvalue=1.,
        reconstruction_error=0.,stable=False,minimum_curvature=-1.)
    with pytest.raises(RuntimeError,match='stabl'):
        m.BasisRHF._checked_info(-1.,info)


def test_df_facade_internal_stability_without_full_eri():
    from gradscf import dft,gto
    mf=dft.RKS(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g'),
               xc='hf',jk_backend='df',auxbasis='def2-universal-jkfit').run()
    check=mf.stability(solver='dense')
    assert check.stable and check.minimum_curvature>0
    assert mf._scf_inputs.eri is None


def test_batch_acceptance_rejects_unstable_finite_gradients():
    from gradscf.model.nnao.training import trainer as m
    info=dict(converged=True,orbital_residual=0.,fixed_point_residual=0.,
              min_overlap_eigenvalue=1.,reconstruction_error=0.,stable=False)
    assert not m.valid_scf(info)


def test_matrix_free_check_handles_a_degenerate_positive_spectrum():
    from gradscf.scf.stability import stabilize_df_rhf
    data=problem();n=24
    data.update(overlap=jnp.eye(n),hcore=jnp.diag(jnp.repeat(jnp.array([-3.,-2.]),12)),
                df_factors=jnp.eye(n)[None],nelectron=24,ao=jnp.zeros((0,n)),
                ao_deriv1=jnp.zeros((4,0,n)))
    initial=run_rks_from_integrals_traceable(**data)
    _,info=jax.jit(lambda:stabilize_df_rhf(initial,**data))()
    assert bool(info['stable']) and int(info['stability_restarts'])==0
    np.testing.assert_allclose(info['minimum_curvature'],4.,atol=1e-8)


def test_stability_does_not_certify_a_different_occupation_topology():
    from gradscf.scf.stability import stabilize_df_rhf
    data=problem();initial=run_rks_from_integrals_traceable(**data)
    for occ in (jnp.array([0.,2.]),jnp.array([1.,1.])):
        changed=replace(initial,mo_occ=occ)
        _,info=stabilize_df_rhf(changed,**data,max_restarts=0)
        assert not bool(info['stable']) and not bool(info['stability_resolved'])
