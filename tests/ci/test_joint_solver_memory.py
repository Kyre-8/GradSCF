"""Prepared CI keeps both diagonal arrays live during the Hamiltonian action."""
import importlib

import jax.numpy as jnp
import pytest

from gradscf import ci
from gradscf._memory import tree_storage_bytes
from gradscf.ci.integrals import CIIntegrals


def test_prepared_solver_reserves_non_kernel_payload_for_its_action(monkeypatch):
    solver = importlib.import_module('gradscf.ci.solver')
    space = ci.make_ci_space(2,1)
    data = CIIntegrals({},jnp.asarray(.2),jnp.full(space.size,.2),jnp.zeros(space.size))
    planned = []
    original = solver.plan_hermitian

    def plan(*args, **kwargs):
        result = original(*args,**kwargs)
        planned.append(result[1])
        return result

    class Prepared(Exception):
        pass

    def action(integrals,space,**kwargs):
        expected = 640+planned[-1]+tree_storage_bytes(data,exclude=(data.constant,data.kernels))
        assert kwargs['reserved_bytes'] == expected
        raise Prepared

    monkeypatch.setattr(solver,'plan_hermitian',plan)
    monkeypatch.setattr(solver,'from_prepared',action)
    with pytest.raises(Prepared):
        solver.solve_ci_from_integrals(data,space,config=ci.CIConfig(solver='dense'),
                                        max_memory=32.,reserved_bytes=640)
