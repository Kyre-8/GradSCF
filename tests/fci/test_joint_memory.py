"""FCI operator tiles and the shared Hermitian solve use one array budget."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace
from dataclasses import replace

from gradscf import fci
from gradscf.fci import hamiltonian as ham
from gradscf.fci import solver
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig
from gradscf.solvers.eigen import primal


def _problem():
    space = fci.make_fci_space(4, (2, 1))
    rng = np.random.default_rng(2901)
    h = np.diag([-1.4, -.8, .1, .6])
    factors = rng.normal(size=(3, 4, 4)) * .035
    factors = (factors + factors.transpose(0, 2, 1)) * .5
    g = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    return space, jnp.asarray(h), jnp.asarray(g)


@pytest.mark.parametrize("method", ["davidson", "dense"])
def test_joint_minimum_rejects_before_hamiltonian_preparation(method, monkeypatch):
    space, h, g = _problem()

    def forbidden(*args, **kwargs):
        pytest.fail("FCI allocated absorption/guesses before its joint minimum check")

    monkeypatch.setattr(ham, "absorb_h1e", forbidden)
    monkeypatch.setattr(jax.random, "normal", forbidden)
    with pytest.raises(ValueError, match="memory|budget|minimum"):
        solver.solve_fci(h, g, space, config=EigenSolverConfig(method=method),
                         max_memory=.0165)


def test_fci_passes_solver_reservation_to_each_action_and_keeps_useful_batches(monkeypatch):
    space, h, g = _problem()
    prepared, action_columns, configurations = [], [], []
    build, davidson = solver.build_hamiltonian, primal._davidson_lowest_symmetric

    def record_build(*args, **kwargs):
        prepared.append(kwargs)
        op = build(*args, **kwargs)
        apply = op.matmat

        def record_apply(v):
            action_columns.append(v.shape[1])
            return apply(v)

        return type(op)(op.shape, op.dtype, op.matvec, diagonal=op.diagonal,
                        matmat=record_apply, transpose_matvec=op.transpose_matvec)

    def record_davidson(*args, **kwargs):
        configurations.append(kwargs)
        return davidson(*args, **kwargs)

    monkeypatch.setattr(solver, "build_hamiltonian", record_build)
    monkeypatch.setattr(primal, "_davidson_lowest_symmetric", record_davidson)
    cfg = EigenSolverConfig(max_subspace=20, maxiter=150, atol=1e-10)
    answer = solver.solve_fci(h, g, space, config=cfg, max_memory=.055)
    assert bool(jnp.all(answer.converged))
    assert prepared[0].get("reserved_bytes", 0) > 0
    assert prepared[0]["max_memory"] == .055
    assert 4 <= configurations[0]["max_subspace"] < 20
    assert max(action_columns) >= 2
    assert max(action_columns) < 20


def test_joint_budget_preserves_fci_values_density_gradient_and_hvp():
    space, h, g = _problem()
    cfg = EigenSolverConfig(max_subspace=20, maxiter=150, atol=1e-10)
    policy = EigenResponseConfig(target="eigenpairs")
    direction = jnp.diag(jnp.array([.01, -.02, .015, -.005]))
    probe = jnp.array([[.2, .1, 0., .05], [.1, -.3, .04, 0.],
                       [0., .04, .4, .02], [.05, 0., .02, -.1]])

    def energy_and_density(hh, budget):
        solved = solver.solve_fci(hh, g, space, config=cfg, response=policy,
                                  max_memory=budget)
        density = fci.make_rdm1(solved.coefficients[0], space, max_memory=budget)
        return solved.total_energies[0] + .03 * jnp.sum(probe * density)

    outputs = []
    for budget in (.055, 1.):
        objective = lambda hh: energy_and_density(hh, budget)
        solved = solver.solve_fci(h, g, space, config=cfg, max_memory=budget)
        assert bool(jnp.all(solved.converged))
        assert np.max(np.asarray(solved.residual_norms)) <= cfg.atol
        value, gradient = jax.value_and_grad(objective)(h)
        hvp = jax.jvp(jax.grad(objective), (h,), (direction,))[1]
        outputs.append((value, gradient, hvp))
    for low, high in zip(*outputs):
        np.testing.assert_allclose(low, high, atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize("reserved", [-1, -.5, True, 1.5])
def test_operator_reservation_is_validated_before_absorption(reserved, monkeypatch):
    space, h, g = _problem()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid reservation reached integral absorption")

    monkeypatch.setattr(ham, "absorb_h1e", forbidden)
    with pytest.raises(ValueError, match="reserved_bytes"):
        ham.build_hamiltonian(h, g, space, max_memory=1., reserved_bytes=reserved)


def test_tight_budget_preserves_affordable_guess_and_action_batches():
    space = fci.make_fci_space(6, (2, 2))
    h = jax.ShapeDtypeStruct((6, 6), jnp.float64)
    g = jax.ShapeDtypeStruct((6,)*4, jnp.float64)
    core = jax.ShapeDtypeStruct((), jnp.float64)
    config, _, batch = solver._plan_fci_memory(space, h.dtype,
        EigenSolverConfig(max_subspace=32), .5, arrays=(h, g, core))
    # The same Hamiltonian was measured with m23/g2/b2 taking 8.11 ms and
    # m17/g4/b6 taking 4.51 ms. Keep normal blocks before growing optional V.
    assert config.initial_guess_count == 4
    assert config.max_trial_vectors == 2
    assert config.collapse_subspace == batch == 6
    assert config.max_subspace == 17


@pytest.mark.parametrize("kind", ["explicit", "scf", "scf-params"])
def test_bound_source_arrays_remain_reserved_during_active_fci(kind, monkeypatch):
    import importlib
    api = importlib.import_module("gradscf.fci.api")
    big_h = jax.ShapeDtypeStruct((20, 20), jnp.float64)
    big_g = jax.ShapeDtypeStruct((20,)*4, jnp.float64)
    if kind == "explicit":
        source = fci.FCIReference(big_h, big_g, (3, 3))
    else:
        # Owner metadata only: this is not a mock numerical SCF reference.
        coefficients = jax.ShapeDtypeStruct((20, 20), jnp.float64)
        source = SimpleNamespace(mo_coeff=coefficients,
            _scf_inputs=SimpleNamespace(hcore=big_h, eri=big_g, overlap=big_h),
            scf_result=SimpleNamespace(mo_coeff=coefficients, fock_matrix=big_h),
            reference=None)
        if kind == "scf-params":
            source.params = {"layer": [jax.ShapeDtypeStruct((8192,), jnp.float64)]}
    h = jnp.diag(jnp.linspace(-1.4, .8, 6))
    g = jnp.zeros((6,)*4)
    monkeypatch.setattr(api, "source_dimensions", lambda *a: (20, (3, 3)))
    monkeypatch.setattr(api, "active_reference", lambda *a, **kw: (h, g, 0.))

    class Captured(Exception):
        pass

    def capture(hh, gg, space, **kw):
        reserved = kw.get("reserved_bytes", 0)
        required = 20**4 * 8 + (8192*8 if kind == "scf-params" else 0)
        assert reserved >= required, "bound source arrays disappeared from the solver budget"
        cfg, _, _ = solver._plan_fci_memory(space, hh.dtype, kw["config"],
            kw["max_memory"], arrays=(hh, gg), reserved_bytes=reserved)
        assert cfg.max_subspace < 40
        raise Captured

    monkeypatch.setattr(api, "solve_fci", capture)
    with pytest.raises(Captured):
        fci.FCI(source, active=range(6), max_memory=2.3 if kind == "scf-params" else 2.1).run()


@pytest.mark.parametrize("storage", ["padding", "groups"])
def test_bound_cartesian_basis_declared_storage_is_counted_once(storage, monkeypatch):
    import importlib
    from gradscf.integrals.basis import basis_from_spec
    from gradscf.scf.inputs.types import RKSIntegralInputs

    api = importlib.import_module("gradscf.fci.api")
    basis = basis_from_spec("H 0 0 0; H 0 0 1.4", basis={"H": [[0, [1., 1.]]]},
                            unit="Bohr", precompute_eri_groups=False)
    if storage == "padding":
        previous = basis.ao_exponents_padded
        payload = jax.ShapeDtypeStruct((8192,), previous.dtype)
        modified = replace(basis, ao_exponents_padded=payload)
    else:
        previous = basis.pair_groups[0].row_idx
        payload = jax.ShapeDtypeStruct((8192,), previous.dtype)
        group = replace(basis.pair_groups[0], row_idx=payload)
        modified = replace(basis, pair_groups=(group,)+basis.pair_groups[1:])
    h, g, coefficients = jnp.eye(2), jnp.zeros((2,)*4), jnp.eye(2)
    inputs = RKSIntegralInputs(basis=basis, overlap=h, hcore=h, eri=None,
        eri_pair_matrix=None, df_factors=None, direct_basis=basis, nelectron=2,
        nuclear_repulsion=0., coords=basis.atom_coords, grid_weights=jnp.zeros(0),
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((0, 2, 3)),
        ao_laplacian=None, dipole_integrals=None)
    source = SimpleNamespace(mo_coeff=coefficients, _scf_inputs=inputs,
        scf_result=SimpleNamespace(fock_matrix=h), reference=SimpleNamespace(direct_basis=basis))

    def forbidden(*args, **kwargs):
        pytest.fail("source memory accounting called an integral/reference builder")

    source.to_reference = forbidden
    monkeypatch.setattr(RKSIntegralInputs, "response_eri_pair_matrix", forbidden)
    monkeypatch.setattr(api, "source_dimensions", lambda *a: (2, (1, 1)))
    monkeypatch.setattr(api, "active_reference", lambda *a, **kw: (h, g, 0.))
    observed = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        observed.append(kwargs["reserved_bytes"])
        raise Captured

    monkeypatch.setattr(api, "solve_fci", capture)
    for selected in (basis, modified):
        source._scf_inputs = replace(inputs, basis=selected, direct_basis=selected)
        source.reference = SimpleNamespace(direct_basis=selected)
        with pytest.raises(Captured):
            fci.FCI(source, max_memory=4.).run()
    expected_increment = payload.shape[0] * np.dtype(payload.dtype).itemsize - previous.nbytes
    assert observed[1] - observed[0] == expected_increment
