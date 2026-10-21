"""Bounded FCI string contractions preserve the real E E convention.

Small references deliberately use the former full excitation formula. The
production action must fit one string tile even when that stack cannot
fit the requested decimal-MB budget. Tests run with JAX float64 on CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import fci
from gradscf.fci import hamiltonian as ham


def _problem(norb=4, nelec=(2, 1)):
    space = fci.make_fci_space(norb, nelec)
    rng = np.random.default_rng(2104)
    h = rng.normal(size=(norb, norb))
    h = (h + h.T) * .5
    factors = rng.normal(size=(norb + 1, norb, norb)) * .1
    factors = (factors + factors.transpose(0, 2, 1)) * .5
    g = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    c = rng.normal(size=space.shape)
    return space, jnp.asarray(h), jnp.asarray(g), jnp.asarray(c)


def _full_ee(g, c, space):
    t = ham.excitation_actions(c, space)
    pairs = space.norb**2
    weighted = (g.reshape(pairs, pairs) @ t.reshape(pairs, c.size)).reshape(t.shape)
    return ham._excitation_transpose(weighted, space)


def test_adequate_workspace_batches_alpha_strings(monkeypatch):
    space, _, g, c = _problem()
    loop_iterations = []
    fori_loop = ham.jax.lax.fori_loop

    def record_loop(lower, upper, body, initial, **kwargs):
        loop_iterations.append(upper - lower)
        return fori_loop(lower, upper, body, initial, **kwargs)

    monkeypatch.setattr(ham.jax.lax, "fori_loop", record_loop)
    actual = ham.contract_2e(g, c, space, max_workspace_elements=16 * 4 * 4)
    np.testing.assert_allclose(actual, _full_ee(g, c, space), atol=2e-12, rtol=2e-12)
    assert len(loop_iterations) == 1
    assert loop_iterations[0] < space.shape[0]


@pytest.mark.parametrize("cap", [16 * 4 * 2, 16 * 4 * 4], ids=["alpha-two", "alpha-four-tail"])
def test_alpha_string_tiles_match_full_one_and_two_electron_actions(cap):
    space, h, g, c = _problem()
    one = ham.contract_1e(h, c, space, max_workspace_elements=cap)
    two = ham.contract_2e(g, c, space, max_workspace_elements=cap)
    reference_one = jnp.einsum("p,pab->ab", h.reshape(-1), ham.excitation_actions(c, space))
    np.testing.assert_allclose(one, reference_one, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(two, _full_ee(g, c, space), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("nelec", [(2, 1), (1, 2), (0, 1), (1, 0), (0, 0), (4, 4)])
def test_one_and_two_electron_actions_use_beta_tiles(nelec):
    space, h, g, c = _problem(nelec=nelec)
    # P=16, B=3; beta spaces of size four/six require a masked tail.
    cap = space.norb**2 * 3
    one = ham.contract_1e(h, c, space, max_workspace_elements=cap)
    two = ham.contract_2e(g, c, space, max_workspace_elements=cap)
    reference_one = jnp.einsum("p,pab->ab", h.reshape(-1), ham.excitation_actions(c, space))
    np.testing.assert_allclose(one, reference_one, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(two, _full_ee(g, c, space), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('cap', [16 * 2, 16 * 3 * 3, 16 * 4 * 2 * 3, 16 * 4 * 4 * 3],
                         ids=['rhs-tail', 'beta-tail', 'alpha-two', 'alpha-four-tail'])
def test_hamiltonian_matmat_and_transpose_with_three_rhs(cap):
    space, h, g, c = _problem()
    vectors = jnp.stack((c.reshape(-1), 2 * c.reshape(-1) + .3,
                         jnp.arange(space.size, dtype=c.dtype) / 17), axis=1)
    operator = ham.build_hamiltonian(h, g, space, max_workspace_elements=cap)
    actual = operator.matmat(vectors)
    effective = ham.absorb_h1e(h, g, space, fac=.5)
    reference = jnp.stack([_full_ee(effective, v.reshape(space.shape), space).reshape(-1)
                           for v in vectors.T], axis=1)
    np.testing.assert_allclose(actual, reference, atol=3e-12, rtol=3e-12)
    columns = jnp.stack([operator.matvec(v) for v in vectors.T], axis=1)
    np.testing.assert_allclose(actual, columns, atol=3e-12, rtol=3e-12)
    left, right = vectors[:, 1], vectors[:, 2]
    np.testing.assert_allclose(left @ operator.matvec(right),
                               operator.T.matvec(left) @ right, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(operator.matmat(vectors[:, :0]), vectors[:, :0])


def test_decimal_mb_budget_fits_tile_but_not_full_excitation_stack():
    space, _, g, c = _problem()
    budget = .010
    full_stack_bytes = space.norb**2 * space.size * c.dtype.itemsize
    # The two former E stacks cannot fit even before output/maps are charged.
    assert budget * 1_000_000 * .8 < g.nbytes + c.nbytes + 2 * full_stack_bytes
    actual = ham.contract_2e(g, c, space, max_workspace_elements=16 * 3,
                             max_memory=budget)
    np.testing.assert_allclose(actual, _full_ee(g, c, space), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("budget", [0., -1., np.nan, np.inf, True, "1", .000001])
def test_invalid_or_impossible_budget_fails_before_link_enumeration(budget, monkeypatch):
    space, _, g, c = _problem()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid FCI memory budget reached string-map allocation")

    monkeypatch.setattr(ham, "_link_maps", forbidden)
    with pytest.raises(ValueError, match="memory|budget|tile"):
        ham.contract_2e(g, c, space, max_memory=budget)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, 15])
def test_invalid_or_impossible_workspace_fails_before_link_enumeration(cap, monkeypatch):
    space, _, g, c = _problem()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid FCI workspace reached string-map allocation")

    monkeypatch.setattr(ham, "_link_maps", forbidden)
    with pytest.raises(ValueError, match="workspace|tile"):
        ham.contract_2e(g, c, space, max_workspace_elements=cap)


def test_zero_orbital_vacuum_with_small_workspace():
    space = fci.make_fci_space(0, (0, 0))
    h, g, c = jnp.zeros((0, 0)), jnp.zeros((0,) * 4), jnp.ones((1, 1))
    np.testing.assert_allclose(ham.contract_1e(h, c, space, max_workspace_elements=1), 0.)
    np.testing.assert_allclose(ham.contract_2e(g, c, space, max_workspace_elements=1), 0.)
    operator = ham.build_hamiltonian(h, g, space, max_workspace_elements=1)
    np.testing.assert_allclose(operator.matvec(c.reshape(-1)), 0.)


@pytest.mark.parametrize("cap", [16 * 3, 16 * 4 * 2, 16 * 4 * 4],
                         ids=["beta-tail", "alpha-two", "alpha-four-tail"])
def test_tiled_h_g_c_jvp_vjp_and_hvp_match_full_formula(cap):
    space, h, g, c = _problem()
    rng = np.random.default_rng(2204)
    directions = tuple(jnp.asarray(rng.normal(size=x.shape)) * .07 for x in (h, g, c))
    probe = jnp.asarray(rng.normal(size=space.shape))

    def tiled(hh, gg, cc):
        return ham.contract_hamiltonian(hh, gg, cc, space, max_workspace_elements=cap)

    def reference(hh, gg, cc):
        return _full_ee(ham.absorb_h1e(hh, gg, space, fac=.5), cc, space)

    actual, tangent = jax.jvp(tiled, (h, g, c), directions)
    expected, reference_tangent = jax.jvp(reference, (h, g, c), directions)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(tangent, reference_tangent, atol=3e-12, rtol=3e-12)
    _, pullback = jax.vjp(tiled, h, g, c)
    _, reference_pullback = jax.vjp(reference, h, g, c)
    for actual_gradient, expected_gradient in zip(pullback(probe), reference_pullback(probe)):
        np.testing.assert_allclose(actual_gradient, expected_gradient, atol=4e-12, rtol=4e-12)

    def loss(action):
        return lambda hh, gg, cc: jnp.sum(jnp.sin(action(hh, gg, cc)) * probe)

    tiled_gradient = jax.grad(loss(tiled), argnums=(0, 1, 2))
    reference_gradient = jax.grad(loss(reference), argnums=(0, 1, 2))
    hvp = jax.jvp(tiled_gradient, (h, g, c), directions)[1]
    reference_hvp = jax.jvp(reference_gradient, (h, g, c), directions)[1]
    for actual_hvp, expected_hvp in zip(hvp, reference_hvp):
        np.testing.assert_allclose(actual_hvp, expected_hvp, atol=1e-11, rtol=1e-11)


def _visit_jaxpr(value, forbidden_shapes):
    if hasattr(value, "jaxpr"):
        _visit_jaxpr(value.jaxpr, forbidden_shapes)
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            for variable in equation.outvars:
                shape = getattr(getattr(variable, "aval", None), "shape", ())
                assert shape not in forbidden_shapes, f"full FCI excitation stack {shape}"
            _visit_jaxpr(equation.params, forbidden_shapes)
    elif isinstance(value, dict):
        for child in value.values():
            _visit_jaxpr(child, forbidden_shapes)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _visit_jaxpr(child, forbidden_shapes)


@pytest.mark.parametrize("full_space_cap", [False, True], ids=["beta-tail", "full-space-cap"])
def test_forward_and_reverse_jaxpr_have_no_full_excitation_stack(full_space_cap):
    space, h, g, c = _problem()
    pairs = space.norb**2
    forbidden = {(pairs, *space.shape), (pairs, space.size),
                 (pairs, *space.shape, 1), (pairs, space.size, 1)}

    def loss(cc):
        cap = pairs * (space.size if full_space_cap else 3)
        action = ham.contract_hamiltonian(h, g, cc, space, max_workspace_elements=cap)
        return jnp.sum(action**2)

    _visit_jaxpr(jax.make_jaxpr(loss)(c), forbidden)
    _visit_jaxpr(jax.make_jaxpr(jax.grad(loss))(c), forbidden)


@pytest.mark.parametrize("operation", ["contract_1e", "contract_2e", "build_hamiltonian", "contract_hamiltonian"])
def test_invalid_space_keeps_explicit_type_error(operation):
    _, h, g, c = _problem()
    arguments = {"contract_1e": (h, c), "contract_2e": (g, c),
                 "build_hamiltonian": (h, g), "contract_hamiltonian": (h, g, c)}
    with pytest.raises(TypeError, match="Expected an FCISpace"):
        getattr(ham, operation)(*arguments[operation], None)


@pytest.mark.parametrize('operation', ['contract_hamiltonian', 'build_hamiltonian'])
def test_effective_eri_preparation_buffers_are_budgeted(operation, monkeypatch):
    space, h, g, c = _problem(norb=16, nelec=(1, 0))
    def forbidden(*args, **kwargs):
        pytest.fail('Insufficient memory reached effective-ERI preparation')
    monkeypatch.setattr(ham, 'absorb_h1e', forbidden)
    arguments = (h, g, c) if operation == 'contract_hamiltonian' else (h, g)
    with pytest.raises(ValueError, match='(?i)(memory|budget|resident|tile)'):
        getattr(ham, operation)(*arguments, space, max_memory=1.45)
