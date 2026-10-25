"""Direct CI actions retain the chosen space without determinant-pair storage."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.ci import hamiltonian
from gradscf.ci.solver import solve_ci
from gradscf.ci.space import make_ci_space, make_uci_space
from gradscf.ci.types import CIConfig


def _integrals(n, unrestricted=False):
    rng = np.random.default_rng(147)
    h = rng.normal(size=(2 if unrestricted else 1, n, n)) * .2
    h = (h + h.swapaxes(-1, -2)) * .5
    factors = rng.normal(size=(2 if unrestricted else 1, n + 1, n, n)) * .1
    factors = (factors + factors.swapaxes(-1, -2)) * .5
    if unrestricted:
        a, b = factors
        return tuple(map(jnp.asarray, h)), tuple(jnp.asarray(g) for g in (
            np.einsum("Lpq,Lrs->pqrs", a, a),
            np.einsum("Lpq,Lrs->pqrs", a, b),
            np.einsum("Lpq,Lrs->pqrs", b, b)))
    return jnp.asarray(h[0]), jnp.asarray(np.einsum("Lpq,Lrs->pqrs", factors[0], factors[0]))


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("rank,selected", [(1, None), (2, None), (2, (0, 2)), (3, None)])
def test_direct_hamiltonian_matches_slater_condon(unrestricted, rank, selected):
    n = 5
    h, g = _integrals(n, unrestricted)
    make = make_uci_space if unrestricted else make_ci_space
    frozen = None if rank == 3 else (((0,), ()) if unrestricted else (0,))
    space = make(n, (2, 1) if unrestricted else 2, max_excitation=rank,
                 excitation_ranks=selected, frozen=frozen)
    rng = np.random.default_rng(62)
    vectors = jnp.asarray(rng.normal(size=(space.size, 3)))
    direct = hamiltonian.build_direct_hamiltonian(h, g, space, block_size=3)
    reference = hamiltonian.build_hamiltonian(h, g, space)
    np.testing.assert_allclose(direct.diagonal, reference.diagonal, atol=2e-12, rtol=0)
    np.testing.assert_allclose(direct.apply(vectors), reference(vectors), atol=2e-12, rtol=0)
    np.testing.assert_allclose(direct.matvec(vectors[:, 0]), reference(vectors[:, 0]), atol=2e-12, rtol=0)
    np.testing.assert_allclose(direct.transpose_matvec(vectors[:, 1]), reference(vectors[:, 1]), atol=2e-12, rtol=0)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_direct_integral_vjp_matches_connections(unrestricted):
    h, g = _integrals(4, unrestricted)
    make = make_uci_space if unrestricted else make_ci_space
    space = make(4, (2, 1) if unrestricted else 2, max_excitation=2)
    rng = np.random.default_rng(29)
    left, right = map(jnp.asarray, rng.normal(size=(2, space.size)))

    def value(t, direct):
        hh = jax.tree.map(lambda x: x * jnp.exp(.1*t), h)
        gg = jax.tree.map(lambda x: x * (1 + .2*t), g)
        operator = (hamiltonian.build_direct_hamiltonian(hh, gg, space, block_size=2)
                    if direct else hamiltonian.build_hamiltonian(hh, gg, space))
        return jnp.dot(left, operator.apply(right) if direct else operator(right))

    actual = jax.jit(jax.value_and_grad(lambda t: value(t, True)))(.17)
    expected = jax.jit(jax.value_and_grad(lambda t: value(t, False)))(.17)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("input_rank,output_rank", [(0, 2), (2, 0), (1, 1), (None, 0)])
def test_static_input_output_rank_projection_and_transpose(unrestricted, input_rank, output_rank):
    h, g = _integrals(4, unrestricted)
    space = (make_uci_space(4, (2, 1), max_excitation=3) if unrestricted
             else make_ci_space(4, 2, max_excitation=3))
    ranks = np.asarray(space.ranks)
    inputs = np.ones(space.size, bool) if input_rank is None else ranks <= input_rank
    outputs = np.ones(space.size, bool) if output_rank is None else ranks <= output_rank
    matrix = np.asarray(hamiltonian.hamiltonian_matrix(h, g, space))
    expected = outputs[:, None]*matrix*inputs[None, :]
    vectors = jnp.asarray(np.random.default_rng(641).normal(size=(space.size, 2)))
    operator = hamiltonian.build_direct_hamiltonian(h, g, space, block_size=2,
        max_input_rank=input_rank, max_output_rank=output_rank)
    np.testing.assert_allclose(operator.apply(vectors), expected@vectors, atol=2e-12, rtol=0)
    np.testing.assert_allclose(operator.T.apply(vectors[:, 0]), expected.T@vectors[:, 0], atol=2e-12, rtol=0)
    np.testing.assert_allclose(operator.diagonal, np.diag(expected), atol=2e-12, rtol=0)


def test_sextuple_to_quadruple_action_and_response_match_connections():
    # Six electrons in six spatial orbitals give just 400 determinants, but
    # include the rank-six ket and rank-four bra needed by the final MP6 node.
    h, g = _integrals(6)
    space = make_ci_space(6, 3, max_excitation=6)
    ranks = np.asarray(space.ranks)
    assert space.size == 400 and ranks.max() == 6
    output = jnp.asarray(ranks <= 4)
    rng = np.random.default_rng(683)
    left, right = map(jnp.asarray, rng.normal(size=(2, space.size)))
    sextuple = jnp.asarray(ranks == 6, dtype=h.dtype)
    reference = hamiltonian.build_hamiltonian(h, g, space)
    operator = hamiltonian.build_direct_hamiltonian(
        h, g, space, block_size=8, max_input_rank=6, max_output_rank=4)
    # A pure sextuple input prevents an accidental lower-rank-only test.
    expected = output*reference(sextuple)
    assert np.linalg.norm(expected) > 1e-5
    np.testing.assert_allclose(operator.apply(sextuple), expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(operator.apply(right), output*reference(right), atol=2e-12, rtol=0)
    np.testing.assert_allclose(operator.T.apply(left), reference(output*left), atol=2e-12, rtol=0)

    def value(t, vector, direct):
        hh, gg = h*jnp.exp(.1*t), g*(1+.2*t)
        if direct:
            action = hamiltonian.build_direct_hamiltonian(
                hh, gg, space, block_size=8, max_input_rank=6, max_output_rank=4)
            return jnp.dot(left, action.apply(vector))
        return jnp.dot(left*output, hamiltonian.build_hamiltonian(hh, gg, space)(vector))

    actual = jax.jit(jax.value_and_grad(lambda t, v: value(t, v, True), (0, 1)))(.17, right)
    expected = jax.jit(jax.value_and_grad(lambda t, v: value(t, v, False), (0, 1)))(.17, right)
    for result, oracle in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(result, oracle, atol=2e-11, rtol=0)


def test_selected_doubles_keep_unretained_single_intermediates():
    # P E P E P vanishes between these determinants, while P E E P does not.
    space = make_ci_space(2, 1, max_excitation=2, excitation_ranks=(0, 2))
    h = jnp.zeros((2, 2))
    d = jnp.array([[0., 1.], [1., 0.]])
    g = jnp.einsum("pq,rs->pqrs", d, d)
    vector = jnp.array([1., 0.])
    actual = hamiltonian.build_direct_hamiltonian(h, g, space, block_size=1).apply(vector)
    np.testing.assert_allclose(actual, [0., 1.], atol=1e-14, rtol=0)
    projected = hamiltonian.build_one_body(d, space, block_size=1)
    np.testing.assert_allclose(projected.apply(vector), 0., atol=1e-14, rtol=0)


def test_single_spin_links_keep_fermionic_phases():
    space = make_uci_space(4, (2, 0), max_excitation=2)
    empty = jnp.zeros((4, 4))
    h = empty.at[3, 0].set(1.).at[0, 3].set(1.)
    reference = (1 << 2) - 1
    lookup = {det: i for i, det in enumerate(space.determinants)}
    state = jnp.zeros(space.size).at[lookup[reference]].set(1.)
    result = hamiltonian.build_one_body((h, empty), space, block_size=2).apply(state)
    # a_3^dagger a_0 |0,1> = -|1,3>.
    assert result[lookup[(1 << 1) | (1 << 3)]] == -1.
    np.testing.assert_allclose(result, hamiltonian.build_hamiltonian(
        (h, empty), (jnp.zeros((4,)*4),)*3, space)(state), atol=1e-14, rtol=0)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_one_body_matches_zero_eri_action_and_response(unrestricted):
    h, g = _integrals(4, unrestricted)
    zero = jax.tree.map(jnp.zeros_like, g)
    space = (make_uci_space(4, (2, 1), max_excitation=2, frozen=((0,), ()))
             if unrestricted else make_ci_space(4, 2, max_excitation=2, frozen=(0,)))
    vector = jnp.arange(space.size, dtype=float) / max(1, space.size)
    expected = hamiltonian.build_hamiltonian(h, zero, space)(vector)
    direct = hamiltonian.build_one_body(h, space, block_size=2)
    np.testing.assert_allclose(direct.apply(vector), expected, atol=2e-12, rtol=0)
    value = lambda t: jnp.sum(hamiltonian.build_one_body(
        jax.tree.map(lambda x: x*t, h), space, block_size=2).apply(vector)**2)
    np.testing.assert_allclose(jax.grad(value)(1.), 2*jnp.sum(expected**2), atol=2e-11, rtol=0)


def test_one_body_transpose_for_a_nonsymmetric_real_matrix():
    h = jnp.array([[.1, .3, -.2], [.7, .2, .4], [.5, -.1, -.3]])
    space = make_uci_space(3, (1, 0), max_excitation=1)
    operator = hamiltonian.build_one_body((h, jnp.zeros_like(h)), space)
    vector = jnp.array([.2, -.3, .7])
    np.testing.assert_allclose(operator.apply(vector), h@vector, atol=2e-14, rtol=0)
    np.testing.assert_allclose(operator.T.apply(vector), h.T@vector, atol=2e-14, rtol=0)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_one_body_block_diagonal_projection_and_response(unrestricted):
    h, g = _integrals(4, unrestricted)
    nocc = (2, 1) if unrestricted else 2
    space = (make_uci_space(4, nocc, max_excitation=2) if unrestricted
             else make_ci_space(4, nocc, max_excitation=2))
    counts = nocc if unrestricted else (nocc,)
    matrices = h if unrestricted else (h,)
    projected = tuple(x.at[:no, no:].set(0.).at[no:, :no].set(0.)
                      for x, no in zip(matrices, counts))
    projected = projected if unrestricted else projected[0]
    expected = hamiltonian.build_hamiltonian(projected, jax.tree.map(jnp.zeros_like, g), space)
    vector = jnp.arange(space.size, dtype=float) / space.size
    actual = hamiltonian.build_one_body(h, space, block_diagonal=True)
    np.testing.assert_allclose(actual.apply(vector), expected(vector), atol=2e-12, rtol=0)
    np.testing.assert_allclose(actual.T.apply(vector), expected(vector), atol=2e-12, rtol=0)
    def value(t):
        varied = tuple(x.at[:no, no:].add(t).at[no:, :no].add(t)
                       for x, no in zip(matrices, counts))
        varied = varied if unrestricted else varied[0]
        return jnp.sum(hamiltonian.build_one_body(varied, space,
                       block_diagonal=True).apply(vector)**2)
    np.testing.assert_allclose(jax.grad(value)(.13), 0., atol=2e-12, rtol=0)


def _array_shapes(value):
    if hasattr(value, "jaxpr"):
        yield from _array_shapes(value.jaxpr)
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            for variable in equation.outvars:
                yield getattr(getattr(variable, "aval", None), "shape", ())
            yield from _array_shapes(equation.params)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _array_shapes(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _array_shapes(item)


def test_direct_action_has_no_full_fci_or_excitation_workspace():
    h, g = _integrals(5)
    space = make_ci_space(5, 2, max_excitation=1)
    assert space.size == 13
    def loss(t):
        operator = hamiltonian.build_direct_hamiltonian(h*t, g, space, block_size=2)
        return jnp.sum(operator.apply(jnp.ones(space.size))**2)
    forbidden = {(space.size, space.size), (25, space.size), (10, 10), (100,)}
    for traced in (jax.make_jaxpr(loss)(1.), jax.make_jaxpr(jax.grad(loss))(1.)):
        assert not (set(_array_shapes(traced)) & forbidden)


def test_direct_action_does_not_build_global_connections(monkeypatch):
    def reject(_space):
        raise AssertionError("The direct path constructed determinant-pair connections")
    monkeypatch.setattr(hamiltonian, "_connections", reject)
    h, g = _integrals(4)
    space = make_ci_space(4, 2)
    result = hamiltonian.build_direct_hamiltonian(h, g, space, block_size=2).apply(jnp.ones(space.size))
    assert np.all(np.isfinite(result))


def test_ci_solver_uses_direct_action(monkeypatch):
    h, g = _integrals(4)
    space = make_ci_space(4, 2)
    reference = hamiltonian.hamiltonian_matrix(h, g, space)
    expected = np.linalg.eigvalsh(np.asarray(reference))[0]
    def reject(_space):
        raise AssertionError("solve_ci used the connection-table oracle")
    monkeypatch.setattr(hamiltonian, "_connections", reject)
    result = solve_ci(h, g, space, config=CIConfig(solver="dense"))
    np.testing.assert_allclose(result.total_energies[0], expected, atol=2e-11, rtol=0)


def test_default_ci_capacity_includes_water_double_zeta_doubles():
    # 5 occupied and 14 virtual Cartesian orbitals: 6861 retained determinants.
    space = make_ci_space(19, 5)
    assert space.size == 6861
    with pytest.raises(ValueError, match="max_determinants"):
        make_ci_space(19, 5, max_determinants=5000)


def test_diagonal_ci_keeps_exact_reference_guess_with_one_cycle():
    h = jnp.diag(jnp.array([-1.5, -.7, .2, .8, 1.6]))
    space = make_ci_space(5, 2)
    result = solve_ci(h, jnp.zeros((5,)*4), space,
                      config=CIConfig(max_cycle=1, max_space=8, conv_tol=1e-11))
    assert bool(result.converged[0])
    np.testing.assert_allclose(result.total_energies[0], -4.4, atol=2e-12, rtol=0)


def test_default_ci_davidson_has_bounded_basis_in_forward_and_backward():
    h, g = _integrals(5)
    space = make_ci_space(5, 2)
    assert space.size > 40
    value = lambda t: solve_ci(h*t, g, space).total_energies[0]
    for traced in (jax.make_jaxpr(value)(1.), jax.make_jaxpr(jax.grad(value))(1.)):
        assert (space.size, space.size) not in set(_array_shapes(traced))


def test_n2_sto3g_cisd_matches_pyscf():
    pyscf = pytest.importorskip("pyscf")
    from gradscf import ci, gto, scf
    atom = "N 0 0 -1.04; N 0 0 1.04"
    molecule = pyscf.gto.M(atom=atom, basis="sto-3g", unit="Bohr", cart=True, verbose=0)
    mf = pyscf.scf.RHF(molecule).run(conv_tol=1e-13, conv_tol_grad=1e-11)
    reference = mf.CISD().run(conv_tol=1e-11)
    native_molecule = gto.M(atom=atom, basis="sto-3g", unit="Bohr", cart=True)
    native = scf.RHF(native_molecule, integral_backend="native", conv_tol=1e-13,
        conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=250,
        init_guess=1.4*np.eye(10)).run()
    assert native.converged
    np.testing.assert_allclose(native.e_tot, mf.e_tot, atol=2e-10, rtol=0)
    result = ci.CISD(native, conv_tol=1e-10, max_cycle=120, max_space=30).run()
    assert result.converged
    np.testing.assert_allclose(result.e_tot, reference.e_tot, atol=2e-9, rtol=0)
