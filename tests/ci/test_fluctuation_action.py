"""Normal-ordered MP W versus an independent Slater--Condon construction."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.ci import hamiltonian
from gradscf.ci.space import make_ci_space, make_uci_space
from gradscf.fci.cistring import excite


def _builder():
    function = getattr(hamiltonian, "build_fluctuation", None)
    assert callable(function), "The shared normal-ordered fluctuation builder is missing"
    return function


def _model(unrestricted):
    rng = np.random.default_rng(854)
    n = 5
    h = rng.normal(size=(2 if unrestricted else 1, n, n)) * .2
    h = .5 * (h + h.swapaxes(-1, -2))
    h += np.diag([-1.2, -.4, .3, .7, 1.1])
    factors = rng.normal(size=(2 if unrestricted else 1, 4, n, n)) * .05
    factors += factors.swapaxes(-1, -2)
    if not unrestricted:
        return jnp.asarray(h[0]), jnp.asarray(np.einsum("Lpq,Lrs->pqrs", factors[0], factors[0]))
    a, b = factors
    return tuple(map(jnp.asarray, h)), tuple(map(jnp.asarray, (
        np.einsum("Lpq,Lrs->pqrs", a, a), np.einsum("Lpq,Lrs->pqrs", a, b),
        np.einsum("Lpq,Lrs->pqrs", b, b))))


def _focks(h, g, nocc):
    if not isinstance(nocc, tuple):
        return h + 2*jnp.einsum("pqii->pq", g[:, :, :nocc, :nocc]) - jnp.einsum(
            "piiq->pq", g[:, :nocc, :nocc, :])
    na, nb = nocc
    aa, ab, bb = g
    return (h[0] + jnp.einsum("pqii->pq", aa[:, :, :na, :na])
            - jnp.einsum("piiq->pq", aa[:, :na, :na, :])
            + jnp.einsum("pqii->pq", ab[:, :, :nb, :nb]),
            h[1] + jnp.einsum("pqii->pq", bb[:, :, :nb, :nb])
            - jnp.einsum("piiq->pq", bb[:, :nb, :nb, :])
            + jnp.einsum("iipq->pq", ab[:na, :na]))


def _oracle(h, g, space):
    """Old determinant connections only; no shared direct one-body kernel."""
    focks = _focks(h, g, space.nocc)
    ns = space.nocc if isinstance(space.nocc, tuple) else (space.nocc,)
    fs = focks if isinstance(space.nocc, tuple) else (focks,)
    blocks = []
    for f, no in zip(fs, ns):
        occupied = jnp.arange(f.shape[0]) < no
        blocks.append(jnp.where(occupied[:, None] == occupied[None, :], f, 0.))
    physical = hamiltonian.build_hamiltonian(h, g, space)
    one_body = hamiltonian.build_hamiltonian(tuple(blocks) if len(blocks) == 2 else blocks[0],
                                            jax.tree.map(jnp.zeros_like, g), space)
    diagonal = physical.diagonal - physical.diagonal[0] - one_body.diagonal + one_body.diagonal[0]
    apply = lambda c: physical(c) - physical.diagonal[0]*c - one_body(c) + one_body.diagonal[0]*c
    return apply, diagonal


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("limits", [(None, None), (2, 0)])
def test_fluctuation_sigma_transpose_and_physical_vjp(unrestricted, frozen, limits):
    build = _builder()
    h, g = _model(unrestricted)
    no = (2, 1) if unrestricted else 2
    selected = (([0], [4]) if unrestricted else [0, 4]) if frozen else None
    make = make_uci_space if unrestricted else make_ci_space
    space = make(5, no, max_excitation=2, frozen=selected)
    input_rank, output_rank = limits
    ranks = np.asarray(space.ranks)
    inputs = jnp.asarray(np.ones(space.size, bool) if input_rank is None else ranks <= input_rank)
    outputs = jnp.asarray(np.ones(space.size, bool) if output_rank is None else ranks <= output_rank)
    rng = np.random.default_rng(855)
    vectors = jnp.asarray(rng.normal(size=(space.size, 2)))
    left, right = vectors[:, 0], vectors[:, 1]
    reference, diagonal = _oracle(h, g, space)
    operator = build(h, g, space, block_size=3,
                     max_input_rank=input_rank, max_output_rank=output_rank)
    expected = outputs[:, None] * reference(inputs[:, None] * vectors)
    np.testing.assert_allclose(operator.apply(vectors), expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(operator.diagonal, diagonal*inputs*outputs, atol=2e-12, rtol=0)
    np.testing.assert_allclose(operator.T.apply(left), inputs*reference(outputs*left), atol=2e-12, rtol=0)

    def value(t, direct):
        hh = jax.tree.map(lambda x: x*(1 + .2*t), h)
        gg = jax.tree.map(lambda x: x*(1 - .1*t), g)
        if direct:
            action = build(hh, gg, space, block_size=3,
                           max_input_rank=input_rank, max_output_rank=output_rank).apply
            return jnp.dot(left, action(right))
        action, _ = _oracle(hh, gg, space)
        return jnp.dot(outputs*left, action(inputs*right))

    actual = jax.jit(jax.value_and_grad(lambda t: value(t, True)))(.13)
    expected = jax.jit(jax.value_and_grad(lambda t: value(t, False)))(.13)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


def test_fluctuation_keeps_nonzero_occupied_virtual_fock_and_its_response():
    build = _builder()
    h, g = _model(False)
    space = make_ci_space(5, 2, max_excitation=2)
    reference = jnp.zeros(space.size).at[0].set(1.)
    determinant, phase = excite(space.determinants[0], (0,), (2,))
    index = space.determinants.index(determinant)
    direction = jnp.zeros_like(h).at[2, 0].set(1.).at[0, 2].set(1.)
    def element(t):
        return build(h + t*direction, g, space).apply(reference)[index]
    value, derivative = jax.jit(jax.value_and_grad(element))(0.)
    expected = phase * _focks(h, g, 2)[2, 0]
    assert abs(expected) > 1e-3
    np.testing.assert_allclose(value, expected, atol=2e-13, rtol=0)
    np.testing.assert_allclose(derivative, phase, atol=2e-13, rtol=0)


def test_fluctuation_does_not_construct_diagonal_fock_kernels(monkeypatch):
    from gradscf.ci import direct
    build = _builder()
    original = direct._compressed_kernel
    one_body_types = []
    def record(tensor, types, *args):
        if len(types) == 2:
            one_body_types.append(types)
        return original(tensor, types, *args)
    monkeypatch.setattr(direct, "_compressed_kernel", record)
    h, g = _model(False)
    build(h, g, make_ci_space(5, 2))
    assert one_body_types
    assert all(a != b for a, b in one_body_types)
