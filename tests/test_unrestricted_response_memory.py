"""Bounded unrestricted response preserves the joint spin-density Hessian."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.scf.molecules import QuadratureGrid
from gradscf.tddft import unrestricted as response
from gradscf.tddft._unrestricted_semilocal_response import UnrestrictedSemilocalResponseFunctional


@pytest.fixture(scope="module")
def unrestricted_problem():
    # Four-AO algebraic response; the native ERIs alone come from H4/STO-3G.
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H 0 0 1.4; H 0 1.7 0; H 1.8 0 0", "sto-3g", cart=True)
    plan = integrals.make_plan(topology)
    return NativeDirectBasis(plan, parameters), plan.evaluate("eri", parameters)


def _reference(problem, beta_occupied, *, direct):
    provider, eri = problem
    alpha = jnp.eye(4)
    beta = jnp.asarray([[.8, -.6, 0., 0.], [.6, .8, 0., 0.],
                        [0., 0., .6, -.8], [0., 0., .8, .6]])
    coeff = jnp.stack((alpha, beta))
    occupations = jnp.asarray([[1., 1., 0., 0.],
                               [float(beta_occupied), 0., 0., 0.]])
    density = jnp.einsum("spi,si,sqi->spq", coeff, occupations, coeff)
    ao = jnp.linspace(-.6, .9, 28).reshape(7, 4)
    return SimpleNamespace(
        mo_coeff=coeff, mo_occ=occupations,
        mo_energy=jnp.asarray([[-1.7, -.8, .5, 1.3], [-1.4, -.3, .7, 1.6]]),
        nocc_alpha=2, nocc_beta=beta_occupied, rdm1=density,
        ao=ao, ao_deriv1=jnp.stack((ao, .11*ao, -.13*ao, .17*ao)),
        grid=QuadratureGrid(jnp.asarray([.2, .3, .4, .25, .35, .15, .45])),
        rep_tensor=jnp.zeros((0, 0, 0, 0)) if direct else eri,
        direct_basis=provider if direct else None, exact_exchange_fraction=1.,
        h1e=jnp.diag(jnp.asarray([-3., -1., 1., 3.])), nuclear_repulsion=0.,
    )


def _parameters():
    layers = []
    for inputs, outputs in ((4, 5), (5, 4), (4, 3), (3, 1)):
        layers.append((jnp.linspace(-.22, .19, inputs*outputs).reshape(inputs, outputs),
                       jnp.linspace(.01, .04, outputs)))
    return {"layers": tuple(layers), "scale": jnp.asarray(.27)}


def _mlp(layers, inputs):
    value = inputs
    for weight, bias in layers[:-1]:
        value = jnp.tanh(value @ weight + bias)
    weight, bias = layers[-1]
    return (value @ weight + bias)[..., 0]


def _network_energy(params, density, ao, weights, coupled):
    density = .5*(density + density.swapaxes(-1, -2))
    rho = jnp.einsum("sgp,gp->sg", jnp.matmul(ao, density), ao)
    inputs = jnp.stack((rho[0], rho[1], jnp.sin(rho[0]), jnp.sin(rho[1])), axis=-1)
    pooled = jnp.dot(weights, _mlp(params["layers"], inputs) + .11*rho[0]*rho[1])
    if coupled:
        joint = jnp.stack((jnp.trace(density[0]), jnp.trace(density[1]),
                           jnp.sum(density[0]), jnp.sum(density[1])))
        pooled = pooled + .23*pooled**2 + .17*_mlp(params["layers"], joint)
    return params["scale"]*pooled


def _functional(coupled):
    def inputs(state):
        assert state.ao.shape == (7, 4), "An opaque callback must retain its complete grid"
        assert state.density_matrix.shape == (2, 4, 4), "Both spin densities are one input"
        return state.density_matrix, state.ao, state.weights

    return dft.custom_xc(input_fn=inputs,
        energy_fn=lambda params, data: _network_energy(params, *data, coupled),
        name="joint_spin_global" if coupled else "joint_spin_local")


def _dense_action(reference, eri, params, density, rows, full_td, energy=None, alpha=1.):
    counts = (reference.nocc_alpha, reference.nocc_beta)
    orbitals = [(reference.mo_coeff[s, :, :n], reference.mo_coeff[s, :, n:])
                for s, n in enumerate(counts)]
    gaps = [reference.mo_energy[s, n:][None, :] - reference.mo_energy[s, :n, None]
            for s, n in enumerate(counts)]
    widths = [int(gap.size) for gap in gaps]
    dim = sum(widths)

    def split(values):
        return (values[:, :widths[0]].reshape(len(values), *gaps[0].shape),
                values[:, widths[0]:].reshape(len(values), *gaps[1].shape))

    x = split(rows[:, :dim])
    y = split(rows[:, dim:]) if full_td else tuple(jnp.zeros_like(v) for v in x)
    transitions = []
    for (occupied, virtual), xx, yy in zip(orbitals, x, y, strict=True):
        transitions.append(jnp.einsum("nia,pa,qi->npq", xx, virtual, occupied)
                           + jnp.einsum("nia,pi,qa->npq", yy, occupied, virtual))
    tangent = jnp.stack(transitions, axis=1)
    coulomb = jnp.einsum("pqrs,nrs->npq", eri, tangent.sum(axis=1))
    exchange = jnp.stack(
        [jnp.einsum("prqs,nrs->npq", eri, value) for value in transitions], axis=1)
    potentials = coulomb[:, None] - alpha*exchange
    if energy is not None:
        hessian = jax.hessian(lambda dm: energy(params, dm.reshape(density.shape)))(density.reshape(-1))
        potentials = potentials + (tangent.reshape(len(rows), -1) @ hessian.T).reshape(tangent.shape)
    upper, lower = [], []
    for s, ((occupied, virtual), xx, yy, gap) in enumerate(zip(orbitals, x, y, gaps, strict=True)):
        def orbital_action(values):
            if getattr(reference, "fock_matrix", None) is None:
                return gap[None]*values
            fock = reference.fock_matrix[s]
            return (jnp.einsum("nia,ab->nib", values, virtual.T @ fock @ virtual)
                    - jnp.einsum("ij,nja->nia", occupied.T @ fock @ occupied, values))

        upper.append((jnp.einsum("npq,qi,pa->nia", potentials[:, s], occupied, virtual)
                      + orbital_action(xx)).reshape(len(rows), -1))
        lower.append((jnp.einsum("npq,pi,qa->nia", potentials[:, s], occupied, virtual)
                      + orbital_action(yy)).reshape(len(rows), -1))
    top = jnp.concatenate(upper, axis=-1)
    return jnp.concatenate((top, -jnp.concatenate(lower, axis=-1)), axis=-1) if full_td else top


def _rows(reference, full_td, count=3):
    dim = sum(n*(4-n) for n in (reference.nocc_alpha, reference.nocc_beta))
    return jnp.linspace(-.3, .4, count*dim*(2 if full_td else 1)).reshape(count, -1)


@pytest.mark.parametrize("full_td", [False, True])
@pytest.mark.parametrize("beta_occupied", [0, 1])
def test_unrestricted_direct_hf_matches_dense_non_symmetric_spin_action(
        unrestricted_problem, full_td, beta_occupied):
    reference = _reference(unrestricted_problem, beta_occupied, direct=True)
    builder = response.build_unrestricted_tdhf_operator if full_td else response.build_unrestricted_tda_operator
    rows = _rows(reference, full_td)
    oracle = lambda values: _dense_action(reference, unrestricted_problem[1], None,
                                         reference.rdm1, values, full_td)
    dense_reference = _reference(unrestricted_problem, beta_occupied, direct=False)
    np.testing.assert_allclose(builder(dense_reference)[0](rows), oracle(rows),
                               atol=3e-11, rtol=3e-11)
    vind = builder(reference)[0]
    np.testing.assert_allclose(jax.jit(vind)(rows), oracle(rows), atol=3e-11, rtol=3e-11)
    direction = jnp.ones_like(rows)*.07
    np.testing.assert_allclose(jax.jvp(vind, (rows,), (direction,))[1],
                               jax.jvp(oracle, (rows,), (direction,))[1], atol=3e-11, rtol=3e-11)
    for actual, expected in zip(jax.vjp(vind, rows)[1](rows),
                                jax.vjp(oracle, rows)[1](rows), strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("full_td", [False, True])
@pytest.mark.parametrize("beta_occupied", [0, 1])
@pytest.mark.parametrize("coupled", [False, True])
def test_unrestricted_external_spin_network_preserves_joint_hessian_and_ad(
        unrestricted_problem, monkeypatch, full_td, beta_occupied, coupled):
    reference = _reference(unrestricted_problem, beta_occupied, direct=True)
    params, rows = _parameters(), _rows(reference, full_td)
    functional = _functional(coupled)

    def forbidden(*args, **kwargs):
        raise AssertionError("Opaque AO kernels must bypass local grid projections")

    monkeypatch.setattr(response, "_unrestricted_grid_xc_response_action", forbidden)
    monkeypatch.setattr(response, "build_unrestricted_semilocal_response_action", forbidden)
    builder = response.build_unrestricted_tdhf_operator if full_td else response.build_unrestricted_tda_operator

    def actual(p, density):
        state = SimpleNamespace(**{**vars(reference), "rdm1": density})
        return builder(state, functional, xc_params=p)[0](rows)

    def oracle(p, density):
        energy = lambda pp, dm: _network_energy(pp, dm, reference.ao, reference.grid.weights, coupled)
        return _dense_action(reference, unrestricted_problem[1], p, density, rows, full_td,
                             energy=energy, alpha=0.)

    dense_reference = _reference(unrestricted_problem, beta_occupied, direct=False)
    dense_action = builder(dense_reference, functional, xc_params=params)[0]
    np.testing.assert_allclose(dense_action(rows), oracle(params, dense_reference.rdm1),
                               atol=4e-11, rtol=4e-11)
    np.testing.assert_allclose(jax.jit(actual)(params, reference.rdm1), oracle(params, reference.rdm1),
                               atol=4e-11, rtol=4e-11)
    params_dot = jax.tree.map(lambda value: jnp.ones_like(value)*.03, params)
    density_dot = jnp.linspace(-.1, .2, 32).reshape(2, 4, 4)
    actual_dot = jax.jvp(actual, (params, reference.rdm1), (params_dot, density_dot))[1]
    expected_dot = jax.jvp(oracle, (params, reference.rdm1), (params_dot, density_dot))[1]
    np.testing.assert_allclose(actual_dot, expected_dot, atol=4e-10, rtol=4e-10)
    cotangent = jnp.linspace(.1, .5, rows.size).reshape(rows.shape)
    actual_grad = jax.vjp(actual, params, reference.rdm1)[1](cotangent)
    expected_grad = jax.vjp(oracle, params, reference.rdm1)[1](cotangent)
    for value, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True):
        np.testing.assert_allclose(value, expected, atol=4e-10, rtol=4e-10)


@pytest.mark.parametrize("full_td", [False, True])
def test_unrestricted_standard_hf_bypasses_grid_factors(unrestricted_problem, monkeypatch, full_td):
    reference = _reference(unrestricted_problem, 1, direct=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("HF has no grid response workspace")

    monkeypatch.setattr(response, "build_unrestricted_semilocal_response_action", forbidden)
    monkeypatch.setattr(response, "_unrestricted_grid_xc_response_action", forbidden)
    builder = response.build_unrestricted_tdhf_operator if full_td else response.build_unrestricted_tda_operator
    vind = builder(reference, UnrestrictedSemilocalResponseFunctional("hf"))[0]
    rows = _rows(reference, full_td)
    np.testing.assert_allclose(vind(rows), _dense_action(reference, unrestricted_problem[1], None,
        reference.rdm1, rows, full_td), atol=3e-11, rtol=3e-11)
    assert vind.memory_plan.grid_block_size == 0


def test_unrestricted_microbatch_bounds_transition_workspace(unrestricted_problem, monkeypatch):
    reference = _reference(unrestricted_problem, 1, direct=False)
    vind = response.build_unrestricted_tda_operator(reference,
        response_kernel_options={"max_memory": .03, "integral_cache": "none"})[0]
    width = vind.memory_plan.batch_size
    rows = _rows(reference, False, count=2*width+1)
    calls, original = [], response._unrestricted_transition_density

    def bounded(orbo, orbv, values, *, bottom):
        calls.append(values.shape[0])
        assert values.shape[0] <= width
        return original(orbo, orbv, values, bottom=bottom)

    monkeypatch.setattr(response, "_unrestricted_transition_density", bounded)
    expected = _dense_action(reference, unrestricted_problem[1], None, reference.rdm1, rows, False)
    np.testing.assert_allclose(vind(rows), expected, atol=3e-11, rtol=3e-11)
    assert calls
    np.testing.assert_allclose(jax.grad(lambda values: jnp.sum(vind(values)**2))(rows),
        jax.grad(lambda values: jnp.sum(_dense_action(reference, unrestricted_problem[1], None,
            reference.rdm1, values, False)**2))(rows), atol=3e-10, rtol=3e-10)


def _grid_graph_shapes(value):
    if hasattr(value, "jaxpr"):
        return _grid_graph_shapes(value.jaxpr)
    if hasattr(value, "eqns"):
        shapes = {v.aval.shape for v in (*value.constvars, *value.invars, *value.outvars)
                  if hasattr(v.aval, "shape")}
        for equation in value.eqns:
            shapes.update(v.aval.shape for v in equation.outvars if hasattr(v.aval, "shape"))
            shapes.update(_grid_graph_shapes(equation.params))
        return shapes
    if isinstance(value, dict):
        return set().union(*(_grid_graph_shapes(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(_grid_graph_shapes(v) for v in value))
    return set()


def test_legacy_unrestricted_lda_keeps_full_grid_callback_without_grid_ov(
        unrestricted_problem, monkeypatch):
    reference = _reference(unrestricted_problem, 1, direct=False)
    calls = []

    class LegacySpinKernel:
        exact_exchange_fraction = 0.

        def spin_grid_kernel(self, state):
            assert state.ao.shape == (7, 4), "Legacy callbacks retain the original full grid"
            calls.append(state.ao.shape[0])
            return tuple(jnp.full(7, coefficient) for coefficient in (1.7, .4, 2.1))

    def energy(_params, density):
        density = .5*(density + density.swapaxes(-1, -2))
        rho = jnp.einsum("sgp,gp->sg", jnp.matmul(reference.ao, density), reference.ao)
        return jnp.dot(reference.grid.weights,
                       .85*rho[0]**2 + .4*rho[0]*rho[1] + 1.05*rho[1]**2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy LDA must not construct grid-by-OV transition features")

    monkeypatch.setattr(response, "_transition_densities_on_grid", forbidden, raising=False)
    vind = response.build_unrestricted_tda_operator(reference, LegacySpinKernel(),
        response_kernel_options={"grid_block_size": 3, "integral_cache": "none"})[0]
    rows = _rows(reference, False, count=8)
    oracle = lambda values: _dense_action(reference, unrestricted_problem[1], None,
        reference.rdm1, values, False, energy=energy, alpha=0.)
    np.testing.assert_allclose(jax.jit(vind)(rows), oracle(rows), atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(jax.grad(lambda values: jnp.sum(vind(values)**2))(rows),
        jax.grad(lambda values: jnp.sum(oracle(values)**2))(rows), atol=3e-10, rtol=3e-10)
    assert calls
    graph = jax.make_jaxpr(vind)(rows)
    shapes = _grid_graph_shapes(graph)
    for grids in (7, 3):
        assert (grids, 2, 2) not in shapes
        assert (grids, 1, 3) not in shapes


@pytest.mark.parametrize("spec,kind", [("lda", "LDA"), ("pbe", "GGA"), ("tpss", "MGGA")])
def test_standard_spin_grid_blocks_match_full_hvp_without_grid_ov(
        unrestricted_problem, monkeypatch, spec, kind):
    from gradscf.tddft import _unrestricted_semilocal_response as semilocal

    reference = _reference(unrestricted_problem, 1, direct=False)
    functional = UnrestrictedSemilocalResponseFunctional(spec)
    fields = {"LDA": 2, "GGA": 8, "MGGA": 10}[kind]
    matrix = .17*jnp.eye(fields) + .03*jnp.ones((fields, fields))
    checking, grid_sizes = [False], []

    def hessian_factory(_spec, _kind):
        def hessian(base):
            if checking[0]:
                assert base.shape[0] <= 3, "Conventional local Hessians must obey the grid block"
                grid_sizes.append(base.shape[0])
            return jnp.broadcast_to(matrix, (base.shape[0], fields, fields))
        return hessian

    monkeypatch.setattr(semilocal, "grid_hessian", hessian_factory)
    orbital_args = (reference, reference.mo_coeff[0, :, :2], reference.mo_coeff[0, :, 2:],
                    reference.mo_coeff[1, :, :1], reference.mo_coeff[1, :, 1:],
                    functional.spin_grid_response_hvp)
    controls = dict(feature_kind=kind, dtype=jnp.float64)
    dense = semilocal.build_unrestricted_semilocal_response_action(*orbital_args, **controls)
    alpha = jnp.linspace(-.2, .3, 32).reshape(8, 2, 2)
    beta = jnp.linspace(.1, -.3, 24).reshape(8, 1, 3)

    def loss(action, a, b):
        return sum(jnp.sum(value**2) for value in action(a, b))

    expected = dense(alpha, beta)
    expected_grad = jax.grad(lambda a, b: loss(dense, a, b), argnums=(0, 1))(alpha, beta)
    original, factor_sizes = semilocal._restricted_response_factors, []

    def bounded_factors(*args, **kwargs):
        size = kwargs.get("grid_size")
        assert size is not None and 1 <= size <= 3, "Do not construct full-grid orbital factors"
        factor_sizes.append(size)
        return original(*args, **kwargs)

    monkeypatch.setattr(semilocal, "_restricted_response_factors", bounded_factors)
    checking[0] = True
    blocked = semilocal.build_unrestricted_semilocal_response_action(
        *orbital_args, block_size=3, **controls)
    for value, target in zip(jax.jit(blocked)(alpha, beta), expected, strict=True):
        np.testing.assert_allclose(value, target, atol=3e-11, rtol=3e-11)
    gradients = jax.grad(lambda a, b: loss(blocked, a, b), argnums=(0, 1))(alpha, beta)
    for value, target in zip(gradients, expected_grad, strict=True):
        np.testing.assert_allclose(value, target, atol=3e-10, rtol=3e-10)
    assert {1, 3}.issubset(factor_sizes) and {1, 3}.issubset(grid_sizes)
    forward = jax.make_jaxpr(blocked)(alpha, beta)
    reverse = jax.make_jaxpr(jax.grad(
        lambda a, b: loss(blocked, a, b), argnums=(0, 1)))(alpha, beta)
    shapes = _grid_graph_shapes(forward) | _grid_graph_shapes(reverse)
    for grids in (7, 3):
        # A valid local LDA spin Hessian is itself (G,2,2); alpha OV has the
        # same dimensions in this fixture. The beta OV shape remains distinct.
        if kind != "LDA":
            assert (grids, 2, 2) not in shapes
        assert (grids, 1, 3) not in shapes
    assert (2, 8, 2, 2) not in shapes and (2, 8, 1, 3) not in shapes


@pytest.mark.parametrize("full_td", [False, True])
def test_unrestricted_noncanonical_response_is_covariant_inside_orbital_subspaces(
        unrestricted_problem, full_td):
    reference = _reference(unrestricted_problem, 1, direct=False)
    fock_mo = jnp.asarray([
        [[-1.8, .16, 0., 0.], [.16, -.7, 0., 0.],
         [0., 0., .6, .12], [0., 0., .12, 1.4]],
        [[-1.5, 0., 0., 0.], [0., -.2, .11, .04],
         [0., .11, .8, .15], [0., .04, .15, 1.7]],
    ])
    reference.fock_matrix = reference.mo_coeff @ fock_mo @ reference.mo_coeff.swapaxes(-1, -2)
    rotations = jnp.asarray([
        [[.8, -.6, 0., 0.], [.6, .8, 0., 0.],
         [0., 0., .6, -.8], [0., 0., .8, .6]],
        [[1., 0., 0., 0.], [0., .8, -.6, 0.],
         [0., .6, .8, 0.], [0., 0., 0., 1.]],
    ])
    rotated = SimpleNamespace(**{**vars(reference), "mo_coeff": reference.mo_coeff @ rotations})
    builder = response.build_unrestricted_tdhf_operator if full_td else response.build_unrestricted_tda_operator
    rows = _rows(reference, full_td)

    def rotate(values):
        blocks = []
        for side in range(2 if full_td else 1):
            offset = side*7
            for spin, (occupied, width) in enumerate(((2, 4), (1, 3))):
                x = values[:, offset:offset+width].reshape(len(values), occupied, 4-occupied)
                rotation = rotations[spin]
                blocks.append((rotation[:occupied, :occupied].T @ x
                               @ rotation[occupied:, occupied:]).reshape(len(values), width))
                offset += width
        return jnp.concatenate(blocks, axis=-1)

    original = builder(reference)[0](rows)
    expected = _dense_action(reference, unrestricted_problem[1], None, reference.rdm1, rows, full_td)
    np.testing.assert_allclose(original, expected, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(builder(rotated)[0](rotate(rows)), rotate(original),
                               atol=3e-11, rtol=3e-11)


def test_unrestricted_global_kernel_preserves_jax_geometry_dependency_chain(unrestricted_problem):
    # An algebraic geometry chain verifies the response projection/kernel AD.
    # It does not certify the separate native direct-integral geometry rule.
    base = _reference(unrestricted_problem, 1, direct=False)
    params, functional, rows = _parameters(), _functional(True), _rows(base, True)

    def state_at(displacement):
        coords = jnp.asarray([[0., 0., 0.], [0., 0., 1.4], [0., 1.7, 0.], [1.8, 0., 0.]])
        coords = coords + displacement*jnp.linspace(-.2, .3, 12).reshape(4, 3)
        pair = jnp.exp(-.2*jnp.sum((coords[:, None]-coords[None, :])**2, axis=-1))
        eri = .08*jnp.einsum("pq,rs->pqrs", pair, pair)
        angle = .13 + .2*displacement
        sine, cosine = jnp.sin(angle), jnp.cos(angle)
        rotation = jnp.asarray([[cosine, 0., -sine, 0.], [0., 1., 0., 0.],
                                [sine, 0., cosine, 0.], [0., 0., 0., 1.]])
        coeff = base.mo_coeff @ rotation
        density = jnp.einsum("spi,si,sqi->spq", coeff, base.mo_occ, coeff)
        ao = base.ao + .07*jnp.sin(jnp.arange(7)[:, None] + coords[:, 0][None, :])
        weights = base.grid.weights*(1. + .03*displacement)
        fock = jnp.stack((base.h1e + .04*pair, base.h1e - .03*pair))
        return SimpleNamespace(**{**vars(base), "atom_coords": coords, "rep_tensor": eri,
            "mo_coeff": coeff, "rdm1": density, "ao": ao,
            "grid": QuadratureGrid(weights), "fock_matrix": fock})

    def actual(displacement):
        state = state_at(displacement)
        return response.build_unrestricted_tdhf_operator(state, functional, xc_params=params)[0](rows)

    def oracle(displacement):
        state = state_at(displacement)
        energy = lambda p, density: _network_energy(p, density, state.ao, state.grid.weights, True)
        return _dense_action(state, state.rep_tensor, params, state.rdm1, rows, True,
                             energy=energy, alpha=0.)

    point, direction = jnp.asarray(.09), jnp.asarray(.7)
    np.testing.assert_allclose(jax.jit(actual)(point), oracle(point), atol=4e-11, rtol=4e-11)
    actual_dot = jax.jvp(actual, (point,), (direction,))[1]
    oracle_dot = jax.jvp(oracle, (point,), (direction,))[1]
    assert jnp.linalg.norm(oracle_dot) > 1e-6
    np.testing.assert_allclose(actual_dot, oracle_dot, atol=4e-10, rtol=4e-10)
    cotangent = jnp.linspace(.1, .5, rows.size).reshape(rows.shape)
    np.testing.assert_allclose(jax.vjp(actual, point)[1](cotangent)[0],
        jax.vjp(oracle, point)[1](cotangent)[0], atol=4e-10, rtol=4e-10)
