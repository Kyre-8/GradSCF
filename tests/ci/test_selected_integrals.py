"""CI retained-rank kernels from selected AO blocks, including raw spin charts."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf.ao2mo import transform_integrals, transform_unrestricted_integrals
from gradscf.ci.direct import build_direct_hamiltonian, build_fluctuation, build_one_body, _reference


def _entry():
    function = getattr(importlib.import_module('gradscf.ci.integrals'), 'from_orbitals', None)
    assert callable(function), 'Missing selected CI integral preparation'
    build = getattr(importlib.import_module('gradscf.ci.direct'), 'from_prepared', None)
    assert callable(build), 'Missing prepared CI sigma action'
    return function, build


def _problem(unrestricted, rank=2, selected=None, frozen=None, nocc=None):
    rng = np.random.default_rng(741)
    c = jnp.asarray(np.eye(4)+rng.normal(scale=.02, size=(4, 4)))
    coefficients = (c, .9*c) if unrestricted else c
    h = jnp.diag(jnp.array([-1.4, -.8, .3, .9]))
    g = jnp.asarray(rng.normal(scale=.01, size=(4,)*4))
    transform = transform_unrestricted_integrals if unrestricted else transform_integrals
    hs, gs = transform(h, coefficients, eri=g)
    no = ((2, 1) if unrestricted else 2) if nocc is None else nocc
    spin_g = ((gs[0], gs[1]), (gs[1].transpose(2, 3, 0, 1), gs[2])) if unrestricted else ((gs, gs), (gs, gs))
    fs, energy = _reference(hs if unrestricted else (hs, hs), spin_g, no if unrestricted else (no, no))
    build_space = ci.make_uci_space if unrestricted else ci.make_ci_space
    space = build_space(4, no, max_excitation=rank, excitation_ranks=selected, frozen=frozen)
    return coefficients, fs if unrestricted else fs[0], energy, hs, gs, g, space


@pytest.mark.parametrize('unrestricted,rank,selected,frozen', [
    (False, 1, None, None), (False, 2, (0, 2), None),
    (False, 2, None, [0, 3]), (False, 3, None, None), (False, 4, None, None),
    (True, 1, None, None), (True, 2, (0, 2), None),
    (True, 2, None, ([0], [3])), (True, 3, None, None)])
def test_selected_sigma_matches_original_raw_chart(unrestricted, rank, selected, frozen):
    prepare, build = _entry()
    c, fs, energy, h, g, ao, space = _problem(unrestricted, rank, selected, frozen)
    data = prepare(c, fs, energy, space, eri=ao, max_memory=32.)
    actual = build(data, space)
    expected = build_direct_hamiltonian(h, g, space)
    probe = jnp.asarray(np.random.default_rng(742).normal(size=(space.size, 3)))
    np.testing.assert_allclose(actual.apply(probe), expected.apply(probe), atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(actual.diagonal, expected.diagonal, atol=2e-11)


def test_selected_preparation_never_requests_full_mo(monkeypatch):
    prepare, build = _entry()
    c, fs, energy, _, _, ao, space = _problem(False, 2, frozen=[0])
    module = importlib.import_module('gradscf.ci.integrals')
    original, seen = module.transform_eri_blocks, []
    def requested(coefficients, requests, **kwargs):
        seen.extend(tuple(coefficients[i].shape[1] for i in request) for request in requests)
        assert all(shape != (4,)*4 for shape in seen)
        return original(coefficients, requests, **kwargs)
    monkeypatch.setattr(module, 'transform_eri_blocks', requested)
    prepared = prepare(c, fs, energy, space, eri=ao, max_memory=32.)
    assert seen and np.isfinite(build(prepared, space).diagonal).all()


def test_selected_uci_nonempty_quadruples_match_raw_action_without_full_mo(monkeypatch):
    prepare, build = _entry()
    c, fs, energy, h, g, ao, space = _problem(True, 4, nocc=(2, 2))
    ranks = np.asarray(space.ranks)
    assert 4 in ranks and space.size == 36
    expected = build_direct_hamiltonian(h, g, space)
    random = np.random.default_rng(746).normal(size=(space.size, 2))
    probe = jnp.asarray(np.column_stack((ranks == 4, random)))
    reference = expected.apply(probe)
    assert np.linalg.norm(np.asarray(reference[:, 0])) > 1e-5
    module = importlib.import_module('gradscf.ci.integrals')
    original, requests_seen = module.transform_eri_blocks, []

    def selected(coefficients, requests, **kwargs):
        shapes = [tuple(coefficients[i].shape[1] for i in request) for request in requests]
        assert all(shape != (space.nmo,)*4 for shape in shapes)
        requests_seen.extend(shapes)
        return original(coefficients, requests, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('Selected UCISDTQ expanded a determinant connection table')

    monkeypatch.setattr(module, 'transform_eri_blocks', selected)
    monkeypatch.setattr(importlib.import_module('gradscf.ci.hamiltonian'), '_connections', forbidden)
    actual = build(prepare(c, fs, energy, space, eri=ao, max_memory=32.), space)
    assert requests_seen
    np.testing.assert_allclose(actual.apply(probe), reference, atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(actual.diagonal, expected.diagonal, atol=2e-11)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_singles_do_not_request_a_four_virtual_diagonal(unrestricted, monkeypatch):
    prepare, build = _entry()
    c, fs, energy, h, g, ao, space = _problem(unrestricted, 1)
    module = importlib.import_module('gradscf.ci.integrals')
    original = module.transform_eri_blocks

    def selected(coefficients, requests, **kwargs):
        assert not any(all(group < 2 for group in request) for request in requests), (
            'A singles determinant cannot contain two virtual particles')
        return original(coefficients, requests, **kwargs)

    monkeypatch.setattr(module, 'transform_eri_blocks', selected)
    actual = build(prepare(c, fs, energy, space, eri=ao, max_memory=32.), space)
    expected = build_direct_hamiltonian(h, g, space)
    probe = jnp.asarray(np.random.default_rng(745).normal(size=(space.size, 2)))
    np.testing.assert_allclose(actual.apply(probe), expected.apply(probe), atol=2e-11)
    np.testing.assert_allclose(actual.diagonal, expected.diagonal, atol=2e-11)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_prepared_raw_c_f_energy_ao_chart_jvp_vjp_hvp(unrestricted):
    prepare, build = _entry()
    c, fs, energy, _, _, ao, space = _problem(unrestricted, 2, frozen=([0], [3]) if unrestricted else [0])
    rng = np.random.default_rng(743)
    x = jnp.asarray(rng.normal(size=space.size))
    point = (c,fs,energy,ao)
    direction = jax.tree.map(lambda a:jnp.asarray(rng.normal(scale=.02,size=a.shape)), point)
    def loss(p, prepared):
        coeff,fock,e,source = p
        if prepared:
            op = build(prepare(coeff,fock,e,space,eri=source,max_memory=32.),space)
            return jnp.sum(op.matvec(x)**2)+.1*jnp.sum(op.diagonal**2)
        transform = transform_unrestricted_integrals if unrestricted else transform_integrals
        raw_h,gs = transform(jnp.zeros((4,4)),coeff,eri=source)
        spin_g = ((gs[0],gs[1]),(gs[1].transpose(2,3,0,1),gs[2])) if unrestricted else ((gs,gs),(gs,gs))
        ns = space.nocc if unrestricted else (space.nocc,)*2
        potentials,_ = _reference(raw_h if unrestricted else (raw_h,raw_h),spin_g,ns)
        hs = tuple(a-b for a,b in zip(fock,potentials)) if unrestricted else fock-potentials[0]
        _,constant = _reference(hs if unrestricted else (hs,hs),spin_g,ns)
        op = build_direct_hamiltonian(hs,gs,space)
        return jnp.sum((op.matvec(x)+(e-constant)*x)**2)+.1*jnp.sum((op.diagonal+e-constant)**2)
    def bundle(p,prepared):
        function=lambda a:loss(a,prepared)
        return jax.value_and_grad(function)(p),jax.jvp(jax.grad(function),(p,),(direction,))[1]
    actual=jax.jit(lambda p:bundle(p,True))(point)
    expected=jax.jit(lambda p:bundle(p,False))(point)
    for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,e,atol=3e-9,rtol=3e-8)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_prepared_normal_ordered_parts_and_rank_transpose(unrestricted):
    prepare, build = _entry()
    c,fs,energy,h,g,ao,space = _problem(unrestricted,2)
    data = prepare(c,fs,energy,space,eri=ao,max_memory=32.)
    x = jnp.asarray(np.random.default_rng(744).normal(size=(space.size,2)))
    actual = build(data,space,part='fluctuation',max_input_rank=1,max_output_rank=2)
    expected = build_fluctuation(h,g,space,max_input_rank=1,max_output_rank=2)
    np.testing.assert_allclose(actual.apply(x),expected.apply(x),atol=2e-11)
    np.testing.assert_allclose(actual.T.apply(x),expected.T.apply(x),atol=2e-11)
    np.testing.assert_allclose(actual.diagonal,expected.diagonal,atol=2e-11)
    fock = build(data,space,part='fock')
    ns = space.nocc if unrestricted else (space.nocc,)*2
    focks = fs if unrestricted else (fs,fs)
    constant = sum(jnp.trace(f[:no,:no]) for f,no in zip(focks,ns))
    expected_f = build_one_body(fs,space,block_diagonal=True)
    np.testing.assert_allclose(fock.apply(x),expected_f.apply(x)-constant*x,atol=2e-11)
