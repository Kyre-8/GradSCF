"""Complete excitation-block actions versus independent determinant operators."""
from itertools import product
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def model(unrestricted):
    rng = np.random.default_rng(31)
    n = 4
    factors = rng.normal(size=(5, n, n))*.12
    factors = .5*(factors+factors.transpose(0, 2, 1))
    g = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    h = np.diag([-1., -.4, .3, .8])
    h += .02*(np.ones((n, n))-np.eye(n))
    if not unrestricted:
        return jnp.asarray(h), jnp.asarray(g), 2
    cb = np.linalg.qr(rng.normal(size=(n, n)))[0]
    gb = np.einsum("pqrs,pP,qQ,rR,sS->PQRS", g, cb, cb, cb, cb)
    gab = np.einsum("pqrs,rR,sS->pqRS", g, cb, cb)
    return (jnp.asarray(h), jnp.asarray(cb.T@h@cb)), tuple(map(jnp.asarray, (g, gab, gb))), (2, 1)


def determinant_map(space, oracle):
    from gradscf.fci.cistring import excite
    n = space.nmo
    na, nb = space.nocc
    reference = (1 << na)-1 | (((1 << nb)-1) << n)
    lookup = {d:i for i,d in enumerate(oracle.determinants)}
    indices, signs = [], []
    va, vb, oa, ob = space.orbitals
    for block in space.blocks:
        rank = sum(block.excitation)
        phase = (-1)**(rank*(rank-1)//2)
        for coords in product(*(range(len(x)) for x in block.indices)):
            values = [tuple(domain[i] for i in choices[c])
                      for domain, choices, c in zip((va, vb, oa, ob), block.indices, coords)]
            particles = values[0]+tuple(n+x for x in values[1])
            holes = values[2]+tuple(n+x for x in values[3])
            det, sign = excite(reference, holes, particles)
            assert sign
            indices.append(lookup[det])
            signs.append(phase*sign)
    return np.asarray(indices), np.asarray(signs)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_complete_tensor_action_matches_determinant_matrix(rank, unrestricted):
    from gradscf.mp._tensor import make_tensor_space, build_tensor_action
    from gradscf.mp.integrals import _reference_data
    from gradscf import ci
    h, g, no = model(unrestricted)
    space = make_tensor_space(4, no, max_excitation=rank)
    oracle = (ci.make_uci_space if unrestricted else ci.make_ci_space)(4, no, max_excitation=rank)
    indices, signs = determinant_map(space, oracle)
    _, _, focks, e0 = _reference_data(h, g, no)
    action = build_tensor_action(g, focks, space)
    full = np.asarray(ci.hamiltonian_matrix(h, g, oracle))[np.ix_(indices, indices)]
    expected = signs[:, None]*full*signs[None, :]-np.diag(np.asarray(e0+space.gaps(focks)))
    actual = jax.jit(jax.vmap(action, in_axes=1, out_axes=1))(jnp.eye(space.size))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0.)


def test_packed_norm_and_frozen_action():
    from gradscf.mp._tensor import make_tensor_space, build_tensor_action
    from gradscf.mp.integrals import _reference_data
    from gradscf import ci
    h, g, no = model(True)
    frozen = ([0], [3])
    space = make_tensor_space(4, no, frozen=frozen, max_excitation=4)
    oracle = ci.make_uci_space(4, no, frozen=frozen, max_excitation=4)
    indices, signs = determinant_map(space, oracle)
    rng = np.random.default_rng(53)
    c = jnp.asarray(rng.normal(size=space.size))
    expanded = space.expand(c)
    norm = sum(jnp.sum(a*a)/np.prod([factorial(k) for k in b.counts])
               for a,b in zip(expanded, space.blocks))
    np.testing.assert_allclose(norm, jnp.vdot(c,c), atol=1e-12)
    np.testing.assert_allclose(space.pack(expanded), c, atol=1e-12)
    _, _, focks, e0 = _reference_data(h,g,no)
    full = np.asarray(ci.hamiltonian_matrix(h,g,oracle))[np.ix_(indices,indices)]
    expected = (signs[:,None]*full*signs[None,:]-np.diag(np.asarray(e0+space.gaps(focks)))) @ np.asarray(c)
    np.testing.assert_allclose(build_tensor_action(g,focks,space)(c), expected, atol=2e-12)


def test_sextuple_block_action_matches_determinant_vector():
    from gradscf.mp._tensor import make_tensor_space, build_tensor_action
    from gradscf.mp.integrals import _reference_data
    from gradscf import ci
    rng = np.random.default_rng(67)
    n = 6
    factors = rng.normal(size=(3,n,n))*.04
    factors = factors+factors.transpose(0,2,1)
    g = jnp.asarray(np.einsum("Lpq,Lrs->pqrs",factors,factors))
    h = jnp.diag(jnp.linspace(-1.,1.,n))
    space = make_tensor_space(n,3,max_excitation=6)
    oracle = ci.make_ci_space(n,3,max_excitation=6)
    indices,signs = determinant_map(space,oracle)
    _,_,fs,e0 = _reference_data(h,g,3)
    c = jnp.asarray(rng.normal(size=space.size))
    determinant_c = jnp.zeros_like(c).at[indices].set(c*signs)
    expected = ci.hamiltonian_action(h,g,oracle,determinant_c)[indices]*signs
    expected = expected-(e0+space.gaps(fs))*c
    action = build_tensor_action(g,fs,space)
    actual = jax.jit(lambda vector: action(vector))(c)
    np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=0.)


def test_dense_expansion_capacity_precedes_allocation():
    from gradscf.mp._tensor import make_tensor_space
    with pytest.raises(ValueError, match="tensor"):
        make_tensor_space(19, 5, frozen=1, max_excitation=4, max_determinants=1_000_000,
                          max_tensor_elements=2_000_000)


def test_reference_only_space_builds_no_unused_integral_nodes():
    from gradscf.mp._tensor import make_tensor_space, build_tensor_action
    from gradscf.mp.integrals import _reference_data
    h,g,no = model(False)
    _,_,fs,_ = _reference_data(h,g,no)
    action = build_tensor_action(g,fs,make_tensor_space(4,no,max_excitation=0))
    assert not action.kernels and not action.plans


def test_order_limited_action_preserves_exact_zero_blocks():
    from gradscf.mp._tensor import make_tensor_space, build_tensor_action
    from gradscf.mp.integrals import _reference_data
    h,g,no = model(False)
    _,_,fs,_ = _reference_data(h,g,no)
    space = make_tensor_space(4,no,max_excitation=4)
    action = build_tensor_action(g,fs,space)
    vector = jnp.where(jnp.asarray(space.ranks) <= 2,jnp.arange(space.size,dtype=float)*.03,0.)
    np.testing.assert_allclose(action.apply(vector,max_input_rank=2),action(vector),atol=2e-12)


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 12])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_tensor_mp_uses_same_lifting_and_matches_series(order, unrestricted):
    from gradscf import mp
    from gradscf.mp.integrals import _reference_data
    from gradscf.mp.series import make_series_space
    h, g, no = model(unrestricted)
    _, _, fs, _ = _reference_data(h, g, no)
    hs = h if unrestricted else (h,)
    canonical = tuple(a-(f-jnp.diag(jnp.diag(f))) for a,f in zip(hs,fs))
    h = canonical if unrestricted else canonical[0]
    old_cfg = mp.MPConfig(order=order, algorithm="series", with_coefficients=True)
    new_cfg = mp.MPConfig(order=order, algorithm="tensor", with_coefficients=True)
    old = mp.run_mp(h,g,nocc=no,nuclear_repulsion=.7,config=old_cfg)
    new = mp.run_mp(h,g,nocc=no,nuclear_repulsion=.7,config=new_cfg)
    assert old.valid and new.valid
    np.testing.assert_allclose(new.corrections, old.corrections, atol=2e-11, rtol=0.)
    np.testing.assert_allclose(new.total_energy, old.total_energy, atol=2e-11, rtol=0.)
    indices, signs = determinant_map(make_series_space(4,no,config=new_cfg),
                                     make_series_space(4,no,config=old_cfg))
    np.testing.assert_allclose(new.wavefunction_coefficients,
        old.wavefunction_coefficients[:,indices]*signs[None,:], atol=2e-10, rtol=0.)
    jax.tree.map(lambda a,b: np.testing.assert_allclose(a,b,atol=2e-10), new.t2, old.t2)
    np.testing.assert_allclose(new.same_spin_energy, old.same_spin_energy, atol=2e-11)
    np.testing.assert_allclose(new.opposite_spin_energy, old.opposite_spin_energy, atol=2e-11)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_tensor_jit_jvp_and_hvp_with_moving_gaps(unrestricted):
    from gradscf import mp
    from gradscf.mp.integrals import _reference_data
    h,g,no = model(unrestricted)
    hs,gs,fs,_ = _reference_data(h,g,no)
    hs = tuple(a-(f-jnp.diag(jnp.diag(f))) for a,f in zip(hs,fs))
    _,_,fs,_ = _reference_data(hs if unrestricted else hs[0],g,no)
    interaction = tuple(f-a for a,f in zip(hs,fs))
    direction = jnp.diag(jnp.linspace(-.01,.02,4))
    def objective(x, algorithm):
        hx = tuple(a+(1-x)*v+((x-1)+.3*(x-1)**2)*direction for a,v in zip(hs,interaction))
        gx = tuple(x*t for t in gs)
        result = mp.run_mp(hx if unrestricted else hx[0], gx if unrestricted else gx[0],
            nocc=no,config=mp.MPConfig(order=5,algorithm=algorithm,with_coefficients=True))
        return result.correlation_energy+.001*jnp.sum(result.wavefunction_coefficients[1:]**2)
    old = lambda x: objective(x,"series")
    new = lambda x: objective(x,"tensor")
    value,gradient = jax.jit(jax.value_and_grad(new))(1.)
    np.testing.assert_allclose(value,old(1.),atol=2e-11,rtol=1e-9)
    np.testing.assert_allclose(gradient,jax.grad(old)(1.),atol=2e-9,rtol=1e-7)
    hvp = jax.jvp(jax.grad(new),(1.,),(.2,))[1]
    np.testing.assert_allclose(hvp,jax.jvp(jax.grad(old),(1.,),(.2,))[1],atol=2e-8,rtol=1e-7)


def test_tensor_invalid_response_and_arraylike_inputs():
    from gradscf import mp
    cfg = mp.MPConfig(order=4,algorithm="tensor",with_t2=False)
    def energy(x):
        return mp.run_mp(jnp.array([[0.,x],[x,0.]]),jnp.zeros((2,)*4),nocc=1,config=cfg).correlation_energy
    assert jnp.isnan(energy(0.)) and jnp.isnan(jax.grad(energy)(0.))
    h,g,no = model(False)
    # Array-like input coercion precedes tensor graph construction.
    result = mp.run_mp(h.tolist(),g.tolist(),nocc=no,config=cfg)
    assert not result.valid  # Deliberately noncanonical; graph still constructs.
