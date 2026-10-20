"""Order-driven MP from a general residual; independent full-space RS oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module", params=[("H 0 0 0; H 0 0 .74", 1),
    ("Li 0 0 0; H 0 0 1.6", 2),
    ("H 0 0 0; H 0 0 .8; H 0 0 1.7; H 0 0 2.6; H 0 0 3.5; H 0 0 4.4", 3)])
def system(request):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, ao2mo
    atom, no = request.param
    mol = gto.M(atom=atom, basis="sto-3g", cart=True, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11, max_cycle=200)
    assert mf.converged
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    g = ao2mo.kernel(mol, c, compact=False).reshape((c.shape[1],)*4)
    return mf, jnp.asarray(h), jnp.asarray(g), no


def reference_series(h, g, eps, no, order, frozen=()):
    """Full PySCF FCI actions + textbook RS recurrence, independent of jet/CI."""
    from pyscf import fci
    unrestricted = isinstance(no, tuple)
    na, nb = no if unrestricted else (no, no)
    ea, eb = eps if unrestricted else (eps, eps)
    n = len(ea)
    aocc = fci.cistring.gen_occslst(range(n), na)
    bocc = fci.cistring.gen_occslst(range(n), nb)
    shape = (len(aocc), len(bocc))
    phi = np.zeros(shape)
    phi[0, 0] = 1.
    engine = fci.direct_uhf if unrestricted else fci.direct_spin1
    tensor = engine.absorb_h1e(h, g, n, (na, nb), .5)
    action = lambda c: np.asarray(engine.contract_2e(tensor, c, n, (na, nb)))
    e0 = action(phi)[0, 0]
    gap = (np.sum(ea[aocc], axis=1)[:, None]+np.sum(eb[bocc], axis=1)[None, :]
           -np.sum(ea[:na])-np.sum(eb[:nb]))
    h0 = e0+gap
    separate = len(frozen) == 2 and all(isinstance(x, (tuple, list)) for x in frozen)
    fa, fb = frozen if separate else (frozen, frozen)
    amask = np.array([all((i in occ) == (i < na) for i in fa) for occ in aocc])
    bmask = np.array([all((i in occ) == (i < nb) for i in fb) for occ in bocc])
    mask = amask[:, None] & bmask[None, :]
    mask[0, 0] = False
    waves, energies = [phi], [e0]
    for k in range(1, order+1):
        source = action(waves[-1])-h0*waves[-1]
        energies.append(source[0, 0])
        for j in range(1, k):
            source = source-energies[j]*waves[k-j]
        waves.append(np.where(mask, -source/np.where(mask, gap, 1.), 0.))
    lookup = {sum(1 << p for p in a)+(sum(1 << p for p in b) << n): (i, j)
              for i, a in enumerate(aocc) for j, b in enumerate(bocc)}
    return np.asarray(energies), waves, lookup


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 12])
def test_all_orders_match_independent_full_space(system, order):
    from gradscf.mp import MPConfig, run_mp
    mf, h, g, no = system
    expected, _, _ = reference_series(np.asarray(h), np.asarray(g), mf.mo_energy, no, order)
    result = run_mp(h, g, nocc=no, config=MPConfig(order=order, algorithm="series"))
    assert result.valid
    np.testing.assert_allclose(result.corrections, expected[2:], atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(result.correlation_energy, sum(expected[2:]), atol=2e-10)
    assert result.e2 == result.corrections[0]


def test_low_orders_preserve_amplitudes_and_spin_components(system):
    from gradscf.mp import MPConfig, run_mp
    _, h, g, no = system
    direct = run_mp(h, g, nocc=no)
    series = run_mp(h, g, nocc=no, config=MPConfig(algorithm="series"))
    np.testing.assert_allclose(series.t2, direct.t2, atol=1e-10)
    np.testing.assert_allclose(series.same_spin_energy, direct.same_spin_energy, atol=1e-10)
    np.testing.assert_allclose(series.opposite_spin_energy, direct.opposite_spin_energy, atol=1e-10)


def test_linear_hamiltonian_nodes_are_shared_outside_taylor(system, monkeypatch):
    from gradscf.mp import MPConfig, run_mp
    from gradscf.ci import hamiltonian
    _, h, g, no = system
    build = hamiltonian.build_direct_hamiltonian
    calls = []
    def counted(*args, **kwargs):
        op = build(*args, **kwargs)
        class Action:
            diagonal = op.diagonal
            def apply(self, c):
                assert type(c).__name__ != "JetTracer", "Share linear actions before Taylor propagation"
                calls.append(c)
                return op.apply(c)
        return Action()
    monkeypatch.setattr(hamiltonian, "build_direct_hamiltonian", counted)
    result = run_mp(h, g, nocc=no, config=MPConfig(order=6, algorithm="series"))
    assert result.valid
    # W Phi0 and W Psi1..3, plus two distinct E2 spin-channel nodes.
    assert len(calls) == 3+1+2


def test_jitted_stored_amplitudes(system):
    from gradscf.mp import MPConfig, run_mp
    _, h, g, no = system
    cfg = MPConfig(order=3, algorithm="series", with_coefficients=True)
    result = jax.jit(lambda h, g: run_mp(h, g, nocc=no, config=cfg))(h, g)
    assert result.valid and result.wavefunction_coefficients.shape[0] == 2
    np.testing.assert_allclose(result.t2, run_mp(h, g, nocc=no).t2, atol=1e-10)


def test_wavefunctions_and_rank_completeness(system):
    from gradscf.mp import MPConfig, run_mp
    from gradscf.mp.series import make_series_space
    mf, h, g, no = system
    config = MPConfig(order=5, algorithm="series", with_coefficients=True)
    space = make_series_space(h.shape[0], no, config=config)
    result = run_mp(h, g, nocc=no, config=config)
    _, waves, lookup = reference_series(np.asarray(h), np.asarray(g), mf.mo_energy, no, 5)
    expected = np.array([[waves[k][lookup[d]] for d in space.determinants] for k in range(3)])
    np.testing.assert_allclose(result.wavefunction_coefficients, expected, atol=2e-9)
    np.testing.assert_allclose(result.wavefunction_coefficients[:, 0], [1., 0., 0.], atol=1e-14)
    assert max(space.ranks) <= 4


def test_high_order_frozen_core_and_virtual(system):
    from gradscf.mp import MPConfig, run_mp
    mf, h, g, no = system
    frozen = (0, h.shape[0]-1)
    expected, _, _ = reference_series(np.asarray(h), np.asarray(g), mf.mo_energy, no, 5, frozen)
    result = run_mp(h, g, nocc=no, frozen=frozen, config=MPConfig(order=5))
    assert result.valid
    np.testing.assert_allclose(result.corrections, expected[2:], atol=2e-10)


def test_jit_gradient_and_hvp(system):
    from gradscf.mp import MPConfig, run_mp
    _, h, g, no = system
    interaction = 2*jnp.einsum("pqii->pq", g[:, :, :no, :no])-jnp.einsum("piiq->pq", g[:, :no, :no, :])
    cfg = MPConfig(order=4, algorithm="series", with_t2=False)
    def energy(x):
        return run_mp(h+(1-x)*interaction, x*g, nocc=no, config=cfg).correlation_energy
    result = run_mp(h, g, nocc=no, config=cfg)
    value, gradient = jax.jit(jax.value_and_grad(energy))(1.)
    powers = jnp.arange(2, 5)
    np.testing.assert_allclose(value, result.correlation_energy, atol=1e-11)
    np.testing.assert_allclose(gradient, jnp.sum(powers*result.corrections), atol=2e-9)
    hvp = jax.jvp(jax.grad(energy), (1.,), (.2,))[1]
    np.testing.assert_allclose(hvp, .2*jnp.sum(powers*(powers-1)*result.corrections), atol=2e-8)


def test_cached_nodes_retain_integral_and_gap_response(system):
    from gradscf.mp import MPConfig, run_mp
    _, h, g, no = system
    interaction = 2*jnp.einsum("pqii->pq", g[:, :, :no, :no])-jnp.einsum("piiq->pq", g[:, :no, :no, :])
    direction = jnp.diag(jnp.linspace(-.02, .03, h.shape[0]))
    cfg = MPConfig(order=6, algorithm="series", with_coefficients=True)
    def objective(x):
        # The physical Fock remains diagonal, but its gaps and W both vary.
        hx = h+(1-x)*interaction+((x-1)+.5*(x-1)**2)*direction
        result = run_mp(hx, x*g, nocc=no, config=cfg)
        return (result.correlation_energy+.01*jnp.sum(result.t2**2)
                +.001*jnp.sum(result.wavefunction_coefficients[1:]**2))
    evaluate = jax.jit(jax.value_and_grad(objective))
    value, gradient = evaluate(1.)
    step = 1e-4
    plus, grad_plus = evaluate(1+step)
    minus, grad_minus = evaluate(1-step)
    assert jnp.isfinite(value)
    np.testing.assert_allclose(gradient, (plus-minus)/(2*step), atol=3e-8, rtol=1e-6)
    hvp = jax.jvp(jax.grad(objective), (1.,), (.2,))[1]
    np.testing.assert_allclose(hvp, .2*(grad_plus-grad_minus)/(2*step), atol=3e-8, rtol=1e-6)


def test_generic_facade_and_arbitrary_order(system):
    from gradscf import mp
    from gradscf.scf.reference import RestrictedReference
    mf, h, g, no = system
    pt = mp.MP(RestrictedReference(h, g, no, mf.mol.energy_nuc()), order=6).run()
    assert pt.converged and len(pt.corrections) == 5
    np.testing.assert_allclose(pt.e_tot, mf.e_tot+jnp.sum(pt.corrections), atol=1e-10)


def test_capacity_precedes_hamiltonian_construction(monkeypatch):
    from gradscf import mp
    from gradscf.ci import hamiltonian
    def forbidden(*args, **kwargs):
        pytest.fail("Do not construct Hamiltonian connections after capacity failure")
    monkeypatch.setattr(hamiltonian, "build_hamiltonian", forbidden)
    with pytest.raises(ValueError, match="determinants"):
        mp.run_mp(jnp.eye(10), jnp.zeros((10,)*4), nocc=5,
                  config=mp.MPConfig(order=6, max_determinants=10))


def test_wavefunction_order_can_exceed_minimum(system):
    from gradscf.mp import MPConfig, run_mp
    mf, h, g, no = system
    cfg = MPConfig(order=3, wavefunction_order=2, with_coefficients=True)
    result = run_mp(h, g, nocc=no, config=cfg)
    assert result.valid and result.wavefunction_coefficients.shape[0] == 3
    expected, _, _ = reference_series(np.asarray(h), np.asarray(g), mf.mo_energy, no, 3)
    np.testing.assert_allclose(result.corrections, expected[2:], atol=1e-10)


def test_unresolved_reference_and_invalid_config():
    from gradscf import mp
    result = mp.run_mp(jnp.zeros((2, 2)), jnp.zeros((2,)*4), nocc=1,
                      config=mp.MPConfig(order=4))
    assert not result.valid and jnp.isnan(result.correlation_energy)
    for kwargs in ({"order":1}, {"order":2.5}, {"algorithm":"unknown"},
                   {"order":6, "wavefunction_order":1}):
        with pytest.raises(ValueError):
            mp.MPConfig(**kwargs)


def test_invalid_series_response_is_nan_not_zero():
    from gradscf import mp
    cfg = mp.MPConfig(order=4, with_t2=False)
    def energy(x):
        h = jnp.array([[0., x], [x, 0.]])
        return mp.run_mp(h, jnp.zeros((2,)*4), nocc=1, config=cfg).correlation_energy
    assert jnp.isnan(energy(0.))
    assert jnp.isnan(jax.grad(energy)(0.))
    assert jnp.isnan(jax.jvp(energy, (0.,), (1.,))[1])


@pytest.mark.parametrize("constant", [jnp.array([0., 1.]), 1.+1.j])
def test_series_requires_real_scalar_constant(constant):
    from gradscf import mp
    with pytest.raises(ValueError, match="real scalar"):
        mp.run_mp(jnp.diag(jnp.array([-1., .5])), jnp.zeros((2,)*4), nocc=1,
                  nuclear_repulsion=constant, config=mp.MPConfig(order=4))


@pytest.fixture(scope="module")
def lithium():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, ao2mo
    mf = scf.UHF(gto.M(atom="Li 0 0 0", basis="6-31g", spin=1, verbose=0)).run(
        conv_tol=1e-14, conv_tol_grad=1e-11, max_cycle=200)
    assert mf.converged
    ca, cb = mf.mo_coeff
    h = tuple(jnp.asarray(c.T @ mf.get_hcore() @ c) for c in (ca, cb))
    g = tuple(jnp.asarray(ao2mo.general(mf.mol, cs, compact=False).reshape((ca.shape[1],)*4))
              for cs in ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)))
    return mf, h, g


@pytest.mark.parametrize("order", [2, 3, 4, 5, 12])
def test_unrestricted_orders(lithium, order):
    from gradscf import mp
    mf, h, g = lithium
    expected, _, _ = reference_series(tuple(map(np.asarray, h)), tuple(map(np.asarray, g)),
                                       mf.mo_energy, (2, 1), order)
    result = mp.run_mp(h, g, nocc=(2, 1), config=mp.MPConfig(order=order, algorithm="series"))
    assert result.valid
    np.testing.assert_allclose(result.corrections, expected[2:], atol=2e-10)
    if order == 2:
        direct = mp.run_mp(h, g, nocc=(2, 1))
        for actual, ref in zip(result.t2, direct.t2):
            np.testing.assert_allclose(actual, ref, atol=1e-10)
        np.testing.assert_allclose(result.same_spin_energy, direct.same_spin_energy, atol=1e-10)
        np.testing.assert_allclose(result.opposite_spin_energy, direct.opposite_spin_energy, atol=1e-10)


def test_auto_unrestricted_third_order_and_arraylike_reference(lithium):
    from gradscf import mp
    from gradscf.scf.reference import UnrestrictedReference
    _, h, g = lithium
    automatic = mp.run_mp(h, g, nocc=(2, 1), config=mp.MPConfig(order=3))
    explicit = mp.run_mp(h, g, nocc=(2, 1), config=mp.MPConfig(order=3, algorithm="series"))
    np.testing.assert_allclose(automatic.corrections, explicit.corrections, atol=1e-12)
    pt = mp.MP3(UnrestrictedReference(tuple(x.tolist() for x in h),
                                     tuple(x.tolist() for x in g), (2, 1))).run()
    assert pt.converged and pt.order == 3
    np.testing.assert_allclose(pt.corrections, explicit.corrections, atol=1e-12)


def test_empty_excitation_space_and_scalar_constant(system):
    from gradscf import mp
    _, h, g, no = system
    cfg = mp.MPConfig(order=6)
    result = mp.run_mp(h, g, nocc=no, frozen=no, config=cfg)
    assert result.valid
    np.testing.assert_allclose(result.corrections, 0., atol=1e-13)
    derivative = jax.grad(lambda x: mp.run_mp(h, g, nocc=no, frozen=no,
        nuclear_repulsion=x, config=cfg).total_energy)(.7)
    np.testing.assert_allclose(derivative, 1., atol=1e-12)


def test_water_public_example_matches_independent_pyscf(monkeypatch):
    import runpy
    from pathlib import Path
    pytest.importorskip("pyscf")
    monkeypatch.syspath_prepend(str(Path("examples/mp").resolve()))
    data = runpy.run_path("examples/mp/compare_water_pyscf.py")
    assert len(data["actual"]) == 7  # Individual E2 through E8.
    np.testing.assert_allclose(data["actual"], data["expected"], atol=2e-10, rtol=0.)


def test_shared_pyscf_reference_preserves_restricted_coefficients(system):
    import runpy
    from pyscf import mp
    mf, h, g, no = system
    reference = runpy.run_path("examples/mp/_pyscf_reference.py")["mp_coefficients"]
    coefficients = reference(mf, 4)
    expected, _, _ = reference_series(np.asarray(h), np.asarray(g), mf.mo_energy, no, 4)
    np.testing.assert_allclose(coefficients, expected[2:], atol=2e-11, rtol=0.)
    np.testing.assert_allclose(coefficients[0], mp.MP2(mf).run().e_corr, atol=2e-11, rtol=0.)


def test_shared_pyscf_reference_preserves_unrestricted_coefficients(lithium):
    import runpy
    from pyscf import mp
    mf, h, g = lithium
    reference = runpy.run_path("examples/mp/_pyscf_reference.py")["mp_coefficients"]
    coefficients = reference(mf, 4, frozen=1)
    expected, _, _ = reference_series(tuple(map(np.asarray, h)), tuple(map(np.asarray, g)),
        mf.mo_energy, (2, 1), 4, frozen=(0,))
    np.testing.assert_allclose(coefficients, expected[2:], atol=2e-11, rtol=0.)
    np.testing.assert_allclose(coefficients[0], mp.MP2(mf, frozen=1).run().e_corr, atol=2e-11, rtol=0.)
