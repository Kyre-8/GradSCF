"""Discrete electric enthalpy: independent dimer and complex-density contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq


def dimer(mesh=(4, 1, 1), *, spins=1, core=False, transformed=False):
    from gradscf.integrals.periodic.fft import PeriodicInputs
    from gradscf.pbc.scf import PeriodicSCFResult
    lattice = np.diag([1., 2., 3.])
    sites = np.array([[.15, 0., 0.], [.55, 0., 0.]])
    if core:
        sites = np.array([[.03, .02, .01], [.15, .12, .09], [.55, .27, -.13]])
    h0 = np.array([[.4, .3], [.3, -.4]])
    if core:
        h0 = np.pad(h0, ((1, 0), (1, 0)))
        h0[0, 0] = -2.
    eigenvalues, orbitals = np.linalg.eigh(h0)
    n = len(h0)
    occupied = n-1
    fractional = np.stack(np.meshgrid(*[np.arange(m)/m for m in mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    fractional -= np.floor(fractional+.5)
    kpoints = fractional@(2*np.pi*np.linalg.inv(lattice).T)
    nk = len(kpoints)
    volume = np.linalg.det(lattice)
    weights = np.full(n, volume/n)
    ao = np.zeros((nk, 4, n, n), dtype=complex)
    ao[:, 0] = np.exp(-1j*kpoints@sites.T)[:, :, None]*np.eye(n)[None]/np.sqrt(volume/n)
    coeff = np.broadcast_to(orbitals, (spins, nk, n, n)).astype(complex).copy()
    hcore = np.broadcast_to(h0, (nk, n, n)).astype(complex).copy()
    overlap = np.broadcast_to(np.eye(n), (nk, n, n)).astype(complex).copy()
    if transformed:
        rng = np.random.default_rng(4381)
        transform = np.eye(n)[None]+.08*(rng.normal(size=(nk, n, n))+1j*rng.normal(size=(nk, n, n)))
        ao = ao@transform[:, None]
        overlap = transform.conj().swapaxes(-1, -2)@overlap@transform
        hcore = transform.conj().swapaxes(-1, -2)@hcore@transform
        coeff = np.linalg.solve(transform, coeff)
        gauge = np.linalg.qr(rng.normal(size=(spins, nk, occupied, occupied))+
                             1j*rng.normal(size=(spins, nk, occupied, occupied)))[0]
        coeff[..., :occupied] = coeff[..., :occupied]@gauge
    occupation = np.broadcast_to((np.arange(n)<occupied).astype(float), (spins, nk, n))
    density = (coeff*occupation[..., None, :])@coeff.conj().swapaxes(-1, -2)
    inputs = PeriodicInputs(jnp.asarray(overlap), jnp.asarray(hcore), jnp.zeros_like(jnp.asarray(hcore)),
        jnp.asarray(ao), jnp.asarray(sites), jnp.asarray(weights), None, jnp.array(0.), jnp.array(0.),
        jnp.asarray(kpoints), jnp.zeros((1, 3)), jnp.full(nk, 1/nk), jnp.asarray(lattice),
        jnp.zeros((1, 3)), jnp.array([float(spins*occupied)]), jnp.array([float(spins*occupied)]))
    state = PeriodicSCFResult(jnp.array(spins*sum(eigenvalues[:occupied])), jnp.asarray(density),
        jnp.asarray(coeff), jnp.asarray(np.broadcast_to(eigenvalues, occupation.shape)),
        jnp.asarray(occupation), jnp.asarray(np.broadcast_to(hcore, density.shape)), jnp.array(True), jnp.array(0))
    valence = orbitals[-2:, occupied-1]
    theta = float(np.arctan2(valence[1], valence[0]))

    def polarization_and_slope(angle):
        q = (2*np.pi*np.linalg.inv(lattice).T)/np.asarray(mesh)[:, None]
        exponent = np.exp(-1j*q@sites[-2:].T)
        link = np.cos(angle)**2*exponent[:, 0]+np.sin(angle)**2*exponent[:, 1]
        link_derivative = np.sin(2*angle)*(exponent[:, 1]-exponent[:, 0])
        phase = np.angle(np.exp(1j*np.asarray(mesh)*np.angle(link)))
        slope = np.asarray(mesh)*np.imag(link_derivative/link)
        return phase@lattice/(2*np.pi*volume), slope@lattice/(2*np.pi*volume)
    _, derivative = polarization_and_slope(theta)
    chi = spins*volume*np.outer(derivative, derivative)/(2*(eigenvalues[-1]-eigenvalues[-2]))
    return inputs, state, tuple([occupied]*spins), theta, polarization_and_slope, chi


def energy_and_fock(density, data, params):
    energy = jnp.einsum('k,skij,kji->', data.kweights, density, data.hcore).real
    return energy, jnp.broadcast_to(data.hcore, density.shape)


@pytest.mark.parametrize('mesh,spins', [((4, 1, 1), 1), ((4, 3, 2), 1), ((4, 1, 1), 2)])
def test_dimer_analytic_dielectric_kweight_copies_and_spin(mesh, spins):
    from gradscf.dfpt.electric import _electric_model, _field_problem
    from gradscf.solvers import LinearSolverConfig
    data, state, counts, _, _, expected = dimer(mesh, spins=spins)
    model, _ = _electric_model(data, state, nocc=counts, evaluate=energy_and_fock)
    problem = _field_problem(model)
    config = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=256)
    respond = jax.jit(lambda direction:problem.response(tangent=direction, config=config))
    outputs = [respond(direction) for direction in jnp.eye(3)]
    assert all(bool(row.converged) for row in outputs)
    actual = np.stack([row.values['polarization'] for row in outputs], axis=1)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-10)
    assert expected[0, 0] > 0
    np.testing.assert_allclose(actual, actual.T, atol=2e-13)


def test_complex_noncanonical_metric_and_field_fock_pairing():
    from gradscf.dfpt.electric import _electric_model, _field_problem, _field_term
    from gradscf.pbc.polarization import make_plan, polarization
    from gradscf.solvers import LinearSolverConfig
    data, state, counts, _, _, expected = dimer((3, 2, 2), core=True, transformed=True)
    model, _ = _electric_model(data, state, nocc=counts, evaluate=energy_and_fock)
    problem = _field_problem(model)
    config = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=256)
    respond = jax.jit(lambda d:problem.response(tangent=d, config=config))
    outputs = [respond(d) for d in jnp.eye(3)]
    assert all(bool(row.converged) for row in outputs)
    actual = np.stack([row.values['polarization'] for row in outputs], axis=1)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-10)
    np.testing.assert_allclose(actual, actual.T, atol=2e-13)
    plan = make_plan(np.asarray(data.kpoints)@np.asarray(data.lattice).T/(2*np.pi))
    anchor = polarization(data, state, plan, nocc=counts).phases
    frames = tuple(state.mo_coeff_spin[s, :, :, :n] for s, n in enumerate(counts))
    rng = np.random.default_rng(811)
    random = jnp.asarray(rng.normal(size=state.density_spin.shape)+1j*rng.normal(size=state.density_spin.shape))
    tangent = .5*(random+random.conj().swapaxes(-1, -2))
    off_shell = state.density_spin+1e-3*tangent
    field = jnp.array([.01, -.007, .004])
    function = lambda density:_field_term(density, data, field, frames, plan, anchor)[0]
    value, potential = _field_term(off_shell, data, field, frames, plan, anchor)
    derivative = jax.jvp(function, (off_shell,), (tangent,))[1]
    trace = jnp.einsum('k,skij,skji->', data.kweights, potential, tangent).real
    finite = (function(off_shell+1e-5*tangent)-function(off_shell-1e-5*tangent))/2e-5
    assert np.isfinite(value)
    np.testing.assert_allclose(potential, potential.conj().swapaxes(-1, -2), atol=2e-13)
    np.testing.assert_allclose(derivative, trace, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(derivative, finite, atol=2e-10, rtol=2e-8)


def test_finite_field_shared_root_matches_independent_scalar_oracle_and_ad():
    from gradscf.dfpt.electric import _electric_model, _solve_field
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear.iterate import NonlinearConfig
    data, state, counts, theta, polar, chi = dimer((4, 1, 1))
    model, diagonal = _electric_model(data, state, nocc=counts, evaluate=energy_and_fock)
    config = NonlinearConfig(maxiter=80, residual_tol=1e-11, energy_tol=1e-13, damping=.2)
    linear = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=128)
    solve = jax.jit(lambda field:_solve_field(model, diagonal, field, config=config, linear_config=linear))
    baseline = np.asarray(model.quantities(model.solution, model.inputs)['polarization'])
    base_p, _ = polar(theta)
    np.testing.assert_allclose(baseline, base_p, atol=2e-14)
    volume = float(np.linalg.det(np.asarray(data.lattice)))
    for field in (1e-3, 3e-4, 1e-4):
        target = jnp.array([field, 0., 0.])
        root, values = solve(target)
        assert root.converged and root.residual_norm < 1e-11
        residual = lambda angle:-.8*np.sin(2*angle)+.6*np.cos(2*angle)-volume*field*polar(angle)[1][0]
        optimum = brentq(residual, theta-.1, theta+.1, xtol=1e-14)
        expected = polar(optimum)[0]
        np.testing.assert_allclose(values['polarization'], expected, atol=2e-11, rtol=2e-9)
        np.testing.assert_allclose(values['electron_number'], 1., atol=2e-11)
    observable = lambda field:solve(field)[1]['polarization']
    direction = jnp.array([.4, -.2, .3])
    forward = jax.jit(lambda f:jax.jvp(observable, (f,), (direction,))[1])(jnp.zeros(3))
    backward = jax.jit(jax.grad(lambda f:jnp.dot(observable(f), direction)))(jnp.zeros(3))
    np.testing.assert_allclose(forward, chi@direction, atol=2e-11, rtol=2e-9)
    np.testing.assert_allclose(backward, chi.T@direction, atol=2e-11, rtol=2e-9)


def test_full_root_electric_enthalpy_hessian_matches_analytic_susceptibility():
    """Constrained density curvature must include the implicit second response."""
    from gradscf.dfpt.electric import _electric_model, _solve_field
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear.iterate import NonlinearConfig
    data, state, counts, _, _, chi = dimer((4, 1, 1))
    model, diagonal = _electric_model(data, state, nocc=counts, evaluate=energy_and_fock)
    config = NonlinearConfig(maxiter=80, residual_tol=1e-11, energy_tol=1e-13, damping=.2)
    linear = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=128)

    def enthalpy(field):
        return _solve_field(model, diagonal, field, config=config,
                            linear_config=linear)[1]['electric_enthalpy']

    zero = jnp.zeros(3)
    gradient = jax.jit(jax.grad(enthalpy))(zero)
    forward_reverse = jax.jit(jax.jacfwd(jax.grad(enthalpy)))(zero)
    reverse_reverse = jax.jit(jax.jacrev(jax.grad(enthalpy)))(zero)
    volume = float(np.linalg.det(np.asarray(data.lattice)))
    baseline = model.quantities(model.solution, model.inputs)['polarization']
    np.testing.assert_allclose(gradient, -volume*baseline, atol=2e-12)
    np.testing.assert_allclose(forward_reverse, -volume*chi, atol=2e-11, rtol=2e-9)
    np.testing.assert_allclose(reverse_reverse, -volume*chi, atol=2e-11, rtol=2e-9)
    np.testing.assert_allclose(forward_reverse, reverse_reverse, atol=2e-12)
    print('Full-root enthalpy Hessian', np.asarray(forward_reverse).tolist(),
          'max analytic error', float(np.max(abs(forward_reverse+volume*chi))), flush=True)


def test_periodic_h2_field_response_and_independent_orbital_root():
    """Independent solver/parameterization on shared integrals, not a second code."""
    from scipy.linalg import expm, expm_frechet
    from scipy.optimize import root as scipy_root
    from gradscf.pbc import gto, scf
    from gradscf.pbc._kpoint import energy_and_fock as hf_energy_fock
    from gradscf.dfpt.electric import electric_problem, _electric_model, _solve_field
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear.iterate import NonlinearConfig
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6., unit='Bohr', mesh=(9,)*3)
    mf = scf.KRHF(cell, kpts=cell.make_kpts((3, 1, 1))).run(
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-11, max_cycle=150, damping=.1)
    assert mf.converged
    evaluate = lambda d,a,p:hf_energy_fock(d, a, mesh=cell.mesh, xc='hf', exxdiv='ewald')
    model, diagonal = _electric_model(mf.inputs, mf.result, nocc=cell.nelec, evaluate=evaluate)
    problem = electric_problem(mf)
    linear = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=128)
    respond = jax.jit(lambda e:problem.response(tangent=e, config=linear))
    responses = [respond(e) for e in jnp.eye(3)]
    assert all(bool(result.converged) for result in responses)
    chi = np.stack([result.values['polarization'] for result in responses], axis=1)
    np.testing.assert_allclose(chi, chi.T, atol=2e-9)
    assert np.linalg.eigvalsh(.5*(chi+chi.T)).min() > -1e-10
    config = NonlinearConfig(maxiter=150, residual_tol=1e-11, energy_tol=1e-13, damping=.2)
    solve = jax.jit(lambda e:_solve_field(model, diagonal, e, config=config, linear_config=linear))
    for axis in range(3):
        for step in (3e-4, 1e-4):
            direction = np.eye(3)[axis]*step
            plus, vp = solve(jnp.asarray(direction))
            minus, vm = solve(jnp.asarray(-direction))
            assert plus.converged & minus.converged
            finite = np.asarray(vp['polarization']-vm['polarization'])/(2*step)
            np.testing.assert_allclose(finite, chi[:, axis], atol=2e-7, rtol=2e-5)

    # A separate NumPy implementation uses full Bloch AOs and exp(-i*q.r).
    data = mf.inputs
    lattice, kpoints, coordinates = map(np.asarray, (data.lattice, data.kpoints, data.coords))
    volume = float(np.linalg.det(lattice))
    weights = np.asarray(data.kweights)
    psi = np.asarray(data.ao[:, 0])*np.exp(1j*kpoints@coordinates.T)[:, :, None]
    fractional = kpoints@lattice.T/(2*np.pi)
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    links, partners, strings = [], [], []
    for axis, count in enumerate((3, 1, 1)):
        q = np.eye(3)[axis]/count
        difference = fractional[None]-fractional[:, None]-q
        difference -= np.rint(difference)
        matches = np.max(abs(difference), axis=-1) < 1e-10
        assert np.all(matches.sum(axis=1) == 1)
        partner = matches.argmax(axis=1)
        phase = np.exp(-1j*coordinates@(q@reciprocal))
        links.append(np.einsum('g,kgp,g,kgq->kpq', np.asarray(data.weights), psi.conj(), phase, psi[partner]))
        partners.append(partner)
        unused, cycles = set(range(len(kpoints))), []
        while unused:
            first = min(unused)
            cycle, current = [], first
            while current in unused:
                cycle.append(current)
                unused.remove(current)
                current = int(partner[current])
            assert current == first and len(cycle) == count
            cycles.append(cycle)
        strings.append(np.asarray(cycles))
    reference = np.asarray(mf.result.mo_coeff_spin[0])
    e0_f0 = jax.jit(lambda d:evaluate(d, data, None))
    field = np.array([3e-4, -2e-4, 1e-4])
    anchor = None

    def oracle(angles):
        nonlocal anchor
        coeff, derivatives = [], []
        for k, (real, imaginary) in enumerate(np.asarray(angles).reshape(-1, 2)):
            generator = np.array([[0., -real+1j*imaginary], [real+1j*imaginary, 0.]])
            coeff.append((reference[k]@expm(generator))[:, 0])
            derivatives.append([(reference[k]@expm_frechet(generator, direction, compute_expm=False))[:, 0]
                for direction in (np.array([[0., -1.], [1., 0.]]), np.array([[0., 1j], [1j, 0.]]))])
        coeff = np.asarray(coeff)
        density = coeff[:, :, None]*coeff[:, None, :].conj()
        energy, fock = e0_f0(jnp.asarray(np.stack([density, density])))
        energy, fock = float(energy), np.asarray(fock)
        matrices = [np.einsum('kp,kpq,kq->k', coeff.conj(), b, coeff[p]) for b, p in zip(links, partners)]
        phases = [np.angle(np.prod(m[cycle], axis=1)) for m, cycle in zip(matrices, strings)]
        if anchor is None:
            anchor = [value.copy() for value in phases]
        phases = [a+np.angle(np.exp(1j*(p-a))) for p, a in zip(phases, anchor)]
        polarization_value = 2*np.array([p.mean() for p in phases])@lattice/(2*np.pi*volume)
        polarization_value += np.asarray(data.ionic_charges)@np.asarray(data.atom_coords)/volume
        gradient = []
        for k in range(len(kpoints)):
            for derivative in derivatives[k]:
                dc = np.zeros_like(coeff)
                dc[k] = derivative
                phases_derivative = []
                for b, p, m, cycle in zip(links, partners, matrices, strings):
                    dm = np.einsum('kp,kpq,kq->k', dc.conj(), b, coeff[p])
                    dm += np.einsum('kp,kpq,kq->k', coeff.conj(), b, dc[p])
                    phases_derivative.append(np.imag(dm/m)[cycle].sum(axis=1).mean())
                dp = 2*np.asarray(phases_derivative)@lattice/(2*np.pi*volume)
                gradient.append(2*weights[k]*np.vdot(derivative, fock[:, k].sum(axis=0)@coeff[k]).real-volume*field@dp)
        return energy-volume*field@polarization_value, np.asarray(gradient), polarization_value

    zero = np.zeros(2*len(kpoints))
    oracle(zero)
    check = np.linspace(-1e-3, 1e-3, len(zero))
    vector = np.linspace(-.7, .5, len(zero))
    finite = (oracle(check+1e-5*vector)[0]-oracle(check-1e-5*vector)[0])/2e-5
    np.testing.assert_allclose(oracle(check)[1]@vector, finite, atol=2e-9, rtol=2e-6)
    independent = scipy_root(lambda x:oracle(x)[1], zero, method='hybr', tol=1e-10)
    actual_residual = np.max(abs(oracle(independent.x)[1]))
    assert actual_residual < 2e-9
    solved, observed = solve(jnp.asarray(field))
    assert solved.converged
    expected = oracle(independent.x)
    np.testing.assert_allclose(observed['polarization'], expected[2], atol=2e-9, rtol=2e-7)
    np.testing.assert_allclose(observed['energy'], expected[0], atol=2e-11)
    print('H2 chi', chi.tolist(), 'independent orbital residual', actual_residual,
          'field polarization max error', float(np.max(abs(np.asarray(observed['polarization'])-expected[2]))))
