"""Discrete Berry links: phase conventions, local response and AO quadrature."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def localized_model(positions, *, mesh=(3, 2, 1), nocc=1, permutation=None):
    """Exactly localized orthonormal grid orbitals, not a molecular AO model."""
    from gradscf.pbc.polarization import make_plan
    lattice = jnp.eye(3)*5.
    fractional = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    fractional -= np.floor(fractional+.5)
    if permutation is not None:
        fractional = fractional[permutation]
    plan = make_plan(fractional)
    k = jnp.asarray(fractional)@(2*jnp.pi*jnp.linalg.inv(lattice).T)
    positions = jnp.asarray(positions)
    n = len(positions)
    weight = jnp.linalg.det(lattice)/n
    values = jnp.exp(-1j*k@positions.T)[:, :, None]*jnp.eye(n)[None]/jnp.sqrt(weight)
    ao = jnp.zeros((len(k), 4, n, n), dtype=complex).at[:, 0].set(values)
    inputs = SimpleNamespace(ao=ao, weights=jnp.full(n, weight), coords=positions, lattice=lattice,
        kpoints=k, kweights=jnp.full(len(k), 1/len(k)), overlap=jnp.broadcast_to(jnp.eye(n), (len(k), n, n)),
        atom_coords=jnp.zeros((1, 3)), ionic_charges=jnp.array([float(nocc)]))
    occupation = jnp.broadcast_to((jnp.arange(n)<nocc).astype(float), (1, len(k), n))
    energies = jnp.broadcast_to(jnp.where(jnp.arange(n)<nocc, -1., 2.), occupation.shape)
    coefficients = jnp.broadcast_to(jnp.eye(n, dtype=complex), (1, len(k), n, n))
    density = coefficients*occupation[..., None, :]
    state = SimpleNamespace(mo_coeff_spin=coefficients, mo_occ_spin=occupation, mo_energy_spin=energies,
        density_spin=density, fock_spin=coefficients*energies[..., None, :], converged=jnp.array(True))
    return inputs, state, plan


@pytest.mark.parametrize('mesh', [(3, 2, 1), (1, 1, 1)])
def test_localized_electron_sign_terminal_link_and_k_permutation(mesh):
    from gradscf.pbc.polarization import ao_links, polarization
    positions = jnp.array([[.7, .9, 1.1], [1.3, 1.4, .4]])
    inputs, state, plan = localized_model(positions, mesh=mesh)
    result = polarization(inputs, state, plan, nocc=(1,))
    np.testing.assert_allclose(result.electronic, -positions[0]/125, atol=2e-15)
    np.testing.assert_allclose(result.total, result.electronic, atol=0)
    assert bool(result.valid)
    np.testing.assert_allclose(result.minimum_link_singular_value, 1., atol=2e-15)
    links = ao_links(inputs.ao, inputs.weights, inputs.coords, inputs.lattice, plan)
    # Direct independently known diagonal matrix, including the last-to-first link.
    for axis, count in enumerate(mesh):
        expected = np.diag(np.exp(-2j*np.pi*np.asarray(positions[:, axis])/5/count))
        np.testing.assert_allclose(links[axis], np.broadcast_to(expected, links[axis].shape), atol=2e-15)
    order = np.arange(np.prod(mesh))[::-1]
    p_inputs, p_state, p_plan = localized_model(positions, mesh=mesh, permutation=order)
    np.testing.assert_allclose(polarization(p_inputs, p_state, p_plan, nocc=(1,)).total, result.total, atol=2e-15)


@pytest.mark.parametrize('degenerate', [True, False])
def test_occupied_arbitrary_unitary_gauge(degenerate):
    from gradscf.pbc.polarization import polarization
    inputs, state, plan = localized_model([[.7, .9, 1.1], [.2, .3, .4], [1.3, 1.4, .4]], nocc=2)
    if not degenerate:
        state.mo_energy_spin = state.mo_energy_spin.at[..., 0].set(-2.)
        state.fock_spin = state.mo_coeff_spin*state.mo_energy_spin[..., None, :]
    original = polarization(inputs, state, plan, nocc=(2,))
    rng = np.random.default_rng(941)
    gauge = np.linalg.qr(rng.normal(size=(len(inputs.kpoints), 2, 2))+1j*rng.normal(size=(len(inputs.kpoints), 2, 2)))[0]
    state.mo_coeff_spin = state.mo_coeff_spin.at[0, :, :2, :2].set(gauge)
    changed = polarization(inputs, state, plan, nocc=(2,))
    np.testing.assert_allclose(changed.total, original.total, atol=3e-15)
    np.testing.assert_allclose(changed.minimum_link_singular_value, original.minimum_link_singular_value, atol=2e-15)


def test_local_branch_crossing_and_jvp_vjp():
    from gradscf.pbc.polarization import polarization
    positions = jnp.array([[2.45, .9, 1.1], [1.3, 1.4, .4]])
    inputs, state, plan = localized_model(positions)
    base = polarization(inputs, state, plan, nocc=(1,))
    def observable(displacement):
        varied, result, mapping = localized_model(positions.at[0, 0].add(displacement))
        return polarization(varied, result, mapping, nocc=(1,), reference_phases=base.phases).total
    difference = observable(.1)-observable(0.)
    np.testing.assert_allclose(difference, [-.1/125, 0., 0.], atol=3e-15)
    forward = jax.jit(lambda x: jax.jvp(observable, (x,), (jnp.array(1.),))[1])(jnp.array(.1))
    backward = jax.jit(jax.grad(lambda x: jnp.dot(observable(x), jnp.array([.3, -.2, .4]))))(jnp.array(.1))
    np.testing.assert_allclose(forward, [-1/125, 0., 0.], atol=3e-14)
    np.testing.assert_allclose(backward, -.3/125, atol=3e-14)
    finite = (observable(.10001)-observable(.09999))/2e-5
    np.testing.assert_allclose(forward, finite, atol=3e-12)


def test_reciprocal_representatives_and_resolved_polarization_quantum():
    from gradscf.pbc.polarization import make_plan, polarization
    positions = jnp.array([[.7, .9, 1.1], [1.3, 1.4, .4]])
    inputs, state, plan = localized_model(positions)
    original = polarization(inputs, state, plan, nocc=(1,))
    integers = np.random.default_rng(319).integers(-2, 3, size=(len(plan.kpoints), 3))
    reciprocal = 2*jnp.pi*jnp.linalg.inv(inputs.lattice).T
    shifts = jnp.asarray(integers)@reciprocal
    inputs.kpoints += shifts
    inputs.ao *= jnp.exp(-1j*shifts@inputs.coords.T)[:, None, :, None]
    changed_plan = make_plan(plan.kpoints+integers)
    changed = polarization(inputs, state, changed_plan, nocc=(1,))
    np.testing.assert_allclose(changed.total, original.total, atol=3e-15)
    # A whole-cell transport needs resolved intermediate branches, not a
    # guessed winding number from its identical periodic endpoint alone.
    anchor = original.phases
    for displacement in np.linspace(.5, 5., 10):
        shifted, current, mapping = localized_model(positions.at[0, 0].add(displacement))
        tracked = polarization(shifted, current, mapping, nocc=(1,), reference_phases=anchor)
        anchor = tracked.phases
    np.testing.assert_allclose(tracked.total-original.total, [-5/125, 0., 0.], atol=3e-15)


@pytest.mark.parametrize('invalid', ['missing', 'duplicate', 'weights', 'uneven'])
def test_invalid_mesh_rejected(invalid):
    from gradscf.pbc.polarization import make_plan
    k = np.array([[0., 0., 0.], [1/3, 0., 0.], [-1/3, 0., 0.]])
    if invalid == 'missing':
        k = k[:-1]
    elif invalid == 'duplicate':
        k[-1] = k[0]
    elif invalid == 'uneven':
        k[1, 0] = .2
    weights = np.array([.2, .4, .4]) if invalid == 'weights' else None
    with pytest.raises(ValueError):
        make_plan(k, kweights=weights)


@pytest.mark.parametrize('invalid', ['fractional', 'gap', 'unconverged', 'metric', 'singular', 'mesh_mismatch'])
def test_invalid_reference_or_link_rejected(invalid):
    from gradscf.pbc.polarization import polarization
    inputs, state, plan = localized_model([[.7, .9, 1.1], [1.3, 1.4, .4]])
    if invalid == 'fractional':
        state.mo_occ_spin = state.mo_occ_spin*.9
    elif invalid == 'gap':
        state.mo_energy_spin = state.mo_energy_spin.at[..., 1].set(-1.)
    elif invalid == 'unconverged':
        state.converged = False
    elif invalid == 'metric':
        state.mo_coeff_spin = state.mo_coeff_spin*1.01
    elif invalid == 'mesh_mismatch':
        inputs.kpoints = inputs.kpoints+.01
    else:
        state.mo_coeff_spin = state.mo_coeff_spin.at[0, 1].set(jnp.array([[0., 1.], [1., 0.]]))
        c = state.mo_coeff_spin
        state.density_spin = (c*state.mo_occ_spin[..., None, :])@c.conj().swapaxes(-1, -2)
        state.fock_spin = (c*state.mo_energy_spin[..., None, :])@c.conj().swapaxes(-1, -2)
    with pytest.raises(ValueError):
        polarization(inputs, state, plan, nocc=(1,))


def test_projector_frame_nonorthogonal_metric_and_response():
    from gradscf.pbc.polarization import occupied_frame
    s = jnp.array([[1.2, .1j, .08], [-.1j, 1.5, .02], [.08, .02, .9]])
    basis = jnp.linalg.inv(jnp.linalg.cholesky(s)).conj().T
    reference = basis[:, :2]
    def matrices(t):
        rotation = jnp.array([[jnp.cos(t), 0., -jnp.sin(t)], [0., 1., 0.], [jnp.sin(t), 0., jnp.cos(t)]])
        c = (basis@rotation)[:, :2]
        return c@c.conj().T
    def restored(t):
        result = occupied_frame(matrices(t), s, reference)
        return result.coefficients@result.coefficients.conj().T
    result = occupied_frame(matrices(.2), s, reference)
    assert result.valid
    np.testing.assert_allclose(result.coefficients.conj().T@s@result.coefficients, jnp.eye(2), atol=2e-14)
    np.testing.assert_allclose(restored(.2), matrices(.2), atol=2e-14)
    actual = jax.jit(lambda t:jax.jvp(restored, (t,), (jnp.array(1.),))[1])(jnp.array(.2))
    expected = jax.jvp(matrices, (jnp.array(.2),), (jnp.array(1.),))[1]
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    with pytest.raises(ValueError):
        occupied_frame(matrices(jnp.pi/2), s, reference)
    with pytest.raises(ValueError):
        occupied_frame(.9*matrices(.2), s, reference)


def test_frame_rejects_indefinite_metric_outside_occupied_space():
    from gradscf.pbc.polarization import occupied_frame
    with pytest.raises(ValueError, match='metric'):
        occupied_frame(jnp.diag(jnp.array([1., 0.])), jnp.diag(jnp.array([1., -1.])), jnp.array([[1.], [0.]]))


def test_reference_phases_must_be_real():
    from gradscf.pbc.polarization import polarization
    inputs, state, plan = localized_model([[.7, .9, 1.1], [1.3, 1.4, .4]])
    base = polarization(inputs, state, plan, nocc=(1,))
    phases = tuple(p+1e-3j for p in base.phases)
    with pytest.raises(ValueError, match='real'):
        polarization(inputs, state, plan, nocc=(1,), reference_phases=phases)


def test_principal_cut_has_forward_value_but_requires_local_branch_for_ad():
    from gradscf.pbc.polarization import polarization
    positions = jnp.array([[2.5, .9, 1.1], [1.3, 1.4, .4]])
    inputs, state, plan = localized_model(positions)
    base = polarization(inputs, state, plan, nocc=(1,))
    assert np.isfinite(base.total).all()
    def observable(displacement, anchor):
        varied, result, mapping = localized_model(positions.at[0, 0].add(displacement))
        return polarization(varied, result, mapping, nocc=(1,), reference_phases=anchor).total
    unanchored = lambda t: observable(t, None)
    tangent = jax.jit(lambda t:jax.jvp(unanchored, (t,), (jnp.array(1.),))[1])(jnp.array(0.))
    cotangent = jax.jit(jax.grad(lambda t:unanchored(t)[0]))(jnp.array(0.))
    assert np.isnan(tangent).all() and np.isnan(cotangent)
    anchored = lambda t: observable(t, base.phases)
    forward = jax.jit(lambda t:jax.jvp(anchored, (t,), (jnp.array(1.),))[1])(jnp.array(0.))
    backward = jax.jit(jax.grad(lambda t:anchored(t)[0]))(jnp.array(0.))
    np.testing.assert_allclose(forward, [-1/125, 0., 0.], atol=3e-14)
    np.testing.assert_allclose(backward, -1/125, atol=3e-14)


def test_invalid_traced_reference_masks_value_jvp_and_vjp():
    from gradscf.pbc.polarization import polarization
    positions = jnp.array([[.7, .9, 1.1], [1.3, 1.4, .4]])
    def observable(t):
        inputs, state, plan = localized_model(positions.at[0, 0].add(t))
        state.mo_occ_spin = state.mo_occ_spin*.9
        return polarization(inputs, state, plan, nocc=(1,)).total
    value, tangent = jax.jit(lambda t:jax.jvp(observable, (t,), (jnp.array(1.),)))(jnp.array(0.))
    cotangent = jax.jit(jax.grad(lambda t:observable(t)[0]))(jnp.array(0.))
    assert np.isnan(value).all() and np.isnan(tangent).all() and np.isnan(cotangent)


def test_independent_pyscf_ao_links_on_identical_quadrature():
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto
    from pyscf.pbc.dft.numint import eval_ao_kpts
    from gradscf.pbc import gto
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.pbc.polarization import make_plan, ao_links
    errors = []
    for fft_size in (31, 61):
        cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6., unit='Bohr',
                     basis='gth-szv', pseudo='gth-pade', mesh=(fft_size,)*3)
        k = cell.make_kpts((3, 1, 1), shift=(.5, 0., 0.))
        data = build_inputs(cell, kpts=k)
        plan = make_plan(np.asarray(k)@np.asarray(cell.lattice).T/(2*np.pi))
        actual = np.asarray(ao_links(data.ao, data.weights, data.coords, data.lattice, plan))
        reference = pgto.M(atom=cell.atom, a=np.asarray(cell.lattice), unit='Bohr',
                          basis='gth-szv', pseudo='gth-pade', cart=True, precision=1e-12, verbose=0)
        psi = np.asarray(eval_ao_kpts(reference, np.asarray(data.coords), kpts=np.asarray(k)))
        expected = []
        reciprocal = 2*np.pi*np.linalg.inv(np.asarray(cell.lattice)).T
        for axis, count in enumerate(plan.shape):
            phase = np.exp(-1j*np.asarray(data.coords)@(reciprocal[axis]/count))
            # Full Bloch functions and exp(-i q.r), independently of the native folding helper.
            expected.append(np.einsum('g,kgp,g,kgq->kpq', np.asarray(data.weights), psi.conj(), phase,
                                      psi[np.asarray(plan.partners[axis])]))
        error = float(np.max(abs(actual-np.asarray(expected))))
        errors.append(error)
        print('FFT', fft_size, 'native/PySCF same-grid cross-k AO error', error)
    assert errors[-1] < 2e-7
    assert errors[-1] < errors[0]
