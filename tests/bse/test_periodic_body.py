"""Independent finite-cell oracles for the first complex periodic BSE body."""
import importlib
import importlib.util
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def api():
    return importlib.import_module('gradscf.bse.pbc')


def test_periodic_body_api_is_available():
    assert importlib.util.find_spec('gradscf.bse.pbc') is not None


@pytest.fixture(scope='module', params=[1, 2])
def reference(request):
    from gradscf.pbc import gto, scf
    from gradscf.pbc._kpoint import energy_and_fock
    cell = gto.M(atom='H .5 .4 .3; H 1.9 .4 .3; H 3.0 3.0 2.8; H 4.4 3.0 2.8',
        a=np.eye(3)*7., unit='Bohr', basis='gth-szv', pseudo='gth-pade', mesh=(7,)*3)
    mf = scf.KRHF(cell, kpts=cell.make_kpts((request.param, 1, 1))).run(
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150)
    assert mf.converged
    c = mf.result.mo_coeff_spin[0]
    _, fock = energy_and_fock(mf.result.density_spin, mf.inputs,
                              mesh=cell.mesh, xc='hf', exxdiv=None)
    # Match the existing periodic TDA convention: its gaps also omit Ewald.
    e = jnp.einsum('kpi,kpq,kqi->ki', c.conj(), fock[0], c).real
    return mf, c, e, cell.nelec[0]


def build(reference, *, inputs=None, coefficients=None, energies=None, **kwargs):
    mf, c, e, no = reference
    return api().build_tda_matrix(mf.inputs if inputs is None else inputs,
        c if coefficients is None else coefficients,
        e if energies is None else energies, e if energies is None else energies,
        no, mesh=mf.cell.mesh, **kwargs)


def dense_oracle(reference, screened, *, reverse=True, return_blocks=False):
    """NumPy Fourier products and a full auxiliary dielectric inverse."""
    mf, c, e, no = reference
    data = mf.inputs
    c, e = np.asarray(c), np.asarray(e)
    nk, _, nmo = c.shape
    k, r, g = map(np.asarray, (data.kpoints, data.coords, data.gvectors))
    fractional = k@np.asarray(data.lattice).T/(2*np.pi)
    table = np.empty((nk, nk), dtype=int)
    for i in range(nk):
        for q in range(nk):
            distance = fractional[i]-fractional[q]-fractional
            table[i, q] = np.flatnonzero(np.max(abs(distance-np.rint(distance)), axis=1)<1e-9)[0]
    zero = np.flatnonzero(np.all(table == np.arange(nk)[:, None], axis=0))[0]
    psi = np.einsum('kgp,kpm->kgm', np.asarray(data.ao[:, 0]), c)
    volume, ngrid = float(mf.cell.volume), len(r)
    factors, inverses, dielectrics = [], [], []
    for q in range(nk):
        block = []
        norm = np.sum((g-k[q])**2, axis=1)
        coulomb = np.zeros_like(norm)
        np.divide(4*np.pi, norm, out=coulomb, where=norm>1e-14)
        for i, j in enumerate(table[:, q]):
            pair = psi[i].conj()[:, :, None]*psi[j][:, None, :]
            pair = pair*np.exp(1j*(r@(k[j]-k[i]+k[q])))[:, None, None]
            transformed = np.fft.fftn(pair.reshape(*mf.cell.mesh, nmo, nmo), axes=(0, 1, 2))
            block.append(transformed.reshape(ngrid, nmo, nmo)*np.sqrt(volume*coulomb)[:, None, None]/ngrid)
        block = np.asarray(block)
        factors.append(block)
        transitions = np.concatenate([block[i, :, :no, no:].reshape(ngrid, -1) for i in range(nk)], axis=1)
        differences = np.concatenate([(e[i, :no, None]-e[j, None, no:]).ravel()
                                       for i, j in enumerate(table[:, q])])
        weights = 4/(nk*differences)
        if reverse:
            backward = np.concatenate([block[i, :, no:, :no].reshape(ngrid, -1)
                                       for i in range(nk)], axis=1)
            reverse_differences = np.concatenate([(e[i, no:, None]-e[j, None, :no]).ravel()
                                                  for i, j in enumerate(table[:, q])])
            transitions = np.concatenate([transitions, backward], axis=1)
            weights = np.concatenate([2/(nk*differences), -2/(nk*reverse_differences)])
        epsilon = np.eye(ngrid)-((transitions*weights)@transitions.conj().T if screened else 0.)
        dielectrics.append(epsilon)
        inverses.append(np.linalg.solve(epsilon, np.eye(ngrid)))
    space = [(k, v, c) for k in range(nk) for v in range(no) for c in range(no, nmo)]
    matrix = np.empty((len(space), len(space)), dtype=complex)
    for row, (k, v, c) in enumerate(space):
        for col, (kp, vp, cp) in enumerate(space):
            q = np.flatnonzero(table[k] == kp)[0]
            exchange = np.vdot(factors[zero][k, :, v, c], factors[zero][kp, :, vp, cp])
            direct = np.vdot(factors[q][k, :, v, vp], inverses[q]@factors[q][k, :, c, cp])
            matrix[row, col] = (2*exchange-direct)/nk
            if row == col:
                matrix[row, col] += e[k, c]-e[k, v]
    return (matrix, factors, dielectrics, inverses, table) if return_blocks else matrix


def test_static_screening_reciprocity_without_time_reversal_reduction(reference):
    """The full static Lehmann response respects q/-q conjugacy by construction."""
    import json
    mf, _, _, _ = reference
    legacy, _, old_eps, old_inv, _ = dense_oracle(reference, True, reverse=False, return_blocks=True)
    complete, _, new_eps, new_inv, _ = dense_oracle(reference, True, reverse=True, return_blocks=True)
    bare = dense_oracle(reference, False)
    frac = np.asarray(mf.kpts)@np.asarray(mf.cell.lattice).T/(2*np.pi)
    indices = np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(n)*n).astype(int)
                                    for n in mf.cell.mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    lookup = {tuple(value): i for i, value in enumerate(indices)}
    half = np.asarray(mf.cell.mesh)//2
    report = {'nk': len(frac), 'legacy_hermiticity': float(np.max(abs(legacy-legacy.conj().T))),
              'full_hermiticity': float(np.max(abs(complete-complete.conj().T))),
              'bare_hermiticity': float(np.max(abs(bare-bare.conj().T))), 'q': []}
    for q in range(len(frac)):
        opposite = np.flatnonzero(np.max(abs(frac+frac[q]-np.rint(frac+frac[q])), axis=1)<1e-10)[0]
        target = np.rint(frac[q]+frac[opposite]).astype(int)-indices
        wrapped = (target+half)%np.asarray(mf.cell.mesh)-half
        order = np.array([lookup[tuple(value)] for value in wrapped])
        exact = np.flatnonzero(np.all(target == wrapped, axis=1))
        item = {'q': q, 'qbar': int(opposite), 'exact_rows': len(exact)}
        for label, blocks in [('legacy_epsilon', old_eps), ('full_epsilon', new_eps),
                              ('legacy_inverse', old_inv), ('full_inverse', new_inv)]:
            defect = blocks[q]-blocks[opposite][order[:, None], order].conj()
            item[label] = float(np.max(abs(defect)))
            item[label+'_exact_rows'] = float(np.max(abs(defect[np.ix_(exact, exact)])))
        report['q'].append(item)
    print(json.dumps(report, indent=2), flush=True)
    assert report['full_hermiticity'] < 1e-12
    assert max(item['full_epsilon'] for item in report['q']) < 1e-12
    assert max(item['full_inverse'] for item in report['q']) < 1e-12
    if len(frac) == 2:
        # Coarse Bloch discretization is not exactly time-reversal paired;
        # doubling only OV must not replace the two physical channels.
        assert report['legacy_hermiticity'] > 1e-6
    # This is the original physical gate, not a symmetrized diagnostic matrix.
    matrix = build(reference)
    np.testing.assert_allclose(matrix, complete, atol=2e-10, rtol=2e-9)


def test_unscreened_limit_matches_existing_hf_tda(reference):
    from gradscf.pbc import tdscf
    matrix = build(reference, screened=False)
    expected, _ = tdscf.TDA(reference[0], nstates=2).get_ab()
    np.testing.assert_allclose(matrix, expected, atol=2e-10, rtol=2e-9)


def test_screened_kernel_matches_independent_auxiliary_solve(reference):
    matrix = build(reference)
    expected = dense_oracle(reference, True)
    np.testing.assert_allclose(matrix, expected, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=2e-10)
    result = api().solve_tda(matrix, nstates=2, tol=1e-10)
    assert np.all(result.converged)
    np.testing.assert_allclose(result.energies, np.linalg.eigvalsh(expected)[:2], atol=2e-9)
    assert np.max(np.asarray(result.residual_norms)) < 1e-10


def test_orbital_phases_preserve_kernel_and_optical_amplitudes(reference):
    _, c, e, no = reference
    rng = np.random.default_rng(20261016)
    phases = np.exp(1j*rng.normal(size=e.shape))
    matrix = build(reference)
    rotated = build(reference, coefficients=c*phases[:, None, :])
    transition_phase = (phases[:, None, no:]/phases[:, :no, None]).ravel()
    np.testing.assert_allclose(rotated, transition_phase.conj()[:, None]*matrix*transition_phase,
                               atol=2e-10, rtol=2e-9)
    result = api().solve_tda(matrix, nstates=2, tol=1e-10)
    vertex = rng.normal(size=(len(c), no, e.shape[1]-no, 3))
    vertex = vertex+1j*rng.normal(size=vertex.shape)
    original = api().transition_dipoles(vertex, result.vectors)
    changed = api().transition_dipoles(vertex*transition_phase.reshape(vertex.shape[:-1])[..., None],
                                       result.vectors*transition_phase.conj()[:, None])
    np.testing.assert_allclose(changed, original, atol=2e-10)


def test_k_order_and_q_zero_location(reference):
    mf, c, e, no = reference
    order = np.arange(len(c))[::-1]
    data = mf.inputs._replace(ao=mf.inputs.ao[order], overlap=mf.inputs.overlap[order],
        hcore=mf.inputs.hcore[order], kpoints=mf.inputs.kpoints[order],
        kweights=mf.inputs.kweights[order])
    matrix = build(reference)
    permuted = build(reference, inputs=data, coefficients=c[order], energies=e[order])
    indices = np.arange(len(matrix)).reshape(len(c), no*(e.shape[1]-no))[order].ravel()
    np.testing.assert_allclose(permuted, matrix[indices[:, None], indices], atol=2e-10, rtol=2e-9)


def test_optical_window_retains_full_screening(reference):
    mf, c, e, no = reference
    full = build(reference)
    qp = np.full(e.shape, np.nan)
    qp[:, [no-1, e.shape[1]-1]] = np.asarray(e)[:, [no-1, e.shape[1]-1]]
    selected = api().build_tda_matrix(mf.inputs, c, e, qp, no, mesh=mf.cell.mesh,
                                     valence=[no-1], conduction=[e.shape[1]-1])
    indices = np.arange(len(full)).reshape(len(c), no, e.shape[1]-no)[:, -1, -1]
    np.testing.assert_allclose(selected, full[indices[:, None], indices], atol=2e-10, rtol=2e-9)


def test_scope_and_autodiff_are_explicit(reference):
    mf, c, e, no = reference
    data = dict(mf.inputs._asdict(), boundary='slab')
    with pytest.raises(NotImplementedError, match='three-dimensional'):
        build(reference, inputs=SimpleNamespace(**data))
    with pytest.raises(ValueError, match='max_transitions'):
        build(reference, max_transitions=1)
    with pytest.raises(NotImplementedError, match='AD|differenti'):
        jax.grad(lambda t: jnp.trace(api().build_tda_matrix(
            mf.inputs, c, e, e*t, no, mesh=mf.cell.mesh)).real)(1.)
    matrix = build(reference)
    with pytest.raises(NotImplementedError, match='AD|differenti'):
        jax.grad(lambda t: api().solve_tda(matrix*t, nstates=1).energies.sum())(1.)


def test_solver_rejects_nonhermitian_input():
    with pytest.raises(ValueError, match='Hermitian'):
        api().solve_tda(jnp.array([[1., 1j], [0., 2.]]), nstates=1)


def test_optical_vertex_matches_independent_small_q_overlap():
    def hamiltonian(k):
        return np.array([[.7+.2*np.cos(k), .4*np.cos(k)-.3j*np.sin(k)],
                         [.4*np.cos(k)+.3j*np.sin(k), -.7-.2*np.cos(k)]])
    k = .31
    energies, orbitals = np.linalg.eigh(hamiltonian(k))
    derivative = np.array([[-.2*np.sin(k), -.4*np.sin(k)-.3j*np.cos(k)],
                           [-.4*np.sin(k)+.3j*np.cos(k), .2*np.sin(k)]])
    velocity = np.vdot(orbitals[:, 0], derivative@orbitals[:, 1])
    gap = energies[1]-energies[0]
    dipole = api().ks_transition_dipoles(jnp.array([velocity, 0., 0.]).reshape(1, 1, 1, 3),
                                        jnp.array([[[gap]]]))[0, 0, 0, 0]
    for step in (1e-4, 3e-5):
        overlaps = []
        for sign in (1., -1.):
            _, shifted = np.linalg.eigh(hamiltonian(k+sign*step))
            column = shifted[:, 1]*np.exp(-1j*np.angle(np.vdot(orbitals[:, 1], shifted[:, 1])))
            overlaps.append(np.vdot(orbitals[:, 0], column))
        np.testing.assert_allclose(dipole, 1j*(overlaps[0]-overlaps[1])/(2*step), atol=2e-9)


def test_optical_nk_normalization_and_scissor():
    vertex = jnp.array([.2+.1j, -.1j, .3])
    for nk in (1, 2, 4):
        dipoles = jnp.broadcast_to(vertex, (nk, 1, 1, 3))
        vector = jnp.ones((nk, 1))/jnp.sqrt(float(nk))
        transition = api().transition_dipoles(dipoles, vector)
        np.testing.assert_allclose(transition[0], jnp.sqrt(2.)*vertex, atol=1e-14)
    velocity = jnp.array([.3, .2j, .1]).reshape(1, 1, 1, 3)
    original = api().ks_transition_dipoles(velocity, jnp.ones((1, 1, 1)))
    qp = api().ks_transition_dipoles(2*velocity, 2*jnp.ones((1, 1, 1)))
    np.testing.assert_allclose(qp, original, atol=1e-14)
    transition = api().transition_dipoles(original, jnp.ones((1, 1)))
    f = api().oscillator_strengths(jnp.array([1.]), transition)
    shifted_f = api().oscillator_strengths(jnp.array([2.]), transition)
    np.testing.assert_allclose(shifted_f, 2*f, atol=1e-14)


def test_ks_reference_dipoles_use_the_existing_complete_velocity():
    assert hasattr(api(), 'ks_dipoles')
    from gradscf.pbc import gto, dft
    from gradscf.pbc.optics import velocity_matrix
    cell = gto.M(atom='H 3 3 2.3; H 3 3 3.7', a=np.eye(3)*6., unit='Bohr',
        basis='gth-szv', pseudo='gth-pbe', mesh=(7,)*3)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='pbe').run(
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10)
    assert mf.converged
    actual = api().ks_dipoles(mf)
    expected = []
    for k, c, e in zip(mf.kpts, mf.mo_coeff, mf.mo_energy):
        velocity = np.asarray(velocity_matrix(cell, kpt=k))
        vertex = np.einsum('pi,dpq,qa->iad', np.asarray(c[:, :1]).conj(), velocity, c[:, 1:])
        expected.append(1j*vertex/(e[1:][None, :, None]-e[:1, None, None]))
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0.)
    np.testing.assert_allclose(api().ks_dipoles(mf, valence=[0], conduction=[1]), actual, atol=1e-13)
    mf.xc = 'svwn'
    with pytest.raises(ValueError, match='again'):
        api().ks_dipoles(mf)


def test_ks_reference_dipoles_reject_unimplemented_exchange_velocity(reference):
    with pytest.raises(NotImplementedError, match='LDA|GGA'):
        api().ks_dipoles(reference[0])


def test_independent_particles_preserve_selected_gaps_and_per_cell_weights():
    qp = jnp.array([[jnp.nan, -.4, jnp.nan, .8], [jnp.nan, -.3, jnp.nan, 1.]])
    vertices = jnp.array([[[[.2+.1j, -.1j, .3]]], [[[.1, .2j, -.2]]]])
    result = api().independent_particles(qp, 2, vertices, valence=[1], conduction=[3])
    np.testing.assert_allclose(result.energies, [1.2, 1.3], atol=1e-14)
    np.testing.assert_allclose(result.dipoles, vertices.reshape(2, 3), atol=1e-14)
    expected = 2*np.array([1.2, 1.3])*np.sum(abs(np.asarray(vertices).reshape(2, 3))**2, axis=1)/3
    np.testing.assert_allclose(result.strengths, expected, atol=1e-14)
    with pytest.raises(ValueError, match='shape|window'):
        api().independent_particles(qp, 2, vertices)
