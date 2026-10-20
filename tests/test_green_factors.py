"""GREEN's pair-adjoint storage must preserve the complete Coulomb metric."""
import numpy as np
import pytest


def _factors(partners, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(2, 3, 2, 2))+1j*rng.normal(size=(2, 3, 2, 2))
    base = (base+base[partners].conj().swapaxes(-1, -2))/2
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    return np.einsum('PQ,kQmn->kPmn', rotation, base)


@pytest.mark.parametrize('partners', [np.array([0, 1]), np.array([1, 0])])
def test_pair_adjoint_storage_preserves_all_cross_k_coulomb_contractions(partners):
    from gradscf_tools.green import pair_adjoint_factors

    original = _factors(partners)
    converted, report = pair_adjoint_factors(original, partners)
    # Compare all four-index pair integrals, rather than only diagonal norms.
    original_eri = np.einsum('kPab,lPcd->kablcd', original.conj(), original)
    converted_eri = np.einsum('kPab,lPcd->kablcd', converted.conj(), converted)
    np.testing.assert_allclose(converted_eri, original_eri, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(converted[partners].conj().swapaxes(-1, -2), converted,
                               atol=0., rtol=0.)
    assert report['relative_gram_error'] < 1e-13
    assert converted.dtype == np.complex128


def test_pair_adjoint_conversion_rejects_an_incompatible_coulomb_metric():
    from gradscf_tools.green import pair_adjoint_factors

    # An arbitrary complex metric has no required pair-adjoint symmetry.
    factor = np.array([[[[1., 2j], [3., 4.]]]], dtype=complex)
    with pytest.raises(ValueError, match='Gram'):
        pair_adjoint_factors(factor, np.array([0]))


def test_pair_adjoint_conversion_requires_a_self_inverse_transfer():
    from gradscf_tools.green import pair_adjoint_factors

    with pytest.raises(ValueError, match='self-inverse'):
        pair_adjoint_factors(np.ones((3, 1, 2, 2), dtype=complex), np.array([1, 2, 0]))


def test_zero_interaction_remains_zero():
    from gradscf_tools.green import pair_adjoint_factors

    converted, report = pair_adjoint_factors(np.zeros((1, 1, 2, 2)), np.array([0]))
    np.testing.assert_array_equal(converted, 0.)
    assert report['relative_gram_error'] == 0.


def test_green_files_preserve_the_source_model_in_the_upstream_reader_layout(tmp_path):
    h5py = pytest.importorskip('h5py')
    from gradscf.gw.pbc.source import PeriodicGWModel
    from gradscf_tools.green import export_source

    table = np.array([[0, 1], [1, 0]])
    energy = np.array([[-.8, .5], [-.7, .6]])
    fock = np.array([np.diag(row) for row in energy]).astype(complex)
    factors = tuple(_factors(table[:, q], seed=17+q) for q in range(2))
    model = PeriodicGWModel(energy=energy, coeff=np.tile(np.eye(2), (2, 1, 1)),
        hcore=.5*fock, reference_potential=np.zeros_like(fock), reference_fock=fock,
        nocc=1, kpoints=np.array([[0., 0., 0.], [-.5, 0., 0.]]),
        lattice=2*np.pi*np.eye(3), table=table, inverse=table, factors=factors)
    directory = tmp_path/'green'
    report = export_source(model, directory, kmesh=(2, 1, 1))
    with h5py.File(directory/'input.h5') as data:
        np.testing.assert_array_equal(data['HF/H-k'][0], model.hcore)
        np.testing.assert_array_equal(data['HF/Fock-k'][0], model.reference_fock)
        np.testing.assert_array_equal(data['params/nel_cell'][()], 2)
        np.testing.assert_array_equal(data['symmetry/k/nk_list'][()], [2, 1, 1])
        np.testing.assert_array_equal(data['symmetry/pairs/conj_pairs_list'][()], [0, 1, 2])
        assert data['HF/S-k'].dtype == np.complex128
    with h5py.File(directory/'df_hf/meta.h5') as data:
        assert data['chunk_size'][()] == 1
    pairs = [(0, 0), (1, 0), (1, 1)]
    restored = {}
    for idx, (left, right) in enumerate(pairs):
        # Upstream df_integral_t reads a double buffer, then reconstructs
        # the opposite triangular pair with an adjoint.
        with h5py.File(directory/f'df_hf/VQ_{idx}.h5') as data:
            raw = data[str(idx)][()]
            assert raw.dtype == np.float64
            value = raw.view(np.complex128).reshape(1, report['naux'], 2, 2)[0]
        restored[left, right] = value
        restored[right, left] = value.conj().swapaxes(-1, -2)
    for q, original in enumerate(factors):
        written = np.asarray([restored[k, table[k, q]] for k in range(2)])
        np.testing.assert_allclose(np.einsum('kPab,lPcd->kablcd', written.conj(), written),
            np.einsum('kPab,lPcd->kablcd', original.conj(), original),
            atol=3e-13, rtol=3e-13)
    with pytest.raises(FileExistsError):
        export_source(model, directory, kmesh=(2, 1, 1))
