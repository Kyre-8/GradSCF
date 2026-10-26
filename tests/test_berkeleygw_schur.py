"""Independent full-frequency binary fixture and block-inverse identity."""
import hashlib
import struct

import numpy as np
import pytest


def _write(stream, value):
    marker = struct.pack('<i', len(value))
    stream.write(marker+value+marker)


def _read(stream):
    marker = stream.read(4)
    size, = struct.unpack('<i', marker)
    value = stream.read(size)
    assert stream.read(4) == marker
    return value


def _fixture(path, *, singular=False):
    dielectric = np.array([
        [[2+.1j, .2-.3j, .1+.1j], [.15+.2j, 4+.2j, .2-.1j], [.12-.1j, .1+.05j, 3-.1j]],
        [[1.8+.2j, .3-.1j, .2+.1j], [.2+.1j, 3+.3j, .1-.2j], [.1-.1j, .2+.03j, 2.8-.2j]]])
    retarded = np.linalg.inv(dielectric)
    v = np.diag([2., 5., 3.])
    advanced = v@retarded.conj().swapaxes(-1, -2)@np.linalg.inv(v)
    matrices = np.stack([retarded, advanced])
    if singular:
        matrices[:, :, 1, 1] = 0.
    integer = lambda value: np.asarray(value, dtype='<i4').tobytes()
    real = lambda value: np.asarray(value, dtype='<f8').tobytes()
    headers = [b'chiGG0'+b'01-Jan-2000', integer([2, 2, 0, 0, 1]), integer([2, 1, 1]),
        real([0., 1.])+np.asarray([.01j, 1j], '<c16').tobytes(), b'', b'', real(8.),
        integer(1)+real([0., 0., .001]),
        integer(3)+integer([[1, 0, 0], [0, 0, 0], [-1, 0, 0]]),
        integer([3, 3])+integer([[3, 3], [2, 2], [1, 1]]), real([1., 0., 1.]), real([0., 0., .001])]
    with path.open('wb') as stream:
        for header in headers:
            _write(stream, header)
        for column in range(3):
            for channel in range(2):
                for row in range(3):
                    _write(stream, np.asarray(matrices[channel, :, row, column], '<c16').tobytes())
    return headers, matrices


def test_schur_copy_matches_inverse_of_original_body_and_preserves_headers(tmp_path):
    from gradscf_tools.berkeleygw import copy_eps0_without_head_feedback

    source, destination = tmp_path/'eps0mat', tmp_path/'eps0mat_body'
    headers, before = _fixture(source)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = copy_eps0_without_head_feedback(source, destination)
    after = np.empty_like(before)
    with destination.open('rb') as stream:
        assert [_read(stream) for _ in range(12)] == headers
        for column in range(3):
            for channel in range(2):
                for row in range(3):
                    after[channel, :, row, column] = np.frombuffer(_read(stream), '<c16')
        assert stream.read(1) == b''
    body = [0, 2]  # G=0 is deliberately not the first matrix entry.
    for channel in range(2):
        for frequency in range(2):
            epsilon = np.linalg.inv(before[channel, frequency])
            expected = np.linalg.inv(epsilon[np.ix_(body, body)])
            np.testing.assert_allclose(after[channel, frequency][np.ix_(body, body)], expected, atol=1e-13)
    np.testing.assert_array_equal(after[:, :, 1, 1], 1.)
    np.testing.assert_array_equal(after[:, :, 1, body], 0.)
    np.testing.assert_array_equal(after[:, :, body, 1], 0.)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert report['head_index'] == 1 and report['matrix_size'] == 3
    assert report['max_identity_relative_error'] < 1e-12


def test_schur_copy_rejects_zero_inverse_head_before_creating_output(tmp_path):
    from gradscf_tools.berkeleygw import copy_eps0_without_head_feedback

    source, destination = tmp_path/'eps0mat', tmp_path/'eps0mat_body'
    _fixture(source, singular=True)
    with pytest.raises(ValueError, match='head'):
        copy_eps0_without_head_feedback(source, destination)
    assert not destination.exists()
