"""The example-only compressed oracle must preserve the full frozen space."""
from pathlib import Path
import numpy as np


def test_symmetry_compressed_rs_matches_full_reference(monkeypatch):
    from pyscf import gto, scf
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"examples/mp"))
    from _pyscf_reference import mp_coefficients, symmetry_reference, rs_coefficients
    mol = gto.M(atom="Ne 0 0 0",basis="3-21g",symmetry="D2h",verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    action,phi,gaps,diagonal,ecore = symmetry_reference(mf,frozen=1)
    assert phi.size < 4900
    index = int(np.argmax(phi))
    np.testing.assert_allclose(action(phi)[index]+ecore,mf.e_tot,atol=1e-10)
    np.testing.assert_allclose(action(phi)[index],diagonal[index],atol=1e-10)
    actual = rs_coefficients(action,phi,gaps,4)
    np.testing.assert_allclose(actual,mp_coefficients(mf,4,frozen=1),atol=1e-11,rtol=0.)
