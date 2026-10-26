import numpy as np
import pytest
from gradscf.integrals import prepare_basis
from gradscf.integrals.basis.data import load_basis_from_snapshot


@pytest.mark.parametrize('name,equivalent,nao',[
    ('6-31G(d)','6-31G*',22),
    ('6-31G(d,p)','6-31G**',34),
    ('6-31+G(d)','6-31+G*',26),
    ('6-311G(d)','6-311G*',30),
])
def test_parenthesized_polarization_matches_star_alias(name,equivalent,nao):
    for symbol in ('C','H'):
        actual=load_basis_from_snapshot(name,symbol)
        expected=load_basis_from_snapshot(equivalent,symbol)
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            assert a[0]==b[0]
            np.testing.assert_allclose(a[1:],b[1:],atol=0,rtol=0)
    top,_=prepare_basis('C 0 0 0; H 1 0 0; H 0 1 0; H 0 0 1; H -1 0 0',name,cart=False)
    assert top.nao==nao


def test_two_d_and_f_extension_is_retained_on_heavy_atoms_only():
    carbon=load_basis_from_snapshot('6-31g(2df)','C')
    assert [s[0] for s in carbon].count(2)==2
    assert [s[0] for s in carbon].count(3)==1
    assert load_basis_from_snapshot('6-31g(2df)','H')==load_basis_from_snapshot('6-31g','H')
