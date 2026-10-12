import importlib
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals


def api():
    module = importlib.import_module('gradscf.model.nnao')
    assert hasattr(module, 'prepare_basis'), 'NNAO basis assembly API is missing'
    return module


def test_main_group_templates_and_halogen_shell_roles():
    m = api()
    assert len(m.supported_elements()) == 34
    for symbol in m.supported_elements():
        layout = m.prepare_basis(f'{symbol} 0 0 0')
        p = layout.bind(jnp.zeros((1, 2, 2)))
        assert p.centers.shape[0] == len(layout.roles)
        for role, a, c in zip(layout.roles, p.exponents, p.coefficients):
            assert c.shape == (len(a), 1)
            assert np.isfinite(c).all() and np.linalg.norm(c) > 0
            if role.startswith('valence'): assert len(a) == 3
    for symbol, nao, ncore_d in [('F',10,0),('Cl',14,0),('Br',23,1),('I',32,2)]:
        b=m.prepare_basis(f'{symbol} 0 0 0',cart=False)
        assert b.topology.nao == nao
        assert sum(l==2 and r=='core' for l,r in zip(b.topology.angular_momenta,b.roles)) == ncore_d
        assert b.roles.count('polarization') == 1
    with pytest.raises(ValueError,match='NNAO'): m.prepare_basis('Fe 0 0 0')


def test_atom_and_shell_mapping_and_jit():
    m=api(); b=m.prepare_basis('F 0 0 0; F 0 0 2')
    x=jnp.zeros((2,2,2)); base=b.bind(x)
    changed=jax.jit(b.bind)(x.at[1,1,0].set(.2))
    for i,(atom,slot) in enumerate(zip(b.shell_atoms,b.slots)):
        equal=np.allclose(base.coefficients[i],changed.coefficients[i])
        assert equal == (not (atom==1 and slot==1))
    shifted=b.bind(x,coords_bohr=base.nuclear_coords+jnp.array([.2,.3,.4]))
    np.testing.assert_allclose(shifted.centers,base.centers+jnp.array([.2,.3,.4]))
    with pytest.raises(ValueError): b.bind(jnp.zeros((2,3)))


def test_native_contraction_and_gradient_match_independent_paths():
    m=api(); b=m.prepare_basis('H 0 0 0; H 0 0 .74')
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix, contract_integrals
    x=jnp.zeros((2,2,2)); p=b.bind(x)
    pt,pp=primitive_basis(b.topology,p)
    primitive=integrals.make_plan(pt,backend='native')
    contracted=integrals.make_plan(b.topology,backend='native')
    tensors={op:primitive.evaluate(op,pp) for op in ('overlap','kinetic','nuclear','dipole','eri')}
    def evaluate(u):
        return contract_integrals(tensors,contraction_matrix(b.topology,b.bind(u)))
    value=evaluate(x.at[0,0,0].set(.13))
    for op,tensor in value.items():
        np.testing.assert_allclose(tensor,contracted.evaluate(op,b.bind(x.at[0,0,0].set(.13))),atol=2e-11,rtol=1e-11)
    np.testing.assert_allclose(np.diag(value['overlap']),1.,atol=1e-12)
    loss=lambda u: evaluate(u)['kinetic'][0,0]+evaluate(u)['eri'][0,0,4,4]
    grad=jax.jit(jax.grad(loss))(x)
    dx=jnp.zeros_like(x).at[0,0,0].set(1e-5)
    np.testing.assert_allclose(grad[0,0,0],(loss(x+dx)-loss(x-dx))/2e-5,atol=1e-7,rtol=1e-6)
    assert abs(grad[0,0,0]) > 1e-5


@pytest.mark.parametrize('symbol',['F','Cl','Br','I'])
@pytest.mark.parametrize('cart',[True,False])
def test_export_matches_independent_native_integrals(symbol,cart):
    pytest.importorskip('pyscf')
    from pyscf import gto
    m=api(); b=m.prepare_basis(f'H 0 0 0; {symbol} 0 0 1.8',cart=cart)
    x=jnp.asarray([[[.1,-.2],[0.,0.]],[[.23,-.11],[-.2,.31]]])
    p=b.bind(x); shells=b.atom_shells(p)
    labels=['H0',symbol+'1']
    mol=gto.M(atom=list(zip(labels,np.asarray(p.nuclear_coords))),unit='Bohr',
              basis=dict(zip(labels,shells)),cart=cart,verbose=0)
    native=integrals.make_plan(b.topology,backend='native')
    for op,name in [('overlap','int1e_ovlp'),('kinetic','int1e_kin'),('nuclear','int1e_nuc')]:
        np.testing.assert_allclose(native.evaluate(op,p),mol.intor(name),atol=1e-9,rtol=1e-12)
    # A bounded ERI reference exercises angular indexing in addition to radial coefficients.
    if symbol=='F':
        np.testing.assert_allclose(native.evaluate('eri',p),mol.intor('int2e'),atol=2e-11,rtol=1e-11)


def test_upstream_snapshot_is_unchanged():
    import json,hashlib
    from pathlib import Path
    root=Path('src/gradscf/model/nnao/vendor/mace-jax')
    for name,expected in json.loads((root/'UPSTREAM.json').read_text())['files_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name


def test_all_electron_template_rejects_effective_nuclear_charges():
    from gradscf.data.molecule import parse_molecule_spec
    spec=replace(parse_molecule_spec('I 0 0 0'),charges=jnp.array([7.]))
    with pytest.raises(ValueError,match='all-electron'):
        api().prepare_basis(spec)


def test_contraction_coordinates_do_not_depend_on_svd_nullspace(monkeypatch):
    def reject(*args,**kwargs):
        raise AssertionError('A degenerate SVD nullspace has no portable orientation')
    monkeypatch.setattr(np.linalg,'svd',reject)
    b=api().prepare_basis('F 0 0 0')
    for c,tangent,slot in zip(b.parameters.coefficients,b.tangents,b.slots):
        if slot>=0:
            np.testing.assert_allclose(tangent.T@tangent,np.eye(2),atol=1e-14)
            np.testing.assert_allclose(c[:,0]@tangent,0.,atol=1e-14)
