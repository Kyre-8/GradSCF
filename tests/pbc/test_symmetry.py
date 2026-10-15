"""Physical space-group maps, Cartesian tensors and lattice-gauge Bloch phases."""
from dataclasses import replace
import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto
from gradscf.integrals.basis import cartesian_angular_tuples
from gradscf.integrals.periodic.ao import fourier_ao


def cell(hexagonal=False,two_atoms=False):
    a=np.array([[5.,0,0],[2.5,np.sqrt(3)*2.5,0],[0,0,7.]]) if hexagonal else np.eye(3)*6
    return gto.M(atom='He 0 0 0'+('; He 3 0 0' if two_atoms else ''),a=a,
        unit='Bohr',basis={'He':[[0,[.8,1.]],[1,[.6,1.]],[2,[.5,1.]]]},
        pseudo='gth-pade',mesh=(9,)*3)


@pytest.mark.parametrize('l',range(5))
def test_cartesian_rotation_is_a_polynomial_representation(l):
    from gradscf.pbc.symmetry import cartesian_rotation
    angle=.37
    q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    p=np.array([[0,0,1.],[1.,0,0],[0,1.,0]])
    points=np.random.default_rng(4).normal(size=(13,3))
    monomials=lambda x:np.stack([np.prod(x**np.array(a),axis=1) for a in cartesian_angular_tuples(l)],axis=1)
    d=cartesian_rotation(l,q)
    np.testing.assert_allclose(monomials(points)@d,monomials(points@q),atol=2e-13)
    np.testing.assert_allclose(cartesian_rotation(l,p@q),cartesian_rotation(l,p)@d,atol=2e-13)
    np.testing.assert_allclose(cartesian_rotation(l,-np.eye(3)),(-1)**l*np.eye(len(d)),atol=1e-14)
    if l==2:
        assert np.linalg.norm(d.T@d-np.eye(len(d)))>.1


@pytest.mark.parametrize('outside',[False,True])
def test_fractional_translation_phase_and_composition(outside):
    from gradscf.pbc.symmetry import Operation
    c=cell(two_atoms=True)
    if outside:
        coords=c.coords.at[0,0].add(6.)
        c.parameters=c.parameters_at(coords);c.coords=coords
    op=Operation.from_cell(c,np.eye(3),[.5,0,0])
    k=jnp.array([.21,-.13,.07])
    matrix=op.matrix(k)
    np.testing.assert_array_equal(op.atom_map,[1,0])
    np.testing.assert_allclose(matrix@matrix,np.exp(-2j*np.pi*k[0])*np.eye(len(matrix)),atol=2e-13)
    np.testing.assert_allclose(op.matrix(k+jnp.array([1.,0,0])),matrix,atol=2e-13)
    shifts=np.array([[0,0,0],[1,0,0]])[::-1] if outside else np.array([[0,0,0],[1,0,0]])
    np.testing.assert_array_equal(op.lattice_shifts,shifts)


@pytest.mark.parametrize('kind',['translation','hexagonal','screw','contracted_f'])
def test_bloch_transform_matches_independent_gaussian_fourier_values(kind):
    from gradscf.pbc.symmetry import Operation
    hexagonal=kind in ('hexagonal','contracted_f')
    c=cell(hexagonal=hexagonal,two_atoms=kind=='translation')
    w=np.array([[0,-1,0],[1,1,0],[0,0,1]]) if hexagonal else np.eye(3)
    t=np.zeros(3) if hexagonal else np.array([.5,0,0])
    if kind=='contracted_f':
        c.basis={'He':[[3,[.8,1.,.2],[.4,.3,1.]]]};c.build()
    if kind=='screw':
        w=np.array([[0,-1,0],[1,0,0],[0,0,1]]);t=np.array([.5,0,.5])
        c.atom=[('He',np.array(f)*6) for f in ((.6,.2,.05),(.3,.6,.55),(.9,.3,.05),(.2,.9,.55))]
        c.build()
    op=Operation.from_cell(c,w,t)
    k=jnp.array([.17,-.11,.03])
    integers=jnp.asarray([[0,0,0],[1,0,1],[-1,2,0],[2,-2,1],[-2,-1,-1]])
    vectors=(integers+k)@c.reciprocal_vectors()
    rotated=vectors@op.cartesian.T
    original=fourier_ao(c.topology,c.parameters,vectors)
    transformed=fourier_ao(c.topology,c.parameters,rotated)@op.matrix(k)
    expected=jnp.exp(-1j*rotated@(jnp.asarray(t)@c.lattice))[:,None]*original
    np.testing.assert_allclose(transformed,expected,atol=2e-11,rtol=2e-11)
    doubled=Operation.from_cell(c,w@w,w@t+t)
    np.testing.assert_allclose(op.matrix(op.rotate_k(k))@op.matrix(k),doubled.matrix(k),atol=2e-12)


def test_cartesian_operator_covariance_against_pyscf():
    from gradscf.pbc.symmetry import Operation
    pgto=pytest.importorskip('pyscf.pbc.gto')
    c=cell(hexagonal=True)
    op=Operation.from_cell(c,[[0,-1,0],[1,1,0],[0,0,1]])
    k=np.array([.12,-.07,.03])
    oracle=pgto.M(atom='He 0 0 0',a=np.asarray(c.lattice),unit='Bohr',
                  basis=c.basis_spec,pseudo='gth-pade',cart=True,precision=1e-10,verbose=0)
    points=np.stack([k,np.asarray(op.rotate_k(k))])@np.asarray(c.reciprocal_vectors())
    overlap=oracle.pbc_intor('int1e_ovlp',kpts=points)
    kinetic=oracle.pbc_intor('int1e_kin',kpts=points)
    for matrices in (overlap,kinetic):
        np.testing.assert_allclose(op.operator(k,matrices[0]),matrices[1],atol=2e-9,rtol=2e-9)
    transform=op.matrix(k)
    assert np.linalg.norm(transform@overlap[0]@transform.conj().T-overlap[1])>.1
    rng=np.random.default_rng(7)
    coeff=jnp.asarray(rng.normal(size=(c.topology.nao,3))+1j*rng.normal(size=(c.topology.nao,3)))
    moved=op.coefficients(k,coeff)
    np.testing.assert_allclose(moved.conj().T@overlap[1]@moved,
                               coeff.conj().T@overlap[0]@coeff,atol=2e-9)


def test_coefficient_density_operator_transforms_retain_jit_ad():
    from gradscf.pbc.symmetry import Operation
    c=cell(two_atoms=True);op=Operation.from_cell(c,np.eye(3),[.5,0,0])
    rng=np.random.default_rng(5);n=c.topology.nao
    x=jnp.asarray(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    d=x@x.conj().T;f=x+x.conj().T;k=jnp.array([.23,.1,-.2])
    moved=jax.jit(op.density)(k,d)
    np.testing.assert_allclose(jnp.trace(moved@op.operator(k,f)),jnp.trace(d@f),atol=1e-10)
    np.testing.assert_allclose(moved,op.coefficients(k,x)@op.coefficients(k,x).conj().T,atol=1e-11)
    loss=lambda point:jnp.vdot(x,op.density(point,d)).real
    gradient=jax.jit(jax.grad(loss))(k)
    direction=jnp.array([.3,-.2,.1]);step=1e-5
    fd=(loss(k+step*direction)-loss(k-step*direction))/(2*step)
    np.testing.assert_allclose(jnp.dot(gradient,direction),fd,atol=1e-5,rtol=1e-7)
    _,tangent=jax.jvp(lambda density:op.density(k,density),(d,),(f,))
    np.testing.assert_allclose(tangent,op.density(k,f),atol=1e-12)


@pytest.mark.parametrize('hexagonal,expected',[(False,48),(True,24)])
def test_detect_space_group_without_requiring_fft_grid_invariance(hexagonal,expected):
    pytest.importorskip('spglib')
    from gradscf.pbc.symmetry import space_group
    c=cell(hexagonal=hexagonal)
    operations=space_group(c)
    assert len(operations)==expected
    assert any(np.linalg.det(op.rotation)<0 for op in operations)
    for op in operations:
        op.validate(c)


@pytest.mark.parametrize('change',['basis','pseudo'])
def test_symmetry_respects_actual_basis_and_pseudopotential(change):
    pytest.importorskip('spglib')
    from gradscf.pbc.symmetry import Operation,space_group
    c=gto.M(atom='He 0 0 0; He 3 3 3',a=np.eye(3)*6,unit='Bohr',mesh=(9,)*3)
    op=Operation.from_cell(c,np.eye(3),[.5,.5,.5])
    assert len(space_group(c))==96
    if change=='basis':
        exponents=list(c.parameters.exponents);exponents[-1]=exponents[-1]*1.01
        c.parameters=replace(c.parameters,exponents=tuple(exponents))
    else:
        c.pseudopotentials=copy.deepcopy(c.pseudopotentials)
        c.pseudopotentials[1][1]*=1.01
    with pytest.raises(ValueError,match='changed'):
        op.validate(c)
    with pytest.raises(ValueError,match='basis|pseudopotential'):
        Operation.from_cell(c,np.eye(3),[.5,.5,.5])
    assert len(space_group(c))==48


def test_invalid_operations_and_floating_centers_are_rejected():
    from gradscf.pbc.symmetry import Operation,cartesian_rotation
    c=cell()
    for w in (np.eye(3)*1.5,np.diag([2.,1.,1.]),[[1,1,0],[0,1,0],[0,0,1]]):
        with pytest.raises(ValueError):Operation.from_cell(c,w)
    with pytest.raises(ValueError):Operation.from_cell(c,np.eye(3),[.2,0,0])
    with pytest.raises(ValueError):Operation.from_cell(c,np.eye(3),tolerance=0.)
    with pytest.raises(ValueError):cartesian_rotation(2,np.ones((3,3)))
    c.parameters=replace(c.parameters,centers=c.parameters.centers.at[0,0].add(.01))
    with pytest.raises(ValueError,match='atom.attached'):
        Operation.from_cell(c,np.eye(3))


@pytest.mark.parametrize('part',['centers','nuclear_coords','coords','lattice'])
def test_nonfinite_geometry_cannot_produce_a_validated_operation(part):
    from gradscf.pbc.symmetry import Operation
    c=cell()
    if part in ('centers','nuclear_coords'):
        c.parameters=replace(c.parameters,**{part:getattr(c.parameters,part).at[0,0].set(jnp.nan)})
    else:
        setattr(c,part,getattr(c,part).at[0,0].set(jnp.nan))
    with pytest.raises(ValueError,match='finite'):
        Operation.from_cell(c,np.eye(3))


def test_nonstandard_lattice_basis_and_stale_geometry():
    from gradscf.pbc.symmetry import Operation
    c=cell(hexagonal=True);w=np.array([[0,-1,0],[1,1,0],[0,0,1]])
    original=Operation.from_cell(c,w)
    change=np.array([[1,2,0],[0,1,0],[0,0,1]])
    c.a=change@np.asarray(c.lattice);c.build()
    transformed=np.linalg.inv(change.T)@w@change.T
    assert np.max(np.abs(transformed))>1
    op=Operation.from_cell(c,transformed)
    np.testing.assert_allclose(op.cartesian,original.cartesian,atol=1e-14)
    k=jnp.array([.13,-.11,.07])
    np.testing.assert_allclose(op.matrix(k@change.T),original.matrix(k),atol=2e-13)
    np.testing.assert_allclose(op.rotate_k(k@change.T)@np.linalg.inv(change.T),original.rotate_k(k),atol=2e-13)
    with pytest.raises(ValueError,match='changed'):original.validate(c)
    with pytest.raises(ValueError):op.rotation[0,0]=3
