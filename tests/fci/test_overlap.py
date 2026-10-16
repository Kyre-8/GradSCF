"""Real FCI wavefunction overlaps, including singular determinant minors."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import fci


def overlap():
    function = getattr(fci, 'overlap', None)
    assert callable(function), 'Expose the differentiable FCI wavefunction overlap'
    return function


def test_public_identity_overlap_and_coefficient_response():
    space = fci.make_fci_space(3,(2,1))
    left = jnp.arange(9.,dtype=jnp.float64).reshape(space.shape)/10
    right = jnp.flip(left)
    function = overlap()
    np.testing.assert_allclose(function(left,right,space),jnp.vdot(left,right),atol=1e-14)
    np.testing.assert_allclose(jax.grad(lambda c:function(c,right,space))(left),right,atol=1e-14)
    # No orbital map is differentiated here; higher occupancy needs no determinant.
    filled = fci.make_fci_space(4,(4,4))
    np.testing.assert_allclose(function(jnp.ones((1,1)),jnp.ones((1,1)),filled),1.)


@pytest.mark.parametrize('norb,nelec',[(0,(0,0)),(3,(1,0)),(3,(2,1)),(4,(3,2))])
def test_nonorthogonal_overlap_matches_independent_pyscf(norb,nelec):
    addons = pytest.importorskip('pyscf.fci.addons')
    space = fci.make_fci_space(norb,nelec)
    rng = np.random.default_rng(83+norb)
    left,right = (rng.normal(size=space.shape) for _ in range(2))
    matrix = np.eye(norb)+.1*rng.normal(size=(norb,norb))
    expected = addons.overlap(left,right,norb,nelec,s=matrix)
    actual = jax.jit(lambda a,b,s:overlap()(a,b,space,orbital_overlap=s))(left,right,matrix)
    np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-12)


@pytest.mark.parametrize('occupied',[1,2,3])
@pytest.mark.parametrize('zero_count',[1,2])
def test_singular_minors_have_correct_jvp_and_vjp(occupied,zero_count):
    space = fci.make_fci_space(occupied,(occupied,0))
    state = jnp.ones(space.shape)
    matrix = np.eye(occupied)
    indices = np.arange(min(zero_count,occupied))
    matrix[indices,indices] = 0.
    cofactor = np.array([[(-1.)**(i+j)*np.linalg.det(np.delete(np.delete(matrix,i,0),j,1))
        for j in range(occupied)] for i in range(occupied)])
    loss = lambda s:overlap()(state,state,space,orbital_overlap=s)
    primal,tangent = jax.jit(lambda s:jax.jvp(loss,(s,),(jnp.eye(occupied),)))(jnp.asarray(matrix))
    gradient = jax.jit(jax.grad(loss))(jnp.asarray(matrix))
    np.testing.assert_allclose(primal,np.linalg.det(matrix),atol=1e-14)
    np.testing.assert_allclose(tangent,np.trace(cofactor),atol=1e-14)
    np.testing.assert_allclose(gradient,cofactor,atol=1e-14)


def test_joint_coefficient_and_orbital_response_and_frame_gauge():
    addons = pytest.importorskip('pyscf.fci.addons')
    space = fci.make_fci_space(4,(3,2))
    rng = np.random.default_rng(731)
    left,right,dl,dr = (rng.normal(size=space.shape) for _ in range(4))
    matrix = np.eye(4)+.1*rng.normal(size=(4,4))
    direction = .2*rng.normal(size=(4,4))
    loss = lambda t:overlap()(left+t*dl,right+t*dr,space,orbital_overlap=matrix+t*direction)
    forward = jax.jit(lambda t:jax.jvp(loss,(t,),(jnp.ones_like(t),))[1])(jnp.asarray(0.))
    reverse = jax.jit(jax.grad(loss))(jnp.asarray(0.))
    step = 1e-5
    independent = lambda t:addons.overlap(left+t*dl,right+t*dr,4,(3,2),s=matrix+t*direction)
    finite_difference = (independent(step)-independent(-step))/(2*step)
    np.testing.assert_allclose(forward,finite_difference,atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(forward,reverse,atol=1e-11)
    ul,ur = (np.linalg.qr(rng.normal(size=(4,4)))[0] for _ in range(2))
    rotated_left = addons.transform_ci_for_orbital_rotation(left,4,(3,2),ul)
    rotated_right = addons.transform_ci_for_orbital_rotation(right,4,(3,2),ur)
    np.testing.assert_allclose(overlap()(rotated_left,rotated_right,space,
        orbital_overlap=ul.T@matrix@ur),loss(0.),atol=1e-10,rtol=1e-12)


def test_cross_overlap_unsupported_occupancy_and_shapes():
    function = overlap()
    full = fci.make_fci_space(4,(4,0))
    with pytest.raises(NotImplementedError,match='three|3'):
        function(jnp.ones(full.shape),jnp.ones(full.shape),full,orbital_overlap=jnp.eye(4))
    space = fci.make_fci_space(3,(2,1))
    states = jnp.ones(space.shape)
    with pytest.raises(ValueError,match='shape|matrix'):
        function(states,states,space,orbital_overlap=jnp.eye(2))
    with pytest.raises(NotImplementedError,match='real'):
        function(states,states,space,orbital_overlap=jnp.eye(3,dtype=complex))
    with pytest.raises(NotImplementedError,match='real'):
        function(states.astype(complex),states,space)
    with pytest.raises(TypeError,match='FCISpace'):
        function(states,states,object())


@pytest.mark.parametrize('capacity',[0,-1,True,2.5])
def test_invalid_workspace_capacity(capacity):
    space = fci.make_fci_space(1,(1,0))
    with pytest.raises(ValueError,match='max_workspace_elements'):
        overlap()(jnp.ones(space.shape),jnp.ones(space.shape),space,
                  max_workspace_elements=capacity)


def test_minor_capacity_is_checked_before_array_conversion():
    space = fci.make_fci_space(20,(3,0))
    class Unavailable:
        def __jax_array__(self,*args,**kwargs):
            raise AssertionError('Capacity must be checked before making arrays')
    with pytest.raises(ValueError,match='max_workspace_elements'):
        overlap()(Unavailable(),Unavailable(),space,orbital_overlap=Unavailable(),
                  max_workspace_elements=1000)
