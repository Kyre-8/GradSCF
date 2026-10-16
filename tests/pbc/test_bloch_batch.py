"""Bloch-grid batching: bounded trace growth, external values and geometry AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto
from gradscf.integrals.periodic.ao import bloch_grid


def _cell(mesh):
    return gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6,
        unit='Bohr', basis='gth-szv', pseudo='gth-pade', mesh=(mesh,)*3)


def test_bloch_trace_size_does_not_scale_with_number_of_kpoints():
    cell = _cell(7)
    evaluate = lambda k: bloch_grid(cell.topology, cell.parameters, cell.lattice, cell.mesh, k)
    one = jax.make_jaxpr(evaluate)(cell.make_kpts((1,1,1)))
    four = jax.make_jaxpr(evaluate)(cell.make_kpts((4,1,1)))
    assert len(four.jaxpr.eqns) <= len(one.jaxpr.eqns)+20


def test_batched_bloch_grid_matches_pyscf():
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto
    from pyscf.pbc.dft.numint import eval_ao_kpts
    # Tight primitives need a finer reciprocal cutoff than the small graph test.
    cell = _cell(61)
    kpts = cell.make_kpts((3,1,1), shift=(.5,0.,0.))
    ao, _, _, coordinates = bloch_grid(cell.topology, cell.parameters, cell.lattice, cell.mesh, kpts)
    reference = pgto.M(atom=cell.atom, a=np.asarray(cell.lattice), unit='Bohr',
        basis='gth-szv', pseudo='gth-pade', cart=True, precision=1e-12, verbose=0)
    expected = np.asarray(eval_ao_kpts(reference, np.asarray(coordinates), kpts=np.asarray(kpts), deriv=1))
    np.testing.assert_allclose(ao, expected, atol=2e-7, rtol=0)


def test_batched_bloch_geometry_jvp_matches_difference():
    cell = _cell(9)
    kpts = cell.make_kpts((3,1,1), shift=(.5,0.,0.))
    direction = jnp.array([[.3,-.2,.1],[-.4,.1,.2]])
    def fields(r):
        return bloch_grid(cell.topology, cell.parameters_at(r), cell.lattice, cell.mesh, kpts)[:2]
    actual = jax.jit(lambda r,u:jax.jvp(fields,(r,),(u,))[1])(cell.coords,direction)
    step = 1e-4
    upper, lower = fields(cell.coords+step*direction), fields(cell.coords-step*direction)
    for derivative, plus, minus in zip(actual,upper,lower):
        np.testing.assert_allclose(derivative,(plus-minus)/(2*step),atol=2e-8,rtol=1e-7)
