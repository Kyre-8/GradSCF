"""Physical boundaries, representation consistency, and public input adapters."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_gaussian_and_grid_energies_match_for_same_amplitude():
    from gradscf import integrals
    from gradscf.ofdft import periodic_inputs, periodic_gaussian_inputs, energy_components, KineticFunctional
    lattice = jnp.eye(3)*6.
    data = periodic_inputs(lattice,(9,9,9),nelectron=2.)
    top, pars = integrals.prepare_basis('H 1 1 1; H 2 1 1','sto-3g',unit='Bohr')
    gaussian = periodic_gaussian_inputs(data,top,pars)
    c = jnp.array([.7,.9]); c *= jnp.sqrt(2/(c@gaussian.overlap@c))
    kw = dict(kinetic=KineticFunctional('wt'),kinetic_params={},xc=None)
    reference = energy_components(gaussian.ao@c,data,**kw)
    projected = energy_components(c,gaussian,**kw)
    np.testing.assert_allclose(projected,reference,rtol=1e-11,atol=1e-11)


def test_lattice_gradient_includes_volume_metric_and_fft():
    from gradscf.ofdft import periodic_inputs, run_ofdft, OFDFTConfig, KineticFunctional
    cfg = OFDFTConfig(xc=None,tolerance=1e-9)
    def energy(length):
        data = periodic_inputs(jnp.eye(3)*length,(3,3,3),nelectron=2.)
        return run_ofdft(data,config=cfg,kinetic=KineticFunctional('wt')).total_energy
    value, derivative = jax.jit(jax.value_and_grad(energy))(5.)
    np.testing.assert_allclose(derivative,-2*value/5,rtol=1e-10)
    np.testing.assert_allclose(derivative,(energy(5.0001)-energy(4.9999))/.0002,rtol=1e-7)


def test_explicit_and_implicit_density_gradients_agree():
    from gradscf.ofdft import periodic_inputs, run_ofdft, OFDFTConfig
    from gradscf.scf import SCFDifferentiationConfig
    data = periodic_inputs(jnp.eye(3)*5.,(3,3,3),nelectron=2.)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    def loss(p, mode):
        cfg = OFDFTConfig(xc=None,maxiter=100,tolerance=1e-10,
                         differentiation=SCFDifferentiationConfig(mode=mode))
        out = run_ofdft(replace(data,external_potential=p*probe),config=cfg)
        return jnp.sum(data.weights*probe*out.density)
    implicit = jax.grad(lambda p:loss(p,'implicit'))(.1)
    explicit = jax.grad(lambda p:loss(p,'explicit'))(.1)
    np.testing.assert_allclose(explicit,implicit,rtol=1e-5,atol=1e-7)


def test_xc_matches_existing_jax_xc_adapter():
    pytest.importorskip("jax_xc", reason="Optional jax-xc runtime required")
    from gradscf.ofdft import periodic_inputs, energy_components, KineticFunctional
    from gradscf.dft.xc import make_classic_xc_functional, RestrictedFeatureBundle
    from gradscf.dft.xc import restricted_feature_bundle_from_rho_grad_tau
    data = periodic_inputs(jnp.eye(3)*5.,(3,3,3),nelectron=2.)
    phi = jnp.ones(27)*jnp.sqrt(2/125)
    for xc in ('svwn','pbe'):
        out = energy_components(phi,data,kinetic=KineticFunctional(),kinetic_params={},xc=xc)
        features = restricted_feature_bundle_from_rho_grad_tau(phi**2)
        expected = jnp.sum(data.weights*make_classic_xc_functional(xc).energy_density(features))
        np.testing.assert_allclose(out.xc,expected,rtol=1e-12)


def test_reject_orbital_dependent_xc_and_nonlocal_pseudopotentials():
    from gradscf.ofdft import periodic_inputs, run_ofdft, OFDFTConfig, inputs_from_cell
    from gradscf.pbc import gto
    data = periodic_inputs(jnp.eye(3)*5.,(3,3,3),nelectron=2.)
    with pytest.raises(ValueError,match='LDA/GGA'):
        run_ofdft(data,config=OFDFTConfig(xc='pbe0'))
    cell = gto.M(atom='C 0 0 0',a=np.eye(3)*5,mesh=(5,5,5))
    with pytest.raises(ValueError,match='nonlocal'):
        inputs_from_cell(cell)


@pytest.mark.parametrize("xc", [None, "svwn", "pbe"])
def test_gaussian_molecular_facade(xc):
    if xc is not None:
        pytest.importorskip("jax_xc", reason="Optional jax-xc runtime required")
    from gradscf import gto, ofdft
    mol = gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g')
    calc = ofdft.OFDFT(mol,xc=xc,grids_level=1,tolerance=1e-8).run()
    assert calc.converged
    np.testing.assert_allclose(calc.result.electron_number,2.,atol=1e-10)
    assert np.isfinite(calc.e_tot)
    assert calc.result.components.kinetic > 0


def test_molecular_coordinate_and_overlap_response():
    from gradscf import gto, ofdft
    mol = gto.M(atom='H 0 0 0; H 0 0 .8',basis='sto-3g')
    coords = jnp.asarray(mol.to_spec().coords_bohr)
    cfg = ofdft.OFDFTConfig(xc=None,tolerance=1e-9)
    def energy(distance):
        inputs = ofdft.gaussian_inputs(mol,coordinates=coords.at[1,2].set(distance),grids_level=0)
        return ofdft.run_ofdft(inputs,config=cfg).total_energy
    distance = coords[1,2]
    derivative = jax.jit(jax.grad(energy))(distance)
    fd = (energy(distance+1e-4)-energy(distance-1e-4))/2e-4
    np.testing.assert_allclose(derivative,fd,rtol=5e-5,atol=1e-6)


def test_periodic_gaussian_solution_and_response():
    from gradscf import integrals, ofdft
    top, pars = integrals.prepare_basis('H 1 1 1; H 3 1 1','sto-3g',unit='Bohr')
    data = ofdft.periodic_inputs(jnp.eye(3)*5.,(7,7,7),nelectron=2.)
    data = ofdft.periodic_gaussian_inputs(data,top,pars)
    probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
    cfg = ofdft.OFDFTConfig(xc=None,tolerance=1e-9)
    def loss(t):
        result = ofdft.run_ofdft(replace(data,external_potential=t*probe),config=cfg)
        return jnp.sum(data.weights*probe*result.density)
    result = ofdft.run_ofdft(data,config=cfg)
    assert result.converged
    np.testing.assert_allclose(result.grid_electron_number,2.,atol=1e-10)
    ad = jax.jit(jax.grad(loss))(.2)
    fd = (loss(.20001)-loss(.19999))/2e-5
    np.testing.assert_allclose(ad,fd,rtol=2e-5,atol=1e-7)


def test_local_gth_periodic_adapter():
    from gradscf.pbc import gto
    from gradscf import ofdft
    cell = gto.M(atom='H 1 1 1; H 2 1 1',a=np.eye(3)*5.,mesh=(7,7,7),unit='Bohr')
    calc = ofdft.OFDFT(cell,xc=None,tolerance=1e-7,maxiter=500).run()
    assert calc.converged
    np.testing.assert_allclose(calc.result.electron_number,2.,atol=1e-12)
    assert np.isfinite(calc.result.components.external)


def test_gaussian_rejects_broadcastable_wrong_potential():
    from gradscf.ofdft import OFDFTInputs, OFDFTConfig, run_ofdft
    data = OFDFTInputs('gaussian',jnp.ones(2),jnp.zeros((2,3)),jnp.array(2.),jnp.array([-1.,-2.]),
        overlap=jnp.eye(2),kinetic_matrix=jnp.eye(2),ao=jnp.eye(2),
        ao_gradient=jnp.zeros((2,2,3)),eri=jnp.zeros((2,2,2,2)))
    with pytest.raises(ValueError,match='external'):
        run_ofdft(data,config=OFDFTConfig(xc=None))


def test_molecular_coulomb_layouts_and_factor_response_agree():
    from gradscf.ofdft import OFDFTInputs, KineticFunctional, energy_components
    factors = jnp.array([[[.7,.1],[.1,.5]], [[.3,-.2],[-.2,.8]]])
    eri = jnp.einsum('Qpq,Qrs->pqrs',factors,factors)
    data = OFDFTInputs('gaussian',jnp.ones(2),jnp.zeros((2,3)),jnp.array(2.),jnp.zeros((2,2)),
        overlap=jnp.eye(2),kinetic_matrix=jnp.eye(2),ao=jnp.eye(2),
        ao_gradient=jnp.zeros((2,2,3)),eri=eri)
    rows,cols=np.tril_indices(2)
    packed=eri[rows[:,None],cols[:,None],rows[None,:],cols[None,:]]
    variants=[data,replace(data,eri=packed),replace(data,eri=None,df_factors=factors)]
    q=jnp.array([.8,1.1])
    def energy(q,inputs):
        return energy_components(q,inputs,kinetic=KineticFunctional(),kinetic_params={}).hartree
    expected=.5*jnp.sum(jnp.einsum('p,Qpq,q->Q',q,factors,q)**2)
    for inputs in variants:
        value,grad=jax.value_and_grad(energy)(q,inputs)
        np.testing.assert_allclose(value,expected,atol=1e-12)
        np.testing.assert_allclose(grad,jax.grad(lambda x:energy(x,data))(q),atol=1e-12)
