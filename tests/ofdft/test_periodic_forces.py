"""The shared geometry derivative driver also covers periodic OFDFT."""
from dataclasses import replace
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import ofdft
from gradscf.pbc import gto
from gradscf.scf import SCFDifferentiationConfig


@pytest.mark.parametrize('representation',['periodic','periodic_gaussian'])
def test_ofdft_force_and_strain_match_finite_differences(representation):
    cell=gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6.,unit='Bohr',mesh=(7,)*3)
    calc=ofdft.OFDFT(cell,representation=representation,xc=None,maxiter=500,tolerance=1e-8,
        differentiation=SCFDifferentiationConfig(tolerance=1e-10,max_iter=100))
    driver=calc.nuc_grad_method()
    gradient=driver.kernel()
    def energy(r,a):
        data=ofdft.inputs_from_cell(cell,representation=representation,lattice=a,
            parameters=replace(cell.parameters,centers=r,nuclear_coords=r))
        out=ofdft.run_ofdft(data,config=calc.config,kinetic=calc.kinetic)
        assert out.converged
        return out.total_energy
    step=1e-4
    fd=(energy(cell.coords.at[1,0].add(step),cell.lattice)-energy(cell.coords.at[1,0].add(-step),cell.lattice))/(2*step)
    np.testing.assert_allclose(gradient[1,0],fd,atol=2e-6,rtol=0)
    strain=jnp.array([[.2,.1,0.],[.1,-.1,.2],[0.,.2,.3]])
    plus=jnp.eye(3)+step*strain;minus=jnp.eye(3)-step*strain
    fd=(energy(cell.coords@plus.T,cell.lattice@plus.T)-energy(cell.coords@minus.T,cell.lattice@minus.T))/(2*step)
    np.testing.assert_allclose(jnp.sum(driver.stress()*strain)*cell.volume,fd,atol=3e-6,rtol=0)
