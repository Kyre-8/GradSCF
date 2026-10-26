"""Independent DFTpy comparison of KEDF terms and self-consistent OFDFT.

Run from the repository root with:
  PYTHONPATH=src:/path/to/DFTpy/src JAX_PLATFORMS=cpu python examples/ofdft/compare_dftpy.py
Reference source: Quantum-MultiScale/DFTpy dev, fbc47b4e (2026-09-27 retrieval).
Both codes use pure Dirac LDA exchange, no correlation, local external potential,
and a neutralizing background. DFTpy supplies independent TF/vW/WT and Hartree
energies/potentials, and its own optimizer. No GradSCF derivative is passed to it.
All units are atomic; potentials are compared in the fixed-N tangent space.
"""
import json
import platform
from time import perf_counter
import dftpy

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import ofdft
from gradscf.ofdft.kinetic import thomas_fermi, wang_teter
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.functional.kedf.tf import TF
from dftpy.functional.kedf.vw import vW
from dftpy.functional.kedf.wt import WT
from dftpy.functional.hartree import Hartree
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.optimization import Optimization
from lda_exchange import exchange_energy, model_inputs


def run_comparison(mesh=(7,7,7)):
    started = perf_counter()
    data = model_inputs(mesh)
    grid = DirectGrid(lattice=np.asarray(data.lattice),nr=mesh,full=True)
    dvol = float(data.weights[0])
    field = lambda rho: DirectField(grid=grid,griddata_3d=np.asarray(rho).reshape(mesh).copy())
    reference_terms = {'tf':TF,'vw':vW,'wt_nonlocal':WT}
    phase = 2*jnp.pi*data.coordinates/6.
    rho = (8/216)*(1+.2*jnp.cos(phase[:,0])+.15*jnp.sin(phase[:,1])+.1*jnp.cos(phase[:,2]))
    direction = .01*(jnp.cos(phase[:,0])+jnp.sin(phase[:,1])*.7)
    direction -= jnp.mean(direction)
    project = lambda v: np.asarray(v)-np.mean(v)
    step = 1e-5
    fixed = {}
    for name, reference in reference_terms.items():
        def energy(n):
            f = ofdft.density_features(jnp.sqrt(n),data)
            if name=='tf':
                return thomas_fermi(n,data.weights)
            if name=='vw':
                return f.vw_energy
            return wang_teter(f)
        evaluate = jax.jit(jax.value_and_grad(energy))
        value, derivative = evaluate(rho)
        potential = derivative/dvol  # JAX discrete derivative -> functional derivative
        hessian_vector = jax.jvp(jax.grad(energy),(rho,),(direction,))[1]/dvol
        ref = reference(field(rho),calcType={'E','V'})
        plus = reference(field(rho+step*direction),calcType={'V'}).potential
        minus = reference(field(rho-step*direction),calcType={'V'}).potential
        ref_hvp = (np.asarray(plus).reshape(-1)-np.asarray(minus).reshape(-1))/(2*step)
        metrics = dict(energy_error=abs(float(value)-float(ref.energy)),
            potential_error=float(np.max(np.abs(project(potential)-project(np.asarray(ref.potential).reshape(-1))))),
            hvp_error=float(np.max(np.abs(project(hessian_vector)-project(ref_hvp)))))
        fixed[name] = metrics
        assert metrics['energy_error'] < 1e-9, (name,metrics)
        assert metrics['potential_error'] < 2e-8, (name,metrics)
        assert metrics['hvp_error'] < 2e-7, (name,metrics)

    # WT's n0 is the cell average. DFTpy's analytic potential holds n0 fixed,
    # while AD differentiates it. Their difference is a constant chemical-
    # potential gauge at fixed N. Projecting out the mean is intentional.
    # Use a zero-integral HVP direction for the same reason.
    hartree = Hartree()
    cx = .75*(3/np.pi)**(1/3)
    ext = field(data.external_potential)
    def reference_energy(name):
        def evaluate(n, calcType=('E','V'), **kwargs):
            del kwargs
            tf = TF(n,calcType={'E','V'})
            vw = vW(n,calcType={'E','V'})
            nonlocal_part = WT(n,calcType={'E','V'}) if name=='wt' else None
            hart = hartree(n,calcType={'E','V'})
            rho_array = np.asarray(n)
            ex = -cx*np.sum(rho_array**(4/3))*dvol
            vx = -(4/3)*cx*rho_array**(1/3)
            energy = tf.energy+vw.energy+hart.energy+ex+float((ext*n).integral())
            potential = tf.potential+vw.potential+hart.potential+ext+vx
            if nonlocal_part is not None:
                energy += nonlocal_part.energy
                potential += nonlocal_part.potential
            return FunctionalOutput(name='TFvW/WT+Hartree+Dirac+external',energy=energy,potential=potential)
        return evaluate

    optimized = {}
    for name in ('tfvw','wt'):
        kinetic = ofdft.KineticFunctional(name)
        config = ofdft.OFDFTConfig(xc=None,tolerance=1e-9,maxiter=700)
        result = ofdft.run_ofdft(data,kinetic=kinetic,xc_energy_fn=exchange_energy,config=config)
        assert result.converged, result.residual_norm
        evaluate_reference = reference_energy(name)
        optimizer = Optimization(optimization_method='LBFGS',EnergyEvaluator=evaluate_reference,
            optimization_options={'maxiter':1000,'econv':1e-13,'ncheck':3})
        # DFTpy prints its own iteration history and uses its own line search.
        ref_density = optimizer.optimize_rho(guess_rho=field(jnp.full_like(rho,8/216)))
        ref_result = evaluate_reference(ref_density)
        # Independently evaluate the constrained stationarity residual using
        # GradSCF's energy; do not trust energy-change stopping alone.
        phi_ref = jnp.sqrt(jnp.asarray(ref_density).reshape(-1))
        x_ref = phi_ref*jnp.sqrt(data.weights/8.)
        def raw_energy(x):
            q = x*jnp.sqrt(8/data.weights)
            return ofdft.energy_components(q,data,kinetic=kinetic,kinetic_params={},
                xc_energy_fn=exchange_energy,xc_params={}).total
        grad_ref = jax.grad(raw_energy)(x_ref)
        residual_ref = float(jnp.linalg.norm(grad_ref-x_ref*jnp.vdot(x_ref,grad_ref)))
        reference_number = float(ref_density.integral())
        density_error = float(jnp.sqrt(jnp.sum(data.weights*(result.density-phi_ref**2)**2)/
                                       jnp.sum(data.weights*phi_ref**4)))
        row = dict(gradscf_energy=float(result.total_energy),dftpy_energy=float(ref_result.energy),
            energy_error=abs(float(result.total_energy)-float(ref_result.energy)),
            relative_density_error=density_error,gradscf_residual=float(result.residual_norm),
            dftpy_residual=residual_ref,dftpy_convergence_code=int(optimizer.converged),
            gradscf_electron_number=float(result.electron_number),dftpy_electron_number=reference_number)
        optimized[name] = row
        if (row['energy_error']>=1e-8 or density_error>=2e-5 or residual_ref>=2e-6 or
                abs(reference_number-8)>1e-10 or optimizer.converged!=0):
            raise AssertionError(row)

    report = dict(mesh=list(mesh),lattice_bohr=np.asarray(data.lattice).tolist(),nelectron=8,
        xc='Dirac LDA exchange only',kinetic_fixed_density=fixed,self_consistent=optimized,
        backend=jax.default_backend(),machine=platform.machine(),jax_version=jax.__version__,
        dftpy_version=dftpy.__version__, dftpy_source=dftpy.__file__,
        reference_target_commit='fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0',
        seconds=perf_counter()-started)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    run_comparison()


# Reference output: CPU arm64, JAX 0.8.1, float64, DFTpy fbc47b4e.
# Grid    KEDF   |delta E| / Ha    relative density L2 error
# 5^3     tfvw    1.332e-15          1.205e-08
# 5^3     wt      1.110e-16          4.142e-09
# 7^3     tfvw    6.439e-15          2.503e-08
# 7^3     wt      3.275e-15          1.375e-08
# 9^3     tfvw    9.548e-15          2.987e-08
# 9^3     wt      1.515e-14          4.478e-08
# Max fixed-density discrepancies over all three grids:
# term       energy / Ha     projected potential   projected HVP
# tf         2.220e-15       3.331e-16             1.016e-11
# vw         4.163e-17       1.499e-15             2.912e-10
# wt_nonlocal 4.163e-17       4.302e-16             2.401e-11
