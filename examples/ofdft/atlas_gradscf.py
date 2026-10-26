"""ATLAS-style fcc Al/hcp Mg: GradSCF TF+lambda*vW, OEPP, PZ-LDA.

Edit the plain settings below. All coordinates/energies sent to GradSCF are in
atomic units. See ATLAS_REPRODUCTION.md for structural and provenance limits.
Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/ofdft/atlas_gradscf.py
"""
import jax
jax.config.update('jax_enable_x64', True)
from gradscf import ofdft
from atlas_inputs import crystal, gaussian_free_inputs, pz_lda_energy

symbol = 'Al'                 # 'Mg' uses an ideal-c/a hcp two-atom cell
spacing = .18                 # maximum lattice-vector grid spacing, Angstrom
atoms, mesh = crystal(symbol, spacing)
inputs = gaussian_free_inputs(atoms, mesh)
config = ofdft.OFDFTConfig(xc=None, maxiter=1000, tolerance=1e-8)

for weight in (1., 1/5, 1/9):
    calculation = ofdft.OFDFT(inputs, xc=None, kinetic='tfvw',
        kinetic_params={'vw':weight}, xc_energy_fn=pz_lda_energy,
        maxiter=config.maxiter, tolerance=config.tolerance).run()
    print(symbol, 'mesh', mesh, 'lambda', weight)
    print('  Energy / Hartree per atom:', float(calculation.e_tot)/len(atoms))
    print('  Electron number:', float(calculation.result.electron_number))
    print('  Residual:', float(calculation.result.residual_norm))
    if not calculation.converged:
        raise RuntimeError('OFDFT did not reach the requested stationarity')


# Measured output, target spacing 0.18 Angstrom, CPU float64:
# system   lambda    energy / Hartree per atom
# Al       1        -2.072264566660577
# Al       1/5      -2.152798390663095
# Al       1/9      -2.187611287981495
# Mg       1        -0.899530058755158
# Mg       1/5      -0.933515515461197
# Mg       1/9      -0.948322546023391
# These are TF+lambda*vW results, not the paper's WGC Table 1.
