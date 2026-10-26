"""Independent WGC99 NumPy adapter with DFTpy density optimization.

The adapter supplies only the WGC nonlocal term. DFTpy supplies TF, full vW,
PZ-LDA, Hartree, OEPP, Ewald, and its optimizer independently of GradSCF.
"""
from atlas_inputs import crystal
from atlas_dftpy import run_reference


for symbol in ('Al', 'Mg'):
    atoms, mesh = crystal(symbol, spacing=.18)
    rho, result, optimizer, _, _ = run_reference(atoms, mesh, kinetic='wgc')
    print(symbol, 'mesh', mesh)
    print('  Energy / Hartree per atom:', float(result.energy) / len(atoms))
    print('  Electrons:', float(rho.integral()))
    if optimizer.converged != 0:
        raise RuntimeError('DFTpy WGC density did not converge')

# CPU float64, target spacing 0.18 Angstrom:
# Al   -2.087391936817966 Ha/atom
# Mg   -0.905887662699798 Ha/atom
