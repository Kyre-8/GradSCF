"""Independent DFTpy version of the ATLAS TF+lambda*vW solid benchmark.

Uses DFTpy's native PZ-LDA implementation, not GradSCF XC or jax-xc.
Run with DFTpy on PYTHONPATH; no PySCF or GradSCF calculation supplies its state.
"""
import numpy as np
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.ions import Ions
from dftpy.functional.pseudo import LocalPseudo
from dftpy.functional.hartree import Hartree
from dftpy.functional.kedf.tf import TF
from dftpy.functional.kedf.vw import vW
from dftpy.functional.xc.semilocal_xc import LDA
from dftpy.optimization import Optimization
from dftpy.ewald import ewald
from atlas_inputs import crystal, pseudo_path


def run_reference(atoms, mesh, weight=1., kinetic="tfvw"):
    ions=Ions.from_ase(atoms)
    symbol=atoms.get_chemical_symbols()[0]
    grid=DirectGrid(lattice=ions.cell,nr=mesh,full=True)
    pseudo=LocalPseudo(grid=grid,ions=ions,PP_list={symbol:str(pseudo_path(symbol))},PME=False)
    hartree=Hartree()
    if kinetic not in ('tfvw','wgc'):
        raise ValueError('Reference KEDF must be tfvw or wgc.')
    nonlocal_wgc=None
    if kinetic=='wgc':
        if weight!=1.:
            raise ValueError('Standard WGC uses full vW.')
        from wgc_dftpy import WGCReference
        vectors=np.asarray(grid.get_reciprocal().g).reshape(3,-1).T
        nonlocal_wgc=WGCReference(vectors,mesh,grid.volume,ions.get_ncharges()/grid.volume)
    enuc=ewald(precision=1e-12,ions=ions,grid=grid,PME=False).energy
    def evaluate(rho,calcType=('E','V'),**kwargs):
        del calcType,kwargs
        result=TF(rho)+vW(rho,y=weight)+hartree(rho)+LDA(rho)+pseudo(rho)
        if nonlocal_wgc is not None:
            result=result+nonlocal_wgc(rho)
        result.energy+=enuc
        return result
    rho=DirectField(grid=grid,griddata_3d=np.full(mesh,ions.get_ncharges()/grid.volume))
    optimizer=Optimization(optimization_method='LBFGS',EnergyEvaluator=evaluate,
        optimization_options={'maxiter':2000,'econv':1e-12,'ncheck':3})
    rho=optimizer.optimize_rho(guess_rho=rho)
    result=evaluate(rho)
    return rho,result,optimizer,pseudo,enuc


def main():
    symbol='Al'
    atoms,mesh=crystal(symbol,.18)
    for weight in (1.,1/5,1/9):
        rho,result,optimizer,_,_=run_reference(atoms,mesh,weight)
        print(symbol,'mesh',mesh,'lambda',weight)
        print('  Energy / Hartree per atom:',float(result.energy)/len(atoms))
        print('  Electron number:',float(rho.integral()))
        if optimizer.converged!=0:
            raise RuntimeError('DFTpy optimizer did not converge')


if __name__=='__main__':
    main()


# Measured output, target spacing 0.18 Angstrom, CPU float64:
# system   lambda    energy / Hartree per atom
# Al       1        -2.072264566660261
# Al       1/5      -2.152798390662868
# Al       1/9      -2.187611287981262
# Mg       1        -0.899530058754829
# Mg       1/5      -0.933515515461072
# Mg       1/9      -0.948322546023274
# These are TF+lambda*vW results, not the paper's WGC Table 1.
