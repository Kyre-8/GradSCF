"""Assemble a MACE-conditioned H-X basis and compute its native overlap.

Requires the optional MACE stack and a built GradSCF native library.
This is a basis-assembly demonstration, not SCF training.
"""
import argparse
import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
from flax import nnx
from gradscf import integrals
from gradscf.data.molecule import atomic_number
from gradscf.model.nnao import MACEBasisModel, build_graph, prepare_direct_basis, supported_elements


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--element',choices=supported_elements(),default='F')
    parser.add_argument('--random-head',action='store_true',help='Use a small random head instead of the exact reference basis')
    args=parser.parse_args()
    atom=[('H',(0.,0.,0.)),(args.element,(0.,0.,1.6))]
    numbers=[1,atomic_number(args.element)]
    layout=prepare_direct_basis(atom,unit='Angstrom')
    model=MACEBasisModel(elements=tuple(sorted(set(numbers))),channels=8,
                         num_interactions=2,max_ell=1,zero_init=not args.random_head,rngs=nnx.Rngs(0))
    graph=build_graph(numbers,[xyz for _,xyz in atom],element_order=model.elements)
    parameters=model.assemble(layout,graph)
    overlap=integrals.make_plan(layout.topology).evaluate('overlap',parameters)
    print('Elements:',layout.symbols,'AO count:',layout.topology.nao)
    for i,(owner,role,l,c) in enumerate(zip(layout.shell_atoms,layout.roles,layout.topology.angular_momenta,parameters.coefficients)):
        print(f'shell={i} atom={owner} role={role} l={l} coefficients={np.asarray(c[:,0])}')
    print('Overlap minimum eigenvalue:',float(np.linalg.eigvalsh(np.asarray(overlap)).min()))


if __name__=='__main__':
    main()
