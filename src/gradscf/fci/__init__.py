"""Differentiable FCI in independent alpha/beta occupation-string spaces."""
from .cistring import FCISpace, make_fci_space, make_strings
from .hamiltonian import (
    contract_1e, contract_2e, absorb_h1e, contract_hamiltonian, make_hdiag,
    build_hamiltonian, energy,
)
from .rdm import (
    make_rdm1, make_rdm2, make_rdm12, make_rdm1s, make_rdm12s,
    trans_rdm1, trans_rdm12, trans_rdm1s, trans_rdm12s, spin_square,
)
from .solver import FCIResult, solve_fci

__all__ = [
    'FCISpace','make_fci_space','make_strings','contract_1e','contract_2e',
    'absorb_h1e','contract_hamiltonian','make_hdiag','build_hamiltonian','energy',
    'make_rdm1','make_rdm2','make_rdm12','make_rdm1s','make_rdm12s',
    'trans_rdm1','trans_rdm12','trans_rdm1s','trans_rdm12s','spin_square',
    'FCIResult','solve_fci',
]
from .reference import FCIReference, fold_core
from .api import FCI, FCISolver, kernel
__all__ += ['FCIReference','fold_core','FCI','FCISolver','kernel']
