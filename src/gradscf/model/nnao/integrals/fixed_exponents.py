"""Fixed-exponent normalized contractions of primitive matrices and DF factors."""
from gradscf.integrals.basis.contraction import contraction_matrix
from gradscf.integrals.molecular.density_fitting import project_factors

def contracted_one_electron(layout, outputs, overlap, hcore):
    transform = contraction_matrix(layout.topology, layout.bind(outputs))
    return transform, transform.T @ overlap @ transform, transform.T @ hcore @ transform

def physical_inputs(layout, outputs, overlap, hcore, factors):
    transform, s, h = contracted_one_electron(layout, outputs, overlap, hcore)
    return s, h, project_factors(factors, transform)
