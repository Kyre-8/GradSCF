"""Pure HF must not construct XC quadrature, including ghost calculations."""
from dataclasses import replace

import jax
import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.data.molecule import parse_molecule_spec

jax.config.update("jax_enable_x64", True)

WATER_DIMER = """
O -1.551007 -0.114520 0.000000
H -1.934259  0.762503 0.000000
H -0.599677  0.040712 0.000000
O  1.350625  0.111469 0.000000
H  1.680398 -0.373741 -0.758561
H  1.680398 -0.373741  0.758561
"""


def test_rhf_ghost_does_not_build_xc_quadrature(monkeypatch):
    from gradscf.scf.inputs import assembly

    def unused_grid(*args, **kwargs):
        raise AssertionError("Pure HF must not build XC quadrature.")

    monkeypatch.setattr(assembly, "build_molecular_grid_from_spec", unused_grid)
    spec = parse_molecule_spec(WATER_DIMER)
    charges = np.asarray(spec.charges).copy()
    charges[3:] = 0
    spec = replace(spec, charges=charges)
    hf = scf.RHF(gto.M(atom=spec, basis="sto-3g")).run()
    assert hf.converged
    assert hf._scf_inputs.coords.shape == (0, 3)
    assert hf._scf_inputs.grid_weights.shape == (0,)
    assert hf._scf_inputs.ao.shape == (0, 14)


@pytest.mark.parametrize("basis", ["sto-3g", "def2-svp"])
def test_rhf_ghost_fragment_matches_independent_pyscf(basis):
    pyscf = pytest.importorskip("pyscf")
    spec = parse_molecule_spec(WATER_DIMER)
    charges = np.asarray(spec.charges).copy()
    charges[3:] = 0
    spec = replace(spec, charges=charges)
    hf = scf.RHF(
        gto.M(atom=spec, basis=basis, cart=True),
        conv_tol=1e-12, conv_tol_density=1e-10,
        conv_tol_grad=1e-9, max_cycle=200,
    ).run()

    atom = [(s if z else "ghost-" + s, tuple(x))
            for s, z, x in zip(spec.symbols, spec.charges, spec.coords_bohr)]
    refmol = pyscf.gto.M(atom=atom, unit="Bohr", basis=basis,
                        cart=True, verbose=0)
    ref = pyscf.scf.RHF(refmol)
    ref.conv_tol, ref.conv_tol_grad, ref.max_cycle = 1e-12, 1e-9, 200
    reference_energy = ref.kernel()
    assert hf.converged and ref.converged and spec.nelectron == 10
    np.testing.assert_allclose(hf.e_tot, reference_energy, atol=1e-8, rtol=0)
    overlap = np.asarray(hf.scf_result.overlap_matrix)
    np.testing.assert_allclose(np.trace(hf.make_rdm1() @ overlap), 10,
                               atol=1e-8, rtol=0)
    np.testing.assert_allclose(hf.scf_result.nuclear_repulsion,
                               refmol.energy_nuc(), atol=1e-10, rtol=0)
