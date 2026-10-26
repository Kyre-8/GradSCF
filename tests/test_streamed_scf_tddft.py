"""Public native-direct SCF/full-TDDFT consumes the same streamed quadrature.

CPU float64; Cartesian STO-3G/6-31G; level-0 moving atom-centered grid. The complete
occupied/virtual spaces are retained.  The dense-grid path is an independent
storage oracle, while both paths use direct J/K and matrix-free Davidson.
"""

import numpy as np
import pytest

from gradscf import dft, gto, tdscf


@pytest.mark.parametrize("unrestricted", [False, True], ids=["water-rks", "heh-uks"])
def test_native_direct_pbe_full_tddft_streamed_grid_matches_dense(unrestricted, record_property):
    # HeH doublet at 1.6 bohr is the established nonzero-root UTDDFT oracle;
    # the OH/STO-3G dense baseline has an unconverged zero-frequency root.
    atom = ("He 0 0 0; H 0 0 .8466835374448" if unrestricted else
            "O 0 0 .117790; H 0 .755453 -.471161; H 0 -.755453 -.471161")
    basis = "6-31g" if unrestricted else "sto-3g"
    molecule = gto.M(atom=atom, basis=basis, unit="Angstrom", cart=True,
                     charge=0, spin=int(unrestricted))
    constructor = dft.UKS if unrestricted else dft.RKS
    results = []
    for block_size in (-1, 64):
        mf = constructor(molecule, xc="pbe", grids_level=0,
            integral_backend="native", jk_backend="direct",
            direct_scf_tol=0., direct_scf_workers=1,
            grid_block_size=block_size, grid_max_memory=64.,
            max_cycle=100, conv_tol=1e-11, conv_tol_density=1e-9,
            conv_tol_grad=1e-7, damp=.15).run()
        assert bool(mf.converged)
        reference = mf.to_reference()
        assert reference.direct_basis is not None
        assert reference.eri_pair_matrix is None
        assert reference.rep_tensor.size == 0
        if block_size > 0:
            assert reference.grid_ao is not None
            assert reference.grid_ao.block_size == block_size
            assert reference.ao.shape[0] == reference.ao_deriv1.shape[1] == 0
        else:
            assert reference.grid_ao is None
            assert reference.ao.shape[0] == reference.grid.weights.shape[0] > block_size
        driver = tdscf.TDDFT(mf, nstates=1, eigensolver="davidson",
            davidson_tol=1e-9, davidson_max_iter=100, davidson_max_subspace=16,
            response_kernel_options={"max_memory": 256., "integral_cache": "none",
                                     "direct_tol": 0., "direct_workers": 1,
                                     "grid_block_size": 64})
        excited = driver.kernel()
        assert bool(np.all(np.asarray(excited.converged)))
        fock = reference.fock_matrix if unrestricted else mf.scf_result.fock_matrix
        results.append((np.asarray(mf.e_tot), np.asarray(fock),
                        np.asarray(excited.excitation_energies)))
        for array in results[-1]:
            assert np.isfinite(array).all()
    for name, value, expected, atol in zip(
            ("total_energy", "fock", "excitation"), results[1], results[0],
            (2e-9, 5e-8, 2e-7)):
        record_property(f"{name}_max_error_ha", float(np.max(np.abs(value - expected))))
        np.testing.assert_allclose(value, expected, atol=atol, rtol=0.)
    record_property("basis", f"Cartesian {basis}")
    record_property("xc", "PBE")
    record_property("grid_level", 0)
    record_property("ao_block_size", 64)
