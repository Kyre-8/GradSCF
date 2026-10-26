"""Forward, non-DF DABNA-2 TDDFT with a level-3 molecular quadrature.

Use ``svwn``, ``pbe`` and ``pbe0`` for the LDA/GGA/global-hybrid comparison.
Each call converges its own XC ground state before using the complete OV space.
Run separate cases in fresh CPU processes when comparing runtime and peak RSS.
"""
from pathlib import Path
from time import perf_counter
import json

import h5py
import jax
import numpy as np

from gradscf import dft, gto, tdscf
from gradscf.data.hdf5_cache import write_restricted_molecule
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)


def run_case(geometry, xc, output_dir, *, nroots=20, max_memory=60000.,
             workers=16, init_density=None):
    """Converge direct RKS and full TDDFT; persist progress before long stages.

    ``geometry`` is an XYZ path in Angstrom. ``init_density`` may supply a
    compatible AO density as a guess; it never replaces the new XC SCF solve.
    ``max_memory`` is the response workspace budget in decimal MB, not an RSS
    limit. The external runner owns CPU affinity, RSS limits and timeouts.
    """
    geometry, output_dir = Path(geometry), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    status = dict(geometry=str(geometry), xc=xc, basis="def2-svp", cart=True,
                  grids_level=3, nroots=nroots, max_memory_mb=max_memory,
                  workers=workers, direct_tol=1e-12, integral_cache="none",
                  davidson_max_subspace=160, davidson_tol=1e-6,
                  grid_block_size=0, grid_max_memory_mb=None,
                  response_grid_block_size=8192)

    def save(phase, **values):
        status.update(values, phase=phase, elapsed_seconds=perf_counter()-started)
        (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print("DABNA_DFT_STATUS " + json.dumps(status), flush=True)

    save("molecule")
    try:
        lines = geometry.read_text().splitlines()
        atom = ";".join(line for line in lines[2:] if line.strip())
        mol = gto.M(atom=atom, basis="def2-svp", unit="Angstrom", cart=True,
                    charge=0, spin=0)
        save("scf")
        phase_started = perf_counter()
        mf = dft.RKS(mol, xc=xc, integral_backend="native", jk_backend="direct",
            execution_device="cpu", grids_level=3, direct_scf_tol=1e-12,
            grid_block_size=0, grid_max_memory=None, max_memory=max_memory,
            direct_scf_workers=workers, conv_tol=1e-9, conv_tol_density=1e-8,
            conv_tol_grad=1e-6, max_cycle=100, damp=.2, level_shift=.2,
            init_guess="hcore" if init_density is None else init_density).run()
        save("scf", scf_seconds=perf_counter()-phase_started,
             scf_converged=bool(mf.converged), scf_cycles=int(mf.cycles),
             scf_energy_hartree=float(mf.e_tot))
        if not mf.converged or not np.isfinite(mf.e_tot):
            raise RuntimeError("The requested XC ground-state SCF did not converge.")

        # A density-only restart is small and independent of the AO/grid cache.
        density_path = output_dir / "scf_density.npz"
        np.savez(density_path, density_matrix=np.asarray(mf.make_rdm1()))
        np.savez(output_dir / "scf_orbitals.npz", mo_coeff=np.asarray(mf.mo_coeff),
                 mo_occ=np.asarray(mf.mo_occ), mo_energy=np.asarray(mf.mo_energy))
        coefficients = np.asarray(mf.mo_coeff)
        nao, nmo = coefficients.shape
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
        nvir = nmo-nocc
        assert nmo == nao, "This benchmark retains the complete MO space"
        save("reference", density_checkpoint=str(density_path), nao=nao,
             nocc=nocc, nvir=nvir, transitions=nocc*nvir)
        phase_started = perf_counter()
        reference = mf.to_reference()
        jax.block_until_ready(reference.dipole_integrals)
        assert reference.direct_basis is not None
        for name in ("rep_tensor", "eri_pair_matrix", "df_factors",
                     "response_df_factors_j", "response_df_factors_k",
                     "eri_ovov", "eri_ovvo", "eri_oovv"):
            value = getattr(reference, name, None)
            assert value is None or value.size == 0, f"Unexpected integral array: {name}"
        assert reference.grid_ao is not None, "The large AO grid must stay streamed"
        with h5py.File(output_dir / "scf_reference.h5", "w") as archive:
            write_restricted_molecule(archive.create_group("reference"), reference)
            archive.attrs["xc"] = xc
        save("td", reference_seconds=perf_counter()-phase_started,
             grid_points=int(reference.grid.weights.size),
             ao_block_size=int(reference.grid_ao.block_size))

        return run_response(reference, xc, output_dir, nroots=nroots,
                            max_memory=max_memory, workers=workers, progress=save)
    except Exception as error:
        save("failed", failed_phase=status["phase"],
             error_type=type(error).__name__, error=str(error))
        raise


def run_response(reference, xc, output_dir, *, nroots=20, max_memory=60000.,
                 workers=16, progress=None):
    """Run full TDDFT on a converged reference, including an HDF5 restart.

    Explicit ``xc`` keeps the response functional when the numerical reference
    is loaded from disk. No SCF or grid is rebuilt in this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    status = dict(xc=xc, nroots=nroots, max_memory_mb=max_memory,
                  davidson_max_subspace=160, davidson_tol=1e-6,
                  response_grid_block_size=8192,
                  grid_points=int(reference.grid.weights.size),
                  nao=int(reference.mo_coeff.shape[-2]), nocc=int(reference.nocc))

    def save(phase, **values):
        if progress is not None:
            progress(phase, **values)
        else:
            status.update(values, phase=phase, elapsed_seconds=perf_counter()-started)
            (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print("DABNA_DFT_STATUS " + json.dumps(status), flush=True)

    save("td")
    td = tdscf.TDDFT(reference, xc_functional=xc, nstates=nroots,
        eigensolver="davidson", davidson_tol=1e-6, davidson_max_iter=512,
        davidson_max_subspace=160, max_memory=max_memory,
        response_kernel_options=dict(two_electron_mode="direct", integral_cache="none",
            direct_tol=1e-12, direct_workers=workers))
    result = td.kernel()
    energies = np.asarray(result.excitation_energies)
    save("td", td_seconds=perf_counter()-started, td_converged=bool(result.converged),
         excitation_energy_hartree=energies.tolist())
    if not bool(result.converged) or energies.shape != (nroots,) or not np.isfinite(energies).all():
        raise RuntimeError("Full TDDFT did not converge all requested roots.")

    save("oscillator_strength")
    phase_started = perf_counter()
    strengths = np.asarray(td.oscillator_strength())
    if strengths.shape != (nroots,) or not np.isfinite(strengths).all():
        raise RuntimeError("TDDFT oscillator strengths are incomplete or non-finite.")
    spectrum = dict(xc=xc, nroots=nroots, excitation_energy_hartree=energies.tolist(),
                    excitation_energy_ev=(energies*HARTREE_TO_EV).tolist(),
                    oscillator_strength=strengths.tolist())
    (output_dir / "spectrum.json").write_text(json.dumps(spectrum, indent=2) + "\n")
    np.savez(output_dir / "spectrum.npz", excitation_energy_hartree=energies,
             excitation_energy_ev=energies*HARTREE_TO_EV, oscillator_strength=strengths)
    x_amplitudes, y_amplitudes = td.xy
    np.savez(output_dir / "td_amplitudes.npz", x=np.asarray(x_amplitudes), y=np.asarray(y_amplitudes))
    save("complete", oscillator_strength_seconds=perf_counter()-phase_started,
         spectrum_json=str(output_dir / "spectrum.json"), spectrum_npz=str(output_dir / "spectrum.npz"))
    return spectrum


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run_case(here / "dabna2_monomer.xyz", "pbe",
             here.parents[2] / "artifacts" / "dabna2" / "monomer_pbe_level3")
