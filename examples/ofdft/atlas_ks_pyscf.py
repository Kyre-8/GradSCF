"""Independent Gaussian KS densities for the shared ATLAS Al/Mg OEPP inputs.

Run from the repository: OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=src \\
    python examples/ofdft/atlas_ks_pyscf.py

This is PySCF KRKS, not CASTEP or a reproduction of its plane-wave reference.
The hcore contains only analytic kinetic energy and the SAME tabulated local
OEPP as the OF examples. No GTH/nonlocal pseudopotential is used. PySCF supplies
FFT Hartree and LibXC unpolarized LDA_X + LDA_C_PZ independently of GradSCF.

calculate() defaults are a small demonstration. The main block reproduces the
recorded basis-level-3 Al k10 / Mg k8 reference (roughly 15 minutes on M4 Pro).
These remain finite-basis, finite-k references. Converge
spacing, kmesh, basis_level, and sigma (Hartree) independently before comparing
to Fig. 4. basis_level=1,2,3,4 adds radial and angular flexibility. For metals,
reduce sigma while increasing kmesh. Compare density and zero-T extrapolated
energy, not just the finite-temperature SCF stopping criterion. Overcomplete
Gaussian directions with overlap eigenvalues below 1e-7 are removed explicitly.
"""
import json
import platform
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, xlogy
import pyscf
from pyscf import lib
from pyscf.scf.addons import remove_linear_dep_
from pyscf.scf import hf as molecular_hf
from pyscf.gto import CHARGE_OF
from pyscf.pbc import dft, gto, scf

from atlas_inputs import VALENCE, crystal, gaussian_free_inputs, provenance


def gaussian_basis(level):
    """Uncontracted even-tempered, core-free Gaussians; exponents in Bohr^-2."""
    if level not in (1, 2, 3, 4):
        raise ValueError('basis_level must be 1, 2, 3, or 4.')
    # Tight core orbitals are unnecessary for a smooth two/three-valence OEPP.
    radial = np.geomspace(.08, 1.25, 3 + level)
    basis = [[l, [float(a), 1.]] for l in (0, 1) for a in radial]
    basis += [[2, [float(a), 1.]] for a in np.geomspace(.12, .6, 1 + level)]
    if level >= 2:
        basis += [[3, [float(a), 1.]] for a in np.geomspace(.15, .5, level - 1)]
    if level == 4:
        basis += [[4, [float(a), 1.]] for a in (.2, .5)]
    return basis


def fermi_occupations(energies, nelectron, sigma):
    """Restricted occupations with exactly N electrons, including odd N/cell."""
    nk = len(energies)
    sizes = [len(e) for e in energies]
    energies = np.concatenate(energies)
    if sigma <= 0:
        raise ValueError('Use a positive smearing width in Hartree.')
    target = nelectron * nk / 2
    mu = brentq(lambda x: expit((x - energies) / sigma).sum() - target,
                energies.min() - 100 * sigma, energies.max() + 100 * sigma,
                xtol=1e-14)
    fractions = expit((mu - energies) / sigma)
    entropy = -2 * np.sum(xlogy(fractions, fractions)
                           + xlogy(1 - fractions, 1 - fractions)) / nk
    occupations = np.split(2 * fractions, np.cumsum(sizes)[:-1])
    return occupations, float(entropy), float(mu)


def calculate(symbol, spacing=.25, kmesh=(2, 2, 2), basis_level=1, sigma=.01,
              max_cycle=80, verbose=3):
    start = perf_counter()
    atoms, mesh = crystal(symbol, spacing)
    inputs = gaussian_free_inputs(atoms, mesh)
    nelectron = len(atoms) * VALENCE[symbol]
    cell = gto.Cell()
    cell.atom = list(zip(atoms.get_chemical_symbols(), atoms.positions))
    cell.a = np.asarray(atoms.cell)
    cell.unit = 'Angstrom'
    cell.basis = {symbol: gaussian_basis(basis_level)}
    cell.mesh = mesh
    cell.precision = 1e-9
    cell.spin = nelectron % 2  # Build parity only; occupations below are unpolarized.
    cell.verbose = verbose
    cell.build()
    # Keep element labels for basis construction, but make effective ions and
    # electron bookkeeping valence-only. No PySCF pseudopotential is attached.
    cell._atm[:, CHARGE_OF] = VALENCE[symbol]
    cell.nelectron = nelectron
    cell.enuc = float(inputs.nuclear_repulsion)
    kpts = cell.make_kpts(kmesh)
    coords = cell.get_uniform_grids(mesh)
    # PySCF wraps coordinates around the origin; compare modulo the lattice.
    shifts = (coords - np.asarray(inputs.coordinates)) @ np.linalg.inv(cell.lattice_vectors())
    if not np.allclose(shifts, np.rint(shifts), atol=1e-8):
        raise RuntimeError('KS and OF integration grids do not agree.')
    weight = cell.vol / len(coords)
    ao_k = dft.numint.eval_ao_kpts(cell, coords, kpts=kpts)
    potential = np.asarray(inputs.external_potential)
    kinetic = np.asarray(cell.pbc_intor('int1e_kin', hermi=1, kpts=kpts))
    overlap = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
    hcore = kinetic + np.asarray([a.conj().T @ (potential[:, None] * a) * weight
                                 for a in ao_k])
    numerical_overlap = np.asarray([a.conj().T @ a * weight for a in ao_k])
    mf = dft.KRKS(cell, kpts=kpts, xc='LDA_X,LDA_C_PZ')
    mf.grids = dft.gen_grid.UniformGrids(cell)
    mf.grids.mesh = mesh
    mf.with_df.mesh = mesh
    mf = scf.addons.smearing(mf, sigma=sigma, method='fermi')
    mf.get_hcore = lambda cell=None, kpts=None: hcore

    def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
        del mo_coeff_kpts
        energies = mf.mo_energy if mo_energy_kpts is None else mo_energy_kpts
        occ, mf.entropy, _ = fermi_occupations(
            energies, nelectron, sigma)
        return occ

    # PySCF's default KRHF smearing rounds odd total electron counts upward.
    # The explicit N/2 occupation solve also makes Gamma-only Al valid.
    mf.get_occ = get_occ
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    mf.max_cycle = max_cycle
    modern_overlap = hasattr(mf, 'check_linear_dependency')
    context = (lib.temporary_env(molecular_hf, remove_overlap_zero_eigenvalue=True,
                               overlap_zero_eigenvalue_threshold=1e-7)
               if modern_overlap else nullcontext())
    with context:
        if modern_overlap:
            # New PySCF pads k-dependent retained ranks only on its x= path.
            orthogonalizers = mf.check_linear_dependency(overlap)
            energies, coefficients = mf.eig(hcore, overlap, x=orthogonalizers)
        else:
            remove_linear_dep_(mf, threshold=1e-7, lindep=1e-7)
            energies, coefficients = mf.eig(hcore, overlap)
        dm0 = mf.make_rdm1(coefficients, mf.get_occ(energies))
        mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError('KS did not reach the requested SCF tolerances.')
    dm = np.asarray(mf.make_rdm1())
    rho = sum(np.einsum('gi,ij,gj->g', a, d, a.conj()).real
              for a, d in zip(ao_k, dm)) / len(kpts)
    trace_electrons = np.einsum('kij,kji->', dm, overlap).real / len(kpts)
    info = dict(symbol=symbol, code='PySCF Gaussian KRKS', pyscf_version=pyscf.__version__,
                backend='CPU', platform=platform.platform(), density_dtype=str(rho.dtype),
                xc='LDA_X,LDA_C_PZ', pseudopotential=provenance(symbol),
                spacing_angstrom=spacing, mesh=list(mesh), kmesh=list(kmesh),
                basis_level=basis_level, nao=cell.nao_nr(), sigma_hartree=sigma,
                overlap_cutoff=1e-7,
                converged=bool(mf.converged), energy_hartree=float(mf.e_tot),
                free_energy_hartree=float(mf.e_free),
                extrapolated_energy_hartree=float(mf.e_zero),
                ewald_hartree=float(cell.enuc), ewald_pyscf_hartree=float(cell.ewald()),
                nelectron=nelectron,
                orbital_electrons=float(trace_electrons),
                grid_electrons=float(rho.sum() * weight),
                density_min_bohr3=float(rho.min()), density_max_bohr3=float(rho.max()),
                overlap_min_eigenvalue=float(np.linalg.eigvalsh(overlap).min()),
                overlap_grid_max_error=float(np.max(abs(numerical_overlap - overlap))),
                elapsed_seconds=perf_counter() - start,
                warning='Finite Gaussian basis, k sampling and smearing; not CASTEP or converged Fig. 4.')
    if not np.isfinite(rho).all() or abs(trace_electrons - nelectron) > 1e-8:
        raise RuntimeError('Invalid KS density/electron count.')
    return info, rho.reshape(mesh)


if __name__ == '__main__':
    lib.num_threads(1)
    results, densities = [], {}
    for symbol in ('Al', 'Mg'):
        nk = 10 if symbol == 'Al' else 8
        info, rho = calculate(symbol, spacing=.18, kmesh=(nk, nk, nk),
                              basis_level=3, sigma=.005, max_cycle=100)
        results.append(info)
        densities[f'{symbol}_rho'] = rho
        print(json.dumps(info, indent=2))
    out = Path(__file__).resolve().parent
    (out / 'atlas_ks_pyscf_results.json').write_text(json.dumps(results, indent=2) + '\n')
    np.savez_compressed(out / 'atlas_ks_pyscf_densities.npz', **densities)
