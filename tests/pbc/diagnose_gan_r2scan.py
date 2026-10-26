"""Fixed-DM component diagnostic for the recorded compact GaN model; no SCF.

Run explicitly on the approved remote CPU pool. Numerical production paths
are imported from src; PySCF independently supplies the comparison arrays.
Outputs belong to ignored artifacts, never to the public result directory.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


def difference(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    delta = a-b
    return dict(max_abs=float(np.max(abs(delta))), rms=float(np.sqrt(np.mean(abs(delta)**2))),
                reference_max_abs=float(np.max(abs(b))))


def fields_from_pyscf(rho):
    """PySCF (spin,rho/gradient/tau,grid) -> existing ten-field SRC layout."""
    assert rho.ndim == 3 and rho.shape[:2] == (2, 5)
    return np.column_stack([rho[0, 0], rho[1, 0], rho[0, 1:4].T,
                            rho[1, 1:4].T, rho[0, 4], rho[1, 4]])


def run(output='artifacts/ad_showcase/development/completion_20261016/gan_components'):
    from gradscf.pbc import gto
    from gradscf.integrals.periodic.fft import build_inputs, get_kpoint_jk
    from gradscf.pbc._kpoint import grid_xc_variables, grid_xc_potential, project_xc_potential
    from gradscf.scf.uks import _point_unrestricted_xc_value_and_grad_kernel
    import pyscf
    from pyscf.pbc import gto as pgto, dft as pdft
    from pyscf.pbc.dft.gen_grid import UniformGrids

    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    symbols = ('Ga', 'N', 'Ga', 'N')
    coordinates = np.array([[0., 0., 0.], [-1.5006514549516698e-17, 1.8411700084457163, .637755],
        [-1.5006514549516698e-17, 1.8411700084457163, 2.5925], [0., 0., 3.2302549999999997]])
    lattice = np.array([[3.189, 0., 0.], [-1.5945, 2.7617550126685746, 0.], [0., 0., 5.185]])
    settings = dict(atom=list(zip(symbols, coordinates)), a=lattice, unit='Angstrom',
                    mesh=(21, 21, 33), cart=True, precision=1e-10)
    report = dict(complete=False, no_scf=True, model='gan_r2scan_compact', xc='r2scan',
        mesh=list(settings['mesh']), kmesh=[2, 1, 1], basis='gth-dzvp-molopt-sr', pseudo='gth-pbe',
        coordinates_angstrom=coordinates.tolist(), lattice_angstrom=lattice.tolist(),
        dm_definition='D_total(k)=(Ne/nao)*inverse(S_PySCF(k)); D_alpha=D_beta=D_total/2',
        hostname=platform.node(), jax=jax.__version__, pyscf=pyscf.__version__, backend=jax.default_backend())
    def save(stage):
        report.update(stage=stage, elapsed_seconds=perf_counter()-started)
        (output/'components.json').write_text(json.dumps(report, indent=2)+'\n')
        print(stage, flush=True)

    cell = gto.M(**settings, basis='gth-dzvp-molopt-sr', pseudo='gth-pbe')
    kpoints = np.asarray(cell.make_kpts((2, 1, 1)))
    # Share physical Bohr coordinates exactly; avoid differences between the
    # packages' Angstrom conversion constants obscuring component attribution.
    reference_settings = dict(settings, atom=list(zip(symbols, np.asarray(cell.coords))),
                              a=np.asarray(cell.lattice), unit='Bohr')
    pc = pgto.M(**reference_settings, basis=cell.basis_spec,
        pseudo=dict(zip(symbols, cell.pseudopotentials)), verbose=0)
    pmf = pdft.KRKS(pc, kpts=kpoints)
    pmf.xc, pmf.max_memory = 'r2scan', 1000
    pmf.grids = UniformGrids(pc)
    pmf.grids.mesh = pmf.with_df.mesh = np.asarray(settings['mesh'])
    pmf.grids.build()
    assert cell.topology.nao == pc.nao == 88 and cell.nelectron == pc.nelectron == 36
    overlap_ref = pmf.get_ovlp()
    hcore_ref = pmf.get_hcore()
    kinetic_ref = pc.pbc_intor('int1e_kin', hermi=1, kpts=kpoints)
    density = (pc.nelectron/pc.nao)*np.linalg.inv(overlap_ref)
    density_spin = np.stack([density/2, density/2])
    report['minimum_dm_eigenvalue'] = float(np.linalg.eigvalsh(density).min())
    assert report['minimum_dm_eigenvalue'] > 0.
    report['electron_count_metric_pyscf'] = float(np.einsum('kij,kji->', density, overlap_ref).real/len(kpoints))
    np.testing.assert_allclose(report['electron_count_metric_pyscf'], pc.nelectron, atol=1e-10)
    report['fractional_kpoints'] = (kpoints@np.asarray(cell.lattice).T/(2*np.pi)).tolist()
    save('pyscf_fixed_density_ready')

    data = build_inputs(cell, kpts=kpoints)
    jax.block_until_ready(data)
    # PySCF wraps grid coordinates into negative representatives. The two
    # arrays describe the same ordered periodic points, modulo lattice rows.
    grid_shift = (np.asarray(data.coords)-pmf.grids.coords)@np.linalg.inv(np.asarray(cell.lattice))
    np.testing.assert_allclose(grid_shift, np.rint(grid_shift), atol=1e-12, rtol=0.)
    report['grid_representative_error'] = float(np.max(abs(grid_shift-np.rint(grid_shift))))
    np.testing.assert_allclose(data.weights, pmf.grids.weights, atol=1e-12, rtol=0.)
    report['integrals'] = dict(overlap=difference(data.overlap, overlap_ref),
        kinetic=difference(data.kinetic, kinetic_ref), hcore=difference(data.hcore, hcore_ref),
        pseudopotential=difference(data.hcore-data.kinetic, hcore_ref-kinetic_ref),
        nuclear_repulsion=dict(native=float(data.nuclear_repulsion), pyscf=float(pc.energy_nuc()),
            signed_error=float(data.nuclear_repulsion)-float(pc.energy_nuc())))
    report['electron_count_metric_native'] = float(np.einsum('kij,kji->', density, data.overlap).real/len(kpoints))
    save('integrals_compared')

    hartree, _ = get_kpoint_jk(data, jnp.asarray(density_spin), mesh=cell.mesh, with_k=False)
    hartree_ref = pmf.get_j(dm_kpts=density, kpts=kpoints)
    report['hartree'] = difference(hartree, hartree_ref)
    trace = lambda d, matrix: float(np.einsum('kij,kji->', d, matrix).real/len(kpoints))
    report['energy_components'] = dict(one_body_native=trace(density, data.hcore),
        one_body_pyscf=trace(density, hcore_ref), hartree_native=.5*trace(density, hartree),
        hartree_pyscf=.5*trace(density, hartree_ref))
    save('hartree_compared')

    ni = pmf._numint
    ao_ref = np.asarray(ni.eval_ao(pc, pmf.grids.coords, kpts=kpoints, deriv=1))
    rho_ref = np.asarray([ni.eval_rho(pc, ao_ref, d, xctype='MGGA', hermi=1, with_lapl=False)
                          for d in density_spin])
    fields_ref = fields_from_pyscf(rho_ref)
    fields = np.asarray(grid_xc_variables(jnp.asarray(density_spin), data, 'MGGA'))
    phase = np.exp(-1j*kpoints@pmf.grids.coords.T)
    ao_periodic = ao_ref*phase[:, None, :, None]
    report['ao'] = dict(values=difference(data.ao[:, 0], ao_periodic[:, 0]),
                        gradients=difference(data.ao[:, 1:4], ao_periodic[:, 1:4]))
    selected = {'rho': slice(0, 2), 'gradient': slice(2, 8), 'tau': slice(8, 10)}
    report['xc_inputs'] = {name: difference(fields[:, part], fields_ref[:, part]) for name, part in selected.items()}
    reference_ao_inputs = data._replace(ao=jnp.asarray(ao_periodic))
    identical_ao_fields = grid_xc_variables(jnp.asarray(density_spin), reference_ao_inputs, 'MGGA')
    report['same_ao_xc_inputs'] = {name: difference(identical_ao_fields[:, part], fields_ref[:, part])
                                    for name, part in selected.items()}
    report['electron_count_grid_native'] = float(np.asarray(data.weights)@fields[:, :2].sum(axis=1))
    report['electron_count_grid_pyscf'] = float(pmf.grids.weights@fields_ref[:, :2].sum(axis=1))
    save('xc_fields_compared')

    exc, point_gradient = grid_xc_potential(jnp.asarray(density_spin), data, xc='r2scan')
    vxc = project_xc_potential(data.ao, data.weights, point_gradient, 'MGGA')
    count_ref, exc_ref, vxc_ref = ni.nr_uks(pc, pmf.grids, 'r2scan', density_spin,
                                          hermi=1, kpts=kpoints, max_memory=1000)
    report['xc'] = dict(energy_native=float(exc), energy_pyscf=float(exc_ref),
                       energy_signed_error=float(exc)-float(exc_ref), potential=difference(vxc, vxc_ref))
    # Feed exactly the same rho/gradient/tau into each functional implementation,
    # then project on the same analytic PySCF AOs to isolate formulas from grids.
    same_energy, same_gradient = _point_unrestricted_xc_value_and_grad_kernel('r2scan', 'MGGA')(jnp.asarray(fields_ref))
    xc_eff = ni.eval_xc_eff('r2scan', rho_ref, deriv=1, xctype='MGGA', spin=1)
    energy_density_ref = np.asarray(xc_eff[0])*rho_ref[:, 0].sum(axis=0)
    gradient_ref = fields_from_pyscf(np.asarray(xc_eff[1]))
    same_potential = project_xc_potential(reference_ao_inputs.ao, data.weights, same_gradient, 'MGGA')
    report['same_fields_xc'] = dict(energy_density=difference(same_energy, energy_density_ref),
        energy_signed_error=float(np.asarray(data.weights)@(np.asarray(same_energy)-energy_density_ref)),
        point_gradient={name:difference(same_gradient[:, part], gradient_ref[:, part]) for name, part in selected.items()},
        potential_same_ao=difference(same_potential, vxc_ref),
        libxc_point_vs_nruks_energy=float(pmf.grids.weights@energy_density_ref)-float(exc_ref))
    report['energy_components'].update(xc_native=float(exc), xc_pyscf=float(exc_ref))
    files = ['src/gradscf/pbc/_kpoint.py', 'src/gradscf/integrals/periodic/fft.py',
             'src/gradscf/integrals/periodic/ao.py', 'src/gradscf/integrals/periodic/pseudo.py',
             'src/gradscf/scf/uks.py', 'src/gradscf/dft/libxc_jax/jax_xc_adapter.py', __file__]
    report['source_sha256'] = {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    report['complete'] = True
    report['scope'] = 'Diagnostic completion only; no parity gate or SCF convergence is asserted.'
    np.savez_compressed(output/'components.npz', density=density, overlap_native=data.overlap,
        overlap_pyscf=overlap_ref, hcore_native=data.hcore, hcore_pyscf=hcore_ref,
        hartree_native=hartree, hartree_pyscf=hartree_ref, fields_native=fields, fields_pyscf=fields_ref,
        vxc_native=vxc, vxc_pyscf=vxc_ref, vxc_same_fields_same_ao=same_potential)
    save('complete')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run()
