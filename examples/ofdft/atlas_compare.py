"""Compare ATLAS-style TF+lambda*vW solids with two independent implementations.

All examples and reference assets live under examples/ofdft. Run:
  PYTHONPATH=src:/path/to/DFTpy/src JAX_PLATFORMS=cpu python examples/ofdft/atlas_compare.py
The report is written beside this script. No WGC Table 1 agreement is implied.
"""
import json
from pathlib import Path
from time import perf_counter
import platform
import jax
import jax.numpy as jnp
import numpy as np
from ase import units

jax.config.update('jax_enable_x64', True)
from gradscf import ofdft
from atlas_inputs import crystal, gaussian_free_inputs, pz_lda_energy, provenance
from atlas_dftpy import run_reference


def compare(spacings=None, output=None, export_slices=None, export_volumes=None,
            kinetic='tfvw'):
    if kinetic not in ('tfvw', 'wgc'):
        raise ValueError('Expected tfvw or wgc.')
    if spacings is None:
        spacings={'Al':(.24,.18,.14,.10), 'Mg':(.24,.18)}
    report={'reference':'arXiv:1507.07373',
        'scope':'fcc Al / ideal-c/a hcp Mg at fixed reference volumes',
        'kinetic':kinetic,
        'dftpy_kinetic':'native TF/vW' if kinetic=='tfvw' else 'independent NumPy WGC adapter',
        'xc':'unpolarized PZ81 LDA + Dirac exchange',
        'hardware':platform.machine(),'backend':jax.default_backend(),'jax':jax.__version__,
        'pseudopotentials':{s:provenance(s) for s in ('Al','Mg')},'calculations':[]}
    slices=[]
    volumes={}
    default_output='atlas_results.json' if kinetic=='tfvw' else 'atlas_wgc_results.json'
    output=Path(__file__).with_name(default_output) if output is None else Path(output)
    for symbol, targets in spacings.items():
        for spacing in targets:
            atoms,mesh=crystal(symbol,spacing)
            inputs=gaussian_free_inputs(atoms,mesh)
            for weight in ((1.,1/5,1/9) if kinetic=='tfvw' else (1.,)):
                start=perf_counter()
                result=ofdft.run_ofdft(inputs,kinetic=ofdft.KineticFunctional(kinetic),
                    kinetic_params={'vw':weight} if kinetic=='tfvw' else {},xc_energy_fn=pz_lda_energy,
                    config=ofdft.OFDFTConfig(xc=None,tolerance=1e-8,maxiter=1000))
                jax.block_until_ready(result.total_energy)
                gradscf_seconds=perf_counter()-start
                print('GradSCF',symbol,mesh,weight,float(result.total_energy),
                      'residual',float(result.residual_norm),flush=True)
                if not result.converged:
                    raise RuntimeError('GradSCF did not reach the requested stationarity')
                start=perf_counter()
                rho,reference,optimizer,pseudo,enuc=run_reference(atoms,mesh,weight,kinetic=kinetic)
                dftpy_seconds=perf_counter()-start
                n=np.asarray(rho).ravel()
                v=np.asarray(reference.potential).ravel()
                w=np.asarray(inputs.weights)
                number=float(np.sum(w*n))
                mu=np.sum(w*n*v)/number
                ref_residual=2*np.sqrt(number)*np.linalg.norm(np.sqrt(w*n)*(v-mu))
                relative_density=float(np.sqrt(np.sum(w*(np.asarray(result.density)-n)**2)/np.sum(w*n*n)))
                nat=len(atoms)
                row=dict(symbol=symbol,spacing_angstrom=spacing,mesh=list(mesh),lambda_vw=weight,
                    lattice_angstrom=np.asarray(atoms.cell).tolist(),fractional_positions=atoms.get_scaled_positions().tolist(),
                    gradscf_energy_ha_per_atom=float(result.total_energy)/nat,
                    dftpy_energy_ha_per_atom=float(reference.energy)/nat,
                    energy_error_ha_per_atom=abs(float(result.total_energy)-float(reference.energy))/nat,
                    relative_density_l2_error=relative_density,
                    gradscf_residual=float(result.residual_norm),dftpy_residual=ref_residual,
                    gradscf_electrons=float(result.electron_number),dftpy_electrons=number,
                    potential_max_error_ha=float(np.max(np.abs(np.asarray(pseudo.vreal).ravel()-inputs.external_potential))),
                    ewald_error_ha=abs(float(enuc)-float(inputs.nuclear_repulsion)),
                    gradscf_seconds=gradscf_seconds,dftpy_seconds=dftpy_seconds)
                if (export_slices is not None or export_volumes is not None) and spacing==.18:
                    grad_grid=np.asarray(result.density).reshape(mesh)
                    ref_grid=np.asarray(rho).reshape(mesh)
                    tag=symbol+'_'+({1.:'1',.2:'1_5',1/9:'1_9'}[weight] if kinetic=='tfvw' else 'wgc')
                    volumes[tag+'_gradscf']=grad_grid
                    volumes[tag+'_dftpy']=ref_grid
                    volumes[tag+'_lattice']=np.asarray(atoms.cell)
                    slices.append(dict(symbol=symbol,lambda_vw=weight,mesh=list(mesh),
                        spacing_angstrom=spacing,lattice_angstrom=np.asarray(atoms.cell).tolist(),
                        plane='fractional w=0, spanned by primitive a1 and a2',
                        density_units='electron / Bohr^3',
                        gradscf_plane=grad_grid[:,:,0].tolist(),dftpy_plane=ref_grid[:,:,0].tolist(),
                        gradscf_planar_mean=grad_grid.mean(axis=(1,2)).tolist(),
                        dftpy_planar_mean=ref_grid.mean(axis=(1,2)).tolist()))
                report['calculations'].append(row)
                output.write_text(json.dumps(report,indent=2)+'\n')
                print('PAIR',json.dumps(row),flush=True)
                assert optimizer.converged==0
                assert row['energy_error_ha_per_atom']<1e-8,row
                assert relative_density<2e-5,row
                assert ref_residual<5e-5,row
                assert abs(number-float(inputs.nelectron))<1e-9,row
                assert row['potential_max_error_ha']<1e-10 and row['ewald_error_ha']<1e-9,row
    if export_slices is not None:
        Path(export_slices).write_text(json.dumps({'density_slices':slices},indent=2)+'\n')
    if export_volumes is not None:
        np.savez_compressed(export_volumes,**volumes)
    print('All paired calculations passed; report:',output)
    for symbol in ('Al','Mg'):
        for weight in ((1.,1/5,1/9) if kinetic=='tfvw' else (1.,)):
            rows=[r for r in report['calculations'] if r['symbol']==symbol and r['lambda_vw']==weight]
            if len(rows)>1:
                error=abs(rows[-1]['gradscf_energy_ha_per_atom']-rows[-2]['gradscf_energy_ha_per_atom'])*units.Hartree*1000
                print(symbol,'lambda',weight,'mesh energy change / meV per atom:',error)
    return report


if __name__=='__main__':
    compare(export_slices=Path(__file__).with_name('atlas_density_slices.json'),
            export_volumes=Path(__file__).with_name('atlas_density_volumes.npz'))
