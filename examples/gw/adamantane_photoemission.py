"""Native GradSCF adamantane: RHF modes/vertices, G0W0 and Fan/DW PES.

This is a STO-3G prototype, not a reproduction of the plane-wave LDA/GW
literature calculation. All SCF, integral and gradient calculations use
GradSCF. Hessians and linear screened HF vertices use native analytic
derivatives with implicit SCF response. Quadratic vertices use central
differences of maximum-overlap transported Fock matrices.
No experimental energies, couplings or linewidths enter the calculation.
No Franck-Condon/HR convolution or phonon feedback is included.
Tr(A) assumes equal photoemission matrix-element weights. For a finite
molecule, eta is a numerical resolution, not a measured lifetime broadening.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu python -u examples/gw/adamantane_photoemission.py
Checkpoints live outside the repository; this example has no CLI.
"""

from itertools import product
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from gradscf_tools.molecular_ep import NativeRHF
from gradscf_tools.molecular_symmetry import signed_permutation_symmetry, coordinate_orbits
from gradscf.scf.rks import RKSConfig
from gradscf.integrals.molecular.factorization import eri_to_df_factors
from gradscf.gw import g0w0_cd_restricted
from gradscf.gw.ep_coupling import PhononModel, fan_retarded, spectral_function
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update('jax_enable_x64', True)
started = time.perf_counter()
cache = Path(os.environ.get('GRADSCF_ADAMANTANE_CACHE', '/private/tmp/gradscf-adamantane-native/calculation'))
cache.mkdir(parents=True, exist_ok=True)
output = Path(__file__).resolve().parents[2]/'reproducibility/gw_bse/adamantane_native'
output.mkdir(parents=True, exist_ok=True)
symbols = ['C']*10 + ['H']*16
angstrom_to_bohr = 1.8897261254578281
mass_au = np.repeat(np.array([12.]*10 + [1.00782503223]*16)*1822.888486209, 3)
step = .002  # Cartesian Bohr; also maximum Cartesian normal-mode displacement.
settings = dict(basis='sto-3g', cartesian_step_bohr=step, charge=0, spin=0,
                method='RHF-screened parallel-transport Fock vertices + G0W0@RHF',
                scf_tol=1e-11, temperature_k=300., gw_nw=64)
key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
manifest = cache/'settings.json'
if manifest.exists():
    assert json.loads(manifest.read_text()) == settings, 'Checkpoint settings differ'
else:
    manifest.write_text(json.dumps(settings, indent=2)+'\n')


def cage(parameters):
    """Five totally symmetric Td coordinates; atom order is fixed."""
    a,b,c,d,e = parameters
    vertices = np.array([v for v in product((-1,1),repeat=3) if np.prod(v)==1], float)
    axes = np.vstack((np.eye(3),-np.eye(3)))
    coords = [*(vertices*a), *(axes*b), *(vertices*c)]
    directions = np.array(list(product((-1,1),repeat=3)), float)
    for axis in axes:
        neighbors = vertices[np.isclose(np.linalg.norm(vertices-2*axis,axis=1),np.sqrt(3))]
        for direction in directions:
            if np.allclose((neighbors-2*axis)@direction, -1):
                coords.append(axis*d + (direction-axis)*e)
    return np.asarray(coords)


initial = np.array([1.54/np.sqrt(3),2*1.54/np.sqrt(3),
                   (1.54+1.09)/np.sqrt(3),(2*1.54+1.09)/np.sqrt(3),1.09/np.sqrt(3)])*angstrom_to_bohr
geometry_map = np.stack([cage(np.eye(5)[p]) for p in range(5)],axis=-1)
atom = ';'.join(f'{s} {x} {y} {z}' for s,(x,y,z) in zip(symbols,cage(initial)))
engine = NativeRHF(atom, basis=settings['basis'], unit='Bohr',
    config=RKSConfig(xc_spec='hf',max_cycle=100,conv_tol=1e-11,
                    conv_tol_density=1e-10,conv_tol_grad=1e-9))


def checkpoint(path, **arrays):
    temporary = path.with_suffix(f'.tmp.{os.getpid()}.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


if not (cache/'geometry.npz').exists():
    calls = [0]
    def objective(parameters):
        result, gradient = engine.evaluate(cage(parameters))
        calls[0] += 1
        print('geometry', calls[0], float(result.total_energy), 'max_gradient',
              float(np.max(abs(gradient))), flush=True)
        return float(result.total_energy), np.einsum('aip,ai->p',geometry_map,gradient)
    optimized = minimize(objective, initial, jac=True, method='BFGS',
                         options=dict(gtol=2e-6,maxiter=80))
    coords = cage(optimized.x)
    result, gradient = engine.evaluate(coords)
    if np.max(abs(gradient)) > 2e-5:
        raise ArithmeticError('Geometry has not reached the requested HF force threshold')
    checkpoint(cache/'geometry.npz', coordinates=coords, gradient=gradient, parameters=optimized.x)
coords = np.load(cache/'geometry.npz')['coordinates']
if not (cache/'reference.npz').exists():
    result, _ = engine.evaluate(coords)
    checkpoint(cache/'reference.npz', **{name:np.asarray(getattr(result,name)) for name in
        ('mo_energy','mo_coeff','mo_occ','density_matrix','fock_matrix','overlap_matrix',
         'hcore_matrix','total_energy','nuclear_repulsion','converged')})
reference = SimpleNamespace(**dict(np.load(cache/'reference.npz')))
assert bool(reference.converged)
nmo = reference.mo_coeff.shape[1]
print('Optimized reference', float(reference.total_energy), 'nmo',nmo,flush=True)

# Reuse the stationary SCF response for each Cartesian direction. These
# products do not construct an integral-coordinate Jacobian or differentiate
# individual eigenvectors within the degenerate molecular orbital spaces.
response = engine.response(coords, reference)
operations = signed_permutation_symmetry(engine, coords, reference)
leaders, orbit_map = coordinate_orbits(operations)
print('Point group operations',len(operations),'independent directions',leaders,flush=True)
for coordinate in leaders:
    target = cache/f'response_{coordinate:03d}.npz'
    if target.exists():
        continue
    direction = np.zeros(coords.size)
    direction[coordinate] = 1.
    column_started = time.perf_counter()
    hvp, fock_derivative = response(direction.reshape(coords.shape))
    checkpoint(target, hessian_column=np.asarray(hvp).ravel(),
               fock_derivative=np.asarray(fock_derivative),
               wall_seconds=time.perf_counter()-column_started)
    print('Analytic response',coordinate+1,'/',coords.size,'elapsed',time.perf_counter()-started,flush=True)

for coordinate, (leader, operation, sign) in enumerate(orbit_map):
    transform, orbital = operations[operation]
    source = np.load(cache/f'response_{leader:03d}.npz')
    column = sign*transform@source['hessian_column']
    derivative = sign*orbital@source['fock_derivative']@orbital.T
    target = cache/f'response_{coordinate:03d}.npz'
    if target.exists():
        existing = np.load(target)
        np.testing.assert_allclose(existing['hessian_column'],column,atol=2e-6,rtol=0.)
        np.testing.assert_allclose(existing['fock_derivative'],derivative,atol=2e-6,rtol=0.)
    else:
        checkpoint(target,hessian_column=column,fock_derivative=derivative,
                   symmetry_source=leader,symmetry_operation=operation)

hessian = np.column_stack([np.load(cache/f'response_{i:03d}.npz')['hessian_column'] for i in range(coords.size)])
asymmetry = float(np.max(abs(hessian-hessian.T)))
if asymmetry > 2e-6:
    raise ArithmeticError(f'SCF Hessian is not symmetric: max difference={asymmetry}')
hessian = (hessian+hessian.T)*.5
translation_error = np.max(abs(hessian.reshape(coords.size,len(coords),3).sum(axis=1)))
if translation_error > 2e-6:
    raise ArithmeticError(f'Hessian translation sum rule failed: {translation_error}')
center = np.average(coords,axis=0,weights=mass_au.reshape(-1,3)[:,0])
rigid = []
for axis in np.eye(3):
    rigid.append((np.tile(axis,(len(coords),1))*np.sqrt(mass_au).reshape(-1,3)).ravel())
    rigid.append((np.cross(coords-center,axis)*np.sqrt(mass_au).reshape(-1,3)).ravel())
q, _ = np.linalg.qr(np.array(rigid).T, mode='complete')
vibrational = q[:,6:]
dynamic = hessian/np.sqrt(mass_au[:,None]*mass_au[None,:])
squared, vectors = np.linalg.eigh(vibrational.T@dynamic@vibrational)
if np.any(squared <= 0):
    raise ArithmeticError(f'Nonpositive internal vibration: min omega²={squared.min()}')
omega = np.sqrt(squared)
modes = vibrational@vectors
derivative = np.stack([np.load(cache/f'response_{i:03d}.npz')['fock_derivative'] for i in range(coords.size)])
couplings = np.einsum('alm,ak->klm',derivative,modes/np.sqrt(mass_au)[:,None])/np.sqrt(2*omega)[:,None,None]
couplings = (couplings+couplings.swapaxes(-1,-2))*.5
print('Vibrations',len(omega),'range_cm-1',omega[[0,-1]]*219474.6313632,'Hessian asymmetry',asymmetry,flush=True)

# Diagonal quadratic mode vertices define the DW term in this effective
# transported HF Hamiltonian model. Each mode uses a controlled small step.
quadratic_diagonal = []
for mode in range(len(omega)):
    target = cache/f'quadratic_{mode:03d}.npz'
    if not target.exists():
        direction = (modes[:,mode]/np.sqrt(mass_au)).reshape(coords.shape)
        dq = step/np.max(abs(direction))
        matrices = []
        for sign in (1.,-1.):
            displaced = coords+sign*dq*direction
            result,_ = engine.evaluate(displaced,init_density=reference.density_matrix,gradient=False)
            fock,_ = engine.transport(result,reference.mo_coeff,coords,displaced)
            matrices.append(fock)
        second = (matrices[0]+matrices[1]-2*np.diag(reference.mo_energy))/dq**2
        checkpoint(target, diagonal_vertex=second/(2*omega[mode]))
        print('Quadratic mode',mode+1,'/',len(omega),flush=True)
    quadratic_diagonal.append(np.load(target)['diagonal_vertex'])
quadratic_diagonal = np.asarray(quadratic_diagonal)
checkpoint(cache/'phonons.npz', energies=omega,couplings=couplings,quadratic_diagonal=quadratic_diagonal,
           modes=modes,hessian=hessian,coordinates=coords,hessian_asymmetry=asymmetry)

if not (cache/'gw_all.npz').exists():
    print('Building GW factors',flush=True)
    _,_,eri,_ = engine.integrals(coords)
    factors = eri_to_df_factors(jnp.asarray(eri),tol=1e-9)
    print('GW auxiliary rank',factors.shape[0],flush=True)
    result = g0w0_cd_restricted(mo_energy=reference.mo_energy,mo_coeff=reference.mo_coeff,nocc=38,
        df_factors=factors, fock_matrix=reference.fock_matrix,hcore_matrix=reference.hcore_matrix,
        density_matrix=reference.density_matrix,nw=settings['gw_nw'],eta=1e-5,
        orbs=tuple(range(nmo)),resolvent_expansion=True)
    if not bool(result.converged):
        raise ArithmeticError('Valence QP roots are not all converged')
    checkpoint(cache/'gw_all.npz', energies=np.asarray(result.mo_energy),residual=np.asarray(result.qp_residual),
               computed=np.asarray(result.qp_computed_mask),converged=np.asarray(result.converged_mask))
print('Native inputs and GW complete',time.perf_counter()-started,flush=True)

# The frozen GW correction defines an effective QP Hamiltonian. Its value
# uses G0W0 levels, while its first/second nuclear derivatives use screened
# HF vertices. Both Fan internal poles and external levels use the same QP
# energies; all 66 levels, including virtual states, are explicitly computed.
gw_data = np.load(cache/'gw_all.npz')
if not np.all(gw_data['computed'] & gw_data['converged']):
    raise ArithmeticError('The Fan internal spectrum requires all QP levels')
qp = gw_data['energies']
if not (np.all(qp[:38] < 0.) and np.all(qp[38:] > 0.)):
    raise ArithmeticError('mu=0 must lie in the QP gap for this closed-shell spectrum')
active = np.arange(10,38)
beta = 1/(3.166811563e-6*settings['temperature_k'])
dw = .5*np.einsum('lij,l->ij',quadratic_diagonal,1+2/np.expm1(beta*omega))
dw_symmetry_error = max(np.max(abs(orbital@dw@orbital.T-dw)) for _,orbital in operations)
if dw_symmetry_error > 2e-5:
    raise ArithmeticError(f'Quadratic vertices break molecular symmetry: {dw_symmetry_error} Ha')
model = PhononModel(jnp.asarray(omega),jnp.asarray(couplings),reference='Native RHF/STO-3G analytic response')
energy_axis = np.linspace(7.,19.,2401)
spectra = {name+suffix:[] for name in ('gw','fan','fan_dw') for suffix in ('','_homo')}
resolution_spectra = {width:[] for width in (.005,.02)}
eta = .01/HARTREE_TO_EV  # Numerical resolution, common to before/after.
for start in range(0,len(energy_axis),64):
    binding = energy_axis[start:start+64]
    frequency = -jnp.asarray(binding)/HARTREE_TO_EV
    sigma = fan_retarded(jnp.asarray(qp),model,frequency,beta=beta,mu=0.,eta=eta)
    sigma = sigma[:,active[:,None],active[None,:]]
    for name, shift, self_energy in (('gw',0.,jnp.zeros_like(sigma)),('fan',0.,sigma),('fan_dw',jnp.asarray(dw[np.ix_(active,active)]),sigma)):
        spectral = spectral_function(jnp.diag(jnp.asarray(qp[active]))+shift,self_energy,frequency,mu=0.,eta=eta)
        diagonal = np.asarray(jnp.diagonal(spectral,axis1=-2,axis2=-1).real)/HARTREE_TO_EV
        if not np.all(np.isfinite(diagonal)) or diagonal.min() < -1e-9:
            raise ArithmeticError('Nonfinite or negative diagonal spectral density')
        spectra[name].append(diagonal.sum(axis=1))
        spectra[name+'_homo'].append(diagonal[:,-3:].sum(axis=1))
    # A finite molecule has discrete vibronic poles. Check how the displayed
    # HOMO envelope depends on the chosen common resolvent resolution.
    for width in resolution_spectra:
        broadening = width/HARTREE_TO_EV
        sigma_check = fan_retarded(jnp.asarray(qp),model,frequency,beta=beta,mu=0.,eta=broadening)
        sigma_check = sigma_check[:,active[:,None],active[None,:]]
        spectral = spectral_function(jnp.diag(jnp.asarray(qp[active]))+jnp.asarray(dw[np.ix_(active,active)]),
                                     sigma_check,frequency,mu=0.,eta=broadening)
        diagonal = np.asarray(jnp.diagonal(spectral,axis1=-2,axis2=-1).real)/HARTREE_TO_EV
        if not np.all(np.isfinite(diagonal)) or diagonal.min() < -1e-9:
            raise ArithmeticError('Resolution check produced invalid spectral density')
        resolution_spectra[width].append(diagonal[:,-3:].sum(axis=1))
for name in spectra:
    spectra[name] = np.concatenate(spectra[name])
np.savetxt(output/'spectra.csv',np.column_stack((energy_axis,*spectra.values())),delimiter=',',
           header='binding_energy_ev,'+','.join(spectra),comments='')
np.savetxt(output/'resolution_check.csv',np.column_stack((energy_axis,
    np.concatenate(resolution_spectra[.005]),spectra['fan_dw_homo'],np.concatenate(resolution_spectra[.02]))),
    delimiter=',',header='binding_energy_ev,eta_5_mev,eta_10_mev,eta_20_mev',comments='')
np.savetxt(output/'mode_energies.csv',np.column_stack((np.arange(1,len(omega)+1),omega*HARTREE_TO_EV*1000,omega*219474.6313632)),
           delimiter=',',header='mode,energy_mev,wavenumber_cm-1',comments='')
checkpoint(output/'phonon_inputs.npz',energies=omega,couplings=couplings,quadratic_diagonal=quadratic_diagonal,
           hf_energies=reference.mo_energy,qp_energies=qp,mo_coeff=reference.mo_coeff,coordinates=coords,
           modes=modes,hessian=hessian,masses_au=mass_au.reshape(-1,3)[:,0],symbols=np.asarray(symbols))
(output/'optimized.xyz').write_text('26\nGradSCF native RHF/STO-3G optimized Td cage\n'+''.join(
    f'{s} {x:.10f} {y:.10f} {z:.10f}\n' for s,(x,y,z) in zip(symbols,coords/angstrom_to_bohr)))
summary = dict(settings,settings_sha256=key,backend=jax.default_backend(),jax_version=jax.__version__,
    platform=platform.platform(),machine=platform.machine(),dtype='float64',
    command='PYTHONPATH=src JAX_PLATFORMS=cpu python -u examples/gw/adamantane_photoemission.py',
    invocation_wall_seconds=time.perf_counter()-started,hf_energy_ha=float(reference.total_energy),
    hessian_method='Native analytic integral Hessian plus implicit SCF response',
    symmetry_operations=len(operations),independent_cartesian_directions=list(map(int,leaders)),
    direct_response_seconds={str(i):float(np.load(cache/f'response_{i:03d}.npz')['wall_seconds']) for i in leaders},
    spectral_eta_ev=.01,spectral_range_ev=[float(energy_axis[0]),float(energy_axis[-1])],
    max_geometry_gradient_ha_bohr=float(np.max(abs(np.load(cache/'geometry.npz')['gradient']))),
    hessian_asymmetry_ha_bohr2=asymmetry,minimum_frequency_cm1=float(omega.min()*219474.6313632),
    hessian_translation_error_ha_bohr2=float(translation_error),
    maximum_frequency_cm1=float(omega.max()*219474.6313632),qp_max_residual_ha=float(abs(gw_data['residual']).max()),
    qp_orbital_count=int(gw_data['computed'].sum()),fan_internal_poles='G0W0, all MOs',
    hf_homo_ip_ev=float(-reference.mo_energy[37]*HARTREE_TO_EV),gw_homo_ip_ev=float(-qp[37]*HARTREE_TO_EV),
    dw_diagonal_ev=(np.diag(dw)[active]*HARTREE_TO_EV).tolist(),
    dw_symmetry_error_ev=float(dw_symmetry_error*HARTREE_TO_EV),
    limitations='Minimal basis; HF-screened vertices and harmonic modes; frozen GW correction under nuclear displacements; unit-weight G0W0 poles; equal photoemission matrix-element weights; eta is numerical resolution, not a lifetime; no HR/multiphonon convolution, no phonon feedback, no fitted experimental shift or coupling.')
(output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
