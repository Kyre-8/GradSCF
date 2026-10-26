"""A mass-normalized Si optical displacement and its static Raman derivative.

The initial Gamma/FFT15/SZV model is a resource and derivative probe, not a
converged silicon spectrum. No phonon frequency or absolute cross section is
assigned. The z optical displacement selects the T2g xy Raman component.
"""
import hashlib
import json
import platform
import resource
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants

jax.config.update('jax_enable_x64', True)
from gradscf.pbc import gto, dft
from gradscf.pbc._kpoint import energy_and_fock
from gradscf.pbc.polarization import make_plan, polarization, _projected_frame
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.dfpt.electric import _electric_model, _solve_electric
from gradscf.solvers import LinearSolverConfig
from gradscf.solvers.nonlinear.iterate import NonlinearConfig


OUTPUT = Path('reproducibility/ad_showcase/P06_silicon_raman')
SETTINGS = dict(a_angstrom=5.43, mesh=[15]*3, kmesh=[1]*3, basis='gth-szv',
                pseudo='gth-pbe', xc='pbe', precision=1e-10)
MASSES_AMU = np.array([28.0855, 28.0855])  # Declared model masses, not an isotope claim.
AMU_TO_ME = physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
CONTROLS = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                max_cycle=200, damping=.1)
LINEAR = LinearSolverConfig(method='gmres', rtol=1e-10, atol=1e-12,
                            restart=40, maxiter=100)
NONLINEAR = NonlinearConfig(maxiter=150, residual_tol=1e-11, energy_tol=1e-13, damping=.2)


def mean_field(settings=None):
    settings = SETTINGS if settings is None else settings
    a = settings['a_angstrom']
    lattice = a/2*np.array([[0.,1.,1.], [1.,0.,1.], [1.,1.,0.]])
    # Put the inversion center at the origin, also for the finite FFT grid.
    coordinates = np.array([[-a/8]*3, [a/8]*3])
    cell = gto.M(atom=list(zip(['Si','Si'],coordinates)), a=lattice, unit='Angstrom',
        basis=settings['basis'], pseudo=settings['pseudo'], mesh=tuple(settings['mesh']),
        precision=settings['precision'], cart=True)
    return dft.KRKS(cell, xc=settings['xc'], kpts=cell.make_kpts(tuple(settings['kmesh'])),
                    temperature=None, exxdiv='ewald', **CONTROLS).run()


def optical_displacement():
    """u=dR/dQ, with sum M*u^2=1 and Q in sqrt(me)*bohr."""
    eigenvector = np.array([[0.,0.,1.], [0.,0.,-1.]])/np.sqrt(2.)
    return eigenvector/np.sqrt(MASSES_AMU[:,None]*AMU_TO_ME)


def mixed_problem(mf, *, state=None, reference_phases=None):
    """Keep the reference chart/anchor fixed while all geometry inputs move."""
    cell = mf.cell
    evaluate = lambda d,a,p:energy_and_fock(d,a,mesh=cell.mesh,xc=mf.xc,
        exxdiv=mf.exxdiv,functional=mf.functional,params=p)
    state = mf.result if state is None else state
    model, diagonal = _electric_model(mf.inputs,state,nocc=cell.nelec,
        evaluate=evaluate,params=mf.params,reference_phases=reference_phases)

    def arguments(q, field, displacement, origin=None):
        origin = cell.coords if origin is None else origin
        data = build_inputs(cell,parameters=cell.parameters_at(origin+q*displacement),
                            kpts=mf.kpts,kweights=mf.kweights)
        return data,mf.params,field

    def response(reference_inputs, origin, displacement, field_direction):
        # Large reference AO/grid arrays and the geometry are runtime JIT
        # operands. Closing over them makes LLVM materialize a large constant
        # graph even though the mathematics does not depend on that staging.
        active = replace(model, inputs=reference_inputs)

        def branch(q, field):
            root, values = _solve_electric(active,diagonal,
                arguments(q,field,displacement,origin),config=NONLINEAR,linear_config=LINEAR)
            return root.solution,values['polarization']

        zero = jnp.array(0.)
        field = jnp.zeros(3)
        field_response = lambda q:jax.jvp(lambda e:branch(q,e),
                                          (field,),(field_direction,))[1]
        (x_e,chi),(x_qe,derivative) = jax.jvp(field_response,(zero,),(jnp.array(1.),))
        (x0,_),(x_q,_) = jax.jvp(lambda q:branch(q,field),
                                (zero,),(jnp.array(1.),))
        # Check the actual differentiated projector equation, independently
        # of any forward iteration or the GMRES internal stopping flag.
        x0,x_q,x_e,x_qe = jax.tree.map(jax.lax.stop_gradient,(x0,x_q,x_e,x_qe))
        def residual(q,e):
            x = x0+q*x_q+e*x_e+q*e*x_qe
            return active.residual(x,arguments(q,e*field_direction,displacement,origin))
        r_e = jax.jvp(lambda e:residual(zero,e),(zero,),(jnp.array(1.),))[1]
        r_qe = jax.jvp(lambda q:jax.jvp(lambda e:residual(q,e),
            (zero,),(jnp.array(1.),))[1],(zero,),(jnp.array(1.),))[1]
        return dict(chi_column=chi, derivative=derivative,
            reference_residual=jnp.max(jnp.abs(residual(zero,zero))),
            field_residual=jnp.max(jnp.abs(r_e)), mixed_residual=jnp.max(jnp.abs(r_qe)))
    return model,arguments,partial(jax.jit(response),model.inputs,cell.coords)


def run_probe(name='gamma_fft15_probe', settings=None):
    settings = SETTINGS if settings is None else settings
    started = perf_counter()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    destination = OUTPUT/(name+'.json')
    destination.write_text('{"complete": false}\n')
    mf = mean_field(settings)
    if not mf.converged:
        raise RuntimeError('The silicon reference SCF did not converge.')
    cell = mf.cell
    plan = make_plan(np.asarray(mf.kpts)@np.asarray(cell.lattice).T/(2*np.pi))
    checked = polarization(mf.inputs,mf.result,plan,nocc=cell.nelec)
    grams = [_projected_frame(mf.result.density_spin[s],mf.inputs.overlap,
             mf.result.mo_coeff_spin[s,:,:,:n])[1] for s,n in enumerate(cell.nelec)]
    minimum_gram = min(float(jnp.min(jnp.linalg.eigvalsh(g))) for g in grams)
    if float(checked.minimum_gap) <= 1e-3 or float(jnp.min(checked.minimum_link_singular_value)) <= 1e-4 or minimum_gram <= .5:
        raise RuntimeError('The reference gap, Berry link or local Gram gate failed.')
    displacement = optical_displacement()
    report = dict(complete=False,reference_complete=True,settings=settings,
        energy_hartree=float(mf.e_tot),coordinates_bohr=np.asarray(cell.coords).tolist(),
        lattice_bohr=np.asarray(cell.lattice).tolist(),volume_bohr3=float(cell.volume),
        occupations=list(cell.nelec),pseudopotential_charges=np.asarray(cell.charges).tolist(),
        nao=cell.topology.nao,basis_data=cell.basis_spec,pseudopotential_data=cell.pseudopotentials,
        masses_amu=MASSES_AMU.tolist(),amu_to_electron_mass=AMU_TO_ME,
        displacement_per_Q=displacement.tolist(),
        mass_norm=float(np.sum(MASSES_AMU[:,None]*AMU_TO_ME*displacement**2)),
        gap_hartree=float(checked.minimum_gap),minimum_gram_eigenvalue=minimum_gram,
        minimum_link_singular_value=float(jnp.min(checked.minimum_link_singular_value)),
        phase_anchor=[np.asarray(p).tolist() for p in checked.phases],
        reference_seconds=perf_counter()-started,
        definition='Q in sqrt(me)*bohr; u=e/sqrt(M), sum|e|^2=1. Static dchi[:,y]/dQ for z optical displacement. No phonon frequency assigned.')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print('Si reference',report['energy_hartree'],'nao',report['nao'],'occupations',cell.nelec,
          'gap',report['gap_hartree'],'link',report['minimum_link_singular_value'],flush=True)
    _,_,respond = mixed_problem(mf)
    before = perf_counter()
    result = respond(jnp.asarray(displacement),jnp.array([0.,1.,0.]))
    jax.block_until_ready(result)
    derivative = np.asarray(result['derivative'])
    residuals = {key:float(result[key]) for key in ['reference_residual','field_residual','mixed_residual']}
    passed = (np.isfinite(derivative).all() and np.isfinite(list(residuals.values())).all()
              and max(residuals.values()) < 1e-10)
    report.update(complete=True,probe_passed=bool(passed),
        chi_column=np.asarray(result['chi_column']).tolist(),dchi_column_dQ=derivative.tolist(),
        depsilon_column_dQ=(4*np.pi*derivative).tolist(),
        dcell_alpha_column_dQ=(float(cell.volume)*derivative).tolist(),
        raman_susceptibility_column=(np.sqrt(float(cell.volume))*derivative).tolist(),
        nonzero_xy=bool(abs(derivative[0])>1e-8),**residuals,
        response_seconds=perf_counter()-before,elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(),python=platform.python_version(),jax=jax.__version__,dtype='float64',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        electric_source_sha256=hashlib.sha256(Path('src/gradscf/dfpt/electric.py').read_bytes()).hexdigest(),
        scope='Single-direction finite-model probe at the recorded k/FFT settings. No FD/gauge/translation verdict yet; no material convergence, frequency, absolute cross section or jet claim.')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print('Si mixed column',derivative.tolist(),'residuals',residuals,
          'time',report['elapsed_seconds'],'peak RSS',report['peak_rss_kib'],flush=True)
    if not passed:
        raise RuntimeError('The mixed response failed finite-value or true-residual checks.')
    return report


if __name__ == '__main__':
    run_probe()
