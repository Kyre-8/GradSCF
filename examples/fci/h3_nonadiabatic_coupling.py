"""X13: lab-frame full-wavefunction FCI NAC near the H3 conical intersection.

The right derivative of <Psi_I(R_left)|Psi_J(R_right)> includes CI coefficients,
the moving Cholesky frame and native AO cross overlaps. No HF is needed.
Individual NACs at the exact crossing are undefined; the complete two-state
subspace is treated with the existing shared subspace response.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_enable_x64',True)
from gradscf import fci
from h3_nac_model import H3FCI

OUTPUT=Path('reproducibility/ad_showcase/X13_h3_fci_nac')


def run():
    started=perf_counter()
    model=H3FCI()
    identity=jnp.eye(3)
    calculate=jax.jit(model.properties)
    states=jax.jit(model.states)
    rows,points=[],[]
    previous=None
    for scale in (1.,.5,.25,.125,.0625,.03125):
        q=jnp.array([.04,.02])*scale
        coordinates=model.coordinates(q)
        point=calculate(coordinates,identity)
        assert np.all(point['converged']) and np.all(point['response_valid'])
        nac=np.asarray(point['nac'])
        assert np.isfinite(nac).all()
        phase=np.ones(2)
        tracking=1.
        if previous is not None:
            matrix=np.asarray(model.state_overlaps(previous,point))
            np.testing.assert_array_equal(np.argmax(abs(matrix),axis=1),[0,1])
            tracking=float(np.min(abs(np.diag(matrix))))
            assert tracking>.999
            phase=np.sign(np.diag(matrix))
        point=dict(point,coefficients=point['coefficients']*phase[:,None,None],
            nac=point['nac']*phase[:,None,None,None]*phase[None,:,None,None],
            ci_only_partial=point['ci_only_partial']*phase[:,None,None,None]*phase[None,:,None,None])
        previous=point
        points.append(point)
        nac=np.asarray(point['nac'])
        anti=float(np.max(abs(nac+nac.swapaxes(0,1))))
        diagonal=float(max(np.max(abs(nac[0,0])),np.max(abs(nac[1,1]))))
        assert anti<1e-8 and diagonal<1e-8
        spin=np.asarray([fci.spin_square(c,model.space)[0] for c in point['coefficients']])
        np.testing.assert_allclose(spin,.75,atol=1e-9,rtol=0)
        modes=np.einsum('ax,qax->q',nac[0,1],np.asarray(model.modes))
        energy=np.asarray(point['energies'])
        rows.append(dict(q_bohr=np.asarray(q).tolist(),radius_bohr=float(np.linalg.norm(q)),
            coordinates_bohr=np.asarray(coordinates).tolist(),energies_hartree=energy.tolist(),
            gap_hartree=float(energy[1]-energy[0]),spin_square=spin.tolist(),
            coefficients=np.asarray(point['coefficients']).tolist(),orbital_frame=np.asarray(point['frame']).tolist(),
            nac_01_per_bohr=nac[0,1].tolist(),mode_nac_per_bohr=modes.tolist(),
            ci_only_partial_01=np.asarray(point['ci_only_partial'])[0,1].tolist(),
            antihermitian_max_error=anti,diagonal_max_error=diagonal,tracking_min_overlap=tracking,
            max_eigen_residual_hartree=float(np.max(point['residual_norms']))))
        print('q',q,'gap',rows[-1]['gap_hartree'],'|d_Q|',np.linalg.norm(modes),flush=True)

    # A geometry-dependent orbital rotation tests cancellation of CI-frame and
    # AO/frame response, rather than just a constant orbital gauge change.
    anchor=jax.tree.map(jax.lax.stop_gradient,points[1])
    coordinates=anchor['coordinates']
    def rotation(right):
        angle=.3+2*jnp.sum((right-model.center)*model.modes[0])
        c,s=jnp.cos(angle),jnp.sin(angle)
        return jnp.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    gauged=lambda right:model.state_overlaps(anchor,model.states(right,rotation(right)))
    overlap0=np.asarray(jax.jit(gauged)(coordinates))
    np.testing.assert_allclose(abs(overlap0),np.eye(2),atol=1e-10,rtol=0)
    gauge_derivative=np.asarray(jax.jit(jax.jacfwd(gauged))(coordinates))
    gauge_derivative=gauge_derivative*np.sign(np.diag(overlap0))[None,:,None,None]
    gauge_error=float(np.max(abs(gauge_derivative-np.asarray(anchor['nac']))))
    assert gauge_error<1e-8

    # These are covariance tests, not a false assertion that raw lab-frame
    # derivative couplings have zero rigid translation/rotation components.
    a,b=.31,.21
    z=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    y=np.array([[np.cos(b),0.,np.sin(b)],[0.,1.,0.],[-np.sin(b),0.,np.cos(b)]])
    spatial_rotation=y@z
    covariance=[]
    for label,moved,back in (
        ('translation',coordinates+jnp.array([.23,-.17,.11]),np.eye(3)),
        ('rotation',coordinates@spatial_rotation.T,spatial_rotation)):
        current=calculate(moved,identity)
        # Three s AOs rotate/translate with their atoms; coefficient-space
        # overlap fixes only the arbitrary eigenvector signs in this check.
        signs=np.sign(np.einsum('iab,iab->i',np.asarray(anchor['coefficients']),np.asarray(current['coefficients'])))
        transformed=np.asarray(current['nac'])*signs[:,None,None,None]*signs[None,:,None,None]
        transformed=np.einsum('ijax,xy->ijay',transformed,back)
        error=float(np.max(abs(transformed-np.asarray(anchor['nac']))))
        assert error<1e-7
        covariance.append(dict(operation=label,max_nac_error_per_bohr=error))

    # At the crossing only the complete isolated pair is differentiated.
    bundle0=jax.tree.map(jax.lax.stop_gradient,jax.jit(model.bundle)(model.center,identity))
    assert bool(bundle0['converged']) and bool(bundle0['response_valid'])
    transported=lambda right:model.transported_bundle(bundle0,right,identity)
    bundle_values=jax.jit(transported)
    tangents=[]
    bundle_fd=[]
    for mode,direction in enumerate(model.modes):
        _,tangent=jax.jit(lambda r,v:jax.jvp(transported,(r,),(v,)))(model.center,direction)
        assert all(np.isfinite(np.asarray(x)).all() for x in tangent)
        tangents.append(tangent)
        for step in (1e-3,3e-4,1e-4):
            plus,minus=bundle_values(model.center+step*direction),bundle_values(model.center-step*direction)
            errors=[float(np.max(abs((np.asarray(p)-np.asarray(m))/(2*step)-np.asarray(d))))
                    for p,m,d in zip(plus,minus,tangent)]
            bundle_fd.append(dict(mode=mode,step_bohr=step,energy_trace_error=errors[0],
                                  projector_error=errors[1]))
        assert max(errors)<3e-6
    dynamic_bundle=lambda right:model.transported_bundle(bundle0,right,rotation(right))
    _,gauge_tangent=jax.jit(lambda r,v:jax.jvp(dynamic_bundle,(r,),(v,)))(model.center,model.modes[0])
    bundle_gauge_error=max(float(np.max(abs(np.asarray(a)-np.asarray(b))))
                          for a,b in zip(gauge_tangent,tangents[0]))
    assert bundle_gauge_error<1e-8
    gaps=np.asarray([row['gap_hartree'] for row in rows])
    coupling=np.asarray([np.linalg.norm(row['mode_nac_per_bohr']) for row in rows])
    slope=float(np.polyfit(np.log(gaps[-4:]),np.log(coupling[-4:]),1)[0])
    assert -.1 < slope+1 < .1
    data=dict(passed=True,method='full FCI',basis='sto-3g',cart=True,nelec=[2,1],norb=3,
        determinant_count=9,side_bohr=1.8,center_coordinates_bohr=np.asarray(model.center).tolist(),
        distortion_modes=np.asarray(model.modes).tolist(),rows=rows,
        geometry_dependent_orbital_gauge_error=gauge_error,rigid_covariance=covariance,
        near_crossing_log_slope=slope,
        exact_crossing=dict(individual_nac=None,subspace_response_valid=True,
            energy_sum_hartree=float(bundle0['energy_sum']),projection=np.asarray(bundle0['projection']).tolist(),
            orbital_frame=np.asarray(bundle0['frame']).tolist(),
            energy_trace_derivative=[float(t[0]) for t in tangents],
            projector_derivatives=[np.asarray(t[1]).tolist() for t in tangents],
            finite_differences=bundle_fd,orbital_gauge_error=bundle_gauge_error,
            definition='Physical pair projector compressed into the fixed nine-determinant anchor space; not generally idempotent away from the anchor. Full pair only.'),
        scope='Lab-frame BO NAC includes CI, orbital-frame and AO cross-overlap response. No ETF/ERF, HF, production TDDFT NAC, or basis-convergence claim.',
        phase_convention='Accepted neighboring geometries tracked by full wavefunction overlap; signs locally continuous, never differentiated.',
        eigen_tolerance_hartree=1e-12,response_rtol=1e-10,response_atol=1e-12,
        hostname=platform.node(),jax=jax.__version__,backend=jax.default_backend(),dtype='float64',
        elapsed_seconds=perf_counter()-started,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__),Path(__file__).with_name('h3_nac_model.py'),Path(fci.__file__).with_name('overlap.py'))})
    source_root,source_hash=Path(fci.__file__).resolve().parents[2],hashlib.sha256()
    for path in sorted(source_root.rglob('*.py')):
        source_hash.update(path.relative_to(source_root.parent).as_posix().encode())
        source_hash.update(path.read_bytes())
    data['source_python_sha256']=source_hash.hexdigest()
    data['fci_solver_sha256']=hashlib.sha256(Path(fci.__file__).with_name('solver.py').read_bytes()).hexdigest()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'native.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (OUTPUT/'gap_coupling.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['q1_bohr','q2_bohr','radius_bohr','gap_hartree','mode_nac_norm_per_bohr'])
        writer.writerows([*row['q_bohr'],row['radius_bohr'],row['gap_hartree'],norm] for row,norm in zip(rows,coupling))
    print('X13 native PASS; near-crossing log slope',slope,'seconds',data['elapsed_seconds'],flush=True)


if __name__=='__main__':
    run()

# Measured on jy_gpu_1, CPU float64, 2026-10-16 (the JSON records exact hashes):
# gap: 0.0332000448 -> 0.00103894028 Eh; |d_Q|: 11.2333 -> 357.823 bohr^-1.
# Near-crossing log(|d_Q|) / log(gap) slope: -0.999674.
# Full wavefunction, not CI-only: single-point omitted-orbital norm 0.604568.
# Individual NAC is undefined at the exact crossing; complete-pair response
# passes independent PySCF projector FD with <=2.17e-9 bohr^-1 error at h=1e-4.
