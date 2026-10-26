"""Independent PySCF FCI/cross-overlap differences for the X13 lab-frame NAC.

Only geometry, state labels and the native result to be compared are read.
All reference integrals, orbital frames and CI states are recomputed in PySCF.
No GradSCF calculation is imported. No electron translation/rotation factor is
added: rigid contractions are checked against electronic generator matrices.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import cholesky,solve_triangular
import pyscf
from pyscf import ao2mo,fci,gto

FOLDER=Path('reproducibility/ad_showcase/X13_h3_fci_nac')


def solve(coordinates):
    mol=gto.M(atom=list(zip(['H']*3,coordinates)),unit='Bohr',basis='sto-3g',spin=1,cart=True,verbose=0)
    overlap=mol.intor('int1e_ovlp')
    frame=solve_triangular(cholesky(overlap,lower=True).T,np.eye(3),lower=False)
    h=frame.T@(mol.intor('int1e_kin')+mol.intor('int1e_nuc'))@frame
    eri=ao2mo.incore.full(mol.intor('int2e'),frame,compact=False).reshape((3,)*4)
    solver=fci.direct_spin1.FCI(mol)
    solver.nroots,solver.conv_tol,solver.pspace_size=3,1e-12,100
    energy,ci=solver.kernel(h,eri,3,(2,1),ecore=mol.energy_nuc())
    assert np.all(solver.converged)
    spin=np.asarray([fci.spin_op.spin_square(c,3,(2,1))[0] for c in ci])
    np.testing.assert_allclose(spin[:2],.75,atol=1e-9,rtol=0)
    assert energy[2]-energy[1]>.4
    return dict(mol=mol,frame=frame,ci=np.asarray(ci),energies=np.asarray(energy),spin=spin)


def overlaps(left,right):
    matrix=left['frame'].T@gto.intor_cross('int1e_ovlp',left['mol'],right['mol'])@right['frame']
    return np.asarray([[fci.addons.overlap(a,b,3,(2,1),s=matrix)
                         for b in right['ci'][:2]] for a in left['ci'][:2]])


def right_difference(anchor,coordinates,direction,step):
    values=[]
    minimum=1.
    for sign in (1,-1):
        moved=solve(coordinates+sign*step*direction)
        matrix=overlaps(anchor,moved)
        np.testing.assert_array_equal(np.argmax(abs(matrix),axis=1),[0,1])
        minimum=min(minimum,float(np.min(abs(np.diag(matrix)))))
        assert minimum>.999
        values.append(matrix*np.sign(np.diag(matrix))[None,:])
    return (values[0]-values[1])/(2*step),minimum


def run():
    started=perf_counter()
    native=json.loads((FOLDER/'native.json').read_text())
    assert native['passed']
    modes=np.asarray(native['distortion_modes'])
    checks=[]
    cartesian_checks=[]
    generators=[]
    for index,point in enumerate(native['rows']):
        coordinates=np.asarray(point['coordinates_bohr'])
        anchor=solve(coordinates)
        np.testing.assert_allclose(anchor['energies'][:2],point['energies_hartree'],atol=1e-9,rtol=0)
        # Native states enter only to select the same real phase convention.
        shared_map=np.asarray(point['orbital_frame']).T@anchor['mol'].intor('int1e_ovlp')@anchor['frame']
        alignment=np.asarray([[fci.addons.overlap(np.asarray(a),b,3,(2,1),s=shared_map)
                               for b in anchor['ci'][:2]] for a in point['coefficients']])
        np.testing.assert_allclose(abs(alignment),np.eye(2),atol=1e-9,rtol=0)
        anchor['ci'][:2]=anchor['ci'][:2]*np.sign(np.diag(alignment))[:,None,None]
        steps=point['radius_bohr']*np.asarray([1e-2,3e-3,1e-3])
        for mode,direction in enumerate(modes):
            ad=point['mode_nac_per_bohr'][mode]
            rows=[]
            for step in steps:
                derivative,minimum=right_difference(anchor,coordinates,direction,step)
                rows.append(dict(step_bohr=float(step),reference_nac_per_bohr=float(derivative[0,1]),
                    absolute_error=float(abs(derivative[0,1]-ad)),minimum_tracked_overlap=minimum))
            np.testing.assert_allclose(rows[-1]['reference_nac_per_bohr'],ad,atol=1e-5,rtol=2e-5)
            assert rows[-1]['absolute_error']<.05*rows[0]['absolute_error']+1e-7
            checks.append(dict(point=index,mode=mode,finite_differences=rows))
        if index in (1,len(native['rows'])-1):
            ad=np.asarray(point['nac_01_per_bohr'])
            rows=[]
            for step in steps:
                numerical=np.empty((3,3))
                minimum=1.
                for component,direction in enumerate(np.eye(9).reshape(9,3,3)):
                    derivative,tracked=right_difference(anchor,coordinates,direction,step)
                    numerical.ravel()[component]=derivative[0,1]
                    minimum=min(minimum,tracked)
                rows.append(dict(step_bohr=float(step),nac_per_bohr=numerical.tolist(),
                    max_absolute_error=float(np.max(abs(numerical-ad))),
                    max_scaled_error=float(np.max(abs(numerical-ad)/(1+abs(ad)))),
                    minimum_tracked_overlap=minimum))
            np.testing.assert_allclose(numerical,ad,atol=1e-5,rtol=2e-5)
            cartesian_checks.append(dict(point=index,finite_differences=rows))

            # int1e_ipovlp is a derivative on the bra. Integration by parts
            # makes it minus the electronic gradient on the ket, hence the
            # derivative of a translated ket orbital center has this sign.
            ip=anchor['mol'].intor('int1e_ipovlp')
            frame=anchor['frame']
            dm=fci.direct_spin1.trans_rdm1(anchor['ci'][0],anchor['ci'][1],3,(2,1))
            translation=np.einsum('xpq,qp->x',np.einsum('pi,xpq,qj->xij',frame,ip,frame),dm)
            rotational=[]
            contractions=[]
            for axis in np.eye(3):
                direction=np.cross(axis,coordinates)
                # All three functions are s AOs: moving their centers under a
                # rigid rotation gives the full electronic rotation generator.
                ao=np.einsum('xpq,qx->pq',ip,direction)
                rotational.append(float(np.einsum('pq,qp->',frame.T@ao@frame,dm)))
                contractions.append(float(np.sum(ad*direction)))
            np.testing.assert_allclose(ad.sum(axis=0),translation,atol=1e-7,rtol=1e-7)
            np.testing.assert_allclose(contractions,rotational,atol=1e-7,rtol=1e-7)
            generators.append(dict(point=index,translation_nac=ad.sum(axis=0).tolist(),
                electronic_translation_generator=translation.tolist(),rotation_nac=contractions,
                electronic_rotation_generator=rotational,
                translation_max_error=float(np.max(abs(ad.sum(axis=0)-translation))),
                rotation_max_error=float(np.max(abs(np.asarray(contractions)-rotational)))))
        print('point',index,'gap',anchor['energies'][1]-anchor['energies'][0],'reference checks pass',flush=True)

    center=np.asarray(native['center_coordinates_bohr'])
    anchor=solve(center)
    np.testing.assert_allclose(anchor['frame'],native['exact_crossing']['orbital_frame'],atol=1e-11,rtol=0)
    basis=np.eye(9).reshape(9,3,3)
    def transported(coordinates):
        right=solve(coordinates)
        matrix=anchor['frame'].T@gto.intor_cross('int1e_ovlp',anchor['mol'],right['mol'])@right['frame']
        vectors=np.asarray([[fci.addons.overlap(left,c,3,(2,1),s=matrix)
                             for c in right['ci'][:2]] for left in basis])
        return right['energies'][:2].sum(),vectors@vectors.T
    bundle_checks=[]
    for mode,direction in enumerate(modes):
        expected=(native['exact_crossing']['energy_trace_derivative'][mode],
                  np.asarray(native['exact_crossing']['projector_derivatives'][mode]))
        for step in (1e-3,3e-4,1e-4):
            plus,minus=transported(center+step*direction),transported(center-step*direction)
            errors=[float(np.max(abs((p-m)/(2*step)-d))) for p,m,d in zip(plus,minus,expected)]
            bundle_checks.append(dict(mode=mode,step_bohr=step,energy_trace_error=errors[0],projector_error=errors[1]))
        assert max(errors)<3e-6
    data=dict(passed=True,mode_checks=checks,cartesian_checks=cartesian_checks,
        generator_checks=generators,complete_pair_checks=bundle_checks,pyscf=pyscf.__version__,
        elapsed_seconds=perf_counter()-started,native_json_sha256=hashlib.sha256((FOLDER/'native.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gates=dict(nac_atol_per_bohr=1e-5,nac_rtol=2e-5,generator_atol=1e-7,complete_pair_fd_atol=3e-6),
        scope='Independent PySCF integrals/FCI and public addons.overlap + gto.intor_cross, including electronic rigid-motion generators. Exact crossing uses the complete pair, never individual NACs.')
    (FOLDER/'pyscf.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print('X13 independent reference PASS; seconds',data['elapsed_seconds'],flush=True)


if __name__=='__main__':
    run()
