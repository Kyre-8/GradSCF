"""Public SCF reference conversion is shared, cached, and freshness-checked."""
from dataclasses import replace
import numpy as np
import pytest

from gradscf import dft, gto, scf
from gradscf.model.neural_xc import training


@pytest.mark.parametrize('spin', [False, True])
def test_samples_and_prediction_accept_solved_scf_without_resolving_it(monkeypatch, spin):
    from gradscf.scf import builders
    mol=gto.M(atom='H 0 0 0' if spin else 'H 0 0 0; H 0 0 .74',basis='sto-3g',spin=1 if spin else 0)
    mf=(dft.UKS if spin else dft.RKS)(mol,xc='hf').run()
    def forbidden(*args,**kwargs):
        raise AssertionError('Preparing a reference must reuse SCF and integrals')
    for name in ('build_rks_integral_inputs','build_uks_integral_inputs','run_rks_from_integrals','run_uks_from_integrals'):
        monkeypatch.setattr(builders,name,forbidden)
    reference=mf.to_reference()
    assert mf.to_reference() is reference
    assert training.Sample(mf,energy=mf.e_tot).molecule is reference
    np.testing.assert_allclose(reference.rdm1.sum(axis=0),mf.make_rdm1().sum(axis=0) if spin else mf.make_rdm1())
    functional=dft.Functional(lambda x:x.rho,lambda p,x:p['a']*(x*x).sum())
    import jax.numpy as jnp
    trainer=training.Trainer(functional,params={'a':jnp.array(.01)})
    _, predicted=trainer.predict(mf)
    np.testing.assert_allclose(predicted.rdm1,reference.rdm1)
    assert scf.as_reference(mf) is reference
    if spin:
        mf.execution_device='cpu'
        moved=mf.to_reference()
        np.testing.assert_allclose(moved.rdm1,reference.rdm1)


def test_reference_rejects_unsolved_unconverged_and_stale_sources():
    mf=dft.RKS(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g'),xc='hf')
    with pytest.raises(RuntimeError,match='SCF|ground.state'):
        training.Sample(mf,energy=-1.)
    mf.run()
    mf.to_reference()
    mf.converged=False
    with pytest.raises(RuntimeError,match='converge'):
        mf.to_reference()
    mf.converged=True
    mf.mol=replace(mf.mol,atom='H 0 0 0; H 0 0 .9')
    with pytest.raises(RuntimeError,match='changed|kernel'):
        training.Sample(mf,energy=-1.)


def test_package_root_has_only_domain_namespaces():
    import gradscf
    assert all(name[0].islower() and not any(x in name for x in ('_from_','_with_','run_','make_'))
               for name in gradscf.__all__)
    assert not hasattr(gradscf,'restricted_molecule_from_spec_with_jax_rks')
    assert not hasattr(scf,'restricted_molecule_from_spec_with_jax_rks')
    assert not hasattr(dft,'restricted_molecule_from_spec_with_jax_rks')


def test_workflow_compatibility_forwarders_are_not_parallel_entrypoints():
    from gradscf.model.neural_xc.workflows import core, pipeline
    for owner, names in ((core, ('run_pipeline_core_from_spec','run_pipeline_core_from_molecule_spec')),
        (pipeline, ('run_neural_xc_spectrum_pipeline_from_spec',
                    'run_neural_xc_spectrum_pipeline_from_molecule_spec',
                    'run_and_report_from_spec','run_and_report_from_molecule_spec'))):
        assert all(not hasattr(owner,name) for name in names)
    assert callable(core.run_pipeline_core)
    assert callable(pipeline.run_and_report)
    assert callable(pipeline.ExperimentPipeline.run)


def test_unrestricted_gw_rejects_stale_scf_before_evaluating_self_energy(monkeypatch):
    from gradscf import gw
    from gradscf.gw import ugw
    mf=dft.UKS(gto.M(atom='H 0 0 0',basis='sto-3g',spin=1),xc='hf').run()
    calculation=gw.UGW(mf,nw=4)
    def forbidden(*args,**kwargs):
        raise AssertionError('Stale source reached the GW driver')
    monkeypatch.setattr(ugw,'g0w0_cd_unrestricted',forbidden)
    mf.conv_tol_density=2e-8
    with pytest.raises(RuntimeError,match='changed|kernel'):
        calculation.kernel()


@pytest.mark.parametrize('spin', [False, True])
def test_reference_rejects_changed_orbitals_including_in_place_edits(spin):
    mol=gto.M(atom='H 0 0 0' if spin else 'H 0 0 0; H 0 0 .74',basis='sto-3g',spin=int(spin))
    mf=(dft.UKS if spin else dft.RKS)(mol,xc='hf').run()
    reference=mf.to_reference()
    for name in ('mo_coeff','mo_occ','mo_energy'):
        original=getattr(mf,name)
        mutable=np.array(original,copy=True)
        setattr(mf,name,mutable)
        assert mf.to_reference() is reference
        mutable.flat[0] += .01
        with pytest.raises(RuntimeError,match='[Oo]rbital'):
            training.Sample(mf,energy=mf.e_tot)
        setattr(mf,name,original)
        assert mf.to_reference() is reference
