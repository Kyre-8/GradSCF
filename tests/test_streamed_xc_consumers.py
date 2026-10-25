"""Secondary molecular consumers must preserve the AO-grid provider."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.basis import CartesianAO, CartesianBasis
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.scf import rks, uks


@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('channel', ['hfx', 'pt2'])
def test_streamed_reference_rejects_legacy_full_ao_feature_caches(monkeypatch, unrestricted, channel):
    from gradscf import gto
    from gradscf.scf import builders
    from gradscf.scf.inputs import build_rks_integral_inputs, build_uks_integral_inputs

    spec = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g').to_spec()
    config = (uks.UKSConfig if unrestricted else rks.RKSConfig)(
        xc_spec='pbe', grid_block_size=3)
    assemble = build_uks_integral_inputs if unrestricted else build_rks_integral_inputs
    data = assemble(atom=spec, basis='sto-3g', config=config,
                    integral_backend='jax', grids_level=0)

    def reject_unnecessary_scf(**kwargs):
        raise AssertionError('Reject incompatible full-AO feature caches before SCF')

    monkeypatch.setattr(builders, 'run_uks_from_integrals' if unrestricted else
                        'run_rks_from_integrals', reject_unnecessary_scf)
    build = builders._build_unrestricted_reference if unrestricted else builders._build_restricted_reference
    options = {'uks_config' if unrestricted else 'rks_config': config,
               'compute_local_hfx_features' if channel == 'hfx' else 'compute_local_pt2_features': True}
    if unrestricted:
        # UKS builds its inputs internally; only the RKS builder accepts an
        # already assembled scf_inputs keyword.
        monkeypatch.setattr(builders, 'build_uks_integral_inputs', lambda **kwargs: data)
    else:
        options['scf_inputs'] = data
    with pytest.raises(NotImplementedError, match='dense|grid_block_size'):
        build(atom=spec, basis='sto-3g', **options)


def inputs(streamed):
    basis = CartesianBasis(aos=(
        CartesianAO(center=jnp.zeros(3), angular=(0, 0, 0),
                    exponents=jnp.asarray([.7]), coefficients=jnp.asarray([1.])),
        CartesianAO(center=jnp.asarray([.6, .1, -.2]), angular=(0, 0, 0),
                    exponents=jnp.asarray([1.1]), coefficients=jnp.asarray([.8])),
    ), precompute_eri_groups=False)
    coords = jnp.asarray([[.1, .2, .3], [.4, -.3, .2], [-.2, .5, .1]])
    weights = jnp.asarray([.2, .3, .4])
    provider = AOGrid.from_basis(basis, coords, weights, block_size=2)
    ao, derivative, _, _, _ = provider.block(0, 3, deriv=1)
    return dict(overlap=jnp.eye(2), hcore=jnp.diag(jnp.asarray([-1., .5])),
                eri=jnp.zeros((2, 2, 2, 2)), nuclear_repulsion=jnp.asarray(.2),
                ao=jnp.zeros((0, 2)) if streamed else ao,
                ao_deriv1=jnp.zeros((4, 0, 2)) if streamed else derivative,
                grid_weights=weights, grid_ao=provider if streamed else None)


@pytest.mark.parametrize('method', ['uks', 'roks'])
def test_orbital_objective_preserves_grid_provider(method):
    from gradscf.scf.orbital_optimization import _problem_functions

    _, _, _, evaluate, _ = _problem_functions(method, 'pbe', 'col', ((1., 0.), (1., 0.)))
    density = jnp.asarray([[[.7, .04], [.04, .3]], [[.5, -.02], [-.02, .2]]])
    results = []
    for streamed in (False, True):
        args = inputs(streamed)
        potential = jnp.asarray([[.1, -.02], [-.02, .03]])
        results.append(jax.jit(lambda dm: (
            *evaluate(dm, args),
            jax.jvp(lambda d: evaluate(d, args)[1], (dm,), (jnp.stack([potential, potential]),))[1]
        ))(density))
    for actual, expected in zip(results[1], results[0], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


def test_unrestricted_stability_problem_preserves_grid_provider():
    from gradscf.scf.stability import _unrestricted_problem

    coeff = jnp.stack([jnp.eye(2), jnp.eye(2)])
    occupations = np.asarray([[1., 0.], [1., 0.]])
    results = []
    for streamed in (False, True):
        data = inputs(streamed)
        data.pop('overlap')
        dimension, _, energy, args, _ = _unrestricted_problem(
            coeff, occupations, config=uks.UKSConfig(xc_spec='pbe'), **data)
        origin = jnp.zeros(dimension)
        function = lambda angle: energy(angle, coeff, args)
        results.append((function(origin), jax.grad(function)(origin),
                        jax.jvp(jax.grad(function), (origin,), (jnp.ones(dimension),))[1]))
    for actual, expected in zip(results[1], results[0], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)


def test_roks_solver_preserves_grid_provider():
    from gradscf.scf.roks import ROKSConfig, run_roks_from_integrals

    results = [run_roks_from_integrals(**inputs(streamed), nalpha=1, nbeta=1,
                 config=ROKSConfig(xc_spec='pbe', max_cycle=2)) for streamed in (False, True)]
    np.testing.assert_allclose(results[1].total_energy, results[0].total_energy, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(results[1].fock_matrix, results[0].fock_matrix, rtol=2e-10, atol=2e-11)


def test_dfpt_adapter_preserves_grid_provider(monkeypatch):
    from gradscf.dfpt import api
    from gradscf.scf.facade import RKS

    data = inputs(True)
    provider = data['grid_ao']
    data.update(eri_pair_matrix=None, df_factors=None, direct_basis=None, nelectron=2)
    source = object.__new__(RKS)
    source._scf_inputs = SimpleNamespace(**data)
    source.mo_coeff, source.converged = jnp.eye(2), True
    source.to_reference = lambda: None
    source._is_external_xc = lambda: False
    source._config = lambda: rks.RKSConfig(xc_spec='pbe')
    captured = {}

    class Captured(Exception):
        pass

    def capture(coefficients, nocc, arguments, fock, energy, **kwargs):
        captured.update(arguments=arguments, fock=fock, energy=energy)
        raise Captured

    monkeypatch.setattr(api, 'ks_problem', capture)
    with pytest.raises(Captured):
        api._molecular(source, 1e-7)
    assert captured['arguments']['grid_ao'] is provider
    dm = jnp.asarray([[[1.2, .1], [.1, .3]]])
    streamed_energy = captured['energy'](dm, captured['arguments'])
    dense = inputs(False)
    dense.update(eri_pair_matrix=None, df_factors=None)
    expected_energy = captured['energy'](dm, dense)
    np.testing.assert_allclose(streamed_energy, expected_energy, rtol=2e-12, atol=2e-12)


def test_gks_facade_keeps_explicit_dense_assembly(monkeypatch):
    from gradscf import gto
    from gradscf.scf import gks

    class Captured(Exception):
        pass

    def capture(**kwargs):
        assert kwargs['config'].grid_block_size == -1
        raise Captured

    monkeypatch.setattr(gks, 'build_uks_integral_inputs', capture)
    method = gks.GKS(gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g'), xc='pbe')
    with pytest.raises(Captured):
        method.kernel()
    method.grid_block_size = 2
    with pytest.raises(NotImplementedError, match='(?i)grid|stream'):
        method.kernel()


def test_bound_uks_global_grid_callback_preserves_provider():
    from gradscf.dft.functional import DensityInputs

    class BoundXC:
        """Legacy protocol with both GGA fields and a whole-grid coupling."""

        def energy_from_molecule(self, molecule):
            fields = DensityInputs(molecule.rdm1, molecule)
            ra, rb = fields.rho_spin
            ga, gb = fields.grad_rho_spin
            qa, qb = fields.rho_spin @ fields.weights
            return jnp.dot(fields.weights, .13 * ra**2 + .17 * rb**2
                           + .02 * jnp.sum(ga**2, axis=1) + .03 * jnp.sum(gb**2, axis=1)) \
                + .03 * qa**2 + .04 * qb**2

        def unrestricted_scf_components(self, molecule):
            fields = DensityInputs(molecule.rdm1, molecule)
            assert fields.weights.shape == (3,)
            ra, rb = fields.rho_spin
            ga, gb = fields.grad_rho_spin
            qa, qb = fields.rho_spin @ fields.weights
            return (.26 * ra + .06 * qa, .34 * rb + .08 * qb,
                    .04 * ga, .06 * gb, 'GGA', jnp.asarray(0.),
                    jnp.zeros((2, 2)), jnp.zeros((2, 2)))

    density = jnp.asarray([[[.7, .04], [.04, .3]], [[.5, -.02], [-.02, .2]]])
    direction = jnp.asarray([[[.1, -.02], [-.02, -.04]], [[-.03, .01], [.01, .02]]])
    results = []
    for streamed in (False, True):
        data = inputs(streamed)

        def evaluate(dm):
            energy, _, fa, fb = uks._raw_fock_and_energy_for_state(
                density_a=dm[0], density_b=dm[1], ao=data['ao'], ao_deriv1=data['ao_deriv1'],
                weights=data['grid_weights'], grid_ao=data['grid_ao'], h=data['hcore'],
                eri=data['eri'], df_factors=None, enuc=data['nuclear_repulsion'], alpha=None,
                cfg=uks.UKSConfig(xc_spec='pbe'), xc_kind=None, overlap=data['overlap'],
                bound_xc=BoundXC(), mo_coeff_a=jnp.eye(2), mo_coeff_b=jnp.eye(2),
                mo_occ_a=jnp.asarray([1., 0.]), mo_occ_b=jnp.asarray([1., 0.]),
                mo_energy_a=jnp.asarray([-1., .5]), mo_energy_b=jnp.asarray([-1., .5]))
            return energy, jnp.stack([fa, fb])

        results.append((*evaluate(density), jax.jvp(lambda dm: evaluate(dm)[1],
            (density,), (direction,))[1]))
    for actual, expected in zip(results[1], results[0], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
